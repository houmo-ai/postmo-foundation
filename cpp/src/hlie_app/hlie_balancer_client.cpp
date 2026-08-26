#include "hlie_app/hlie_balancer_client.h"

#include <chrono>

#include "hlie_base/hlie_logger.h"
#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

HLIE_NAMESPACE_BEGIN

HLIEBalancerClient::~HLIEBalancerClient() {
    stop();
}

void HLIEBalancerClient::start(const HLIEBalancerConfig& config, const int local_port, const std::string& instance_id) {
    stop();
    if (!config.enabled) {
        return;
    }

    config_ = config;
    local_port_ = local_port;
    instance_id_ = instance_id.empty() ? "0" : instance_id;
    management_ = parse_management_url(config.management_url);
    registered_once_ = false;
    stop_.store(false);

    heartbeat_thread_ = std::make_unique<std::thread>(&HLIEBalancerClient::heartbeat_loop, this);
}

void HLIEBalancerClient::stop() {
    stop_.store(true);
    if (heartbeat_thread_ && heartbeat_thread_->joinable()) {
        heartbeat_thread_->join();
    }
    heartbeat_thread_.reset();
}

HLIEBalancerClient::ParsedUrl HLIEBalancerClient::parse_management_url(const std::string& url) {
    ParsedUrl parsed;
    std::string rest = url;
    if (const auto scheme_pos = rest.find("://"); scheme_pos != std::string::npos) {
        parsed.scheme = rest.substr(0, scheme_pos);
        rest = rest.substr(scheme_pos + 3);
    }

    if (const auto path_pos = rest.find('/'); path_pos != std::string::npos) {
        rest = rest.substr(0, path_pos);
    }

    if (const auto port_pos = rest.rfind(':'); port_pos != std::string::npos) {
        parsed.host = rest.substr(0, port_pos);
        try {
            parsed.port = std::stoi(rest.substr(port_pos + 1));
        } catch (...) {
            parsed.port = 8085;
        }
    } else if (!rest.empty()) {
        parsed.host = rest;
    }

    if (parsed.host.empty()) {
        parsed.host = "127.0.0.1";
    }
    return parsed;
}

bool HLIEBalancerClient::register_once() {
    if (management_.scheme != "http") {
        LOG_ERROR("hibalancer registration only supports http management_url currently: {}", config_.management_url);
        return false;
    }

    httplib::Client client(management_.host, management_.port);
    client.set_connection_timeout(2, 0);
    client.set_read_timeout(2, 0);
    client.set_write_timeout(2, 0);

    const std::string target_id = "hi-cpp-" + config_.target_host + "-" + std::to_string(local_port_);
    const json body = {
        {"llama_cpp_slots_aggregated_status", {
            {"status", "ok"},
            {"slots_idle", 1},
            {"slots_processing", 0},
        }},
        {"llama_cpp_target_configuration", {
            {"Id", target_id},
            {"Name", "hi.cpp-" + instance_id_},
            {"llama_cpp_configuration", {
                {"http_address", {
                    {"scheme", "http"},
                    {"host", config_.target_host},
                    {"port", local_port_},
                }},
            }},
        }},
        {"paddler_agent_name", "hi.cpp-" + instance_id_},
    };

    const auto response = client.Post("/register/target", body.dump(), "application/json");
    if (!response) {
        LOG_WARN("hibalancer registration failed: target_id={}, management={}:{}", target_id, management_.host, management_.port);
        return false;
    }
    if (response->status != 200) {
        LOG_WARN("hibalancer registration returned status {}: target_id={}, body={}", response->status, target_id, response->body);
        return false;
    }

    if (!registered_once_) {
        LOG_INFO("hibalancer registration succeeded: target_id={}, management={}:{}, port={}",
                 target_id, management_.host, management_.port, local_port_);
        registered_once_ = true;
    } else {
        LOG_DEBUG("hibalancer registration heartbeat sent: target_id={}, port={}", target_id, local_port_);
    }
    return true;
}

void HLIEBalancerClient::heartbeat_loop() {
    LOG_INFO("hibalancer heartbeat starting: management_url={}, target_host={}, local_port={}, interval_ms={}",
             config_.management_url, config_.target_host, local_port_, config_.heartbeat_interval_ms);

    while (!stop_.load()) {
        register_once();
        const auto sleep_step = std::chrono::milliseconds(100);
        int slept_ms = 0;
        while (!stop_.load() && slept_ms < config_.heartbeat_interval_ms) {
            std::this_thread::sleep_for(sleep_step);
            slept_ms += static_cast<int>(sleep_step.count());
        }
    }

    LOG_INFO("hibalancer heartbeat stopped: local_port={}", local_port_);
}

HLIE_NAMESPACE_END

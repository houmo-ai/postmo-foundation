#include "hlie_app/hlie_web_service.h"

#include <string>

#include "hlie_app/hlie_workflow.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_exception.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_singleton.h"
#include "hlie_base/hlie_util.h"
#include "json.hpp"
#include "utils.hpp"

HLIE_NAMESPACE_BEGIN

namespace {
    std::string process_body(const std::string& body) {
        try {
            if(body.empty()) {
                return body;
            }
            auto body_json = nlohmann::json::parse(body);
            // request
            if(body_json.contains("messages") && body_json["messages"].is_array() && !body_json["messages"].empty()) {
                for(auto& message: body_json["messages"]) {
                    if(message.contains("content") && message["content"].is_array() && !message["content"].empty()) {
                        for (auto& content_item : message["content"]) {
                            if (content_item.contains("type") && content_item["type"] == "image_url" &&
                            content_item.contains("image_url") && content_item["image_url"].contains("url")){
                                std::string b64_json_data = content_item["image_url"]["url"];
                                LOG_TRACE("Request b64_json: {}", b64_json_data);
                                if(b64_json_data.length() > 100) {
                                    content_item["image_url"]["url"] = b64_json_data.substr(0, 100) + "..., length=" + std::to_string(b64_json_data.length());
                                }   
                            }
                        }
                    }
                }
            }
            // response
            if(body_json.contains("data") && body_json["data"].is_array() && !body_json["data"].empty()) {
                for(auto& item: body_json["data"]) {
                    if(item.contains("b64_json")) {
                        std::string b64_json_data = item["b64_json"];
                        LOG_TRACE("Response b64_json: {}", b64_json_data);
                        if(b64_json_data.length() > 100) {
                            item["b64_json"] = b64_json_data.substr(0, 100) + "..., length=" + std::to_string(b64_json_data.length());
                        }
                    }
                }
            }
            return body_json.dump();
        } catch (const json::exception& e) {
            LOG_WARN("Failed to parse body as JSON: {}", e.what());
            return body; 
        }
    }
    
    void log_server_request(const httplib::Request& req, const httplib::Response& res) {
        // skip GH copilot requests when using default port
        if (req.path == "/health" || req.path == "/v1/completions") {
            return;
        }

        const std::string request_id = res.has_header("X-Request-ID") ? res.get_header_value("X-Request-ID") : "null";
        const auto access_logger = HLIELogger::get_logger(HLIE_ACCESS);
        HLIE_CHECK_NULL(access_logger);
        access_logger->info("{} - - \"{} {}\" {} [RequestID={}]", req.remote_addr, req.method, req.path, res.status, request_id);
        access_logger->flush();

        LOG_INFO("Request[{}]: {} {} {} {} \n{}", request_id, req.method, req.path, req.remote_addr, res.status, process_body(req.body));
        LOG_INFO("Response[{}]: {}", request_id, process_body(res.body));
    }

    void res_error(httplib::Response& res, const json& error_data) {
        const json final_response{{"error", error_data}};
        res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
        res.status = json_value(error_data, "code", 500);
    }

    void exception_handler(const httplib::Request&, httplib::Response& res, const std::exception_ptr& ep) {
        std::string message;
        int ret = -1;
        try {
            std::rethrow_exception(ep);
        } catch (const WebServerException& e) {
            message = e.get_err_msg();
            ret = e.get_ret_code();
        } catch (const std::exception& e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        const std::string request_id = res.has_header("X-Request-ID")
                ? res.get_header_value("X-Request-ID")
                : "null";
        const nlohmann::json formatted_error = format_error_response(message, ERROR_TYPE_SERVER, ret);
        LOG_ERROR("[Request-{}] got server exception: {}", request_id, formatted_error.dump().c_str());
        res_error(res, formatted_error);
    }
} // namespace

HLIEWebService::HLIEWebService(const std::shared_ptr<HLIEWorkflow>& workflow) {
    workflow_ = workflow;
}

void HLIEWebService::init(int n_threads_http) {
    web_server_ = std::make_shared<httplib::Server>();
    web_server_->set_default_headers({{"Server", "HLIECpp"}});

    web_server_->Get("/hi", [](const httplib::Request&, httplib::Response& res) {
        LOG_INFO("Got a request /hi"); // For touch
        res.set_content("Hello World!", "text/plain");
    });

    web_server_->set_logger(log_server_request);
    web_server_->set_exception_handler(exception_handler);

    // set timeouts and change hostname and port
    web_server_->set_read_timeout(600);
    web_server_->set_write_timeout(600);


    web_server_->set_pre_routing_handler(
        [this](const httplib::Request& req, httplib::Response& res) { return this->pre_routing_handler(req, res); });

    web_server_->new_task_queue = [n_threads_http] { return new httplib::ThreadPool(n_threads_http); };
    LOG_INFO("Web Service init done, n_threads_http: {}", n_threads_http);
}

int HLIEWebService::run(const std::string& host_name, int port) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    const auto& args = config_services->get_args();
    if (const auto cfg_port = config_services->get_int("server", "port", -1); cfg_port < 0) {
        if (args.find("--port") != args.end()) {
            try {
                port = std::stoi(args.at("--port"));
                LOG_DEBUG("Using arg port: {}", port);
            } catch (...) {
                LOG_ERROR("Cannot found port config, default port will be used");
            }
        } else {
            LOG_DEBUG("Cannot found port config, default port will be used");
        }
    } else {
        port = cfg_port;
    }

    if (args.find("--instance_id") != args.end()) {
        port += std::stoi(args.at("--instance_id"));
        LOG_DEBUG("Instance[{}] port: {}", args.at("--instance_id"), port);
    }

    if (const bool was_bound = web_server_->bind_to_port(host_name, port); !was_bound) {
        LOG_ERROR("couldn't bind HTTP server socket, hostname: {}, port: %d\n", host_name, port);
        stop();

        return -1;
    }

    worker_thread_ = std::make_unique<std::thread>([this] {
        int ret = this->web_server_->listen_after_bind();
        spdlog::debug("webserver listen ret: {}", ret);
    });
    web_server_->wait_until_ready();

    LOG_INFO("HTTP server is listening, hostname: {}, port: {}", host_name, port);

    return 0;
}

void HLIEWebService::register_routers() const {
    HLIE_CHECK_NULL(workflow_);

    const auto& routes = workflow_->get_route_configs();

    for (const auto& [paths, method, handler, require_model, model_type] : routes) {
        // 检查是否需要特定模型服务
        if (require_model) {
            const auto model_service = workflow_->get_model_service(model_type);
            if (!model_service) {
                LOG_INFO("Skipping route {} and related - required model service not available", paths[0]);
                continue;
            }

            // 检查模型服务状态
            if (model_service->get_state() != SERVER_STATE_READY) {
                LOG_INFO("Skipping route {} and related - model service not ready", paths[0]);
                continue;
            }
        }

        // 注册路由
        if (method == GET) {
            for (const auto& path : paths) {
                LOG_INFO("Registered GET route: {}", path);
                web_server_->Get(path, handler);
            }
        } else if (method == POST) {
            for (const auto& path : paths) {
                LOG_INFO("Registered POST route: {}", path);
                web_server_->Post(path, handler);
            }
        }
    }
}


void HLIEWebService::register_post(const std::string& path, httplib::Server::Handler&& func) const {
    web_server_->Post(path, func);
}

void HLIEWebService::register_get(const std::string& path, httplib::Server::Handler&& func) const {
    web_server_->Get(path, func);
}

void HLIEWebService::stop() {
    if (web_server_) {
        web_server_->stop();
    }

    if (!worker_thread_) {
        return;
    }

    if (worker_thread_->joinable()) {
        worker_thread_->join();
    }
    worker_thread_.reset();
}

httplib::Server::HandlerResponse HLIEWebService::pre_routing_handler(const httplib::Request& req, httplib::Response& res) const {
    LOG_TRACE("Pre-routing handler called for {} {}", req.method, req.path);
    res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
    // If this is OPTIONS request, skip validation because browsers don't include Authorization header
    if (req.method == "OPTIONS") {
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "GET, POST");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.set_content("", "text/html"); // blank response, no data
        return httplib::Server::HandlerResponse::Handled; // skip further processing
    }

    if (!middleware_server_state(req, res)) {
        return httplib::Server::HandlerResponse::Handled;
    }

    if (!middleware_validate_api_key(req, res)) {
        return httplib::Server::HandlerResponse::Handled;
    }

    return httplib::Server::HandlerResponse::Unhandled;
}

bool HLIEWebService::middleware_server_state(const httplib::Request& req, httplib::Response& res) const {
    LOG_TRACE("middleware_server_state called for path: {}", req.path);
    HLIE_CHECK_NULL_RETURN(workflow_, 0);
    const int state = workflow_->get_model_service_state();
    if (req.path != "/health") {
        const auto request_id = HlieUtil::uuid();
        res.set_header("X-Request-ID", request_id);
        LOG_INFO("Request[{}] come in, current state[0: LOADING_MODEL 1:READY]: {}", request_id, state);
        if (const auto logger = HLIELogger::get_logger(HLIE_ACCESS)) {
            logger->info("{} - - \"{} {}\" [RequestID={}]", req.remote_addr, req.method, req.path, request_id);
        }
    }

    if (req.path == "/models" || req.path == "/v1/models" || req.path == "/api/tags") {
        // allow the models endpoint to be accessed during loading
        return true;
    }

    if (state == SERVER_STATE_LOADING_MODEL) {
        res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
        return false;
    }
    return true;
}

bool HLIEWebService::middleware_validate_api_key(const httplib::Request& req, httplib::Response& res) const {
    HLIE_CHECK_NULL_RETURN(workflow_, 0);
    static const std::unordered_set<std::string> public_endpoints = {"/health", "/models", "/v1/models", "/api/tags"};

    // If API key is not set, skip validation
    if (workflow_->get_api_keys().empty()) {
        return true;
    }

    // If path is public or is static file, skip validation
    if (public_endpoints.find(req.path) != public_endpoints.end() || req.path == "/") {
        return true;
    }

    // Check for API key in the header
    const auto auth_header = req.get_header_value("Authorization");

    if (const std::string prefix = "Bearer "; auth_header.substr(0, prefix.size()) == prefix) {
        if (const std::string received_api_key = auth_header.substr(prefix.size());
            std::find(workflow_->get_api_keys().begin(), workflow_->get_api_keys().end(), received_api_key) !=
            workflow_->get_api_keys().end()) {
            return true; // API key is valid
        }
    }

    // API key is invalid or not provided
    res_error(res, format_error_response("Invalid API Key", ERROR_TYPE_AUTHENTICATION));

    LOG_WARN("Unauthorized: Invalid API Key");

    return false;
}

HLIE_NAMESPACE_END

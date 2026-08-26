#include "hlie_app/hlie_replica_config.h"

#include <algorithm>
#include <cctype>

#include "hlie_base/hlie_config_service.h"

HLIE_NAMESPACE_BEGIN

namespace {
    int clamp_min(const int value, const int min_value) {
        return value < min_value ? min_value : value;
    }

    std::string normalized_policy(std::string policy) {
        std::transform(policy.begin(), policy.end(), policy.begin(), [](const unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        if (policy != "prefer_idle_card" && policy != "prefer_used_card" && policy != "exclusive_card") {
            return "prefer_idle_card";
        }
        return policy;
    }
}

HLIEReplicaConfig HLIEReplicaConfig::load() {
    const auto& config = Singleton<HLIEConfigService>::instance();
    HLIEReplicaConfig result;
    result.count = clamp_min(config->get_int("replica", "count", 1, true), 1);
    result.memory_saving = config->get_bool("replica", "memory_saving", false, true);
    result.startup_delay_sec = clamp_min(config->get_int("replica", "startup_delay_sec", 45, true), 0);
    result.device_policy = normalized_policy(config->get_string("replica", "device_policy", "prefer_idle_card", true));
    result.max_instances_per_device = clamp_min(config->get_int("replica", "max_instances_per_device", 2, true), 1);
    return result;
}

HLIEBalancerConfig HLIEBalancerConfig::load() {
    const auto& config = Singleton<HLIEConfigService>::instance();
    HLIEBalancerConfig result;
    result.enabled = config->get_bool("balancer", "enabled", false, true);
    result.management_url = config->get_string("balancer", "management_url", "http://127.0.0.1:8085", true);
    result.target_host = config->get_string("balancer", "target_host", "127.0.0.1", true);
    result.heartbeat_interval_ms = clamp_min(config->get_int("balancer", "heartbeat_interval_ms", 1000, true), 100);
    return result;
}

HLIE_NAMESPACE_END

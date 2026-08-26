#ifndef HLIE_REPLICA_CONFIG_H
#define HLIE_REPLICA_CONFIG_H

#include <string>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

struct HLIEReplicaConfig {
    int count = 1;
    bool memory_saving = false;
    int startup_delay_sec = 45;
    std::string device_policy = "prefer_idle_card";
    int max_instances_per_device = 2;

    [[nodiscard]] bool enabled() const { return count > 1; }

    static HLIEReplicaConfig load();
};

struct HLIEBalancerConfig {
    bool enabled = false;
    std::string management_url = "http://127.0.0.1:8085";
    std::string target_host = "127.0.0.1";
    int heartbeat_interval_ms = 1000;

    static HLIEBalancerConfig load();
};

HLIE_NAMESPACE_END

#endif // HLIE_REPLICA_CONFIG_H

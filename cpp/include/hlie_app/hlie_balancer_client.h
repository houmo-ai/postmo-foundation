#ifndef HLIE_BALANCER_CLIENT_H
#define HLIE_BALANCER_CLIENT_H

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "hlie_app/hlie_replica_config.h"
#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class HLIE_API HLIEBalancerClient {
public:
    HLIEBalancerClient() = default;
    ~HLIEBalancerClient();

    void start(const HLIEBalancerConfig& config, int local_port, const std::string& instance_id);
    void stop();

private:
    struct ParsedUrl {
        std::string scheme = "http";
        std::string host = "127.0.0.1";
        int port = 8085;
    };

    static ParsedUrl parse_management_url(const std::string& url);
    bool register_once();
    void heartbeat_loop();

    HLIEBalancerConfig config_;
    int local_port_ = 0;
    std::string instance_id_;
    ParsedUrl management_;
    std::unique_ptr<std::thread> heartbeat_thread_;
    std::atomic<bool> stop_{false};
    bool registered_once_ = false;
};

HLIE_NAMESPACE_END

#endif // HLIE_BALANCER_CLIENT_H

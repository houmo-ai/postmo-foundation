#ifndef HLIE_APP_MODELS_STOP_SERVICE_APPLICATION_SERVICE_H
#define HLIE_APP_MODELS_STOP_SERVICE_APPLICATION_SERVICE_H

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"
#include "hlie_core/domain/hlie_model_launch_ticket.h"
#include "server/server-http.h"

HLIE_NAMESPACE_BEGIN

struct HLIEStopServiceWorkerSnapshot {
    std::uint64_t process_id = 0;
    int instance_id = 0;
    int port = 0;
    std::vector<std::string> model_names;
    HLIEModelLaunchTicket launch_ticket;
};

struct HLIEStopServiceApplicationDependencies {
    std::function<bool()> replicas_mode_provider;
    std::function<std::vector<std::string>()> local_model_keys_provider;
    std::function<bool(const std::string&)> try_begin_eviction;
    std::function<void(const std::string&)> abort_eviction;
    std::function<bool(const std::string&)> evict_claimed_model;
    std::function<std::optional<HLIEStopServiceWorkerSnapshot>(std::uint64_t)>
        worker_provider;
    std::function<bool(const HLIEModelLaunchTicket&)> stop_ticketed_worker;
    std::function<bool(const HLIEStopServiceWorkerSnapshot&)>
        stop_worker_fallback;
    std::function<void(const HLIEModelLaunchTicket&)>
        publish_confirmed_stop;
    std::function<std::optional<std::string>(const std::string&)>
        resolve_model_key;
    std::function<void(const std::string&)>
        publish_confirmed_legacy_stop;
};

// Owns the legacy /hlie/v1/stop DTO and coordinates local batch eviction or
// Worker stop confirmation. Controller and Supervisor remain behind ports.
class HLIEStopServiceApplicationService final {
public:
    explicit HLIEStopServiceApplicationService(
        HLIEStopServiceApplicationDependencies dependencies);

    [[nodiscard]] server_http_res_ptr handle(
        const server_http_req& request) const;

private:
    [[nodiscard]] server_http_res_ptr stop_local_models(
        int instance_id, int process_id) const;
    [[nodiscard]] server_http_res_ptr stop_worker(
        int instance_id, int process_id) const;

    HLIEStopServiceApplicationDependencies dependencies_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_MODELS_STOP_SERVICE_APPLICATION_SERVICE_H

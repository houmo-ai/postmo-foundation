#ifndef HLIE_APP_CONTROL_WORKER_CONTROL_APPLICATION_SERVICE_H
#define HLIE_APP_CONTROL_WORKER_CONTROL_APPLICATION_SERVICE_H

#include <functional>

#include "hlie_app/control/hlie_worker_control_service.h"
#include "server/server-http.h"

HLIE_NAMESPACE_BEGIN

struct HLIEWorkerControlApplicationDependencies {
    std::function<HLIEModelLaunchTicket()> launch_ticket_provider;
};

// Owns the legacy HTTP/JSON compatibility surface above the transport-neutral
// WorkerControlService. Launch tokens are parsed for fencing and never logged.
class HLIEWorkerControlApplicationService final {
public:
    HLIEWorkerControlApplicationService(
        HLIEWorkerControlService& worker_control,
        HLIEWorkerControlApplicationDependencies dependencies);

    [[nodiscard]] server_http_res_ptr handle_get_status() const;
    [[nodiscard]] server_http_res_ptr handle_post_status(
        const server_http_req& request) const;
    [[nodiscard]] server_http_res_ptr handle_shutdown(
        const server_http_req& request) const;

private:
    HLIEWorkerControlService& worker_control_;
    HLIEWorkerControlApplicationDependencies dependencies_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_CONTROL_WORKER_CONTROL_APPLICATION_SERVICE_H

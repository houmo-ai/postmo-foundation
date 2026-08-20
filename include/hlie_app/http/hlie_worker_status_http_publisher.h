#ifndef HLIE_APP_HTTP_WORKER_STATUS_HTTP_PUBLISHER_H
#define HLIE_APP_HTTP_WORKER_STATUS_HTTP_PUBLISHER_H

#include "hlie_app/control/hlie_worker_control_service.h"
#include "httplib.h"

HLIE_NAMESPACE_BEGIN

class HLIEHttpWorkerStatusPublisher final
    : public IHLIEWorkerStatusPublisher {
public:
    [[nodiscard]] bool publish(
        const HLIEWorkerStatusPublication& publication) override;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_HTTP_WORKER_STATUS_HTTP_PUBLISHER_H

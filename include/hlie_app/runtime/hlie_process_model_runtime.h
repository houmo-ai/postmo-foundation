#ifndef HLIE_APP_PROCESS_MODEL_RUNTIME_H
#define HLIE_APP_PROCESS_MODEL_RUNTIME_H

#include <memory>

#include "hlie_app/runtime/hlie_worker_supervisor.h"
#include "hlie_core/runtime/hlie_model_runtime.h"

HLIE_NAMESPACE_BEGIN

class HLIEProcessModelRuntime final : public IHLIEModelRuntime {
public:
    HLIEProcessModelRuntime(HLIEWorkerSupervisor& supervisor,
                            HLIEWorkerRuntime worker,
                            HLIEModelCapabilities capabilities);
    ~HLIEProcessModelRuntime() override;

    HLIEProcessModelRuntime(const HLIEProcessModelRuntime&) = delete;
    HLIEProcessModelRuntime& operator=(const HLIEProcessModelRuntime&) = delete;

    [[nodiscard]] HLIERuntimeIdentity identity() const override;
    [[nodiscard]] HLIEModelCapabilities capabilities() const override;
    [[nodiscard]] HLIERuntimeHealth health() const override;
    void begin_draining() override;
    [[nodiscard]] HLIERuntimeStopResult stop(
        const HLIERuntimeStopRequest& request) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_PROCESS_MODEL_RUNTIME_H

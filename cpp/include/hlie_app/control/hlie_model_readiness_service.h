#pragma once

#include <chrono>
#include <functional>
#include <optional>
#include <string>

#include "hlie_app/control/hlie_model_controller.h"
#include "hlie_app/control/hlie_model_load_service.h"
#include "hlie_app/util/hlie_background_task_group.h"

HLIE_NAMESPACE_BEGIN

enum class HLIEModelReadinessStatus {
    Ready,
    Missing,
    Unavailable,
    InvalidLoadRequest,
    ScheduleFailed,
    TimedOut,
    Cancelled,
    Orphaned,
    Failed,
    Unloaded,
};

struct HLIEModelReadinessRequest {
    std::string model_key;
    std::optional<HLIEModelLoadRequest> load_request;
    int local_port = 0;
    std::chrono::steady_clock::time_point deadline{};
    std::function<bool()> should_stop;
};

struct HLIEModelReadinessResult {
    HLIEModelReadinessStatus status = HLIEModelReadinessStatus::Missing;
    std::string model_key;
};

// Owns asynchronous auto-load task lifetime and coalesced readiness waits.
// Controller remains the generation/state owner; ModelLoadService remains the
// reservation/load transaction preparer.
class HLIEModelReadinessService {
public:
    using LoadFunction = std::function<HLIEModelLoadResult(
        HLIEModelLoadRequest,
        int,
        const std::optional<HLIEModelLaunchTicket>&)>;
    using DrainingProvider = std::function<bool()>;
    using LoadSucceededCallback =
        std::function<void(const HLIEModelLoadReceipt&)>;

    HLIEModelReadinessService(
        HLIEModelController& controller,
        LoadFunction load,
        DrainingProvider draining_provider,
        LoadSucceededCallback load_succeeded = {});

    ~HLIEModelReadinessService();

    HLIEModelReadinessService(const HLIEModelReadinessService&) = delete;
    HLIEModelReadinessService& operator=(
        const HLIEModelReadinessService&) = delete;

    [[nodiscard]] HLIEModelReadinessResult ensure_ready(
        HLIEModelReadinessRequest request);

    void stop_and_join();

private:
    [[nodiscard]] bool is_draining() const;

    HLIEModelController& controller_;
    LoadFunction load_;
    DrainingProvider draining_provider_;
    LoadSucceededCallback load_succeeded_;
    BackgroundTaskGroup tasks_;
};

HLIE_NAMESPACE_END

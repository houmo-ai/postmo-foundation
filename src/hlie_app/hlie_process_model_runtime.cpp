#include "hlie_app/runtime/hlie_process_model_runtime.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <utility>

HLIE_NAMESPACE_BEGIN

namespace {
std::string stop_reason_name(const HLIERuntimeStopReason reason) {
    switch (reason) {
    case HLIERuntimeStopReason::ManualUnload:
        return "manual_unload";
    case HLIERuntimeStopReason::Eviction:
        return "eviction";
    case HLIERuntimeStopReason::Shutdown:
        return "shutdown";
    case HLIERuntimeStopReason::StartupRollback:
        return "startup_rollback";
    case HLIERuntimeStopReason::Recovery:
        return "recovery";
    }
    return "unknown";
}
} // namespace

class HLIEProcessModelRuntime::Impl {
public:
    Impl(HLIEWorkerSupervisor& supervisor,
         HLIEWorkerRuntime worker,
         const HLIEModelCapabilities capabilities)
        : supervisor_(supervisor), worker_(std::move(worker)), capabilities_(capabilities) {
        identity_ = {
            worker_.runtime_id,
            HLIERuntimeKind::Process,
            worker_.launch_ticket,
            worker_.instance_id,
            worker_.port,
        };
        if (!worker_.valid() || !identity_.valid()) {
            throw std::invalid_argument("invalid process runtime identity");
        }
    }

    [[nodiscard]] HLIERuntimeIdentity identity() const { return identity_; }
    [[nodiscard]] HLIEModelCapabilities capabilities() const { return capabilities_; }

    [[nodiscard]] HLIERuntimeHealth health() const {
        std::lock_guard lock(mutex_);
        if (stopped_) {
            return {identity_, HLIERuntimeHealthState::Stopped, 0};
        }
        if (draining_) {
            return {identity_, HLIERuntimeHealthState::Draining, 0};
        }
        const auto current = supervisor_.find_runtime(worker_.runtime_id);
        if (!current || !current->launch_ticket.matches(worker_.launch_ticket)) {
            return {identity_, HLIERuntimeHealthState::Failed, -1};
        }
        return {identity_, current->ready ? HLIERuntimeHealthState::Ready
                                          : HLIERuntimeHealthState::Starting, 0};
    }

    void begin_draining() {
        std::lock_guard lock(mutex_);
        if (!stopped_) {
            draining_ = true;
        }
    }

    [[nodiscard]] HLIERuntimeStopResult stop(const HLIERuntimeStopRequest& request) {
        HLIERuntimeStopResult result;
        result.identity = identity_;
        if (!request.valid()) {
            result.native_code = -1;
            return result;
        }

        std::lock_guard lock(mutex_);
        if (stopped_) {
            result.confirmed = true;
            result.graceful = true;
            result.already_stopped = true;
            result.deadline_exceeded = std::chrono::steady_clock::now() > request.deadline;
            return result;
        }
        draining_ = true;
        if (!supervisor_.find_runtime(worker_.runtime_id)) {
            stopped_ = true;
            result.confirmed = true;
            result.already_stopped = true;
            result.deadline_exceeded = std::chrono::steady_clock::now() > request.deadline;
            result.graceful = !result.deadline_exceeded;
            return result;
        }

        const auto now = std::chrono::steady_clock::now();
        const auto remaining = request.deadline > now
            ? std::chrono::duration_cast<std::chrono::milliseconds>(request.deadline - now)
            : std::chrono::milliseconds::zero();
        const auto stopped = supervisor_.request_stop(worker_, HLIEWorkerStopRequest{
            stop_reason_name(request.reason), remaining,
            request.force || remaining == std::chrono::milliseconds::zero()});
        result.confirmed = stopped.confirmed;
        result.graceful = stopped.graceful;
        result.deadline_exceeded = std::chrono::steady_clock::now() > request.deadline;
        if (result.confirmed) {
            stopped_ = true;
        } else {
            result.native_code = -2;
        }
        return result;
    }

private:
    HLIEWorkerSupervisor& supervisor_;
    HLIEWorkerRuntime worker_;
    HLIEModelCapabilities capabilities_;
    HLIERuntimeIdentity identity_;
    mutable std::mutex mutex_;
    bool draining_ = false;
    bool stopped_ = false;
};

HLIEProcessModelRuntime::HLIEProcessModelRuntime(
    HLIEWorkerSupervisor& supervisor,
    HLIEWorkerRuntime worker,
    const HLIEModelCapabilities capabilities)
    : impl_(std::make_unique<Impl>(supervisor, std::move(worker), capabilities)) {}

HLIEProcessModelRuntime::~HLIEProcessModelRuntime() = default;

HLIERuntimeIdentity HLIEProcessModelRuntime::identity() const { return impl_->identity(); }
HLIEModelCapabilities HLIEProcessModelRuntime::capabilities() const { return impl_->capabilities(); }
HLIERuntimeHealth HLIEProcessModelRuntime::health() const { return impl_->health(); }
void HLIEProcessModelRuntime::begin_draining() { impl_->begin_draining(); }
HLIERuntimeStopResult HLIEProcessModelRuntime::stop(
    const HLIERuntimeStopRequest& request) {
    return impl_->stop(request);
}

HLIE_NAMESPACE_END

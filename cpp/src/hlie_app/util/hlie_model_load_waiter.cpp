#include "hlie_app/util/hlie_model_load_waiter.h"

#include <algorithm>

HLIE_NAMESPACE_BEGIN

ModelLoadWaitResult wait_for_model_load(
    std::condition_variable& state_changed,
    std::unique_lock<std::mutex>& wait_lock,
    const ModelLoadStatusProbe& probe,
    const std::chrono::steady_clock::time_point deadline,
    const std::function<bool()>& should_stop,
    const std::chrono::milliseconds recheck_interval) {
    const auto effective_interval = std::max(recheck_interval, std::chrono::milliseconds(1));

    while (true) {
        if (should_stop && should_stop()) {
            return ModelLoadWaitResult::CANCELLED;
        }

        const auto status = probe();
        if (!status) {
            return ModelLoadWaitResult::MISSING;
        }
        if (*status == SERVER_MODEL_STATUS_LOADED) {
            return ModelLoadWaitResult::LOADED;
        }
        if (*status != SERVER_MODEL_STATUS_LOADING) {
            return ModelLoadWaitResult::STOPPED;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return ModelLoadWaitResult::TIMED_OUT;
        }

        state_changed.wait_until(wait_lock, std::min(deadline, now + effective_interval));
    }
}

HLIE_NAMESPACE_END

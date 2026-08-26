#ifndef HLIE_MODEL_LOAD_WAITER_H
#define HLIE_MODEL_LOAD_WAITER_H

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>

#include <server/server-models.h>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

enum class ModelLoadWaitResult {
    LOADED,
    STOPPED,
    MISSING,
    CANCELLED,
    TIMED_OUT,
};

using ModelLoadStatusProbe = std::function<std::optional<server_model_status>()>;

ModelLoadWaitResult wait_for_model_load(
    std::condition_variable& state_changed,
    std::unique_lock<std::mutex>& wait_lock,
    const ModelLoadStatusProbe& probe,
    std::chrono::steady_clock::time_point deadline,
    const std::function<bool()>& should_stop = {},
    std::chrono::milliseconds recheck_interval = std::chrono::milliseconds(100));

HLIE_NAMESPACE_END

#endif // HLIE_MODEL_LOAD_WAITER_H

#pragma once

#include <chrono>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

// Transport- and backend-neutral cancellation fact consumed by core metrics.
// Concrete request controls stay in hlie-app and adapt into this value.
enum class HLIECancellationStopReason {
    None = 0,
    Cancelled = 1,
    Deadline = 2,
    NONE = None,
    CANCELLED = Cancelled,
    DEADLINE = Deadline,
};

struct HLIECancellationObservation {
    HLIECancellationStopReason reason = HLIECancellationStopReason::None;
    std::chrono::steady_clock::time_point requested_at{};
};

HLIE_NAMESPACE_END

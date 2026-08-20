#include "hlie_app/runtime/hlie_cancellation_metrics.h"

#include <algorithm>
#include <stdexcept>

HLIE_NAMESPACE_BEGIN

HLIECancellationMetrics& HLIECancellationMetrics::instance() {
    static HLIECancellationMetrics metrics;
    return metrics;
}

bool HLIECancellationMetrics::observe_backend_completion(
    const HLIEBackendCancellationLane lane,
    const HLIECancellationObservation observation,
    const std::chrono::steady_clock::time_point completed_at) {
    if (observation.reason == HLIECancellationStopReason::None ||
        observation.requested_at.time_since_epoch().count() == 0) {
        return false;
    }

    const auto lag = completed_at > observation.requested_at
        ? std::chrono::duration_cast<std::chrono::milliseconds>(
              completed_at - observation.requested_at).count()
        : 0;
    const auto lag_ms = static_cast<std::uint64_t>(std::max<std::int64_t>(0, lag));
    auto& stats = stats_.at(index(lane, observation.reason));
    stats.observations.fetch_add(1);
    stats.total_lag_ms.fetch_add(lag_ms);
    auto maximum = stats.max_lag_ms.load();
    while (maximum < lag_ms &&
           !stats.max_lag_ms.compare_exchange_weak(maximum, lag_ms)) {
    }
    return true;
}

HLIECancellationLagSnapshot HLIECancellationMetrics::snapshot(
    const HLIEBackendCancellationLane lane,
    const HLIECancellationStopReason reason) const {
    const auto& stats = stats_.at(index(lane, reason));
    return {
        stats.observations.load(),
        stats.total_lag_ms.load(),
        stats.max_lag_ms.load(),
    };
}

std::string_view HLIECancellationMetrics::lane_name(
    const HLIEBackendCancellationLane lane) {
    switch (lane) {
    case HLIEBackendCancellationLane::WHISPER:
        return "whisper";
    case HLIEBackendCancellationLane::COSYVOICE:
        return "cosyvoice";
    case HLIEBackendCancellationLane::STABLE_DIFFUSION:
        return "sd";
    }
    return "unknown";
}

std::string_view HLIECancellationMetrics::reason_name(
    const HLIECancellationStopReason reason) {
    switch (reason) {
    case HLIECancellationStopReason::Cancelled:
        return "cancelled";
    case HLIECancellationStopReason::Deadline:
        return "deadline";
    case HLIECancellationStopReason::None:
        break;
    }
    return "none";
}

void HLIECancellationMetrics::reset_for_tests() {
    for (auto& stats : stats_) {
        stats.observations.store(0);
        stats.total_lag_ms.store(0);
        stats.max_lag_ms.store(0);
    }
}

std::size_t HLIECancellationMetrics::index(
    const HLIEBackendCancellationLane lane,
    const HLIECancellationStopReason reason) {
    const auto lane_index = static_cast<std::size_t>(lane);
    std::size_t reason_index = 0;
    switch (reason) {
    case HLIECancellationStopReason::Cancelled:
        reason_index = 0;
        break;
    case HLIECancellationStopReason::Deadline:
        reason_index = 1;
        break;
    case HLIECancellationStopReason::None:
        throw std::invalid_argument("cancellation metrics require a stop reason");
    }
    if (lane_index >= LANE_COUNT) {
        throw std::out_of_range("invalid cancellation metrics lane");
    }
    return lane_index * REASON_COUNT + reason_index;
}

HLIE_NAMESPACE_END

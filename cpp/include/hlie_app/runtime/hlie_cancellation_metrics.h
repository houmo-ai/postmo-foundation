#ifndef HLIE_CANCELLATION_METRICS_H
#define HLIE_CANCELLATION_METRICS_H

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string_view>

#include "hlie_base/hlie_base.h"
#include "hlie_core/domain/hlie_cancellation_observation.h"

HLIE_NAMESPACE_BEGIN

enum class HLIEBackendCancellationLane {
    WHISPER = 0,
    COSYVOICE = 1,
    STABLE_DIFFUSION = 2,
};

struct HLIECancellationLagSnapshot {
    std::uint64_t observations = 0;
    std::uint64_t total_lag_ms = 0;
    std::uint64_t max_lag_ms = 0;
};

class HLIECancellationMetrics final {
public:
    static HLIECancellationMetrics& instance();

    bool observe_backend_completion(
        HLIEBackendCancellationLane lane,
        HLIECancellationObservation observation,
        std::chrono::steady_clock::time_point completed_at = std::chrono::steady_clock::now());

    [[nodiscard]] HLIECancellationLagSnapshot snapshot(
        HLIEBackendCancellationLane lane,
        HLIECancellationStopReason reason) const;

    static std::string_view lane_name(HLIEBackendCancellationLane lane);
    static std::string_view reason_name(HLIECancellationStopReason reason);

    void reset_for_tests();

private:
    static constexpr std::size_t LANE_COUNT = 3;
    static constexpr std::size_t REASON_COUNT = 2;

    struct AtomicStats {
        std::atomic<std::uint64_t> observations{0};
        std::atomic<std::uint64_t> total_lag_ms{0};
        std::atomic<std::uint64_t> max_lag_ms{0};
    };

    static std::size_t index(HLIEBackendCancellationLane lane,
                             HLIECancellationStopReason reason);

    std::array<AtomicStats, LANE_COUNT * REASON_COUNT> stats_{};
};

HLIE_NAMESPACE_END

#endif // HLIE_CANCELLATION_METRICS_H

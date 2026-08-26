#include <catch_amalgamated.hpp>

#include <atomic>
#include <chrono>
#include <thread>

#include "hlie_app/hlie_model_service.h"
#include "hlie_app/runtime/hlie_cancellation_metrics.h"

using namespace hm_HLIE;

TEST_CASE("queued request control propagates explicit cancellation", "[admission][cancellation]") {
    HLIEQueuedRequestControl control;
    CHECK_FALSE(control.is_cancelled());
    control.cancel();
    CHECK(control.is_cancelled());
}

TEST_CASE("queued request control observes connection cancellation callback", "[admission][cancellation]") {
    std::atomic<bool> stopped{false};
    HLIEQueuedRequestControl control;
    control.should_stop = [&] { return stopped.load(); };
    CHECK_FALSE(control.is_cancelled());
    stopped.store(true);
    CHECK(control.is_cancelled());
}

TEST_CASE("queued request control enforces queue deadline", "[admission][deadline]") {
    HLIEQueuedRequestControl control;
    control.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
    CHECK_FALSE(control.is_expired());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(control.is_expired());
}

TEST_CASE("queued request control preserves the first stop reason across copies",
          "[admission][cancellation][deadline]") {
    HLIEQueuedRequestControl control;
    const auto deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds(25);
    control.deadline = deadline;
    control.expire();

    auto copied = control;
    copied.cancel();
    const auto observation = copied.stop_observation();
    CHECK(observation.reason == HLIEQueuedRequestStopReason::DEADLINE);
    CHECK(observation.requested_at == deadline);
    CHECK(control.is_cancelled());
}

TEST_CASE("backend cancellation flag remains valid after the HTTP-side control is destroyed",
          "[admission][cancellation][lifetime]") {
    HLIEQueuedRequestControl queued_control;
    const std::atomic<bool>* backend_flag = nullptr;
    {
        HLIEQueuedRequestControl http_control;
        queued_control = http_control;
        backend_flag = http_control.cancelled.get();
        REQUIRE(backend_flag != nullptr);
        CHECK_FALSE(backend_flag->load());
    }

    queued_control.cancel();
    CHECK(backend_flag->load());
    CHECK(queued_control.stop_observation().reason ==
          HLIEQueuedRequestStopReason::CANCELLED);
}

TEST_CASE("backend completion records deterministic cancellation lag",
          "[admission][cancellation][metrics]") {
    auto& metrics = HLIECancellationMetrics::instance();
    metrics.reset_for_tests();

    HLIEQueuedRequestControl cancelled;
    cancelled.cancel();
    const auto cancelled_at = cancelled.stop_observation().requested_at;
    REQUIRE(metrics.observe_backend_completion(
        HLIEBackendCancellationLane::STABLE_DIFFUSION,
        cancelled.observe_for_completion(),
        cancelled_at + std::chrono::milliseconds(25)));

    const auto cancelled_snapshot = metrics.snapshot(
        HLIEBackendCancellationLane::STABLE_DIFFUSION,
        HLIEQueuedRequestStopReason::CANCELLED);
    CHECK(cancelled_snapshot.observations == 1);
    CHECK(cancelled_snapshot.total_lag_ms == 25);
    CHECK(cancelled_snapshot.max_lag_ms == 25);

    HLIEQueuedRequestControl expired;
    expired.deadline = std::chrono::steady_clock::now();
    expired.expire();
    REQUIRE(metrics.observe_backend_completion(
        HLIEBackendCancellationLane::STABLE_DIFFUSION,
        expired.observe_for_completion(),
        expired.deadline + std::chrono::milliseconds(80)));

    const auto deadline_snapshot = metrics.snapshot(
        HLIEBackendCancellationLane::STABLE_DIFFUSION,
        HLIEQueuedRequestStopReason::DEADLINE);
    CHECK(deadline_snapshot.observations == 1);
    CHECK(deadline_snapshot.total_lag_ms == 80);
    CHECK(deadline_snapshot.max_lag_ms == 80);
}

TEST_CASE("backend completion ignores requests that were not stopped",
          "[admission][cancellation][metrics]") {
    auto& metrics = HLIECancellationMetrics::instance();
    metrics.reset_for_tests();
    const HLIEQueuedRequestControl active;
    CHECK_FALSE(metrics.observe_backend_completion(
        HLIEBackendCancellationLane::WHISPER,
        active.observe_for_completion()));
    CHECK(metrics.snapshot(HLIEBackendCancellationLane::WHISPER,
                           HLIEQueuedRequestStopReason::CANCELLED).observations == 0);
}

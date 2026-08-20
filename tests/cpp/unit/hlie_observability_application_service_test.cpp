#include <catch_amalgamated.hpp>

#include <string>

#include "hlie_app/observability/hlie_observability_application_service.h"
#include "json.hpp"

namespace {

using namespace hm_HLIE;

nlohmann::json response_body(const server_http_res_ptr& response) {
    REQUIRE(response);
    return nlohmann::json::parse(response->data);
}

HLIEObservabilitySnapshot populated_snapshot() {
    HLIEObservabilitySnapshot snapshot;
    snapshot.models = {1, 2, 3};
    snapshot.llm_admission = {4, 5, 6, 7};
    snapshot.qwen3_tts_admission = {8, 9, 10, 11};
    snapshot.workers = {12, 13, 14, 15, 16, 17};
    snapshot.runtimes = {18, 19, 20, 21, 22, 23, 24};
    snapshot.inference = {25, 26, 27, 28, 29, 30, 31, 32, 33, 34};
    snapshot.reconciliation = {true, 35, 36, 37, 38};
    snapshot.discovery = {false, true, 39, 40, 41, 42, 43, 44};
    snapshot.resources = {45, 46, 47, 48};
    snapshot.devices.push_back({49, 50, 51, 52, 53, 54, 55, 56});
    snapshot.cancellation_lag.push_back(
        {"whisper", "cancelled", 57, 58, 59});
    return snapshot;
}

} // namespace

TEST_CASE("Observability application renders the complete health contract",
          "[p1][observability][application-service]") {
    const auto snapshot = populated_snapshot();
    HLIEObservabilityApplicationService service({
        [snapshot] { return snapshot.models; },
        [snapshot] { return snapshot; },
    });

    const auto response = service.handle_health();
    const auto body = response_body(response);
    CHECK(response->status == 200);
    CHECK(response->content_type == "application/json; charset=utf-8");
    CHECK(body["status"] == "degraded");
    CHECK(body["ready"] == true);
    CHECK(body["models"] == nlohmann::json{
        {"loaded", 1}, {"loading", 2}, {"failed", 3}});
    CHECK(body["admission"]["llm"] == nlohmann::json{
        {"max_in_flight", 4}, {"in_flight", 5},
        {"accepted", 6}, {"rejected", 7}});
    CHECK(body["admission"]["qwen3_tts"] == nlohmann::json{
        {"max_in_flight", 8}, {"in_flight", 9},
        {"accepted", 10}, {"rejected", 11}});
    CHECK(body["workers"] == nlohmann::json{
        {"running", 12}, {"unexpected_exits", 13},
        {"ready_timeouts", 14}, {"graceful_shutdowns", 15},
        {"forced_shutdowns", 16}, {"port_conflicts", 17}});
    CHECK(body["runtimes"] == nlohmann::json{
        {"total", 18}, {"embedded", 19}, {"process", 20},
        {"ready", 21}, {"starting", 22}, {"draining", 23},
        {"failed", 24}});
    CHECK(body["inference_dispatch"] == nlohmann::json{
        {"active_sessions", 25}, {"dispatched", 26},
        {"draining_rejected", 27}, {"model_not_found", 28},
        {"model_not_ready", 29}, {"runtime_unavailable", 30},
        {"endpoint_unavailable", 31}, {"unsupported_operation", 32},
        {"endpoint_failures", 33}, {"cached_endpoints", 34}});
    CHECK(body["control_plane_reconciliation"] == nlohmann::json{
        {"running", true}, {"ticks", 35}, {"failures", 36},
        {"policy_refreshes", 37}, {"idle_ttl_evictions", 38}});
    CHECK(body["model_discovery"] == nlohmann::json{
        {"running", false}, {"pending_stability", true},
        {"scan_attempts", 39}, {"parse_attempts", 40},
        {"scan_failures", 41}, {"published_models", 42},
        {"requested_generation", 43}, {"completed_generation", 44}});
    CHECK(body["resources"] == nlohmann::json{
        {"expired_pending_rollbacks", 45},
        {"expired_pending_protected", 46},
        {"tracked_pending", 47}, {"tracked_committed", 48}});
    REQUIRE(body["devices"].size() == 1);
    CHECK(body["devices"].at(0) == nlohmann::json{
        {"id", 49}, {"total_mb", 50}, {"observed_used_mb", 51},
        {"pending_reserved_mb", 52}, {"committed_reserved_mb", 53},
        {"committed_unobserved_mb", 54}, {"effective_used_mb", 55},
        {"admissible_mb", 56}});
    CHECK(body["backend_cancellation_lag"]["whisper"]["cancelled"] ==
          nlohmann::json{{"observations", 57}, {"total_lag_ms", 58},
                         {"max_lag_ms", 59}});
}

TEST_CASE("Observability application keeps live and readiness semantics",
          "[p1][observability][application-service]") {
    HLIEObservabilityModelSummary summary;
    int summary_calls = 0;
    int snapshot_calls = 0;
    HLIEObservabilityApplicationService service({
        [&] {
            ++summary_calls;
            return summary;
        },
        [&] {
            ++snapshot_calls;
            return HLIEObservabilitySnapshot{};
        },
    });

    const auto live = service.handle_live();
    CHECK(live->status == 200);
    CHECK(response_body(live)["status"] == "live");
    CHECK(summary_calls == 0);
    CHECK(snapshot_calls == 0);

    auto ready = service.handle_ready();
    CHECK(ready->status == 503);
    CHECK(response_body(ready) == nlohmann::json{
        {"status", "not_ready"}, {"loaded_models", 0},
        {"loading_models", 0}});

    summary = {2, 1, 0};
    ready = service.handle_ready();
    CHECK(ready->status == 503);
    CHECK(response_body(ready)["loading_models"] == 1);

    summary = {2, 0, 0};
    ready = service.handle_ready();
    const auto ready_body = response_body(ready);
    CHECK(ready->status == 200);
    CHECK(ready_body["status"] == "ready");
    CHECK(ready_body["loaded_models"] == 2);
    CHECK_FALSE(ready_body.contains("loading_models"));
    CHECK(summary_calls == 3);
    CHECK(snapshot_calls == 0);
}

TEST_CASE("Observability application preserves Prometheus metric names",
          "[p1][observability][application-service]") {
    const auto snapshot = populated_snapshot();
    HLIEObservabilityApplicationService service({
        [snapshot] { return snapshot.models; },
        [snapshot] { return snapshot; },
    });

    const auto response = service.handle_metrics();
    REQUIRE(response);
    CHECK(response->status == 200);
    CHECK(response->content_type == "text/plain; version=0.0.4");
    const std::string expected =
        "hlie_models_loaded 1\n"
        "hlie_runtimes_total 18\n"
        "hlie_runtimes{kind=\"embedded\"} 19\n"
        "hlie_runtimes{kind=\"process\"} 20\n"
        "hlie_runtimes{state=\"ready\"} 21\n"
        "hlie_runtimes{state=\"starting\"} 22\n"
        "hlie_runtimes{state=\"draining\"} 23\n"
        "hlie_runtimes{state=\"failed\"} 24\n"
        "hlie_admission_in_flight{lane=\"llm\"} 5\n"
        "hlie_admission_rejected_total{lane=\"llm\"} 7\n"
        "hlie_admission_in_flight{lane=\"qwen3_tts\"} 9\n"
        "hlie_admission_rejected_total{lane=\"qwen3_tts\"} 11\n"
        "hlie_inference_dispatch_active_sessions 25\n"
        "hlie_inference_dispatch_total 26\n"
        "hlie_inference_dispatch_rejected_total{reason=\"draining\"} 27\n"
        "hlie_inference_dispatch_rejected_total{reason=\"model_not_found\"} 28\n"
        "hlie_inference_dispatch_rejected_total{reason=\"model_not_ready\"} 29\n"
        "hlie_inference_dispatch_rejected_total{reason=\"runtime_unavailable\"} 30\n"
        "hlie_inference_dispatch_rejected_total{reason=\"endpoint_unavailable\"} 31\n"
        "hlie_inference_dispatch_rejected_total{reason=\"unsupported_operation\"} 32\n"
        "hlie_inference_dispatch_failures_total 33\n"
        "hlie_inference_dispatch_cached_endpoints 34\n"
        "hlie_control_plane_reconciliation_running 1\n"
        "hlie_control_plane_reconciliation_ticks_total 35\n"
        "hlie_control_plane_reconciliation_failures_total 36\n"
        "hlie_control_plane_policy_refreshes_total 37\n"
        "hlie_control_plane_idle_ttl_evictions_total 38\n"
        "hlie_model_discovery_running 0\n"
        "hlie_model_discovery_pending_stability 1\n"
        "hlie_model_discovery_scan_attempts_total 39\n"
        "hlie_model_discovery_parse_attempts_total 40\n"
        "hlie_model_discovery_scan_failures_total 41\n"
        "hlie_model_discovery_published_models_total 42\n"
        "hlie_worker_unexpected_exits_total 13\n"
        "hlie_worker_ready_timeouts_total 14\n"
        "hlie_worker_graceful_shutdowns_total 15\n"
        "hlie_worker_forced_shutdowns_total 16\n"
        "hlie_worker_port_conflicts_total 17\n"
        "hlie_resource_expired_pending_rollbacks_total 45\n"
        "hlie_resource_expired_pending_protected 46\n"
        "hlie_resource_tracked_reservations{state=\"pending\"} 47\n"
        "hlie_resource_tracked_reservations{state=\"committed\"} 48\n"
        "hlie_backend_cancellation_lag_observations_total{lane=\"whisper\",reason=\"cancelled\"} 57\n"
        "hlie_backend_cancellation_lag_milliseconds_sum{lane=\"whisper\",reason=\"cancelled\"} 58\n"
        "hlie_backend_cancellation_lag_milliseconds_max{lane=\"whisper\",reason=\"cancelled\"} 59\n";
    CHECK(response->data == expected);
}

TEST_CASE("Observability health is ok only without model or runtime failures",
          "[p1][observability][application-service]") {
    HLIEObservabilitySnapshot snapshot;
    snapshot.models.loaded = 1;
    HLIEObservabilityApplicationService service({
        [&snapshot] { return snapshot.models; },
        [&snapshot] { return snapshot; },
    });

    CHECK(response_body(service.handle_health())["status"] == "ok");
    snapshot.runtimes.failed = 1;
    CHECK(response_body(service.handle_health())["status"] == "degraded");
}

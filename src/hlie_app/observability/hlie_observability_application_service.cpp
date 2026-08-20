#include "hlie_app/observability/hlie_observability_application_service.h"

#include <sstream>
#include <utility>

#include "hlie_base/hlie_common_define.h"

HLIE_NAMESPACE_BEGIN

namespace {

server_http_res_ptr json_response(json payload, const int status = 200) {
    auto response = std::make_unique<server_http_res>();
    response->status = status;
    response->content_type = MIMETYPE_JSON;
    response->data = payload.dump();
    return response;
}

json admission_json(const HLIEObservabilityAdmissionMetrics& metrics) {
    return {
        {"max_in_flight", metrics.max_in_flight},
        {"in_flight", metrics.in_flight},
        {"accepted", metrics.accepted},
        {"rejected", metrics.rejected},
    };
}

} // namespace

HLIEObservabilityApplicationService::HLIEObservabilityApplicationService(
    HLIEObservabilityApplicationDependencies dependencies)
    : dependencies_(std::move(dependencies)) {}

server_http_res_ptr HLIEObservabilityApplicationService::handle_live() const {
    return json_response({{"status", "live"}});
}

server_http_res_ptr HLIEObservabilityApplicationService::handle_ready() const {
    const auto models = dependencies_.model_summary_provider
        ? dependencies_.model_summary_provider()
        : HLIEObservabilityModelSummary{};
    if (models.loaded == 0 || models.loading != 0) {
        return json_response({
            {"status", "not_ready"},
            {"loaded_models", models.loaded},
            {"loading_models", models.loading},
        }, 503);
    }
    return json_response({
        {"status", "ready"},
        {"loaded_models", models.loaded},
    });
}

server_http_res_ptr HLIEObservabilityApplicationService::handle_health() const {
    const auto snapshot = dependencies_.snapshot_provider
        ? dependencies_.snapshot_provider()
        : HLIEObservabilitySnapshot{};
    json devices = json::array();
    for (const auto& device : snapshot.devices) {
        devices.push_back({
            {"id", device.id},
            {"total_mb", device.total_mb},
            {"observed_used_mb", device.observed_used_mb},
            {"pending_reserved_mb", device.pending_reserved_mb},
            {"committed_reserved_mb", device.committed_reserved_mb},
            {"committed_unobserved_mb", device.committed_unobserved_mb},
            {"effective_used_mb", device.effective_used_mb},
            {"admissible_mb", device.admissible_mb},
        });
    }
    json cancellation_lag = json::object();
    for (const auto& metric : snapshot.cancellation_lag) {
        cancellation_lag[metric.lane][metric.reason] = {
            {"observations", metric.observations},
            {"total_lag_ms", metric.total_lag_ms},
            {"max_lag_ms", metric.max_lag_ms},
        };
    }

    return json_response({
        {"status", snapshot.models.failed == 0 && snapshot.runtimes.failed == 0
            ? "ok" : "degraded"},
        {"ready", snapshot.models.loaded > 0},
        {"models", {
            {"loaded", snapshot.models.loaded},
            {"loading", snapshot.models.loading},
            {"failed", snapshot.models.failed},
        }},
        {"admission", {
            {"llm", admission_json(snapshot.llm_admission)},
            {"qwen3_tts", admission_json(snapshot.qwen3_tts_admission)},
        }},
        {"workers", {
            {"running", snapshot.workers.running},
            {"unexpected_exits", snapshot.workers.unexpected_exits},
            {"ready_timeouts", snapshot.workers.ready_timeouts},
            {"graceful_shutdowns", snapshot.workers.graceful_shutdowns},
            {"forced_shutdowns", snapshot.workers.forced_shutdowns},
            {"port_conflicts", snapshot.workers.port_conflicts},
        }},
        {"runtimes", {
            {"total", snapshot.runtimes.total},
            {"embedded", snapshot.runtimes.embedded},
            {"process", snapshot.runtimes.process},
            {"ready", snapshot.runtimes.ready},
            {"starting", snapshot.runtimes.starting},
            {"draining", snapshot.runtimes.draining},
            {"failed", snapshot.runtimes.failed},
        }},
        {"inference_dispatch", {
            {"active_sessions", snapshot.inference.active_sessions},
            {"dispatched", snapshot.inference.dispatched},
            {"draining_rejected", snapshot.inference.draining_rejected},
            {"model_not_found", snapshot.inference.model_not_found},
            {"model_not_ready", snapshot.inference.model_not_ready},
            {"runtime_unavailable", snapshot.inference.runtime_unavailable},
            {"endpoint_unavailable", snapshot.inference.endpoint_unavailable},
            {"unsupported_operation", snapshot.inference.unsupported_operation},
            {"endpoint_failures", snapshot.inference.endpoint_failures},
            {"cached_endpoints", snapshot.inference.cached_endpoints},
        }},
        {"control_plane_reconciliation", {
            {"running", snapshot.reconciliation.running},
            {"ticks", snapshot.reconciliation.ticks},
            {"failures", snapshot.reconciliation.failures},
            {"policy_refreshes", snapshot.reconciliation.policy_refreshes},
            {"idle_ttl_evictions", snapshot.reconciliation.idle_ttl_evictions},
        }},
        {"model_discovery", {
            {"running", snapshot.discovery.running},
            {"pending_stability", snapshot.discovery.pending_stability},
            {"scan_attempts", snapshot.discovery.scan_attempts},
            {"parse_attempts", snapshot.discovery.parse_attempts},
            {"scan_failures", snapshot.discovery.scan_failures},
            {"published_models", snapshot.discovery.published_models},
            {"requested_generation", snapshot.discovery.requested_generation},
            {"completed_generation", snapshot.discovery.completed_generation},
        }},
        {"resources", {
            {"expired_pending_rollbacks", snapshot.resources.expired_pending_rollbacks},
            {"expired_pending_protected", snapshot.resources.expired_pending_protected},
            {"tracked_pending", snapshot.resources.tracked_pending},
            {"tracked_committed", snapshot.resources.tracked_committed},
        }},
        {"backend_cancellation_lag", std::move(cancellation_lag)},
        {"devices", std::move(devices)},
    });
}

server_http_res_ptr HLIEObservabilityApplicationService::handle_metrics() const {
    const auto snapshot = dependencies_.snapshot_provider
        ? dependencies_.snapshot_provider()
        : HLIEObservabilitySnapshot{};
    std::ostringstream metrics;
    metrics << "hlie_models_loaded " << snapshot.models.loaded << '\n'
            << "hlie_runtimes_total " << snapshot.runtimes.total << '\n'
            << "hlie_runtimes{kind=\"embedded\"} " << snapshot.runtimes.embedded << '\n'
            << "hlie_runtimes{kind=\"process\"} " << snapshot.runtimes.process << '\n'
            << "hlie_runtimes{state=\"ready\"} " << snapshot.runtimes.ready << '\n'
            << "hlie_runtimes{state=\"starting\"} " << snapshot.runtimes.starting << '\n'
            << "hlie_runtimes{state=\"draining\"} " << snapshot.runtimes.draining << '\n'
            << "hlie_runtimes{state=\"failed\"} " << snapshot.runtimes.failed << '\n'
            << "hlie_admission_in_flight{lane=\"llm\"} " << snapshot.llm_admission.in_flight << '\n'
            << "hlie_admission_rejected_total{lane=\"llm\"} " << snapshot.llm_admission.rejected << '\n'
            << "hlie_admission_in_flight{lane=\"qwen3_tts\"} " << snapshot.qwen3_tts_admission.in_flight << '\n'
            << "hlie_admission_rejected_total{lane=\"qwen3_tts\"} " << snapshot.qwen3_tts_admission.rejected << '\n'
            << "hlie_inference_dispatch_active_sessions " << snapshot.inference.active_sessions << '\n'
            << "hlie_inference_dispatch_total " << snapshot.inference.dispatched << '\n'
            << "hlie_inference_dispatch_rejected_total{reason=\"draining\"} "
            << snapshot.inference.draining_rejected << '\n'
            << "hlie_inference_dispatch_rejected_total{reason=\"model_not_found\"} "
            << snapshot.inference.model_not_found << '\n'
            << "hlie_inference_dispatch_rejected_total{reason=\"model_not_ready\"} "
            << snapshot.inference.model_not_ready << '\n'
            << "hlie_inference_dispatch_rejected_total{reason=\"runtime_unavailable\"} "
            << snapshot.inference.runtime_unavailable << '\n'
            << "hlie_inference_dispatch_rejected_total{reason=\"endpoint_unavailable\"} "
            << snapshot.inference.endpoint_unavailable << '\n'
            << "hlie_inference_dispatch_rejected_total{reason=\"unsupported_operation\"} "
            << snapshot.inference.unsupported_operation << '\n'
            << "hlie_inference_dispatch_failures_total " << snapshot.inference.endpoint_failures << '\n'
            << "hlie_inference_dispatch_cached_endpoints " << snapshot.inference.cached_endpoints << '\n'
            << "hlie_control_plane_reconciliation_running " << (snapshot.reconciliation.running ? 1 : 0) << '\n'
            << "hlie_control_plane_reconciliation_ticks_total " << snapshot.reconciliation.ticks << '\n'
            << "hlie_control_plane_reconciliation_failures_total " << snapshot.reconciliation.failures << '\n'
            << "hlie_control_plane_policy_refreshes_total " << snapshot.reconciliation.policy_refreshes << '\n'
            << "hlie_control_plane_idle_ttl_evictions_total " << snapshot.reconciliation.idle_ttl_evictions << '\n'
            << "hlie_model_discovery_running " << (snapshot.discovery.running ? 1 : 0) << '\n'
            << "hlie_model_discovery_pending_stability " << (snapshot.discovery.pending_stability ? 1 : 0) << '\n'
            << "hlie_model_discovery_scan_attempts_total " << snapshot.discovery.scan_attempts << '\n'
            << "hlie_model_discovery_parse_attempts_total " << snapshot.discovery.parse_attempts << '\n'
            << "hlie_model_discovery_scan_failures_total " << snapshot.discovery.scan_failures << '\n'
            << "hlie_model_discovery_published_models_total " << snapshot.discovery.published_models << '\n'
            << "hlie_worker_unexpected_exits_total " << snapshot.workers.unexpected_exits << '\n'
            << "hlie_worker_ready_timeouts_total " << snapshot.workers.ready_timeouts << '\n'
            << "hlie_worker_graceful_shutdowns_total " << snapshot.workers.graceful_shutdowns << '\n'
            << "hlie_worker_forced_shutdowns_total " << snapshot.workers.forced_shutdowns << '\n'
            << "hlie_worker_port_conflicts_total " << snapshot.workers.port_conflicts << '\n'
            << "hlie_resource_expired_pending_rollbacks_total " << snapshot.resources.expired_pending_rollbacks << '\n'
            << "hlie_resource_expired_pending_protected " << snapshot.resources.expired_pending_protected << '\n'
            << "hlie_resource_tracked_reservations{state=\"pending\"} " << snapshot.resources.tracked_pending << '\n'
            << "hlie_resource_tracked_reservations{state=\"committed\"} " << snapshot.resources.tracked_committed << '\n';
    for (const auto& metric : snapshot.cancellation_lag) {
        metrics << "hlie_backend_cancellation_lag_observations_total{lane=\""
                << metric.lane << "\",reason=\"" << metric.reason << "\"} "
                << metric.observations << '\n'
                << "hlie_backend_cancellation_lag_milliseconds_sum{lane=\""
                << metric.lane << "\",reason=\"" << metric.reason << "\"} "
                << metric.total_lag_ms << '\n'
                << "hlie_backend_cancellation_lag_milliseconds_max{lane=\""
                << metric.lane << "\",reason=\"" << metric.reason << "\"} "
                << metric.max_lag_ms << '\n';
    }

    auto response = std::make_unique<server_http_res>();
    response->status = 200;
    response->content_type = "text/plain; version=0.0.4";
    response->data = metrics.str();
    return response;
}

HLIE_NAMESPACE_END

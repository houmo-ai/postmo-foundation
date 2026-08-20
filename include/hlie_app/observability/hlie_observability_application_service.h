#ifndef HLIE_APP_OBSERVABILITY_APPLICATION_SERVICE_H
#define HLIE_APP_OBSERVABILITY_APPLICATION_SERVICE_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"
#include "server/server-http.h"

HLIE_NAMESPACE_BEGIN

struct HLIEObservabilityModelSummary {
    std::size_t loaded = 0;
    std::size_t loading = 0;
    std::size_t failed = 0;
};

struct HLIEObservabilityAdmissionMetrics {
    std::size_t max_in_flight = 0;
    std::size_t in_flight = 0;
    std::uint64_t accepted = 0;
    std::uint64_t rejected = 0;
};

struct HLIEObservabilityWorkerMetrics {
    std::size_t running = 0;
    std::uint64_t unexpected_exits = 0;
    std::uint64_t ready_timeouts = 0;
    std::uint64_t graceful_shutdowns = 0;
    std::uint64_t forced_shutdowns = 0;
    std::uint64_t port_conflicts = 0;
};

struct HLIEObservabilityRuntimeMetrics {
    std::size_t total = 0;
    std::size_t embedded = 0;
    std::size_t process = 0;
    std::size_t ready = 0;
    std::size_t starting = 0;
    std::size_t draining = 0;
    std::size_t failed = 0;
};

struct HLIEObservabilityInferenceMetrics {
    std::uint64_t active_sessions = 0;
    std::uint64_t dispatched = 0;
    std::uint64_t draining_rejected = 0;
    std::uint64_t model_not_found = 0;
    std::uint64_t model_not_ready = 0;
    std::uint64_t runtime_unavailable = 0;
    std::uint64_t endpoint_unavailable = 0;
    std::uint64_t unsupported_operation = 0;
    std::uint64_t endpoint_failures = 0;
    std::uint64_t cached_endpoints = 0;
};

struct HLIEObservabilityReconciliationMetrics {
    bool running = false;
    std::uint64_t ticks = 0;
    std::uint64_t failures = 0;
    std::uint64_t policy_refreshes = 0;
    std::uint64_t idle_ttl_evictions = 0;
};

struct HLIEObservabilityDiscoveryMetrics {
    bool running = false;
    bool pending_stability = false;
    std::uint64_t scan_attempts = 0;
    std::uint64_t parse_attempts = 0;
    std::uint64_t scan_failures = 0;
    std::uint64_t published_models = 0;
    std::uint64_t requested_generation = 0;
    std::uint64_t completed_generation = 0;
};

struct HLIEObservabilityResourceMetrics {
    std::uint64_t expired_pending_rollbacks = 0;
    std::uint64_t expired_pending_protected = 0;
    std::size_t tracked_pending = 0;
    std::size_t tracked_committed = 0;
};

struct HLIEObservabilityDeviceMetrics {
    int id = -1;
    std::uint64_t total_mb = 0;
    std::uint64_t observed_used_mb = 0;
    std::uint64_t pending_reserved_mb = 0;
    std::uint64_t committed_reserved_mb = 0;
    std::uint64_t committed_unobserved_mb = 0;
    std::uint64_t effective_used_mb = 0;
    std::uint64_t admissible_mb = 0;
};

struct HLIEObservabilityCancellationLag {
    std::string lane;
    std::string reason;
    std::uint64_t observations = 0;
    std::uint64_t total_lag_ms = 0;
    std::uint64_t max_lag_ms = 0;
};

struct HLIEObservabilitySnapshot {
    HLIEObservabilityModelSummary models;
    HLIEObservabilityAdmissionMetrics llm_admission;
    HLIEObservabilityAdmissionMetrics qwen3_tts_admission;
    HLIEObservabilityWorkerMetrics workers;
    HLIEObservabilityRuntimeMetrics runtimes;
    HLIEObservabilityInferenceMetrics inference;
    HLIEObservabilityReconciliationMetrics reconciliation;
    HLIEObservabilityDiscoveryMetrics discovery;
    HLIEObservabilityResourceMetrics resources;
    std::vector<HLIEObservabilityDeviceMetrics> devices;
    std::vector<HLIEObservabilityCancellationLag> cancellation_lag;
};

struct HLIEObservabilityApplicationDependencies {
    std::function<HLIEObservabilityModelSummary()> model_summary_provider;
    std::function<HLIEObservabilitySnapshot()> snapshot_provider;
};

// Owns health/readiness JSON and Prometheus text rendering from neutral,
// immutable control-plane observations.
class HLIEObservabilityApplicationService final {
public:
    explicit HLIEObservabilityApplicationService(
        HLIEObservabilityApplicationDependencies dependencies);

    [[nodiscard]] server_http_res_ptr handle_live() const;
    [[nodiscard]] server_http_res_ptr handle_ready() const;
    [[nodiscard]] server_http_res_ptr handle_health() const;
    [[nodiscard]] server_http_res_ptr handle_metrics() const;

private:
    HLIEObservabilityApplicationDependencies dependencies_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_OBSERVABILITY_APPLICATION_SERVICE_H

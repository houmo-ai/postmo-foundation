#ifndef HLIE_APP_CONTROL_PLANE_RECONCILER_H
#define HLIE_APP_CONTROL_PLANE_RECONCILER_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "hlie_app/resource/hlie_resource_scheduler.h"

HLIE_NAMESPACE_BEGIN

class HLIEInferenceDispatcher;
class HLIEModelController;

struct HLIEControlPlaneReconcilerOptions {
    std::chrono::milliseconds reconciliation_interval{1000};
    std::chrono::seconds disabled_ttl_check_interval{30};
    std::chrono::seconds minimum_ttl_check_interval{5};
    std::chrono::seconds maximum_ttl_check_interval{60};
};

struct HLIEControlPlaneReconcilerMetrics {
    std::uint64_t reconciliation_ticks = 0;
    std::uint64_t reconciliation_failures = 0;
    std::uint64_t policy_refreshes = 0;
    std::uint64_t idle_ttl_evictions = 0;
    bool running = false;
};

// Owns the periodic control-plane timer only. Lifecycle state remains in the
// Controller, reservation state in the Scheduler, and endpoint state in the
// Dispatcher. Environment/config and process observations enter as callbacks.
class HLIEControlPlaneReconciler {
public:
    using ResidencyPolicyProvider = std::function<HLIEModelResidencyPolicy()>;
    using ExternalLiveOwnerProvider =
        std::function<std::vector<HLIEModelLaunchTicket>()>;

    HLIEControlPlaneReconciler(
        HLIEResourceScheduler& scheduler,
        HLIEModelController& controller,
        HLIEInferenceDispatcher* dispatcher,
        ResidencyPolicyProvider residency_policy_provider,
        ExternalLiveOwnerProvider external_live_owner_provider = {},
        HLIEControlPlaneReconcilerOptions options = {});
    ~HLIEControlPlaneReconciler();

    HLIEControlPlaneReconciler(const HLIEControlPlaneReconciler&) = delete;
    HLIEControlPlaneReconciler& operator=(
        const HLIEControlPlaneReconciler&) = delete;

    void start();
    void stop();

    // Exposed for deterministic tests and operational one-shot recovery.
    // A TTL evaluation also refreshes the externally supplied policy.
    [[nodiscard]] bool reconcile_once(
        bool evaluate_idle_ttl = true,
        std::time_t now = std::time(nullptr));
    [[nodiscard]] HLIEControlPlaneReconcilerMetrics metrics() const noexcept;

private:
    [[nodiscard]] std::chrono::seconds ttl_check_interval(
        const HLIEModelResidencyPolicy& policy) const noexcept;
    void run_loop();

    HLIEResourceScheduler& scheduler_;
    HLIEModelController& controller_;
    HLIEInferenceDispatcher* dispatcher_ = nullptr;
    ResidencyPolicyProvider residency_policy_provider_;
    ExternalLiveOwnerProvider external_live_owner_provider_;
    HLIEControlPlaneReconcilerOptions options_;

    mutable std::mutex lifecycle_mutex_;
    std::mutex wait_mutex_;
    std::condition_variable wait_cv_;
    std::thread worker_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> running_{false};
    std::atomic<std::uint64_t> reconciliation_ticks_{0};
    std::atomic<std::uint64_t> reconciliation_failures_{0};
    std::atomic<std::uint64_t> policy_refreshes_{0};
    std::atomic<std::uint64_t> idle_ttl_evictions_{0};
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_CONTROL_PLANE_RECONCILER_H

#include "hlie_app/control/hlie_control_plane_reconciler.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "hlie_app/control/hlie_model_controller.h"
#include "hlie_app/inference/hlie_inference_dispatcher.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

namespace {
HLIEControlPlaneReconcilerOptions normalize_options(
    HLIEControlPlaneReconcilerOptions options) {
    options.reconciliation_interval = std::max(
        options.reconciliation_interval, std::chrono::milliseconds(10));
    options.minimum_ttl_check_interval = std::max(
        options.minimum_ttl_check_interval, std::chrono::seconds(1));
    options.maximum_ttl_check_interval = std::max(
        options.maximum_ttl_check_interval,
        options.minimum_ttl_check_interval);
    options.disabled_ttl_check_interval = std::clamp(
        options.disabled_ttl_check_interval,
        options.minimum_ttl_check_interval,
        options.maximum_ttl_check_interval);
    return options;
}
} // namespace

HLIEControlPlaneReconciler::HLIEControlPlaneReconciler(
    HLIEResourceScheduler& scheduler,
    HLIEModelController& controller,
    HLIEInferenceDispatcher* dispatcher,
    ResidencyPolicyProvider residency_policy_provider,
    ExternalLiveOwnerProvider external_live_owner_provider,
    HLIEControlPlaneReconcilerOptions options)
    : scheduler_(scheduler),
      controller_(controller),
      dispatcher_(dispatcher),
      residency_policy_provider_(std::move(residency_policy_provider)),
      external_live_owner_provider_(std::move(external_live_owner_provider)),
      options_(normalize_options(options)) {
    if (!residency_policy_provider_) {
        throw std::invalid_argument(
            "ControlPlaneReconciler requires a residency policy provider");
    }
}

HLIEControlPlaneReconciler::~HLIEControlPlaneReconciler() {
    stop();
}

void HLIEControlPlaneReconciler::start() {
    std::lock_guard lifecycle_lock(lifecycle_mutex_);
    if (worker_.joinable()) {
        return;
    }

    try {
        scheduler_.update_residency_policy(residency_policy_provider_());
        policy_refreshes_.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception& ex) {
        reconciliation_failures_.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR("Control-plane policy initialization failed: {}", ex.what());
    } catch (...) {
        reconciliation_failures_.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR("Control-plane policy initialization failed with an unknown exception");
    }

    stop_requested_.store(false, std::memory_order_release);
    running_.store(true, std::memory_order_release);
    worker_ = std::thread(&HLIEControlPlaneReconciler::run_loop, this);
    const auto policy = scheduler_.residency_policy();
    LOG_INFO("Control-plane reconciler started: interval_ms={}, auto_unload={}, idle_ttl_sec={}",
             options_.reconciliation_interval.count(), policy.auto_unload,
             policy.idle_ttl_sec);
}

void HLIEControlPlaneReconciler::stop() {
    std::lock_guard lifecycle_lock(lifecycle_mutex_);
    if (!worker_.joinable()) {
        running_.store(false, std::memory_order_release);
        return;
    }

    {
        // Publish the predicate while holding the CV mutex so notify cannot
        // land between the worker's predicate check and its wait transition.
        std::lock_guard wait_lock(wait_mutex_);
        stop_requested_.store(true, std::memory_order_release);
    }
    wait_cv_.notify_all();
    worker_.join();
    running_.store(false, std::memory_order_release);
    LOG_INFO("Control-plane reconciler stopped");
}

bool HLIEControlPlaneReconciler::reconcile_once(
    const bool evaluate_idle_ttl, const std::time_t now) {
    reconciliation_ticks_.fetch_add(1, std::memory_order_relaxed);
    try {
        if (evaluate_idle_ttl) {
            scheduler_.update_residency_policy(residency_policy_provider_());
            policy_refreshes_.fetch_add(1, std::memory_order_relaxed);
        }

        std::vector<HLIEModelLaunchTicket> protecting_owners;
        if (external_live_owner_provider_) {
            protecting_owners = external_live_owner_provider_();
        }

        const auto runtime_snapshots = controller_.get_all_runtime_snapshots();
        protecting_owners.reserve(
            protecting_owners.size() + runtime_snapshots.size());
        for (const auto& snapshot : runtime_snapshots) {
            if (!hlie_model_state_protects_reservation(snapshot.state)) {
                continue;
            }
            const auto ticket = controller_.get_launch_ticket(
                snapshot.identity.model_key);
            if (ticket && ticket->generation == snapshot.generation) {
                protecting_owners.push_back(*ticket);
            }
        }

        scheduler_.reconcile_expired_pending(protecting_owners);
        if (dispatcher_) {
            dispatcher_->reconcile_endpoints();
        }

        if (!evaluate_idle_ttl) {
            return true;
        }
        const auto policy = scheduler_.residency_policy();
        if (!policy.auto_unload || policy.idle_ttl_sec <= 0) {
            return true;
        }

        const auto plan = scheduler_.plan_for_idle_ttl(runtime_snapshots, now);
        if (!plan.candidate) {
            return true;
        }

        const auto& candidate = *plan.candidate;
        LOG_INFO("Model residency idle TTL eviction selected: key={}, model_name={}, ttl_sec={}, hits={}",
                 candidate.key, candidate.model_name, policy.idle_ttl_sec,
                 candidate.hit_count);
        const auto result = controller_.evict_runtime(candidate.key, plan.reason);
        if (result.evicted()) {
            idle_ttl_evictions_.fetch_add(1, std::memory_order_relaxed);
        } else {
            LOG_WARN("Model residency idle TTL eviction was not completed: key={}, status={}",
                     candidate.key, static_cast<int>(result.status));
        }
        return true;
    } catch (const std::exception& ex) {
        reconciliation_failures_.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR("Control-plane reconciliation failed: {}", ex.what());
    } catch (...) {
        reconciliation_failures_.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR("Control-plane reconciliation failed with an unknown exception");
    }
    return false;
}

HLIEControlPlaneReconcilerMetrics
HLIEControlPlaneReconciler::metrics() const noexcept {
    return {
        reconciliation_ticks_.load(std::memory_order_relaxed),
        reconciliation_failures_.load(std::memory_order_relaxed),
        policy_refreshes_.load(std::memory_order_relaxed),
        idle_ttl_evictions_.load(std::memory_order_relaxed),
        running_.load(std::memory_order_acquire),
    };
}

std::chrono::seconds HLIEControlPlaneReconciler::ttl_check_interval(
    const HLIEModelResidencyPolicy& policy) const noexcept {
    if (policy.idle_ttl_sec <= 0) {
        return options_.disabled_ttl_check_interval;
    }
    return std::chrono::seconds(std::clamp(
        policy.idle_ttl_sec / 2,
        static_cast<int>(options_.minimum_ttl_check_interval.count()),
        static_cast<int>(options_.maximum_ttl_check_interval.count())));
}

void HLIEControlPlaneReconciler::run_loop() {
    auto next_ttl_check = std::chrono::steady_clock::now() +
        ttl_check_interval(scheduler_.residency_policy());

    while (!stop_requested_.load(std::memory_order_acquire)) {
        std::unique_lock wait_lock(wait_mutex_);
        wait_cv_.wait_for(wait_lock, options_.reconciliation_interval, [this] {
            return stop_requested_.load(std::memory_order_acquire);
        });
        wait_lock.unlock();
        if (stop_requested_.load(std::memory_order_acquire)) {
            break;
        }

        const auto steady_now = std::chrono::steady_clock::now();
        const bool evaluate_idle_ttl = steady_now >= next_ttl_check;
        (void)reconcile_once(evaluate_idle_ttl, std::time(nullptr));
        if (evaluate_idle_ttl) {
            next_ttl_check = steady_now +
                ttl_check_interval(scheduler_.residency_policy());
        }
    }
    running_.store(false, std::memory_order_release);
}

HLIE_NAMESPACE_END

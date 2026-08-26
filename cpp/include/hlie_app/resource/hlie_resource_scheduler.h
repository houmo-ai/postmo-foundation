#ifndef HLIE_APP_RESOURCE_SCHEDULER_H
#define HLIE_APP_RESOURCE_SCHEDULER_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "hlie_core/domain/hlie_model_launch_ticket.h"
#include "hlie_core/domain/hlie_model_runtime_snapshot.h"
#include "hlie_core/resource/hlie_device_reservation_provider.h"
#include "hlie_core/resource/hlie_model_residency_manager.h"
#include "hlie_core/resource/hlie_placement.h"
#include "hlie_core/resource/hlie_reservation_handle.h"

HLIE_NAMESPACE_BEGIN

struct HLIEResourceSchedulerMetrics {
    std::uint64_t expired_pending_rollbacks = 0;
    std::uint64_t expired_pending_protected = 0;
    std::size_t tracked_pending = 0;
    std::size_t tracked_committed = 0;
};

struct HLIEResidencyPlan {
    HLIEModelResidencyPolicy policy;
    std::optional<HLIEModelEvictionCandidate> candidate;
    std::string reason;

    [[nodiscard]] bool should_evict() const noexcept {
        return policy.auto_unload && candidate.has_value();
    }
};

// Owns placement grants, reservation state transitions, reconciliation and
// residency planning. It never starts/stops runtimes and has no dependency on
// Workflow, Registry, ProcessManager, HTTP or proprietary SDK types.
class HLIEResourceScheduler {
public:
    explicit HLIEResourceScheduler(
        std::shared_ptr<IHLIEDeviceReservationProvider> provider,
        HLIEModelResidencyPolicy residency_policy = {});

    [[nodiscard]] std::optional<HLIEPendingReservationHandle> reserve(
        const std::string& model_key, std::uint64_t total_mb,
        const HLIEPlacementIntent& placement, bool exclusive = false);

    [[nodiscard]] bool bind(HLIEPendingReservationHandle& reservation,
                            const HLIEModelLaunchTicket& ticket);
    [[nodiscard]] bool track(HLIEPendingReservationHandle& reservation,
                             const HLIEModelLaunchTicket& ticket);
    [[nodiscard]] bool commit(const HLIEModelLaunchTicket& ticket);
    void rollback(HLIEPendingReservationHandle& reservation);
    void release(const HLIEModelLaunchTicket& ticket);
    void release(const std::string& model_key);
    void release_all();

    void update_residency_policy(HLIEModelResidencyPolicy policy);
    [[nodiscard]] HLIEModelResidencyPolicy residency_policy() const;
    [[nodiscard]] HLIEResidencyPlan plan_for_resource_pressure(
        const std::vector<HLIEModelRuntimeSnapshot>& models,
        const std::string& requested_model) const;
    [[nodiscard]] HLIEResidencyPlan plan_for_loaded_limit(
        const std::vector<HLIEModelRuntimeSnapshot>& models,
        const std::string& requested_model) const;
    [[nodiscard]] HLIEResidencyPlan plan_for_idle_ttl(
        const std::vector<HLIEModelRuntimeSnapshot>& models, std::time_t now) const;

    void reconcile_expired_pending(
        const std::vector<HLIEModelLaunchTicket>& protecting_owners);
    [[nodiscard]] HLIEResourceSchedulerMetrics metrics() const;
    [[nodiscard]] std::vector<HLIEDeviceLedgerSnapshot> ledger_snapshots() const;

private:
    using TrackedReservation = std::variant<HLIEPendingReservationHandle,
                                            HLIECommittedReservationHandle>;
    struct Binding {
        HLIEModelLaunchTicket ticket;
        TrackedReservation reservation;
    };

    [[nodiscard]] static HLIEDeviceReservationId binding_id(const Binding& binding);
    [[nodiscard]] static bool owner_matches(const HLIEDeviceReservation& reservation,
                                            const HLIEModelLaunchTicket& owner);

    std::shared_ptr<IHLIEDeviceReservationProvider> provider_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Binding> reservations_;
    HLIEModelResidencyPolicy residency_policy_;
    std::uint64_t expired_pending_rollbacks_ = 0;
    std::uint64_t expired_pending_protected_ = 0;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_RESOURCE_SCHEDULER_H

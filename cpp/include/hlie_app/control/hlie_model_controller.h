#ifndef HLIE_APP_MODEL_CONTROLLER_H
#define HLIE_APP_MODEL_CONTROLLER_H

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <optional>
#include <mutex>
#include <string>
#include <vector>

#include "hlie_app/hlie_model_registry.h"
#include "hlie_app/resource/hlie_resource_scheduler.h"
#include "hlie_app/runtime/hlie_model_backend_launcher.h"
#include "hlie_app/runtime/hlie_model_runtime_factory.h"
#include "hlie_app/runtime/hlie_runtime_directory.h"
#include "hlie_core/runtime/hlie_runtime_event.h"

HLIE_NAMESPACE_BEGIN

struct HLIEControllerStopSummary {
    std::size_t stopped = 0;
    std::size_t unconfirmed = 0;
    std::size_t stale_registry_updates = 0;

    [[nodiscard]] bool complete() const noexcept { return unconfirmed == 0; }
};

struct HLIEControllerRuntimePublishResult {
    bool published = false;
    bool cleanup_confirmed = false;
    HLIERuntimeIdentity identity;
};

enum class HLIEControllerLoadBeginStatus {
    Started,
    BusyOrMissing,
    StalePreclaim,
    ReservationBindFailed,
};

struct HLIEControllerLoadBeginResult {
    HLIEControllerLoadBeginStatus status =
        HLIEControllerLoadBeginStatus::BusyOrMissing;
    std::optional<HLIEModelLaunchTicket> launch_ticket;

    [[nodiscard]] bool started() const noexcept {
        return status == HLIEControllerLoadBeginStatus::Started &&
               launch_ticket.has_value();
    }
};

enum class HLIEControllerRuntimeAdoptionStatus {
    Ready,
    AwaitingReady,
    StaleLaunch,
    RuntimePublicationFailed,
    ReservationTransitionFailed,
    ReadyPublicationFailed,
};

struct HLIEControllerRuntimeAdoptionResult {
    HLIEControllerRuntimeAdoptionStatus status =
        HLIEControllerRuntimeAdoptionStatus::StaleLaunch;
    bool cleanup_confirmed = false;

    [[nodiscard]] bool accepted() const noexcept {
        return status == HLIEControllerRuntimeAdoptionStatus::Ready ||
               status == HLIEControllerRuntimeAdoptionStatus::AwaitingReady;
    }
};

enum class HLIEControllerLoadStatus {
    Ready,
    AwaitingReady,
    BackendLauncherUnavailable,
    BusyOrMissing,
    StalePreclaim,
    ReservationBindFailed,
    BackendStartFailed,
    RuntimeAdoptionFailed,
    CleanupUnconfirmed,
};

struct HLIEControllerLoadResult {
    HLIEControllerLoadStatus status =
        HLIEControllerLoadStatus::BusyOrMissing;
    std::optional<HLIEModelLaunchTicket> launch_ticket;
    HLIEBackendLaunchArtifact backend;

    [[nodiscard]] bool accepted() const noexcept {
        return status == HLIEControllerLoadStatus::Ready ||
               status == HLIEControllerLoadStatus::AwaitingReady;
    }
};

enum class HLIEControllerEvictionStatus {
    Evicted,
    Rejected,
    RuntimeMissing,
    StopUnconfirmed,
};

struct HLIEControllerEvictionResult {
    HLIEControllerEvictionStatus status =
        HLIEControllerEvictionStatus::Rejected;
    HLIERuntimeIdentity identity;
    bool registry_update_stale = false;

    [[nodiscard]] bool evicted() const noexcept {
        return status == HLIEControllerEvictionStatus::Evicted;
    }
};

enum class HLIEEnsureReadyDisposition {
    Ready,
    LaunchRequired,
    WaitForExistingLaunch,
    Missing,
    Unavailable,
};

struct HLIEEnsureReadyDecision {
    HLIEEnsureReadyDisposition disposition =
        HLIEEnsureReadyDisposition::Missing;
    std::optional<HLIEModelLaunchTicket> launch_ticket;
    std::optional<HLIEModelRuntimeSnapshot> snapshot;
};

enum class HLIEWaitReadyStatus {
    Ready,
    Failed,
    Unloaded,
    Orphaned,
    Unavailable,
    Missing,
    Cancelled,
    TimedOut,
};

struct HLIEWaitReadyResult {
    HLIEWaitReadyStatus status = HLIEWaitReadyStatus::Missing;
    std::optional<HLIEModelRuntimeSnapshot> snapshot;
};

// Application-service owner for model lifecycle state. Workflow may still
// adapt transport requests during P1-5, but Registry writes, runtime events,
// confirmed-stop reconciliation and request/eviction leases enter here.
class HLIEModelController final : public IHLIERuntimeEventSink {
public:
    using ModelInstance = HLIEModelRegistry::ModelInstance;
    using StateChangedCallback = std::function<void()>;

    HLIEModelController(HLIEResourceScheduler& scheduler,
                        HLIERuntimeDirectory& runtime_directory,
                        IHLIEModelRuntimeFactory& runtime_factory,
                        StateChangedCallback state_changed = {},
                        IHLIEModelBackendLauncher* backend_launcher = nullptr,
                        std::shared_ptr<HLIEModelCatalog> model_catalog = {});

    // Catalog and immutable read model. Registry itself remains private.
    void register_model(HLIEModelInfo& model_info, int estimated_mem_mb);
    [[nodiscard]] bool register_model_if_absent(HLIEModelInfo& model_info,
                                                int estimated_mem_mb);
    [[nodiscard]] bool contains(const std::string& model_key) const;
    [[nodiscard]] std::optional<server_model_meta> get_meta(
        const std::string& model_name) const;
    [[nodiscard]] std::optional<ModelInstance> get_instance_snapshot(
        const std::string& model_name) const;
    [[nodiscard]] std::optional<ModelInstance> get_exact_instance_snapshot(
        const HLIEModelInfo& model_info) const;
    [[nodiscard]] std::vector<server_model_meta> get_all_metas() const;
    [[nodiscard]] std::vector<ModelInstance> get_all_instances() const;
    [[nodiscard]] std::optional<HLIEModelRuntimeSnapshot> get_runtime_snapshot(
        const std::string& model_name) const;
    [[nodiscard]] std::vector<HLIEModelRuntimeSnapshot>
    get_all_runtime_snapshots() const;
    [[nodiscard]] std::optional<HLIEModelSpec> resolve_model_spec(
        const std::string& selector) const;
    [[nodiscard]] std::vector<HLIEModelSpec> get_catalog_snapshots() const;

    // Generation-fenced lifecycle transaction primitives. These compatibility
    // entry points shrink as complete load/unload operations move here.
    [[nodiscard]] std::optional<HLIEModelInfo> prepare_model_info_with_devices(
        const std::string& model_name, const std::string& devices);
    [[nodiscard]] HLIEControllerLoadBeginResult begin_load(
        const std::string& model_name,
        HLIEPendingReservationHandle& reservation,
        const std::optional<HLIEModelLaunchTicket>& preclaimed = std::nullopt);
    [[nodiscard]] HLIEControllerLoadResult load_model(
        HLIEPendingReservationHandle& reservation,
        HLIEModelInfo model_info,
        std::string service_kind,
        bool include_parent_arguments = true,
        const std::optional<HLIEModelLaunchTicket>& preclaimed = std::nullopt);
    void rollback_pending_reservation(HLIEPendingReservationHandle& reservation);
    [[nodiscard]] std::optional<HLIEModelLaunchTicket> try_begin_loading_ticket(
        const std::string& model_name);
    [[nodiscard]] HLIEEnsureReadyDecision ensure_ready(
        const std::string& model_name);
    [[nodiscard]] HLIEWaitReadyResult wait_until_ready(
        const std::string& model_name,
        std::chrono::steady_clock::time_point deadline,
        const std::function<bool()>& should_stop = {});
    [[nodiscard]] std::optional<HLIEModelLaunchTicket> get_launch_ticket(
        const std::string& model_name) const;
    [[nodiscard]] bool is_current_launch(
        const HLIEModelLaunchTicket& ticket) const;
    [[nodiscard]] bool mark_loaded_if_current(
        const HLIEModelLaunchTicket& ticket, int port);
    [[nodiscard]] bool mark_unloaded_if_current(
        const HLIEModelLaunchTicket& ticket, int port = 0);
    [[nodiscard]] bool mark_failed_if_current(
        const HLIEModelLaunchTicket& ticket, int exit_code);
    [[nodiscard]] bool mark_orphaned_if_current(
        const HLIEModelLaunchTicket& ticket, int failure_code);
    [[nodiscard]] bool mark_unloaded(const std::string& model_name,
                                     int port = 0);
    [[nodiscard]] bool update_status_if_current(
        const HLIEModelLaunchTicket& ticket, server_model_status status,
        int port);
    [[nodiscard]] bool apply_worker_status(
        const HLIEModelLaunchTicket& ticket, server_model_status status,
        int port);
    [[nodiscard]] bool publish_confirmed_stop(
        const HLIEModelLaunchTicket& ticket);
    [[nodiscard]] bool publish_confirmed_legacy_stop(
        const std::string& model_key);
    [[nodiscard]] bool update_port(const std::string& model_name, int port);
    [[nodiscard]] bool update_model_info(const std::string& model_name,
                                         const HLIEModelInfo& model_info);
    [[nodiscard]] bool update_runtime_devices(
        const std::string& model_name, const std::vector<int>& device_ids);

    [[nodiscard]] std::optional<HLIEModelLaunchTicket> begin_request_ticket(
        const std::string& model_name);
    void end_request(const HLIEModelLaunchTicket& ticket);
    [[nodiscard]] bool try_begin_eviction(const std::string& model_name);
    void abort_eviction(const std::string& model_name);

    [[nodiscard]] HLIEControllerEvictionResult evict_runtime(
        const std::string& model_key, const std::string& reason,
        bool eviction_already_claimed = false,
        std::chrono::milliseconds timeout = std::chrono::seconds(5));

    [[nodiscard]] HLIEControllerRuntimePublishResult publish_embedded_runtime(
        const HLIEEmbeddedRuntimeCreateRequest& request);
    [[nodiscard]] HLIEControllerRuntimePublishResult publish_process_runtime(
        const HLIEProcessRuntimeCreateRequest& request);
    [[nodiscard]] HLIEControllerRuntimeAdoptionResult adopt_embedded_runtime(
        HLIEPendingReservationHandle& reservation,
        const HLIEModelLaunchTicket& ticket,
        const HLIEEmbeddedRuntimeCreateRequest& request,
        int ready_port);
    [[nodiscard]] HLIEControllerRuntimeAdoptionResult adopt_process_runtime(
        HLIEPendingReservationHandle& reservation,
        const HLIEModelLaunchTicket& ticket,
        const HLIEProcessRuntimeCreateRequest& request);

    [[nodiscard]] bool stop_and_detach_runtime(
        const HLIEModelLaunchTicket& ticket, HLIERuntimeStopReason reason,
        bool force = false,
        std::chrono::milliseconds timeout = std::chrono::seconds(5));
    void reconcile_runtime_publish_failure(
        const HLIEModelLaunchTicket& ticket, bool cleanup_confirmed,
        const std::string& reason);
    void rollback_model_start(const HLIEModelLaunchTicket& ticket,
                              const std::string& reason);
    void rollback_model_start(const std::string& model_key,
                              const std::string& reason);
    void begin_draining();
    [[nodiscard]] HLIEControllerStopSummary stop_all(
        HLIERuntimeStopReason reason,
        std::chrono::milliseconds timeout = std::chrono::seconds(5));

    void on_runtime_event(const HLIERuntimeEvent& event) override;

private:
    [[nodiscard]] HLIEControllerRuntimePublishResult publish_runtime(
        std::shared_ptr<IHLIEModelRuntime> runtime);
    void notify_state_changed() const;

    HLIEModelRegistry registry_;
    HLIEResourceScheduler& scheduler_;
    HLIERuntimeDirectory& runtime_directory_;
    IHLIEModelRuntimeFactory& runtime_factory_;
    IHLIEModelBackendLauncher* backend_launcher_ = nullptr;
    StateChangedCallback state_changed_;
    mutable std::mutex state_wait_mutex_;
    mutable std::condition_variable state_changed_cv_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_MODEL_CONTROLLER_H

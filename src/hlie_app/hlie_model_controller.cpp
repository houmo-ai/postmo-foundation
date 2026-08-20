#include "hlie_app/control/hlie_model_controller.h"

#include <algorithm>
#include <utility>

#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

HLIEModelController::HLIEModelController(
    HLIEResourceScheduler& scheduler,
    HLIERuntimeDirectory& runtime_directory,
    IHLIEModelRuntimeFactory& runtime_factory,
    StateChangedCallback state_changed,
    IHLIEModelBackendLauncher* backend_launcher,
    std::shared_ptr<HLIEModelCatalog> model_catalog)
    : registry_(std::move(model_catalog)),
      scheduler_(scheduler), runtime_directory_(runtime_directory),
      runtime_factory_(runtime_factory),
      backend_launcher_(backend_launcher),
      state_changed_(std::move(state_changed)) {}

void HLIEModelController::register_model(HLIEModelInfo& model_info,
                                         const int estimated_mem_mb) {
    registry_.register_model(model_info, estimated_mem_mb);
}

bool HLIEModelController::register_model_if_absent(
    HLIEModelInfo& model_info, const int estimated_mem_mb) {
    return registry_.register_model_if_absent(model_info, estimated_mem_mb);
}

bool HLIEModelController::contains(const std::string& model_key) const {
    return registry_.contains(model_key);
}

std::optional<server_model_meta> HLIEModelController::get_meta(
    const std::string& model_name) const {
    return registry_.get_meta(model_name);
}

std::optional<HLIEModelController::ModelInstance>
HLIEModelController::get_instance_snapshot(const std::string& model_name) const {
    return registry_.get_instance_snapshot(model_name);
}

std::optional<HLIEModelController::ModelInstance>
HLIEModelController::get_exact_instance_snapshot(
    const HLIEModelInfo& model_info) const {
    return registry_.get_exact_instance_snapshot(model_info);
}

std::vector<server_model_meta> HLIEModelController::get_all_metas() const {
    return registry_.get_all_metas();
}

std::vector<HLIEModelController::ModelInstance>
HLIEModelController::get_all_instances() const {
    return registry_.get_all_instances();
}

std::optional<HLIEModelRuntimeSnapshot>
HLIEModelController::get_runtime_snapshot(const std::string& model_name) const {
    return registry_.get_runtime_snapshot(model_name);
}

std::vector<HLIEModelRuntimeSnapshot>
HLIEModelController::get_all_runtime_snapshots() const {
    return registry_.get_all_runtime_snapshots();
}

std::optional<HLIEModelSpec> HLIEModelController::resolve_model_spec(
    const std::string& selector) const {
    return registry_.catalog().resolve(selector);
}

std::vector<HLIEModelSpec> HLIEModelController::get_catalog_snapshots() const {
    return registry_.catalog().snapshots();
}

std::optional<HLIEModelInfo>
HLIEModelController::prepare_model_info_with_devices(
    const std::string& model_name, const std::string& devices) {
    return registry_.prepare_model_info_with_devices(model_name, devices);
}

HLIEControllerLoadBeginResult HLIEModelController::begin_load(
    const std::string& model_name,
    HLIEPendingReservationHandle& reservation,
    const std::optional<HLIEModelLaunchTicket>& preclaimed) {
    std::optional<HLIEModelLaunchTicket> ticket;
    if (preclaimed) {
        if (!preclaimed->valid() ||
            !registry_.is_current_launch(*preclaimed)) {
            return {HLIEControllerLoadBeginStatus::StalePreclaim,
                    std::nullopt};
        }
        ticket = preclaimed;
    } else {
        ticket = try_begin_loading_ticket(model_name);
        if (!ticket) {
            return {HLIEControllerLoadBeginStatus::BusyOrMissing,
                    std::nullopt};
        }
    }

    if (!scheduler_.bind(reservation, *ticket)) {
        scheduler_.rollback(reservation);
        (void)mark_unloaded_if_current(*ticket);
        return {HLIEControllerLoadBeginStatus::ReservationBindFailed,
                std::move(ticket)};
    }
    return {HLIEControllerLoadBeginStatus::Started, std::move(ticket)};
}

HLIEControllerLoadResult HLIEModelController::load_model(
    HLIEPendingReservationHandle& reservation,
    HLIEModelInfo model_info,
    std::string service_kind,
    const bool include_parent_arguments,
    const std::optional<HLIEModelLaunchTicket>& preclaimed) {
    if (!backend_launcher_) {
        scheduler_.rollback(reservation);
        if (preclaimed) {
            (void)mark_unloaded_if_current(*preclaimed);
        }
        return {HLIEControllerLoadStatus::BackendLauncherUnavailable};
    }

    const auto begin = begin_load(
        model_info.model_key_, reservation, preclaimed);
    if (!begin.started()) {
        scheduler_.rollback(reservation);
        HLIEControllerLoadStatus status = HLIEControllerLoadStatus::BusyOrMissing;
        if (begin.status == HLIEControllerLoadBeginStatus::StalePreclaim) {
            status = HLIEControllerLoadStatus::StalePreclaim;
        } else if (begin.status ==
                   HLIEControllerLoadBeginStatus::ReservationBindFailed) {
            status = HLIEControllerLoadStatus::ReservationBindFailed;
        }
        return {status, begin.launch_ticket};
    }

    const auto ticket = *begin.launch_ticket;
    model_info.set_device_reservation_id(reservation.id());
    auto artifact = backend_launcher_->launch({
        ticket, std::move(model_info), std::move(service_kind),
        include_parent_arguments,
    });
    if (!artifact.started()) {
        scheduler_.rollback(reservation);
        rollback_model_start(ticket, "backend_start_failed");
        return {HLIEControllerLoadStatus::BackendStartFailed, ticket,
                std::move(artifact)};
    }

    const auto adoption = artifact.runtime_kind == HLIERuntimeKind::Process
        ? adopt_process_runtime(
              reservation, ticket, artifact.process_runtime)
        : adopt_embedded_runtime(
              reservation, ticket, artifact.embedded_runtime,
              artifact.ready_port);
    if (adoption.accepted()) {
        return {
            adoption.status == HLIEControllerRuntimeAdoptionStatus::Ready
                ? HLIEControllerLoadStatus::Ready
                : HLIEControllerLoadStatus::AwaitingReady,
            ticket,
            std::move(artifact),
        };
    }

    bool cleanup_confirmed = adoption.cleanup_confirmed;
    if (!cleanup_confirmed) {
        cleanup_confirmed = backend_launcher_->cleanup_unadopted(
            artifact, "runtime_adoption_failed");
        if (cleanup_confirmed) {
            rollback_model_start(ticket, "runtime_adoption_failed");
        }
    }
    return {
        cleanup_confirmed
            ? HLIEControllerLoadStatus::RuntimeAdoptionFailed
            : HLIEControllerLoadStatus::CleanupUnconfirmed,
        ticket,
        std::move(artifact),
    };
}

void HLIEModelController::rollback_pending_reservation(
    HLIEPendingReservationHandle& reservation) {
    scheduler_.rollback(reservation);
}

std::optional<HLIEModelLaunchTicket>
HLIEModelController::try_begin_loading_ticket(const std::string& model_name) {
    const auto ticket = registry_.try_begin_loading_ticket(model_name);
    if (ticket) {
        notify_state_changed();
    }
    return ticket;
}

HLIEEnsureReadyDecision HLIEModelController::ensure_ready(
    const std::string& model_name) {
    auto snapshot = registry_.get_runtime_snapshot(model_name);
    if (!snapshot) {
        return {HLIEEnsureReadyDisposition::Missing, std::nullopt,
                std::nullopt};
    }
    if (hlie_model_state_is_ready(snapshot->state)) {
        return {HLIEEnsureReadyDisposition::Ready, std::nullopt,
                std::move(snapshot)};
    }
    if (hlie_model_state_is_loading(snapshot->state)) {
        return {HLIEEnsureReadyDisposition::WaitForExistingLaunch,
                std::nullopt, std::move(snapshot)};
    }
    if (snapshot->state == HLIEModelLifecycleState::Orphaned ||
        snapshot->state == HLIEModelLifecycleState::Draining ||
        snapshot->state == HLIEModelLifecycleState::Stopping) {
        return {HLIEEnsureReadyDisposition::Unavailable, std::nullopt,
                std::move(snapshot)};
    }

    if (auto ticket = try_begin_loading_ticket(model_name)) {
        auto claimed = registry_.get_runtime_snapshot(model_name);
        return {HLIEEnsureReadyDisposition::LaunchRequired,
                std::move(ticket), std::move(claimed)};
    }

    snapshot = registry_.get_runtime_snapshot(model_name);
    if (snapshot && hlie_model_state_is_loading(snapshot->state)) {
        return {HLIEEnsureReadyDisposition::WaitForExistingLaunch,
                std::nullopt, std::move(snapshot)};
    }
    if (snapshot && hlie_model_state_is_ready(snapshot->state)) {
        return {HLIEEnsureReadyDisposition::Ready, std::nullopt,
                std::move(snapshot)};
    }
    return {snapshot ? HLIEEnsureReadyDisposition::Unavailable
                     : HLIEEnsureReadyDisposition::Missing,
            std::nullopt, std::move(snapshot)};
}

HLIEWaitReadyResult HLIEModelController::wait_until_ready(
    const std::string& model_name,
    const std::chrono::steady_clock::time_point deadline,
    const std::function<bool()>& should_stop) {
    constexpr auto recheck_interval = std::chrono::milliseconds(100);
    std::unique_lock lock(state_wait_mutex_);
    while (true) {
        auto snapshot = registry_.get_runtime_snapshot(model_name);
        if (!snapshot) {
            return {HLIEWaitReadyStatus::Missing, std::nullopt};
        }
        if (hlie_model_state_is_ready(snapshot->state)) {
            return {HLIEWaitReadyStatus::Ready, std::move(snapshot)};
        }
        if (snapshot->state == HLIEModelLifecycleState::Failed) {
            return {HLIEWaitReadyStatus::Failed, std::move(snapshot)};
        }
        if (snapshot->state == HLIEModelLifecycleState::Unloaded) {
            return {HLIEWaitReadyStatus::Unloaded, std::move(snapshot)};
        }
        if (snapshot->state == HLIEModelLifecycleState::Orphaned) {
            return {HLIEWaitReadyStatus::Orphaned, std::move(snapshot)};
        }
        if (snapshot->state == HLIEModelLifecycleState::Draining ||
            snapshot->state == HLIEModelLifecycleState::Stopping ||
            snapshot->state == HLIEModelLifecycleState::Sleeping) {
            return {HLIEWaitReadyStatus::Unavailable, std::move(snapshot)};
        }
        if (should_stop && should_stop()) {
            return {HLIEWaitReadyStatus::Cancelled, std::move(snapshot)};
        }

        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return {HLIEWaitReadyStatus::TimedOut, std::move(snapshot)};
        }
        state_changed_cv_.wait_until(
            lock, std::min(deadline, now + recheck_interval));
    }
}

std::optional<HLIEModelLaunchTicket> HLIEModelController::get_launch_ticket(
    const std::string& model_name) const {
    return registry_.get_launch_ticket(model_name);
}

bool HLIEModelController::is_current_launch(
    const HLIEModelLaunchTicket& ticket) const {
    return registry_.is_current_launch(ticket);
}

bool HLIEModelController::mark_loaded_if_current(
    const HLIEModelLaunchTicket& ticket, const int port) {
    const bool updated = registry_.mark_loaded_if_current(ticket, port);
    if (updated) {
        notify_state_changed();
    }
    return updated;
}

bool HLIEModelController::mark_unloaded_if_current(
    const HLIEModelLaunchTicket& ticket, const int port) {
    const bool updated = registry_.mark_unloaded_if_current(ticket, port);
    if (updated) {
        notify_state_changed();
    }
    return updated;
}

bool HLIEModelController::mark_failed_if_current(
    const HLIEModelLaunchTicket& ticket, const int exit_code) {
    const bool updated = registry_.mark_failed_if_current(ticket, exit_code);
    if (updated) {
        notify_state_changed();
    }
    return updated;
}

bool HLIEModelController::mark_orphaned_if_current(
    const HLIEModelLaunchTicket& ticket, const int failure_code) {
    const bool updated = registry_.mark_orphaned_if_current(ticket, failure_code);
    if (updated) {
        notify_state_changed();
    }
    return updated;
}

bool HLIEModelController::mark_unloaded(const std::string& model_name,
                                        const int port) {
    const bool updated = registry_.mark_unloaded(model_name, port);
    if (updated) {
        notify_state_changed();
    }
    return updated;
}

bool HLIEModelController::update_status_if_current(
    const HLIEModelLaunchTicket& ticket, const server_model_status status,
    const int port) {
    const bool updated = registry_.update_status_if_current(ticket, status, port);
    if (updated) {
        notify_state_changed();
    }
    return updated;
}

bool HLIEModelController::apply_worker_status(
    const HLIEModelLaunchTicket& ticket, const server_model_status status,
    const int port) {
    if (!registry_.is_current_launch(ticket)) {
        return false;
    }
    if (status == SERVER_MODEL_STATUS_LOADED && !scheduler_.commit(ticket)) {
        return false;
    }
    if (update_status_if_current(ticket, status, port)) {
        if (status == SERVER_MODEL_STATUS_UNLOADED) {
            scheduler_.release(ticket);
        }
        return true;
    }
    if (status == SERVER_MODEL_STATUS_LOADED) {
        scheduler_.release(ticket);
    }
    return false;
}

bool HLIEModelController::publish_confirmed_stop(
    const HLIEModelLaunchTicket& ticket) {
    (void)runtime_directory_.detach(ticket);
    scheduler_.release(ticket);
    return mark_unloaded_if_current(ticket);
}

bool HLIEModelController::publish_confirmed_legacy_stop(
    const std::string& model_key) {
    scheduler_.release(model_key);
    return mark_unloaded(model_key);
}

bool HLIEModelController::update_port(const std::string& model_name,
                                      const int port) {
    return registry_.update_port(model_name, port);
}

bool HLIEModelController::update_model_info(
    const std::string& model_name, const HLIEModelInfo& model_info) {
    return registry_.update_model_info(model_name, model_info);
}

bool HLIEModelController::update_runtime_devices(
    const std::string& model_name, const std::vector<int>& device_ids) {
    return registry_.update_runtime_devices(model_name, device_ids);
}

std::optional<HLIEModelLaunchTicket>
HLIEModelController::begin_request_ticket(const std::string& model_name) {
    return registry_.begin_request_ticket(model_name);
}

void HLIEModelController::end_request(const HLIEModelLaunchTicket& ticket) {
    registry_.end_request(ticket);
}

bool HLIEModelController::try_begin_eviction(const std::string& model_name) {
    const bool claimed = registry_.try_begin_eviction(model_name);
    if (claimed) {
        notify_state_changed();
    }
    return claimed;
}

void HLIEModelController::abort_eviction(const std::string& model_name) {
    registry_.abort_eviction(model_name);
    notify_state_changed();
}

HLIEControllerEvictionResult HLIEModelController::evict_runtime(
    const std::string& model_key, const std::string& reason,
    const bool eviction_already_claimed,
    const std::chrono::milliseconds timeout) {
    if (!eviction_already_claimed && !try_begin_eviction(model_key)) {
        return {HLIEControllerEvictionStatus::Rejected, {}, false};
    }

    const auto ticket = registry_.get_launch_ticket(model_key);
    if (!ticket) {
        abort_eviction(model_key);
        return {HLIEControllerEvictionStatus::RuntimeMissing, {}, false};
    }
    const auto runtime = runtime_directory_.find(*ticket);
    if (!runtime) {
        abort_eviction(model_key);
        return {HLIEControllerEvictionStatus::RuntimeMissing, {}, false};
    }
    const auto identity = runtime->identity();
    if (!stop_and_detach_runtime(*ticket, HLIERuntimeStopReason::Eviction,
                                 false, timeout)) {
        LOG_ERROR("Model eviction retained unconfirmed runtime ownership: model={}, generation={}, runtime_id={}, reason={}",
                  model_key, ticket->generation, identity.runtime_id, reason);
        return {HLIEControllerEvictionStatus::StopUnconfirmed, identity, false};
    }

    scheduler_.release(*ticket);
    const bool registry_updated = mark_unloaded_if_current(*ticket);
    LOG_INFO("Model runtime eviction confirmed: model={}, generation={}, runtime_id={}, kind={}, reason={}, registry_update_stale={}",
             model_key, ticket->generation, identity.runtime_id,
             identity.kind == HLIERuntimeKind::Process ? "process" : "embedded",
             reason, !registry_updated);
    return {HLIEControllerEvictionStatus::Evicted, identity,
            !registry_updated};
}

HLIEControllerRuntimePublishResult
HLIEModelController::publish_embedded_runtime(
    const HLIEEmbeddedRuntimeCreateRequest& request) {
    return publish_runtime(runtime_factory_.create_embedded(request));
}

HLIEControllerRuntimePublishResult
HLIEModelController::publish_process_runtime(
    const HLIEProcessRuntimeCreateRequest& request) {
    return publish_runtime(runtime_factory_.create_process(request));
}

HLIEControllerRuntimeAdoptionResult
HLIEModelController::adopt_embedded_runtime(
    HLIEPendingReservationHandle& reservation,
    const HLIEModelLaunchTicket& ticket,
    const HLIEEmbeddedRuntimeCreateRequest& request,
    const int ready_port) {
    if (!ticket.valid() || !request.launch_ticket.matches(ticket) ||
        !registry_.is_current_launch(ticket)) {
        scheduler_.rollback(reservation);
        return {HLIEControllerRuntimeAdoptionStatus::StaleLaunch, false};
    }

    const auto published = publish_embedded_runtime(request);
    if (!published.published) {
        if (published.cleanup_confirmed) {
            scheduler_.rollback(reservation);
        } else if (!scheduler_.track(reservation, ticket)) {
            LOG_ERROR("Failed to retain reservation after uncertain embedded runtime publication: model={}, generation={}",
                      ticket.model_key, ticket.generation);
        }
        reconcile_runtime_publish_failure(
            ticket, published.cleanup_confirmed,
            "embedded_runtime_publish_failed");
        return {HLIEControllerRuntimeAdoptionStatus::RuntimePublicationFailed,
                published.cleanup_confirmed};
    }

    if (!scheduler_.track(reservation, ticket) || !scheduler_.commit(ticket)) {
        const bool stopped = stop_and_detach_runtime(
            ticket, HLIERuntimeStopReason::StartupRollback, true);
        reconcile_runtime_publish_failure(
            ticket, stopped, "embedded_reservation_commit_failed");
        return {HLIEControllerRuntimeAdoptionStatus::ReservationTransitionFailed,
                stopped};
    }

    if (!mark_loaded_if_current(ticket, ready_port)) {
        const bool stopped = stop_and_detach_runtime(
            ticket, HLIERuntimeStopReason::StartupRollback, true);
        reconcile_runtime_publish_failure(
            ticket, stopped, "embedded_runtime_stale_publish");
        return {HLIEControllerRuntimeAdoptionStatus::ReadyPublicationFailed,
                stopped};
    }
    return {HLIEControllerRuntimeAdoptionStatus::Ready, true};
}

HLIEControllerRuntimeAdoptionResult
HLIEModelController::adopt_process_runtime(
    HLIEPendingReservationHandle& reservation,
    const HLIEModelLaunchTicket& ticket,
    const HLIEProcessRuntimeCreateRequest& request) {
    auto runtime = runtime_factory_.create_process(request);
    if (!runtime) {
        if (!scheduler_.track(reservation, ticket)) {
            LOG_ERROR("Failed to retain reservation after process RuntimeFactory failure: model={}, generation={}",
                      ticket.model_key, ticket.generation);
        }
        reconcile_runtime_publish_failure(
            ticket, false, "process_runtime_factory_failed");
        return {
            HLIEControllerRuntimeAdoptionStatus::RuntimePublicationFailed,
            false};
    }
    const auto cleanup_unpublished = [&](const std::string& reason) {
        runtime->begin_draining();
        const auto stopped = runtime->stop({
            std::chrono::steady_clock::now() + std::chrono::seconds(5),
            HLIERuntimeStopReason::StartupRollback,
            true,
        });
        if (stopped.confirmed) {
            scheduler_.rollback(reservation);
            rollback_model_start(ticket, reason);
        } else {
            if (!scheduler_.track(reservation, ticket)) {
                LOG_ERROR("Failed to retain reservation after unconfirmed process cleanup: model={}, generation={}",
                          ticket.model_key, ticket.generation);
            }
            (void)mark_orphaned_if_current(ticket, stopped.native_code);
        }
        return stopped.confirmed;
    };

    if (!ticket.valid() || !request.identity.launch_ticket.matches(ticket) ||
        !registry_.is_current_launch(ticket)) {
        return {HLIEControllerRuntimeAdoptionStatus::StaleLaunch,
                cleanup_unpublished("process_runtime_stale_launch")};
    }
    if (!scheduler_.track(reservation, ticket)) {
        return {
            HLIEControllerRuntimeAdoptionStatus::ReservationTransitionFailed,
            cleanup_unpublished("process_reservation_track_failed")};
    }

    const auto published = publish_runtime(std::move(runtime));
    if (!published.published) {
        reconcile_runtime_publish_failure(
            ticket, published.cleanup_confirmed,
            "process_runtime_publish_failed");
        return {HLIEControllerRuntimeAdoptionStatus::RuntimePublicationFailed,
                published.cleanup_confirmed};
    }
    return {HLIEControllerRuntimeAdoptionStatus::AwaitingReady, true};
}

HLIEControllerRuntimePublishResult HLIEModelController::publish_runtime(
    std::shared_ptr<IHLIEModelRuntime> runtime) {
    if (!runtime) {
        return {};
    }
    const auto identity = runtime->identity();
    if (runtime_directory_.publish(runtime)) {
        LOG_INFO("Controller published model runtime: model={}, generation={}, runtime_id={}, kind={}",
                 identity.launch_ticket.model_key,
                 identity.launch_ticket.generation, identity.runtime_id,
                 identity.kind == HLIERuntimeKind::Process ? "process" : "embedded");
        return {true, false, identity};
    }

    LOG_ERROR("Controller rejected model runtime publication: model={}, generation={}, runtime_id={}",
              identity.launch_ticket.model_key,
              identity.launch_ticket.generation, identity.runtime_id);
    runtime->begin_draining();
    const auto stopped = runtime->stop({
        std::chrono::steady_clock::now() + std::chrono::seconds(5),
        HLIERuntimeStopReason::StartupRollback,
        true,
    });
    if (!stopped.confirmed && identity.launch_ticket.valid()) {
        (void)mark_orphaned_if_current(
            identity.launch_ticket, stopped.native_code);
    }
    return {false, stopped.confirmed, identity};
}

bool HLIEModelController::stop_and_detach_runtime(
    const HLIEModelLaunchTicket& ticket, const HLIERuntimeStopReason reason,
    const bool force, const std::chrono::milliseconds timeout) {
    auto runtime = runtime_directory_.detach(ticket);
    if (!runtime) {
        return false;
    }

    runtime->begin_draining();
    const auto stopped = runtime->stop({
        std::chrono::steady_clock::now() + timeout,
        reason,
        force,
    });
    if (stopped.confirmed) {
        return true;
    }

    const auto identity = runtime->identity();
    LOG_ERROR("Runtime stop was not confirmed; retaining handle: model={}, generation={}, runtime_id={}",
              ticket.model_key, ticket.generation, identity.runtime_id);
    if (!runtime_directory_.publish(runtime)) {
        LOG_ERROR("Failed to restore unconfirmed runtime handle: model={}, generation={}, runtime_id={}",
                  ticket.model_key, ticket.generation, identity.runtime_id);
    }
    (void)mark_orphaned_if_current(ticket, stopped.native_code);
    return false;
}

void HLIEModelController::reconcile_runtime_publish_failure(
    const HLIEModelLaunchTicket& ticket, const bool cleanup_confirmed,
    const std::string& reason) {
    if (cleanup_confirmed) {
        rollback_model_start(ticket, reason);
        return;
    }

    LOG_ERROR("Runtime publication failed without confirmed cleanup; retaining reservation: model={}, generation={}, reason={}",
              ticket.model_key, ticket.generation, reason);
    (void)mark_orphaned_if_current(ticket, -4);
}

void HLIEModelController::rollback_model_start(
    const HLIEModelLaunchTicket& ticket, const std::string& reason) {
    LOG_WARN("Model start rollback: key={}, generation={}, reason={}",
             ticket.model_key, ticket.generation, reason);
    if (runtime_directory_.find(ticket) &&
        !stop_and_detach_runtime(ticket, HLIERuntimeStopReason::StartupRollback,
                                 true)) {
        LOG_ERROR("Model start rollback retained an unconfirmed runtime: key={}, generation={}, reason={}",
                  ticket.model_key, ticket.generation, reason);
        (void)mark_orphaned_if_current(ticket, -5);
        return;
    }

    scheduler_.release(ticket);
    if (!mark_unloaded_if_current(ticket)) {
        LOG_DEBUG("Ignored stale model rollback: key={}, generation={}",
                  ticket.model_key, ticket.generation);
    }
}

void HLIEModelController::rollback_model_start(
    const std::string& model_key, const std::string& reason) {
    if (const auto ticket = registry_.get_launch_ticket(model_key)) {
        rollback_model_start(*ticket, reason);
        return;
    }
    LOG_WARN("Model start rollback without launch ticket: key={}, reason={}",
             model_key, reason);
    scheduler_.release(model_key);
    (void)mark_unloaded(model_key);
}

void HLIEModelController::begin_draining() {
    for (const auto& health : runtime_directory_.health_snapshots()) {
        if (auto runtime = runtime_directory_.find(health.identity.launch_ticket)) {
            runtime->begin_draining();
        }
    }
}

HLIEControllerStopSummary HLIEModelController::stop_all(
    const HLIERuntimeStopReason reason,
    const std::chrono::milliseconds timeout) {
    HLIEControllerStopSummary summary;
    for (auto& runtime : runtime_directory_.detach_all()) {
        const auto identity = runtime->identity();
        runtime->begin_draining();
        const auto stopped = runtime->stop({
            std::chrono::steady_clock::now() + timeout,
            reason,
            false,
        });
        if (!stopped.confirmed) {
            ++summary.unconfirmed;
            LOG_ERROR("Controller runtime stop was not confirmed: model={}, generation={}, runtime_id={}",
                      identity.launch_ticket.model_key,
                      identity.launch_ticket.generation, identity.runtime_id);
            if (!runtime_directory_.publish(runtime)) {
                LOG_ERROR("Controller failed to restore unconfirmed runtime: model={}, generation={}, runtime_id={}",
                          identity.launch_ticket.model_key,
                          identity.launch_ticket.generation,
                          identity.runtime_id);
            }
            (void)mark_orphaned_if_current(
                identity.launch_ticket, stopped.native_code);
            continue;
        }

        ++summary.stopped;
        scheduler_.release(identity.launch_ticket);
        if (!registry_.mark_unloaded_if_current(identity.launch_ticket)) {
            ++summary.stale_registry_updates;
        }
    }
    if (summary.stopped > summary.stale_registry_updates) {
        notify_state_changed();
    }
    return summary;
}

void HLIEModelController::on_runtime_event(const HLIERuntimeEvent& event) {
    const auto& ticket = event.launch_ticket;
    if (!ticket.valid()) {
        LOG_DEBUG("Worker runtime event has no launch ticket; state unchanged: runtime_id={}",
                  event.runtime_id);
        return;
    }

    if (event.kind == HLIERuntimeEventKind::Exited) {
        if (event.expected_exit) {
            // The synchronous stop transaction owns expected-exit publication.
            return;
        }
        if (!event.exit_confirmed) {
            LOG_ERROR("Worker exit was not confirmed; retaining runtime and reservation: model={}, generation={}, runtime_id={}",
                      ticket.model_key, ticket.generation, event.runtime_id);
            (void)mark_orphaned_if_current(ticket, event.exit_code);
            return;
        }
        (void)runtime_directory_.detach(ticket);
        scheduler_.release(ticket);
        if (!mark_failed_if_current(ticket, event.exit_code)) {
            LOG_DEBUG("Ignored stale worker exit: model={}, generation={}, runtime_id={}",
                      ticket.model_key, ticket.generation, event.runtime_id);
        }
        return;
    }

    if (event.kind == HLIERuntimeEventKind::ReadyTimeout) {
        if (!event.exit_confirmed) {
            LOG_ERROR("Worker READY timeout cleanup was not confirmed; retaining reservation: runtime_id={}, generation={}",
                      event.runtime_id, ticket.generation);
            (void)mark_orphaned_if_current(ticket, -2);
            return;
        }
        (void)runtime_directory_.detach(ticket);
        scheduler_.release(ticket);
        (void)mark_failed_if_current(ticket, -2);
        return;
    }

    const auto snapshot = registry_.get_runtime_snapshot(ticket.model_key);
    if (snapshot && snapshot->state == HLIEModelLifecycleState::Ready &&
        registry_.is_current_launch(ticket)) {
        LOG_DEBUG("Ignored duplicate Worker READY event: model={}, generation={}, runtime_id={}",
                  ticket.model_key, ticket.generation, event.runtime_id);
        return;
    }

    if (snapshot && hlie_model_state_is_loading(snapshot->state) &&
        registry_.is_current_launch(ticket) && scheduler_.commit(ticket) &&
        mark_loaded_if_current(ticket, event.port)) {
        return;
    }

    LOG_ERROR("Worker readiness reconciliation failed: runtime_id={}, model={}, generation={}",
              event.runtime_id, ticket.model_key, ticket.generation);
    if (!stop_and_detach_runtime(ticket, HLIERuntimeStopReason::Recovery, true)) {
        LOG_ERROR("Worker readiness reconciliation stop was not confirmed; retaining reservation: runtime_id={}, generation={}",
                  event.runtime_id, ticket.generation);
        return;
    }
    scheduler_.release(ticket);
    (void)mark_failed_if_current(ticket, -3);
}

void HLIEModelController::notify_state_changed() const {
    state_changed_cv_.notify_all();
    if (state_changed_) {
        state_changed_();
    }
}

HLIE_NAMESPACE_END

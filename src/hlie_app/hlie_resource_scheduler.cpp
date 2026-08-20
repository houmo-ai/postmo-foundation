#include "hlie_app/resource/hlie_resource_scheduler.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <unordered_set>

#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

HLIEResourceScheduler::HLIEResourceScheduler(
    std::shared_ptr<IHLIEDeviceReservationProvider> provider,
    HLIEModelResidencyPolicy residency_policy)
    : provider_(std::move(provider)), residency_policy_(std::move(residency_policy)) {
    if (!provider_) {
        throw std::invalid_argument("ResourceScheduler requires a reservation provider");
    }
}

std::optional<HLIEPendingReservationHandle> HLIEResourceScheduler::reserve(
    const std::string& model_key, const std::uint64_t total_mb,
    const HLIEPlacementIntent& placement, const bool exclusive) {
    if (model_key.empty() || total_mb == 0 || !placement.valid()) {
        return std::nullopt;
    }

    HLIEDeviceReservationRequest request;
    request.model_key = model_key;
    request.total_mb = total_mb;
    request.device_count = static_cast<std::size_t>(placement.required_device_count);
    request.requested_devices = placement.requested_device_ids;
    request.exclusive = exclusive;
    // Preserve the established single-card admission behavior. The provider
    // remains responsible for applying its safety margin and observations.
    request.allow_single_device_capacity_limit = placement.required_device_count == 1;
    auto reservation = provider_->reserve_resources(request);
    if (!reservation) {
        return std::nullopt;
    }
    return HLIEPendingReservationHandle(std::move(*reservation));
}

bool HLIEResourceScheduler::bind(HLIEPendingReservationHandle& reservation,
                                 const HLIEModelLaunchTicket& ticket) {
    if (!reservation.valid() || !ticket.valid()) {
        return false;
    }
    if (!provider_->bind_reservation(reservation.id(), ticket.model_key,
                                     ticket.generation)) {
        return false;
    }
    return reservation.bind(ticket);
}

bool HLIEResourceScheduler::track(HLIEPendingReservationHandle& reservation,
                                  const HLIEModelLaunchTicket& ticket) {
    if (!reservation.valid() || !ticket.valid() || !reservation.ticket() ||
        !reservation.ticket()->matches(ticket)) {
        return false;
    }

    HLIEDeviceReservationId replaced = 0;
    {
        std::lock_guard lock(mutex_);
        const auto current = reservations_.find(ticket.model_key);
        if (current != reservations_.end()) {
            replaced = binding_id(current->second);
            reservations_.erase(current);
        }
        reservations_.emplace(
            ticket.model_key,
            Binding{ticket, TrackedReservation(std::move(reservation))});
    }
    if (replaced != 0) {
        (void)provider_->release_reservation(replaced);
    }
    return true;
}

bool HLIEResourceScheduler::commit(const HLIEModelLaunchTicket& ticket) {
    if (!ticket.valid()) {
        return false;
    }
    std::lock_guard lock(mutex_);
    const auto it = reservations_.find(ticket.model_key);
    if (it == reservations_.end() || !it->second.ticket.matches(ticket)) {
        return false;
    }
    auto* pending = std::get_if<HLIEPendingReservationHandle>(&it->second.reservation);
    if (!pending) {
        const auto* committed = std::get_if<HLIECommittedReservationHandle>(
            &it->second.reservation);
        return committed && committed->valid() && committed->ticket().matches(ticket);
    }
    if (!provider_->commit_reservation(pending->id())) {
        return false;
    }
    it->second.reservation = HLIECommittedReservationHandle(
        pending->reservation(), ticket);
    return true;
}

void HLIEResourceScheduler::rollback(HLIEPendingReservationHandle& reservation) {
    if (!reservation.valid()) {
        return;
    }
    const auto id = reservation.id();
    reservation.reset();
    (void)provider_->rollback_reservation(id);
}

void HLIEResourceScheduler::release(const HLIEModelLaunchTicket& ticket) {
    if (!ticket.valid()) {
        return;
    }
    HLIEDeviceReservationId id = 0;
    {
        std::lock_guard lock(mutex_);
        const auto it = reservations_.find(ticket.model_key);
        if (it == reservations_.end() || !it->second.ticket.matches(ticket)) {
            return;
        }
        id = binding_id(it->second);
        reservations_.erase(it);
    }
    (void)provider_->release_reservation(id);
}

void HLIEResourceScheduler::release(const std::string& model_key) {
    HLIEDeviceReservationId id = 0;
    {
        std::lock_guard lock(mutex_);
        const auto it = reservations_.find(model_key);
        if (it == reservations_.end()) {
            return;
        }
        id = binding_id(it->second);
        reservations_.erase(it);
    }
    (void)provider_->release_reservation(id);
}

void HLIEResourceScheduler::release_all() {
    std::vector<HLIEDeviceReservationId> ids;
    {
        std::lock_guard lock(mutex_);
        ids.reserve(reservations_.size());
        for (const auto& [_, binding] : reservations_) {
            ids.push_back(binding_id(binding));
        }
        reservations_.clear();
    }
    for (const auto id : ids) {
        (void)provider_->release_reservation(id);
    }
}

void HLIEResourceScheduler::update_residency_policy(HLIEModelResidencyPolicy policy) {
    std::lock_guard lock(mutex_);
    residency_policy_ = std::move(policy);
}

HLIEModelResidencyPolicy HLIEResourceScheduler::residency_policy() const {
    std::lock_guard lock(mutex_);
    return residency_policy_;
}

HLIEResidencyPlan HLIEResourceScheduler::plan_for_resource_pressure(
    const std::vector<HLIEModelRuntimeSnapshot>& models,
    const std::string& requested_model) const {
    const auto policy = residency_policy();
    const HLIEModelResidencyManager residency(policy);
    return {policy, residency.select_lru_candidate(models, requested_model),
            "resource_pressure"};
}

HLIEResidencyPlan HLIEResourceScheduler::plan_for_loaded_limit(
    const std::vector<HLIEModelRuntimeSnapshot>& models,
    const std::string& requested_model) const {
    const auto policy = residency_policy();
    const HLIEModelResidencyManager residency(policy);
    return {policy, residency.select_for_loaded_limit(models, requested_model),
            "max_loaded_models"};
}

HLIEResidencyPlan HLIEResourceScheduler::plan_for_idle_ttl(
    const std::vector<HLIEModelRuntimeSnapshot>& models, const std::time_t now) const {
    const auto policy = residency_policy();
    const HLIEModelResidencyManager residency(policy);
    return {policy, residency.select_for_idle_ttl(models, now), "idle_ttl"};
}

void HLIEResourceScheduler::reconcile_expired_pending(
    const std::vector<HLIEModelLaunchTicket>& protecting_owners) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    const auto all = provider_->get_reservations();
    std::unordered_set<HLIEDeviceReservationId> protected_ids;
    for (const auto& reservation : all) {
        if (reservation.state != HLIEDeviceReservationState::Pending ||
            reservation.expires_at > now) {
            continue;
        }
        if (std::any_of(protecting_owners.begin(), protecting_owners.end(),
                        [&](const auto& owner) { return owner_matches(reservation, owner); })) {
            protected_ids.insert(reservation.id);
        }
    }

    const auto previous_protected = expired_pending_protected_;
    expired_pending_protected_ = protected_ids.size();
    if (expired_pending_protected_ > 0 && previous_protected == 0) {
        LOG_WARN("Expired pending device reservations remain protected by live owners: count={}",
                 expired_pending_protected_);
    } else if (expired_pending_protected_ == 0 && previous_protected > 0) {
        LOG_INFO("Expired pending device reservation owner protection cleared: previous_count={}",
                 previous_protected);
    } else if (expired_pending_protected_ != previous_protected) {
        LOG_DEBUG("Expired protected pending reservation count changed: previous={}, current={}",
                  previous_protected, expired_pending_protected_);
    }

    const auto expired = provider_->rollback_expired_pending_reservations(protected_ids);
    if (expired.empty()) {
        return;
    }
    std::unordered_set<HLIEDeviceReservationId> expired_ids;
    for (const auto& reservation : expired) {
        expired_ids.insert(reservation.id);
    }
    for (auto it = reservations_.begin(); it != reservations_.end();) {
        if (expired_ids.find(binding_id(it->second)) != expired_ids.end()) {
            it = reservations_.erase(it);
        } else {
            ++it;
        }
    }
    expired_pending_rollbacks_ += expired.size();
    LOG_WARN("Reconciled {} expired orphan pending device reservation(s)", expired.size());
}

HLIEResourceSchedulerMetrics HLIEResourceScheduler::metrics() const {
    std::lock_guard lock(mutex_);
    HLIEResourceSchedulerMetrics result;
    result.expired_pending_rollbacks = expired_pending_rollbacks_;
    result.expired_pending_protected = expired_pending_protected_;
    for (const auto& [_, binding] : reservations_) {
        if (std::holds_alternative<HLIEPendingReservationHandle>(binding.reservation)) {
            ++result.tracked_pending;
        } else {
            ++result.tracked_committed;
        }
    }
    return result;
}

std::vector<HLIEDeviceLedgerSnapshot> HLIEResourceScheduler::ledger_snapshots() const {
    return provider_->get_ledger_snapshots();
}

HLIEDeviceReservationId HLIEResourceScheduler::binding_id(const Binding& binding) {
    return std::visit([](const auto& reservation) { return reservation.id(); },
                      binding.reservation);
}

bool HLIEResourceScheduler::owner_matches(
    const HLIEDeviceReservation& reservation,
    const HLIEModelLaunchTicket& owner) {
    return owner.valid() && reservation.generation != 0 &&
           reservation.model_key == owner.model_key &&
           reservation.generation == owner.generation;
}

HLIE_NAMESPACE_END

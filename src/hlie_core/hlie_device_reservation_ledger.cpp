#include "hlie_core/resource/hlie_device_reservation_ledger.h"

#include <algorithm>
#include <limits>
#include <set>

HLIE_NAMESPACE_BEGIN

namespace {
std::uint64_t saturating_add(const std::uint64_t lhs, const std::uint64_t rhs) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return lhs + rhs;
}

void saturating_subtract(std::uint64_t& value, const std::uint64_t amount) {
    value = amount >= value ? 0 : value - amount;
}
} // namespace

HLIEDeviceReservationLedger::HLIEDeviceReservationLedger(const std::uint64_t safety_margin_mb)
    : safety_margin_mb_(safety_margin_mb) {
}

void HLIEDeviceReservationLedger::reset() {
    std::lock_guard lock(mutex_);
    devices_.clear();
    reservations_.clear();
    next_id_ = 1;
}

void HLIEDeviceReservationLedger::set_pending_ttl(const std::chrono::seconds pending_ttl) {
    std::lock_guard lock(mutex_);
    pending_ttl_ = std::max(std::chrono::seconds{1}, pending_ttl);
}

void HLIEDeviceReservationLedger::update_observation(const HLIEDeviceObservation& observation) {
    if (observation.device_id < 0 || observation.total_mb == 0) {
        return;
    }

    std::lock_guard lock(mutex_);
    auto& entry = devices_[observation.device_id];
    // HAL reports aggregate device usage and cannot attribute a positive delta
    // to this process, a Worker, or an external/root service. Keep committed
    // memory as unobserved for admission purposes until its reservation is
    // released. This intentionally trades utilization for the safety invariant
    // that unrelated HAL growth can never erase framework-owned capacity.
    entry.total_mb = observation.total_mb;
    entry.observed_used_mb = std::min(observation.used_mb, observation.total_mb);
    entry.observed_available_mb = std::min(observation.available_mb, observation.total_mb);
}

std::uint64_t HLIEDeviceReservationLedger::effective_used_locked(const DeviceEntry& entry) const {
    return saturating_add(entry.observed_used_mb,
                          saturating_add(entry.pending_reserved_mb, entry.committed_unobserved_mb));
}

std::uint64_t HLIEDeviceReservationLedger::reserved_size_locked(
    const DeviceEntry& entry, const std::uint64_t requested_mb, const bool allow_capacity_limit) const {
    return allow_capacity_limit ? std::min(requested_mb, entry.total_mb) : requested_mb;
}

bool HLIEDeviceReservationLedger::can_reserve_locked(const int device_id, const std::uint64_t size_mb,
                                                     const bool exclusive, const bool allow_capacity_limit) const {
    const auto it = devices_.find(device_id);
    if (it == devices_.end()) {
        return false;
    }

    const auto& entry = it->second;
    if (entry.exclusive_reservations > 0) {
        return false;
    }

    const std::uint64_t effective_used = effective_used_locked(entry);
    if (exclusive && effective_used != 0) {
        return false;
    }

    if (allow_capacity_limit && size_mb > entry.total_mb) {
        return effective_used == 0;
    }

    const std::uint64_t required = saturating_add(size_mb, safety_margin_mb_);
    return effective_used <= entry.total_mb && required <= entry.total_mb - effective_used;
}

std::optional<HLIEDeviceReservation> HLIEDeviceReservationLedger::reserve(
    const HLIEDeviceReservationRequest& request) {
    if (request.total_mb == 0 || request.device_count == 0) {
        return std::nullopt;
    }

    std::lock_guard lock(mutex_);
    std::vector<int> selected;
    if (!request.requested_devices.empty()) {
        std::set<int> unique(request.requested_devices.begin(), request.requested_devices.end());
        selected.assign(unique.begin(), unique.end());
        if (selected.size() != request.device_count) {
            return std::nullopt;
        }
    } else {
        const std::uint64_t per_device = (request.total_mb + request.device_count - 1) / request.device_count;
        std::vector<int> idle;
        std::vector<int> used;
        for (const auto& [device_id, entry] : devices_) {
            if (!can_reserve_locked(device_id, per_device, request.exclusive,
                                    request.allow_single_device_capacity_limit && request.device_count == 1)) {
                continue;
            }
            (effective_used_locked(entry) == 0 ? idle : used).push_back(device_id);
        }
        selected.insert(selected.end(), idle.begin(), idle.end());
        selected.insert(selected.end(), used.begin(), used.end());
        if (selected.size() < request.device_count) {
            return std::nullopt;
        }
        selected.resize(request.device_count);
    }

    const std::uint64_t per_device = (request.total_mb + selected.size() - 1) / selected.size();
    const bool allow_capacity_limit = request.allow_single_device_capacity_limit && selected.size() == 1;
    for (const int device_id : selected) {
        if (!can_reserve_locked(device_id, per_device, request.exclusive, allow_capacity_limit)) {
            return std::nullopt;
        }
    }

    HLIEDeviceReservation reservation;
    reservation.id = next_id_++;
    if (reservation.id == 0) {
        reservation.id = next_id_++;
    }
    reservation.model_key = request.model_key;
    reservation.generation = request.generation;
    reservation.total_mb = request.total_mb;
    reservation.devices = std::move(selected);
    reservation.exclusive = request.exclusive;
    reservation.per_device_mb.reserve(reservation.devices.size());
    reservation.created_at = std::chrono::steady_clock::now();
    reservation.expires_at = reservation.created_at + pending_ttl_;

    for (const int device_id : reservation.devices) {
        auto& entry = devices_.at(device_id);
        const std::uint64_t reserved_mb = reserved_size_locked(entry, per_device, allow_capacity_limit);
        reservation.per_device_mb.push_back(reserved_mb);
        entry.pending_reserved_mb = saturating_add(entry.pending_reserved_mb, reserved_mb);
        if (reservation.exclusive) {
            ++entry.exclusive_reservations;
        }
    }

    reservations_.emplace(reservation.id, reservation);
    return reservation;
}

void HLIEDeviceReservationLedger::remove_pending_locked(const HLIEDeviceReservation& reservation) {
    for (std::size_t i = 0; i < reservation.devices.size(); ++i) {
        auto it = devices_.find(reservation.devices[i]);
        if (it == devices_.end()) {
            continue;
        }
        saturating_subtract(it->second.pending_reserved_mb, reservation.per_device_mb[i]);
        if (reservation.exclusive && it->second.exclusive_reservations > 0) {
            --it->second.exclusive_reservations;
        }
    }
}

void HLIEDeviceReservationLedger::remove_committed_locked(const HLIEDeviceReservation& reservation) {
    for (std::size_t i = 0; i < reservation.devices.size(); ++i) {
        auto it = devices_.find(reservation.devices[i]);
        if (it == devices_.end()) {
            continue;
        }
        saturating_subtract(it->second.committed_reserved_mb, reservation.per_device_mb[i]);
        saturating_subtract(it->second.committed_unobserved_mb, reservation.per_device_mb[i]);
        if (reservation.exclusive && it->second.exclusive_reservations > 0) {
            --it->second.exclusive_reservations;
        }
    }
}

bool HLIEDeviceReservationLedger::bind(const HLIEDeviceReservationId reservation_id,
                                       const std::string& model_key,
                                       const std::uint64_t generation) {
    if (reservation_id == 0 || model_key.empty() || generation == 0) {
        return false;
    }

    std::lock_guard lock(mutex_);
    const auto it = reservations_.find(reservation_id);
    if (it == reservations_.end() || it->second.state != HLIEDeviceReservationState::Pending) {
        return false;
    }

    auto& reservation = it->second;
    if (reservation.generation != 0 &&
        (reservation.generation != generation || reservation.model_key != model_key)) {
        return false;
    }
    reservation.model_key = model_key;
    reservation.generation = generation;
    // Binding proves that the startup transaction is still making progress.
    // Renew the lease so an expiry snapshot racing with bind cannot roll back
    // a reservation that has just acquired a live owner.
    reservation.expires_at = std::chrono::steady_clock::now() + pending_ttl_;
    return true;
}

bool HLIEDeviceReservationLedger::commit(const HLIEDeviceReservationId reservation_id) {
    std::lock_guard lock(mutex_);
    const auto it = reservations_.find(reservation_id);
    if (it == reservations_.end()) {
        return false;
    }
    if (it->second.state == HLIEDeviceReservationState::Committed) {
        return true;
    }

    auto& reservation = it->second;
    for (std::size_t i = 0; i < reservation.devices.size(); ++i) {
        auto& entry = devices_.at(reservation.devices[i]);
        saturating_subtract(entry.pending_reserved_mb, reservation.per_device_mb[i]);
        entry.committed_reserved_mb = saturating_add(entry.committed_reserved_mb, reservation.per_device_mb[i]);
        entry.committed_unobserved_mb = saturating_add(entry.committed_unobserved_mb, reservation.per_device_mb[i]);
    }
    reservation.state = HLIEDeviceReservationState::Committed;
    return true;
}

bool HLIEDeviceReservationLedger::rollback(const HLIEDeviceReservationId reservation_id) {
    std::lock_guard lock(mutex_);
    const auto it = reservations_.find(reservation_id);
    if (it == reservations_.end() || it->second.state != HLIEDeviceReservationState::Pending) {
        return false;
    }
    remove_pending_locked(it->second);
    reservations_.erase(it);
    return true;
}

bool HLIEDeviceReservationLedger::release(const HLIEDeviceReservationId reservation_id) {
    std::lock_guard lock(mutex_);
    const auto it = reservations_.find(reservation_id);
    if (it == reservations_.end()) {
        return false;
    }
    if (it->second.state == HLIEDeviceReservationState::Pending) {
        remove_pending_locked(it->second);
    } else {
        remove_committed_locked(it->second);
    }
    reservations_.erase(it);
    return true;
}

bool HLIEDeviceReservationLedger::covers(
    const HLIEDeviceReservationId reservation_id,
    const std::vector<int>& devices,
    const std::uint32_t minimum_total_mb,
    const std::string& model_key) const {
    std::lock_guard lock(mutex_);
    const auto it = reservations_.find(reservation_id);
    if (it == reservations_.end() || it->second.total_mb < minimum_total_mb ||
        (!model_key.empty() && it->second.model_key != model_key)) {
        return false;
    }

    std::vector<int> expected = devices;
    std::sort(expected.begin(), expected.end());
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    auto actual = it->second.devices;
    std::sort(actual.begin(), actual.end());
    return expected == actual;
}

std::vector<HLIEDeviceReservation> HLIEDeviceReservationLedger::rollback_expired_pending(
    const std::unordered_set<HLIEDeviceReservationId>& protected_reservations,
    const std::chrono::steady_clock::time_point now) {
    std::lock_guard lock(mutex_);
    std::vector<HLIEDeviceReservation> rolled_back;
    for (auto it = reservations_.begin(); it != reservations_.end();) {
        const auto& reservation = it->second;
        if (reservation.state != HLIEDeviceReservationState::Pending ||
            reservation.expires_at > now || protected_reservations.contains(reservation.id)) {
            ++it;
            continue;
        }

        remove_pending_locked(reservation);
        rolled_back.push_back(reservation);
        it = reservations_.erase(it);
    }
    return rolled_back;
}

std::optional<HLIEDeviceReservation> HLIEDeviceReservationLedger::get(
    const HLIEDeviceReservationId reservation_id) const {
    std::lock_guard lock(mutex_);
    const auto it = reservations_.find(reservation_id);
    return it == reservations_.end() ? std::nullopt : std::optional<HLIEDeviceReservation>{it->second};
}

std::vector<HLIEDeviceReservation> HLIEDeviceReservationLedger::reservations() const {
    std::lock_guard lock(mutex_);
    std::vector<HLIEDeviceReservation> result;
    result.reserve(reservations_.size());
    for (const auto& [_, reservation] : reservations_) {
        result.push_back(reservation);
    }
    return result;
}

std::vector<HLIEDeviceLedgerSnapshot> HLIEDeviceReservationLedger::snapshots() const {
    std::lock_guard lock(mutex_);
    std::vector<HLIEDeviceLedgerSnapshot> result;
    result.reserve(devices_.size());
    for (const auto& [device_id, entry] : devices_) {
        HLIEDeviceLedgerSnapshot snapshot;
        snapshot.device_id = device_id;
        snapshot.total_mb = entry.total_mb;
        snapshot.observed_used_mb = entry.observed_used_mb;
        snapshot.observed_available_mb = entry.observed_available_mb;
        snapshot.pending_reserved_mb = entry.pending_reserved_mb;
        snapshot.committed_reserved_mb = entry.committed_reserved_mb;
        snapshot.committed_unobserved_mb = entry.committed_unobserved_mb;
        snapshot.effective_used_mb = effective_used_locked(entry);
        const std::uint64_t unavailable = saturating_add(snapshot.effective_used_mb, safety_margin_mb_);
        snapshot.admissible_mb = unavailable >= entry.total_mb ? 0 : entry.total_mb - unavailable;
        snapshot.exclusive = entry.exclusive_reservations > 0;
        result.push_back(snapshot);
    }
    return result;
}

HLIE_NAMESPACE_END

#ifndef HLIE_CORE_RESERVATION_HANDLE_H
#define HLIE_CORE_RESERVATION_HANDLE_H

#include <optional>
#include <utility>

#include "hlie_core/domain/hlie_model_launch_ticket.h"
#include "hlie_core/resource/hlie_device_reservation_ledger.h"

HLIE_NAMESPACE_BEGIN

// Move-only startup grant. Destruction is deliberately passive: rollback is
// performed by ResourceScheduler so ownership changes remain explicit.
class HLIEPendingReservationHandle {
public:
    HLIEPendingReservationHandle() = default;
    explicit HLIEPendingReservationHandle(HLIEDeviceReservation reservation)
        : reservation_(std::move(reservation)) {}

    HLIEPendingReservationHandle(const HLIEPendingReservationHandle&) = delete;
    HLIEPendingReservationHandle& operator=(const HLIEPendingReservationHandle&) = delete;

    HLIEPendingReservationHandle(HLIEPendingReservationHandle&& other) noexcept
        : reservation_(std::move(other.reservation_)), ticket_(std::move(other.ticket_)) {
        other.reset();
    }
    HLIEPendingReservationHandle& operator=(HLIEPendingReservationHandle&& other) noexcept {
        if (this != &other) {
            reservation_ = std::move(other.reservation_);
            ticket_ = std::move(other.ticket_);
            other.reset();
        }
        return *this;
    }

    [[nodiscard]] bool valid() const noexcept { return reservation_.id != 0; }
    [[nodiscard]] HLIEDeviceReservationId id() const noexcept { return reservation_.id; }
    [[nodiscard]] const HLIEDeviceReservation& reservation() const noexcept { return reservation_; }
    [[nodiscard]] const std::optional<HLIEModelLaunchTicket>& ticket() const noexcept { return ticket_; }

    [[nodiscard]] bool bind(const HLIEModelLaunchTicket& ticket) {
        if (!valid() || !ticket.valid() ||
            (reservation_.generation != 0 &&
             (reservation_.generation != ticket.generation ||
              reservation_.model_key != ticket.model_key))) {
            return false;
        }
        reservation_.model_key = ticket.model_key;
        reservation_.generation = ticket.generation;
        ticket_ = ticket;
        return true;
    }

    void reset() noexcept {
        reservation_ = {};
        ticket_.reset();
    }

private:
    HLIEDeviceReservation reservation_;
    std::optional<HLIEModelLaunchTicket> ticket_;
};

class HLIECommittedReservationHandle {
public:
    HLIECommittedReservationHandle() = default;
    HLIECommittedReservationHandle(HLIEDeviceReservation reservation,
                                   HLIEModelLaunchTicket ticket)
        : reservation_(std::move(reservation)), ticket_(std::move(ticket)) {
        reservation_.state = HLIEDeviceReservationState::Committed;
    }

    HLIECommittedReservationHandle(const HLIECommittedReservationHandle&) = delete;
    HLIECommittedReservationHandle& operator=(const HLIECommittedReservationHandle&) = delete;
    HLIECommittedReservationHandle(HLIECommittedReservationHandle&&) noexcept = default;
    HLIECommittedReservationHandle& operator=(HLIECommittedReservationHandle&&) noexcept = default;

    [[nodiscard]] bool valid() const noexcept {
        return reservation_.id != 0 && ticket_.valid();
    }
    [[nodiscard]] HLIEDeviceReservationId id() const noexcept { return reservation_.id; }
    [[nodiscard]] const HLIEDeviceReservation& reservation() const noexcept { return reservation_; }
    [[nodiscard]] const HLIEModelLaunchTicket& ticket() const noexcept { return ticket_; }

private:
    HLIEDeviceReservation reservation_;
    HLIEModelLaunchTicket ticket_;
};

HLIE_NAMESPACE_END

#endif // HLIE_CORE_RESERVATION_HANDLE_H

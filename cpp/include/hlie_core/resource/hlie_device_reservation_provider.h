#ifndef HLIE_CORE_DEVICE_RESERVATION_PROVIDER_H
#define HLIE_CORE_DEVICE_RESERVATION_PROVIDER_H

#include <optional>
#include <unordered_set>
#include <vector>

#include "hlie_core/resource/hlie_device_reservation_ledger.h"

HLIE_NAMESPACE_BEGIN

// Resource-side port consumed by the control plane. Implementations may use
// HAL/TCIM observations, but Scheduler never depends on those SDK types.
class IHLIEDeviceReservationProvider {
public:
    virtual ~IHLIEDeviceReservationProvider() = default;

    [[nodiscard]] virtual std::optional<HLIEDeviceReservation> reserve_resources(
        const HLIEDeviceReservationRequest& request) = 0;
    [[nodiscard]] virtual bool bind_reservation(
        HLIEDeviceReservationId reservation_id, const std::string& model_key,
        std::uint64_t generation) = 0;
    [[nodiscard]] virtual bool commit_reservation(
        HLIEDeviceReservationId reservation_id) = 0;
    [[nodiscard]] virtual bool rollback_reservation(
        HLIEDeviceReservationId reservation_id) = 0;
    [[nodiscard]] virtual bool release_reservation(
        HLIEDeviceReservationId reservation_id) = 0;
    [[nodiscard]] virtual std::vector<HLIEDeviceReservation>
    rollback_expired_pending_reservations(
        const std::unordered_set<HLIEDeviceReservationId>& protected_reservations) = 0;
    [[nodiscard]] virtual std::vector<HLIEDeviceReservation> get_reservations() const = 0;
    [[nodiscard]] virtual std::vector<HLIEDeviceLedgerSnapshot> get_ledger_snapshots() const = 0;
};

HLIE_NAMESPACE_END

#endif // HLIE_CORE_DEVICE_RESERVATION_PROVIDER_H

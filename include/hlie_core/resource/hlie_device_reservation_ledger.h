#ifndef HLIE_CORE_DEVICE_RESERVATION_LEDGER_H
#define HLIE_CORE_DEVICE_RESERVATION_LEDGER_H

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

using HLIEDeviceReservationId = std::uint64_t;

enum class HLIEDeviceReservationState {
    Pending,
    Committed,
};

struct HLIEDeviceObservation {
    int device_id = -1;
    std::uint64_t total_mb = 0;
    std::uint64_t used_mb = 0;
    std::uint64_t available_mb = 0;
};

struct HLIEDeviceReservationRequest {
    std::string model_key;
    std::uint64_t generation = 0;
    std::uint64_t total_mb = 0;
    std::size_t device_count = 1;
    std::vector<int> requested_devices;
    bool exclusive = false;
    bool allow_single_device_capacity_limit = false;
};

struct HLIEDeviceReservation {
    HLIEDeviceReservationId id = 0;
    std::string model_key;
    std::uint64_t generation = 0;
    std::uint64_t total_mb = 0;
    std::vector<int> devices;
    std::vector<std::uint64_t> per_device_mb;
    bool exclusive = false;
    HLIEDeviceReservationState state = HLIEDeviceReservationState::Pending;
    std::chrono::steady_clock::time_point created_at{};
    std::chrono::steady_clock::time_point expires_at{};
};

struct HLIEDeviceLedgerSnapshot {
    int device_id = -1;
    std::uint64_t total_mb = 0;
    std::uint64_t observed_used_mb = 0;
    std::uint64_t observed_available_mb = 0;
    std::uint64_t pending_reserved_mb = 0;
    std::uint64_t committed_reserved_mb = 0;
    std::uint64_t committed_unobserved_mb = 0;
    std::uint64_t effective_used_mb = 0;
    std::uint64_t admissible_mb = 0;
    bool exclusive = false;
};

// Thread-safe logical device ledger. HAL observations and framework-owned
// reservations are kept separate so a refresh cannot erase an in-flight load.
class HLIEDeviceReservationLedger {
public:
    explicit HLIEDeviceReservationLedger(std::uint64_t safety_margin_mb = 200);

    void reset();
    void set_pending_ttl(std::chrono::seconds pending_ttl);
    void update_observation(const HLIEDeviceObservation& observation);

    [[nodiscard]] std::optional<HLIEDeviceReservation> reserve(
        const HLIEDeviceReservationRequest& request);
    [[nodiscard]] bool bind(HLIEDeviceReservationId reservation_id,
                            const std::string& model_key, std::uint64_t generation);
    [[nodiscard]] bool commit(HLIEDeviceReservationId reservation_id);
    [[nodiscard]] bool rollback(HLIEDeviceReservationId reservation_id);
    [[nodiscard]] bool release(HLIEDeviceReservationId reservation_id);
    [[nodiscard]] bool covers(HLIEDeviceReservationId reservation_id,
                              const std::vector<int>& devices,
                              std::uint32_t minimum_total_mb,
                              const std::string& model_key = "") const;
    [[nodiscard]] std::vector<HLIEDeviceReservation> rollback_expired_pending(
        const std::unordered_set<HLIEDeviceReservationId>& protected_reservations,
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

    [[nodiscard]] std::optional<HLIEDeviceReservation> get(
        HLIEDeviceReservationId reservation_id) const;
    [[nodiscard]] std::vector<HLIEDeviceReservation> reservations() const;
    [[nodiscard]] std::vector<HLIEDeviceLedgerSnapshot> snapshots() const;

private:
    struct DeviceEntry {
        std::uint64_t total_mb = 0;
        std::uint64_t observed_used_mb = 0;
        std::uint64_t observed_available_mb = 0;
        std::uint64_t pending_reserved_mb = 0;
        std::uint64_t committed_reserved_mb = 0;
        std::uint64_t committed_unobserved_mb = 0;
        std::size_t exclusive_reservations = 0;
    };

    [[nodiscard]] bool can_reserve_locked(int device_id, std::uint64_t size_mb,
                                          bool exclusive, bool allow_capacity_limit) const;
    [[nodiscard]] std::uint64_t effective_used_locked(const DeviceEntry& entry) const;
    [[nodiscard]] std::uint64_t reserved_size_locked(const DeviceEntry& entry,
                                                     std::uint64_t requested_mb,
                                                     bool allow_capacity_limit) const;
    void remove_pending_locked(const HLIEDeviceReservation& reservation);
    void remove_committed_locked(const HLIEDeviceReservation& reservation);

    std::uint64_t safety_margin_mb_ = 200;
    std::chrono::seconds pending_ttl_{300};
    HLIEDeviceReservationId next_id_ = 1;
    std::map<int, DeviceEntry> devices_;
    std::unordered_map<HLIEDeviceReservationId, HLIEDeviceReservation> reservations_;
    mutable std::mutex mutex_;
};

HLIE_NAMESPACE_END

#endif // HLIE_CORE_DEVICE_RESERVATION_LEDGER_H

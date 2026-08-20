#ifndef HLIE_DEVICE_SERVICE_H
#define HLIE_DEVICE_SERVICE_H

#include <future>
#include <server/server-http.h>
#include <set>
#include <unordered_set>
#include <vector>

#include "hlie_core/resource/hlie_device_reservation_provider.h"
#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

struct HLIEDeviceInfo {
    uint8_t id;
    uint8_t chip_id;
    uint8_t group_id;
    uint32_t mem_total;
    uint32_t mem_used;
    uint32_t mem_avail;
    bool exclusive = false;
    std::set<std::string> loaded_models;
};

enum class MemApplyMode {
    PreferIdleCard,     // 按卡 Index 顺序，优先选择显存未被使用的卡
    PreferUsedCard,     // 优先在已有负载的卡上继续分配显存
    ExclusiveCard,      // 独占一张卡
};

class HLIEDeviceService : public IHLIEDeviceReservationProvider {
public:
    HLIEDeviceService() = default;
    ~HLIEDeviceService() = default;

    int init();

    int acquire_mem(uint32_t size, uint32_t& out_device, bool dry_run = false, bool allow_capacity_limit = false,
                    const std::string& model_name = "");
    server_http_res_ptr acquire_mem(const server_http_req& req);
    int acquire_mem_on_devices(const std::vector<int>& devices, uint32_t total_size, bool exclusive = false, bool dry_run = false,
                               const std::string& model_name = "");
    int acquire_mem_on_device_count(size_t device_count, uint32_t total_size, std::vector<int>& out_devices, bool dry_run = false,
                                    const std::string& model_name = "");

    [[nodiscard]] std::optional<HLIEDeviceReservation> reserve_mem(
        uint32_t size, const std::string& model_key = "", uint64_t generation = 0,
        bool allow_capacity_limit = false);
    [[nodiscard]] std::optional<HLIEDeviceReservation> reserve_mem_on_devices(
        const std::vector<int>& devices, uint32_t total_size, bool exclusive = false,
        const std::string& model_key = "", uint64_t generation = 0);
    [[nodiscard]] std::optional<HLIEDeviceReservation> reserve_mem_on_device_count(
        size_t device_count, uint32_t total_size, const std::string& model_key = "",
        uint64_t generation = 0);
    [[nodiscard]] std::optional<HLIEDeviceReservation> reserve_resources(
        const HLIEDeviceReservationRequest& request) override;
    [[nodiscard]] bool bind_reservation(HLIEDeviceReservationId reservation_id,
                                        const std::string& model_key, uint64_t generation) override;
    [[nodiscard]] bool commit_reservation(HLIEDeviceReservationId reservation_id) override;
    [[nodiscard]] bool rollback_reservation(HLIEDeviceReservationId reservation_id) override;
    [[nodiscard]] bool release_reservation(HLIEDeviceReservationId reservation_id) override;
    [[nodiscard]] std::vector<HLIEDeviceReservation> rollback_expired_pending_reservations(
        const std::unordered_set<HLIEDeviceReservationId>& protected_reservations) override;
    [[nodiscard]] std::vector<HLIEDeviceReservation> get_reservations() const override;
    [[nodiscard]] std::optional<HLIEDeviceReservation> get_reservation(
        HLIEDeviceReservationId reservation_id) const;
    [[nodiscard]] bool reservation_matches(HLIEDeviceReservationId reservation_id,
                                           const std::vector<int>& devices,
                                           uint32_t total_size,
                                           const std::string& model_key = "") const;
    [[nodiscard]] std::vector<HLIEDeviceLedgerSnapshot> get_ledger_snapshots() const override;

    int get_device_infos(std::vector<HLIEDeviceInfo>& device_infos);

    void release_mem_on_devices(const std::vector<int>& devices, uint32_t total_size, bool exclusive = false);
    void release_model(uint32_t device, const std::string& model_name);
    void refresh_device_usage(uint32_t device);

    // API-HANDLERS
    server_http_res_ptr handle_devices(const server_http_req& req);
    static bool check_mem_available(uint32_t device, uint32_t size, bool allow_capacity_limit = false,
                                    const std::string& model_name = "");

private:
    int acquire_card_first(uint32_t size, uint32_t& out_device, bool dry_run, bool allow_capacity_limit,
                           const std::string& model_name);
    int acquire_mem_first(uint32_t size, uint32_t& out_device, bool dry_run, bool allow_capacity_limit,
                          const std::string& model_name);
    int acquire_exclusive(uint32_t size, uint32_t& out_device, bool dry_run, bool allow_capacity_limit,
                          const std::string& model_name);
    int try_idle_first(uint32_t size, uint32_t& out_device, bool dry_run, bool allow_capacity_limit,
                       const std::string& model_name);
    int try_used_first(uint32_t size, uint32_t& out_device, bool dry_run, bool allow_capacity_limit,
                       const std::string& model_name);

    std::map<int, HLIEDeviceInfo> devices_;
    HLIEDeviceReservationLedger reservation_ledger_;
    MemApplyMode mode_ = MemApplyMode::PreferIdleCard;
    int total_mem_ = 0;
    std::mutex mutex_;
};

HLIE_NAMESPACE_END

#endif

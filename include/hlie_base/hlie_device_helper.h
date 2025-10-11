#ifndef HLIE_DEVICE_HELPER_H
#define HLIE_DEVICE_HELPER_H

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "hlie_base.h"

HLIE_NAMESPACE_BEGIN


class HLIEDeviceHelper {
public:
    void init(const std::string& devices_str);

    /**
     * 申请设备ID（返回optional明确表示可能失败）
     *
     * @param applicant
     * @return
     */
    std::optional<int> acquire_device(const std::string& applicant, bool dry_run = false);

    /**
     * 申请设备ID（返回optional明确表示可能失败）
     *
     * @param applicant
     * @param device_id
     * @return
     */
    std::optional<int> acquire_device(const std::string& applicant, int device_id, bool dry_run = false);

    /**
     * 尝试获取指定的设备
     *
     * @param applicant
     * @param devices - 设备ID列表Str
     * @param acquiring_devices - 获取到的设备ID列表
     * @param dry_run - 尝试获取设备，不修改设备状态
     * @return
     */
    bool acquire_devices(const std::string& applicant, const std::string& devices, std::vector<int>& acquiring_devices, bool dry_run = false);

    /**
     * 释放设备ID
     *
     * @param device_ids
     * @return
     */
    size_t release_devices(const std::vector<int>& device_ids);

    void print_status() const;

    int get_device_count() const;

private:
    std::unordered_map<int, std::string> devices_;
    mutable std::mutex mutex_;
};


HLIE_NAMESPACE_END

#endif // HLIE_DEVICE_HELPER_H

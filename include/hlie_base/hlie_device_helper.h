#ifndef HLIE_DEVICE_HELPER_H
#define HLIE_DEVICE_HELPER_H

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>


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
    std::optional<int> acquire_device(const std::string& applicant);

    /**
     * 申请设备ID（返回optional明确表示可能失败）
     *
     * @param applicant
     * @param device_id
     * @return
     */
    std::optional<int> acquire_device(const std::string& applicant, int device_id);

    /**
     * 尝试获取指定的设备
     *
     * @param applicant
     * @param devices - 设备ID列表
     * @return
     */
    bool acquire_devices(const std::string& applicant, const std::string& devices);

    /**
     * 释放设备ID
     *
     * @param device_id
     * @return
     */
    bool release_device(int device_id);

    void print_status() const;

    int get_device_count() const;

private:
    std::unordered_map<int, std::string> devices_;
    mutable std::mutex mutex_;
};


HLIE_NAMESPACE_END

#endif // HLIE_DEVICE_HELPER_H

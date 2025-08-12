#include "hlie_base/hlie_device_helper.h"

#include <sstream>

#include "hlie_base/hlie_logger.h"


HLIE_NAMESPACE_BEGIN


void HLIEDeviceHelper::init(const std::string& devices_str) {
    devices_.clear();
    try {
        std::stringstream ss(devices_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // 去掉可能的空格
            token.erase(0, token.find_first_not_of(' '));
            token.erase(token.find_last_not_of(' ') + 1);

            if (!token.empty()) {
                devices_[std::stoi(token)];
            }
        }
    } catch (const std::exception& ex) {
        devices_.clear();
        devices_[0];
        LOG_ERROR("failed to parse devices from \"{}\", ex: {}, Device 0 will be used", devices_str, ex.what());
    }
}

std::optional<int> HLIEDeviceHelper::acquire_device(const std::string& applicant) {
    if (applicant.empty()) {
        return std::nullopt;
    }

    std::lock_guard lock(mutex_);

    // 查找第一个空闲设备
    if (const auto it = std::find_if(devices_.begin(), devices_.end(), [](const auto& pair) { return pair.second.empty(); });
        it != devices_.end()) {
        it->second = applicant;
        return it->first; // 返回分配的设备ID
    }

    return std::nullopt;
}

bool HLIEDeviceHelper::release_device(const int device_id) {
    std::lock_guard lock(mutex_);
    if (const auto it = devices_.find(device_id); it != devices_.end()) {
        it->second.clear();
        return true;
    }
    return false;
}

void HLIEDeviceHelper::print_status() const {
    std::lock_guard lock(mutex_);
    LOG_INFO("=== Device Status ===");
    for (const auto& [id, user] : devices_) {
        LOG_INFO("Device {}:{}", id, user.empty() ? "[Available]" : "Used by " + user);
    }
}

HLIE_NAMESPACE_END

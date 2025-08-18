#include "hlie_base/hlie_device_helper.h"

#include <sstream>

#include "hlie_base/hlie_logger.h"


HLIE_NAMESPACE_BEGIN


void HLIEDeviceHelper::init(const std::string& devices_str) {
    LOG_INFO("Begin to init devices with devices_str[{}]", devices_str);
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
        LOG_ERROR("applicant cannot be empty");
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

std::optional<int> HLIEDeviceHelper::acquire_device(const std::string& applicant, const int device_id) {
    if (applicant.empty()) {
        LOG_ERROR("applicant cannot be empty");
        return std::nullopt;
    }

    std::lock_guard lock(mutex_);

    // 查找第一个空闲设备且等于device_id
    if (const auto it = std::find_if(devices_.begin(), devices_.end(),
                                     [device_id](const auto& pair) { return pair.first == device_id && pair.second.empty(); });
        it != devices_.end()) {
        it->second = applicant;
        return it->first; // 返回分配的设备ID
    }

    return std::nullopt;
}

bool HLIEDeviceHelper::acquire_devices(const std::string& applicant, const std::string& devices) {
    if (applicant.empty()) {
        LOG_ERROR("applicant cannot be empty");
        return false;
    }

    std::lock_guard lock(mutex_);

    std::vector<int> acquiring_devices;
    try {
        std::stringstream ss(devices);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // 去掉可能的空格
            token.erase(0, token.find_first_not_of(' '));
            token.erase(token.find_last_not_of(' ') + 1);

            if (!token.empty()) {
                acquiring_devices.push_back(std::stoi(token));
            }
        }
    } catch (const std::exception& ex) {
        LOG_ERROR("failed to parse acquiring devices \"{}\", ex: {}", devices, ex.what());
        return false;
    }

    if (acquiring_devices.empty()) {
        LOG_ERROR("no acquiring devices parsed with {}", devices);
        return false;
    }

    if (std::all_of(acquiring_devices.begin(), acquiring_devices.end(),
                    [&](const int device_id) { return devices_.find(device_id) != devices_.end(); })) {
        for (auto device : acquiring_devices) {
            devices_[device] = applicant;
        }
        return true;
    }

    LOG_ERROR("cannot found devices in global devices pool with {}, see the sys.cfg to check it", devices);
    return false;
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

int HLIEDeviceHelper::get_device_count() const {
    return static_cast<int>(devices_.size());
}

HLIE_NAMESPACE_END

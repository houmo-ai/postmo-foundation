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

std::optional<int> HLIEDeviceHelper::acquire_device(const std::string& applicant, const bool dry_run) {
    if (applicant.empty()) {
        LOG_ERROR("applicant cannot be empty");
        return std::nullopt;
    }

    std::lock_guard lock(mutex_);

    // 查找第一个空闲设备
    if (const auto it = std::find_if(devices_.begin(), devices_.end(), [](const auto& pair) { return pair.second.empty(); });
        it != devices_.end()) {
        if (!dry_run) {
            it->second = applicant;
        }
        return it->first; // 返回分配的设备ID
    }

    return std::nullopt;
}

std::optional<int> HLIEDeviceHelper::acquire_device(const std::string& applicant, const int device_id, const bool dry_run) {
    if (applicant.empty()) {
        LOG_ERROR("applicant cannot be empty");
        return std::nullopt;
    }

    std::lock_guard lock(mutex_);

    // 查找第一个空闲设备且等于device_id
    if (const auto it = std::find_if(devices_.begin(), devices_.end(),
                                     [device_id](const auto& pair) { return pair.first == device_id && pair.second.empty(); });
        it != devices_.end()) {
        if (!dry_run) {
            it->second = applicant;
        }
        return it->first; // 返回分配的设备ID
    }

    return std::nullopt;
}

bool HLIEDeviceHelper::acquire_devices(const std::string& applicant, int device_num, std::vector<int>& acquiring_devices,
                                       const bool dry_run) {
    if (applicant.empty()) {
        LOG_ERROR("applicant cannot be empty");
        return false;
    }

    std::lock_guard lock(mutex_);

    acquiring_devices.clear();
    std::vector<int> tmp;;
    for (const auto& [fst, snd] : devices_) {
        if (snd.empty()) {
            tmp.push_back(fst);
            if (tmp.size() == device_num) {
                break;
            }
        }
    }

    if (tmp.size() < device_num) {
        LOG_ERROR("cannot acquire specified device num {} in global devices pool with {}, see the sys.cfg to check it", device_num);
        return false;
    }

    for (const auto& device : tmp) {
        if (!dry_run) {
            devices_.at(device) = applicant;
        }
        acquiring_devices.push_back(device);
    }

    return true;
}

bool HLIEDeviceHelper::acquire_devices(const std::string& applicant, const std::string& devices, std::vector<int>& acquiring_devices, const bool dry_run) {
    if (applicant.empty()) {
        LOG_ERROR("applicant cannot be empty");
        return false;
    }

    std::lock_guard lock(mutex_);

    acquiring_devices.clear();
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
                    [&](const int device_id) { return devices_.find(device_id) != devices_.end() && devices_.at(device_id).empty(); })) {
        if (!dry_run) {
            for (auto device : acquiring_devices) {
                devices_[device] = applicant;
            }
        }

        return true;
    }

    LOG_ERROR("cannot found devices in global devices pool with {}, see the sys.cfg to check it", devices);
    return false;
}

size_t HLIEDeviceHelper::release_devices(const std::vector<int>& device_ids) {
    std::lock_guard lock(mutex_);
    size_t count = 0;
    for (int id : device_ids) {
        auto it = devices_.find(id);
        if (it == devices_.end()) continue;

        if (!it->second.empty()) {
            it->second.clear();
            LOG_INFO("Device {} released", id);
            ++count;
        }
    }

    return count;
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

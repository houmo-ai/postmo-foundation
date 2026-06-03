#include "hlie_app/hlie_device_service.h"

#include <cstdlib>
#include <cstdint>
#include <string>
#include <unordered_set>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "hlie_app/hlie_model_info.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

HLIE_NAMESPACE_BEGIN

namespace {
    constexpr auto BACK_UP_MEM = 200;

    constexpr uint32_t HM_SYS_MAX_DEVICES = 32;

    struct hm_device_info {
        uint32_t num_devices;
        uint32_t device_ids[HM_SYS_MAX_DEVICES];
    };

    struct hm_mem_info {
        uint32_t mem_total;
        uint32_t mem_used;
        uint32_t mem_avail;
    };

    using hm_sys_get_device_info_t = uint32_t (*)(hm_device_info* info);
    using hm_sys_get_mem_info_t = int (*)(int dev_index, hm_mem_info* mem_info);

    struct HmSysSdk {
        void* handle = nullptr;
        hm_sys_get_device_info_t get_device_info = nullptr;
        hm_sys_get_mem_info_t get_mem_info = nullptr;
        bool load_attempted = false;

        static const char* library_name() {
#ifdef _WIN32
            return "libhal_xh2a.dll";
#else
            return "libhal_xh2a.so";
#endif
        }

        bool load() {
            if (handle) {
                return true;
            }

            if (load_attempted) {
                return false;
            }
            load_attempted = true;

            LOG_INFO("loading Houmo HAL library {}", library_name());

#ifdef _WIN32
            const char* env_paths[] = {
                std::getenv("HOUMO_SDK_PATH"),
                std::getenv("HOUMO_DRV_PATH"),
                std::getenv("TCIM_RUNTIME_PATH"),
                ".",
                nullptr,
            };

            for (const char** env_path = env_paths; *env_path != nullptr; ++env_path) {
                if (!*env_path || !**env_path) {
                    continue;
                }

                const std::string base_path = *env_path;
                const std::string candidate_paths[] = {
                    base_path + "/hal/lib/" + library_name(),
                    base_path + "/lib/" + library_name(),
                    base_path + "/" + library_name(),
                };

                for (const auto& path : candidate_paths) {
                    handle = LoadLibraryA(path.c_str());
                    if (handle) {
                        LOG_INFO("loaded Houmo HAL library from {}", path);
                        break;
                    }
                }

                if (handle) {
                    break;
                }
            }

            if (!handle) {
                handle = LoadLibraryA(library_name());
                if (!handle) {
                    LOG_ERROR("failed to load Houmo HAL library {}, error {}", library_name(), GetLastError());
                    return false;
                }
                LOG_INFO("loaded Houmo HAL library from system path");
            }

            get_device_info =
                reinterpret_cast<hm_sys_get_device_info_t>(GetProcAddress(static_cast<HMODULE>(handle), "hm_sys_get_device_info"));
            get_mem_info =
                reinterpret_cast<hm_sys_get_mem_info_t>(GetProcAddress(static_cast<HMODULE>(handle), "hm_sys_get_mem_info"));
#else
            const char* env_paths[] = {
                std::getenv("HOUMO_SDK_PATH"),
                std::getenv("HOUMO_DRV_PATH"),
                std::getenv("TCIM_RUNTIME_PATH"),
                "/usr/local/houmo-sdk/hal/lib",
                "/usr/local/houmo-drv/hal/lib",
                ".",
                nullptr,
            };

            for (const char** env_path = env_paths; *env_path != nullptr; ++env_path) {
                if (!*env_path || !**env_path) {
                    continue;
                }

                const std::string path = std::string(*env_path) + "/" + library_name();
                handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
                if (handle) {
                    LOG_INFO("loaded Houmo HAL library from {}", path);
                    break;
                }
            }

            if (!handle) {
                handle = dlopen(library_name(), RTLD_LAZY | RTLD_LOCAL);
                if (!handle) {
                    LOG_ERROR("failed to load Houmo HAL library {}, error {}", library_name(), dlerror());
                    return false;
                }
                LOG_INFO("loaded Houmo HAL library from system path");
            }

            get_device_info = reinterpret_cast<hm_sys_get_device_info_t>(dlsym(handle, "hm_sys_get_device_info"));
            get_mem_info = reinterpret_cast<hm_sys_get_mem_info_t>(dlsym(handle, "hm_sys_get_mem_info"));
#endif

            if (!get_device_info || !get_mem_info) {
                LOG_ERROR("Houmo HAL library missing required symbols");
#ifdef _WIN32
                FreeLibrary(static_cast<HMODULE>(handle));
#else
                dlclose(handle);
#endif
                handle = nullptr;
                get_device_info = nullptr;
                get_mem_info = nullptr;
                return false;
            }

            return true;
        }

        ~HmSysSdk() {
            if (handle) {
#ifdef _WIN32
                FreeLibrary(static_cast<HMODULE>(handle));
#else
                dlclose(handle);
#endif
            }
        }
    };

    HmSysSdk& hm_sys_sdk() {
        static HmSysSdk sdk;
        return sdk;
    }
}

int HLIEDeviceService::init() {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    auto devices_str = config_services->get_string("server", "devices", "0", true);

    LOG_INFO("Begin to init devices with config [{}]", devices_str);

    devices_.clear();

    // 1. parse config ids
    std::vector<int> config_devices;

    try {
        std::stringstream ss(devices_str);
        std::string token;

        while (std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(' '));
            token.erase(token.find_last_not_of(' ') + 1);

            if (!token.empty()) {
                config_devices.push_back(std::stoi(token));
            }
        }
    } catch (const std::exception& ex) {
        LOG_ERROR("failed to parse devices config {}, ex: {}", devices_str, ex.what());
        config_devices = {0};
    }

    // 2. query system devices
    auto& sdk = hm_sys_sdk();
    if (!sdk.load()) {
        LOG_ERROR("failed to load Houmo HAL SDK");
        return -1;
    }

    hm_device_info sys_info{};
    const uint32_t ret = sdk.get_device_info(&sys_info);
    const uint32_t dev_count = sys_info.num_devices ? sys_info.num_devices : ret;

    if (dev_count == 0) {
        LOG_ERROR("no device found by SDK");
        return -1;
    }

    std::unordered_set<int> valid_ids;
    for (uint32_t i = 0; i < dev_count; ++i) {
        valid_ids.insert(sys_info.device_ids[i]);
    }

    // 3. validate & init devices
    for (int dev : config_devices) {
        if (!valid_ids.count(dev)) {
            LOG_WARN("configured device {} not found, skip", dev);
            continue;
        }

        hm_mem_info mem{};
        if (sdk.get_mem_info(dev, &mem) < 0) {
            LOG_WARN("failed to query mem info of {}, skip", dev);
            continue;
        }

        HLIEDeviceInfo info{};
        info.id = dev;
        info.mem_total = mem.mem_total;
        info.mem_used = mem.mem_used;
        info.mem_avail = mem.mem_avail;
        info.exclusive = false;

        devices_.emplace(dev, info);

        LOG_INFO("device {} ready: total {}MB used {}MB avail {}MB", dev, mem.mem_total, mem.mem_used, mem.mem_avail);
    }

    if (devices_.empty()) {
        LOG_ERROR("no usable devices after init");
        return -1;
    }

    LOG_INFO("init finished, {} devices available", devices_.size());
    return 0;
}


int HLIEDeviceService::acquire_mem(uint32_t size, uint32_t& out_device, bool dry_run) {
    std::unique_lock lock(mutex_);
    LOG_INFO("acquire_mem");
    for (auto [dev, _] : devices_) {
        refresh_device_usage(dev);
    }
    LOG_INFO("acquire_mem refresh_device_usage done");

    switch (mode_) {
    case MemApplyMode::PreferIdleCard:
        return acquire_card_first(size, out_device, dry_run);
    case MemApplyMode::PreferUsedCard:
        return acquire_mem_first(size, out_device, dry_run);
    case MemApplyMode::ExclusiveCard:
        return acquire_exclusive(size, out_device, dry_run);
    default:
        return -1;
    }
}

server_http_res_ptr HLIEDeviceService::acquire_mem(const server_http_req& req) {
    const json body = json::parse(req.body);
    auto res = std::make_unique<server_res_generator_light>();
    const int size = json_value(body, "port", 0);
    if (size == 0) {
        const json resp = {{"code", 0}};
        res->ok(resp);
        return res;
    }

    uint32_t device;
    if (acquire_mem(size, device) < 0) {
        LOG_ERROR("acquire_mem failed");
        res->error("acquire_mem failed");
        return res;
    }

    const json resp = {{"code", 0, "device", device}};
    res->ok(resp);
    return res;
}

int HLIEDeviceService::acquire_mem_on_devices(const std::vector<int>& devices, const uint32_t total_size, const bool exclusive,
                                              const bool dry_run) {

    if (devices.empty()) {
        LOG_ERROR("acquire_mem_on_devices: empty device list");
        return -1;
    }

    std::unique_lock lock(mutex_);

    // 去重
    std::unordered_set<uint32_t> uniq(devices.begin(), devices.end());
    const size_t n = uniq.size();

    const uint32_t per_card = (total_size + n - 1) / n; // ceil

    // ---------- Phase 1: check ----------
    for (auto dev : uniq) {
        auto it = devices_.find(dev);
        if (it == devices_.end()) {
            LOG_ERROR("device {} not managed", dev);
            return -1;
        }

        auto& info = it->second;

        if (info.exclusive && info.mem_used > 0) {
            LOG_ERROR("device {} already exclusive occupied", dev);
            return -1;
        }

        if (!check_mem_available(dev, per_card)) {
            LOG_ERROR("device {} insufficient mem for shard {}MB", dev, per_card);
            return -1;
        }
    }

    if (dry_run) {
        return 0;
    }

    // ---------- Phase 2: commit ----------
    for (auto dev : uniq) {
        auto& info = devices_[dev];
        info.mem_used += per_card;

        if (exclusive) {
            info.exclusive = true;
        }

        LOG_INFO("device {} acquire {}MB (multi-card), used {}MB", dev, per_card, info.mem_used);
    }

    return 0;
}

int HLIEDeviceService::get_device_infos(std::vector<HLIEDeviceInfo>& device_infos) {
    std::unique_lock lock(mutex_);
    auto& sdk = hm_sys_sdk();
    if (!sdk.load()) {
        LOG_ERROR("failed to load Houmo HAL SDK");
        return -1;
    }

    hm_device_info info = {};
    const uint32_t ret_code = sdk.get_device_info(&info);
    /* prefer info.num_devices if filled, otherwise use return value */
    const uint32_t dev_count = info.num_devices ? info.num_devices : ret_code;
    if (dev_count <= 0) {
        LOG_ERROR("no device found!");
        return -1;
    }

    LOG_INFO("found {} devices", dev_count);

    for (int i = 0; i < dev_count; i++) {
        int dev = static_cast<int>(info.device_ids[i]);
        LOG_INFO("try to get mem info of device {}", dev);
        hm_mem_info mem_info{};
        if (const auto ret = sdk.get_mem_info(dev, &mem_info); ret < 0) {
            LOG_ERROR("failed to get mem info of device {}, ret: {}", dev, ret);
        }

        LOG_INFO("get mem info of device {}: total {}MB, used {}MB, avail {}MB", dev, mem_info.mem_total, mem_info.mem_used,
                 mem_info.mem_avail);

        HLIEDeviceInfo device_info{};
        device_info.id = dev;
        device_info.mem_total = mem_info.mem_total;
        device_info.mem_used = mem_info.mem_used;
        device_info.mem_avail = mem_info.mem_avail;
        device_info.exclusive = false;

        device_infos.emplace_back(device_info);
    }

    return static_cast<int>(dev_count);
}

void HLIEDeviceService::release_mem_on_devices(const std::vector<int>& devices, uint32_t total_size, bool exclusive) {

    if (devices.empty()) {
        LOG_WARN("release_mem_on_devices: empty device list");
        return;
    }

    std::unique_lock lock(mutex_);

    std::unordered_set<uint32_t> uniq(devices.begin(), devices.end());
    const size_t n = uniq.size();

    const uint32_t per_card = (total_size + n - 1) / n;

    for (auto dev : uniq) {
        auto it = devices_.find(dev);
        if (it == devices_.end()) {
            LOG_WARN("release: device {} not found", dev);
            continue;
        }

        auto& info = it->second;

        if (info.mem_used < per_card) {
            LOG_ERROR("device {} mem underflow: used {} < release {}", dev, info.mem_used, per_card);

            info.mem_used = 0;
        } else {
            info.mem_used -= per_card;
        }

        if (exclusive && info.mem_used == 0) {
            info.exclusive = false;
        }

        LOG_INFO("device {} release {}MB, used {}MB", dev, per_card, info.mem_used);
        refresh_device_usage(dev);
    }
}

void HLIEDeviceService::release_model(uint32_t device, const std::string& model_name) {
}

void HLIEDeviceService::refresh_device_usage(uint32_t device) {
    auto it = devices_.find(device);
    if (it == devices_.end())
        return;

    auto& sdk = hm_sys_sdk();
    if (!sdk.load()) {
        LOG_WARN("refresh_device_usage: Houmo HAL SDK unavailable");
        return;
    }

    hm_mem_info mem{};
    if (sdk.get_mem_info(device, &mem) == 0) {
        it->second.mem_used = mem.mem_used;
        it->second.mem_avail = mem.mem_avail;
        it->second.mem_total = mem.mem_total;
    } else {
        LOG_WARN("refresh_device_usage: failed to query device {}", device);
    }
}

server_http_res_ptr HLIEDeviceService::handle_devices(const server_http_req& req) {
    std::vector<HLIEDeviceInfo> device_infos;
    get_device_infos(device_infos);

    json devices = json::array();
    for (const auto& device_info : device_infos) {
        devices.push_back({{"id", device_info.id},
                           {"mem_total", device_info.mem_total},
                           {"mem_used", device_info.mem_used},
                           {"mem_avail", device_info.mem_avail}});
    }

    const json resp = {{"code", 0}, {"devices", devices}};
    auto res = std::make_unique<server_res_generator_light>();
    res->ok(resp);
    return res;
}

bool HLIEDeviceService::check_mem_available(const uint32_t device, const uint32_t size) {
    auto& sdk = hm_sys_sdk();
    if (!sdk.load()) {
        return false;
    }

    hm_mem_info mem_info{};
    if (sdk.get_mem_info(device, &mem_info) < 0)
        return false;

    return mem_info.mem_avail > size + BACK_UP_MEM;
}

int HLIEDeviceService::acquire_card_first(const uint32_t size, uint32_t& out_device, const bool dry_run) {
    if (try_idle_first(size, out_device, dry_run) == 0)
        return 0;

    LOG_INFO("CARD_FIRST fallback to used cards");
    return try_used_first(size, out_device, dry_run);
}

int HLIEDeviceService::acquire_mem_first(const uint32_t size, uint32_t& out_device, const bool dry_run) {
    if (try_used_first(size, out_device, dry_run) == 0)
        return 0;

    LOG_INFO("MEM_FIRST fallback to idle cards");
    return try_idle_first(size, out_device, dry_run);
}

int HLIEDeviceService::acquire_exclusive(uint32_t size, uint32_t& out_device, const bool dry_run) {
    for (auto& [id, info] : devices_) {
        if (info.mem_used != 0 || info.exclusive)
            continue;

        if (check_mem_available(id, size)) {
            if (!dry_run) {
                info.mem_used += size;
                info.exclusive = true;
            }
            out_device = id;
            return 0;
        }

        LOG_WARN("acquire_exclusive: device {} insufficient mem, used: {}", id, info.mem_used);
    }

    return -1;
}

int HLIEDeviceService::try_idle_first(const uint32_t size, uint32_t& out_device, const bool dry_run) {
    for (auto& [id, info] : devices_) {

        if (info.mem_used != 0 || info.exclusive)
            continue;

        if (!check_mem_available(id, size)) {
            LOG_WARN("try_idle_first: device {} insufficient mem", id);
            continue;
        }

        if (!dry_run) {
            info.mem_used += size;
        }

        out_device = id;
        LOG_INFO("try_idle_first: choose device {} for {} MB", id, size);
        return 0;
    }

    return -1;
}

int HLIEDeviceService::try_used_first(const uint32_t size, uint32_t& out_device, const bool dry_run) {
    for (auto& [id, info] : devices_) {

        if (info.mem_used == 0 || info.exclusive)
            continue;

        if (!check_mem_available(id, size)) {
            LOG_WARN("try_used_first: device {} insufficient mem", id);
            continue;
        }

        if (!dry_run) {
            info.mem_used += size;
        }

        out_device = id;
        LOG_INFO("try_used_first: choose device {} for {} MB", id, size);
        return 0;
    }

    return -1;
}


HLIE_NAMESPACE_END

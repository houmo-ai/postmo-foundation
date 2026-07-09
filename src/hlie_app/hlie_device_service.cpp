#include "hlie_app/hlie_device_service.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

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
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

HLIE_NAMESPACE_BEGIN

namespace {
    constexpr auto BACK_UP_MEM = 200;

    constexpr uint32_t HM_SYS_MAX_DEVICES = 32;

#ifdef _WIN32
    std::string windows_error_message(const DWORD error) {
        char* message = nullptr;
        const DWORD size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                          nullptr, error, 0, reinterpret_cast<char*>(&message), 0, nullptr);
        std::string result = size && message ? std::string(message, size) : "unknown error";
        if (message) {
            LocalFree(message);
        }
        while (!result.empty() && (result.back() == '\r' || result.back() == '\n')) {
            result.pop_back();
        }
        return result;
    }

    void append_installed_driver_paths(std::vector<std::string>& base_paths) {
        const char* program_files_roots[] = {
            std::getenv("ProgramFiles(x86)"),
            std::getenv("ProgramFiles"),
        };

        std::vector<std::string> discovered;
        for (const char* root : program_files_roots) {
            if (!root || !*root) {
                continue;
            }

            std::error_code error;
            for (const auto& entry : std::filesystem::directory_iterator(root, error)) {
                if (error) {
                    break;
                }
                if (!entry.is_directory(error) || error) {
                    error.clear();
                    continue;
                }

                const auto directory_name = entry.path().filename().string();
                if (directory_name.rfind("houmo-drv-xh2", 0) == 0) {
                    discovered.push_back(entry.path().string());
                }
            }
        }

        std::sort(discovered.rbegin(), discovered.rend());
        for (const auto& path : discovered) {
            if (std::find(base_paths.begin(), base_paths.end(), path) == base_paths.end()) {
                LOG_INFO("discovered Houmo driver installation: {}", path);
                base_paths.push_back(path);
            }
        }
    }
#endif

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
            char current_dir[MAX_PATH] = {};
            GetCurrentDirectoryA(MAX_PATH, current_dir);
            LOG_INFO("HAL loader context: current_dir={}, HOUMO_SDK_PATH={}, HOUMO_DRV_PATH={}, TCIM_RUNTIME_PATH={}", current_dir,
                     std::getenv("HOUMO_SDK_PATH") ? std::getenv("HOUMO_SDK_PATH") : "<unset>",
                     std::getenv("HOUMO_DRV_PATH") ? std::getenv("HOUMO_DRV_PATH") : "<unset>",
                     std::getenv("TCIM_RUNTIME_PATH") ? std::getenv("TCIM_RUNTIME_PATH") : "<unset>");

            std::vector<std::string> base_paths = {
                std::getenv("HOUMO_SDK_PATH") ? std::getenv("HOUMO_SDK_PATH") : "",
                std::getenv("HOUMO_DRV_PATH") ? std::getenv("HOUMO_DRV_PATH") : "",
                std::getenv("TCIM_RUNTIME_PATH") ? std::getenv("TCIM_RUNTIME_PATH") : "",
            };
            append_installed_driver_paths(base_paths);

            for (const auto& base_path : base_paths) {
                if (base_path.empty()) {
                    continue;
                }

                const std::string candidate_paths[] = {
                    base_path + "/hal/bin/" + library_name(),
                    base_path + "/hal/lib/" + library_name(),
                    base_path + "/bin/" + library_name(),
                    base_path + "/lib/" + library_name(),
                    base_path + "/" + library_name(),
                };

                for (const auto& path : candidate_paths) {
                    const DWORD attributes = GetFileAttributesA(path.c_str());
                    if (attributes == INVALID_FILE_ATTRIBUTES) {
                        LOG_INFO("HAL candidate does not exist: {}", path);
                        continue;
                    }

                    handle = LoadLibraryExA(path.c_str(), nullptr,
                                            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
                    if (handle) {
                        LOG_INFO("loaded Houmo HAL library from {}", path);
                        HlieUtil::set_env("TCIM_RUNTIME_PATH", std::filesystem::path(path).parent_path().string());
                        break;
                    }
                    const DWORD error = GetLastError();
                    LOG_ERROR("HAL candidate exists but failed to load: {}, error {} ({})", path, error, windows_error_message(error));
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
            const std::filesystem::path custom_hal_dirs[] = {
                "/opt/system/lib/xpu/houmo",
                "/usr/lib/xpu/houmo",
            };

            for (const auto& dir : custom_hal_dirs) {
                const auto path = dir / library_name();
                std::error_code error;
                if (!std::filesystem::exists(path, error) || error) {
                    LOG_INFO("custom HAL candidate does not exist: {}", path.string());
                    continue;
                }

                handle = dlopen(path.string().c_str(), RTLD_LAZY | RTLD_LOCAL);
                if (handle) {
                    LOG_INFO("loaded custom Houmo HAL library from {}", path.string());
                    HlieUtil::set_env("TCIM_RUNTIME_PATH", path.parent_path().string());
                    break;
                }

                LOG_ERROR("custom HAL candidate exists but failed to load: {}, error {}", path.string(), dlerror());
            }

            const char* env_paths[] = {
                std::getenv("HOUMO_SDK_PATH"),
                std::getenv("HOUMO_DRV_PATH"),
                std::getenv("TCIM_RUNTIME_PATH"),
                "/usr/local/houmo-sdk/hal/lib",
                "/usr/local/houmo-drv/hal/lib",
                ".",
                nullptr,
            };

            if (!handle) {
                for (const char** env_path = env_paths; *env_path != nullptr; ++env_path) {
                    if (!*env_path || !**env_path) {
                        continue;
                    }

                    const auto path = std::filesystem::path(*env_path) / library_name();
                    handle = dlopen(path.string().c_str(), RTLD_LAZY | RTLD_LOCAL);
                    if (handle) {
                        LOG_INFO("loaded Houmo HAL library from {}", path.string());
                        HlieUtil::set_env("TCIM_RUNTIME_PATH", path.parent_path().string());
                        break;
                    }
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


int HLIEDeviceService::acquire_mem(uint32_t size, uint32_t& out_device, bool dry_run, bool allow_capacity_limit,
                                   const std::string& model_name) {
    std::unique_lock lock(mutex_);
    LOG_INFO("acquire_mem");
    for (auto [dev, _] : devices_) {
        refresh_device_usage(dev);
    }
    LOG_INFO("acquire_mem refresh_device_usage done");

    switch (mode_) {
    case MemApplyMode::PreferIdleCard:
        return acquire_card_first(size, out_device, dry_run, allow_capacity_limit, model_name);
    case MemApplyMode::PreferUsedCard:
        return acquire_mem_first(size, out_device, dry_run, allow_capacity_limit, model_name);
    case MemApplyMode::ExclusiveCard:
        return acquire_exclusive(size, out_device, dry_run, allow_capacity_limit, model_name);
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
                                              const bool dry_run, const std::string& model_name) {

    if (devices.empty()) {
        LOG_ERROR("acquire_mem_on_devices: empty device list");
        return -1;
    }

    std::unique_lock lock(mutex_);

    // 去重
    std::unordered_set<uint32_t> uniq(devices.begin(), devices.end());
    const size_t n = uniq.size();

    const uint32_t per_card = (total_size + n - 1) / n; // ceil
    const bool allow_capacity_limit = n == 1;

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

        if (!check_mem_available(dev, per_card, allow_capacity_limit, model_name)) {
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
        info.mem_used += allow_capacity_limit ? std::min(per_card, info.mem_total) : per_card;

        if (exclusive) {
            info.exclusive = true;
        }

        LOG_INFO("device {} acquire {}MB (multi-card), used {}MB", dev, per_card, info.mem_used);
    }

    return 0;
}

int HLIEDeviceService::acquire_mem_on_device_count(const size_t device_count, const uint32_t total_size,
                                                   std::vector<int>& out_devices, const bool dry_run,
                                                   const std::string& model_name) {
    out_devices.clear();
    if (device_count == 0) {
        LOG_ERROR("acquire_mem_on_device_count: device count is zero");
        return -1;
    }

    std::unique_lock lock(mutex_);
    if (devices_.size() < device_count) {
        LOG_ERROR("model {} requires {} devices, but only {} devices are configured", model_name, device_count, devices_.size());
        return -1;
    }

    const uint32_t per_device = (total_size + device_count - 1) / device_count;
    const bool allow_capacity_limit = device_count == 1;
    for (auto& [id, info] : devices_) {
        refresh_device_usage(id);
        if (info.exclusive || !check_mem_available(id, per_device, allow_capacity_limit, model_name)) {
            continue;
        }
        out_devices.push_back(id);
        if (out_devices.size() == device_count) {
            break;
        }
    }

    if (out_devices.size() != device_count) {
        LOG_ERROR("model {} cannot acquire {} configured devices with {}MB required per device",
                  model_name, device_count, per_device);
        out_devices.clear();
        return -1;
    }

    if (!dry_run) {
        for (const int device : out_devices) {
            auto& info = devices_.at(device);
            info.mem_used += allow_capacity_limit ? std::min(per_device, info.mem_total) : per_device;
            LOG_INFO("device {} acquire {}MB for model {}", device,
                     allow_capacity_limit ? std::min(per_device, info.mem_total) : per_device, model_name);
        }
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
        const uint32_t release_size = n == 1 ? std::min(per_card, info.mem_total) : per_card;

        if (info.mem_used < release_size) {
            LOG_ERROR("device {} mem underflow: used {} < release {}", dev, info.mem_used, release_size);

            info.mem_used = 0;
        } else {
            info.mem_used -= release_size;
        }

        if (exclusive && info.mem_used == 0) {
            info.exclusive = false;
        }

        LOG_INFO("device {} release {}MB, used {}MB", dev, release_size, info.mem_used);
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

bool HLIEDeviceService::check_mem_available(const uint32_t device, const uint32_t size, const bool allow_capacity_limit,
                                            const std::string& model_name) {
    auto& sdk = hm_sys_sdk();
    if (!sdk.load()) {
        return false;
    }

    hm_mem_info mem_info{};
    if (sdk.get_mem_info(device, &mem_info) < 0)
        return false;

    if (allow_capacity_limit && size > mem_info.mem_total) {
        LOG_WARN("model {} estimated memory {}MB exceeds single device {} capacity {}MB; limit estimate to device capacity and continue availability check",
                 model_name, size, device, mem_info.mem_total);
        return mem_info.mem_avail >= mem_info.mem_total;
    }

    return mem_info.mem_avail > size + BACK_UP_MEM;
}

int HLIEDeviceService::acquire_card_first(const uint32_t size, uint32_t& out_device, const bool dry_run,
                                          const bool allow_capacity_limit, const std::string& model_name) {
    if (try_idle_first(size, out_device, dry_run, allow_capacity_limit, model_name) == 0)
        return 0;

    LOG_INFO("CARD_FIRST fallback to used cards");
    return try_used_first(size, out_device, dry_run, allow_capacity_limit, model_name);
}

int HLIEDeviceService::acquire_mem_first(const uint32_t size, uint32_t& out_device, const bool dry_run,
                                         const bool allow_capacity_limit, const std::string& model_name) {
    if (try_used_first(size, out_device, dry_run, allow_capacity_limit, model_name) == 0)
        return 0;

    LOG_INFO("MEM_FIRST fallback to idle cards");
    return try_idle_first(size, out_device, dry_run, allow_capacity_limit, model_name);
}

int HLIEDeviceService::acquire_exclusive(uint32_t size, uint32_t& out_device, const bool dry_run,
                                         const bool allow_capacity_limit, const std::string& model_name) {
    for (auto& [id, info] : devices_) {
        if (info.mem_used != 0 || info.exclusive)
            continue;

        if (check_mem_available(id, size, allow_capacity_limit, model_name)) {
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

int HLIEDeviceService::try_idle_first(const uint32_t size, uint32_t& out_device, const bool dry_run,
                                      const bool allow_capacity_limit, const std::string& model_name) {
    for (auto& [id, info] : devices_) {

        if (info.mem_used != 0 || info.exclusive)
            continue;

        if (!check_mem_available(id, size, allow_capacity_limit, model_name)) {
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

int HLIEDeviceService::try_used_first(const uint32_t size, uint32_t& out_device, const bool dry_run,
                                      const bool allow_capacity_limit, const std::string& model_name) {
    for (auto& [id, info] : devices_) {

        if (info.mem_used == 0 || info.exclusive)
            continue;

        if (!check_mem_available(id, size, allow_capacity_limit, model_name)) {
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

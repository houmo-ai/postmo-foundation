#include "hlie_base/hlie_rk_performance.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_singleton.h"
#include "hlie_base/hlie_util.h"

#ifndef _WIN32
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

HLIE_NAMESPACE_BEGIN

namespace {
    constexpr const char* RK_PERF_SECTION = "server";
    constexpr const char* RK_PERF_KEY = "rk_perf";
    constexpr const char* RK_PERF_CORES_KEY = "rk_perf_cores";
    constexpr const char* RK3588_COMPATIBLE = "rockchip,rk3588";
    constexpr const char* TCIM_SYNC_MODE_ENV = "TCIM_XH2_USE_SYNC_MODE";

    std::string to_lower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

#ifndef _WIN32
    std::string read_file(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return "";
        }

        std::ostringstream oss;
        oss << in.rdbuf();
        return oss.str();
    }

    bool is_rk3588_platform() {
        const auto compatible = to_lower(read_file("/proc/device-tree/compatible"));
        return compatible.find(RK3588_COMPATIBLE) != std::string::npos;
    }

    std::vector<int> parse_core_list(const std::string& cores) {
        std::vector<int> result;
        std::stringstream ss(cores);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t"));
            const auto last = token.find_last_not_of(" \t");
            if (last == std::string::npos) {
                continue;
            }
            token.erase(last + 1);

            try {
                const int core = std::stoi(token);
                if (core >= 0) {
                    result.push_back(core);
                }
            } catch (const std::exception& ex) {
                LOG_WARN("Ignore invalid rk perf core '{}', ex: {}", token, ex.what());
            }
        }

        if (result.empty()) {
            result = {4, 5, 6, 7};
        }
        return result;
    }

    std::vector<int> get_rk_perf_cores() {
        const auto& config_service = Singleton<HLIEConfigService>::instance();
        return parse_core_list(config_service->get_string(RK_PERF_SECTION, RK_PERF_CORES_KEY, "4,5,6,7", true));
    }

    void apply_affinity_to_current_thread(const std::vector<int>& cores, const char* scope) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (const int core : cores) {
            if (core >= CPU_SETSIZE) {
                LOG_WARN("Ignore RK performance core {}, it is greater than CPU_SETSIZE {}", core, CPU_SETSIZE);
                continue;
            }
            CPU_SET(core, &cpuset);
        }

        const int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        if (ret != 0) {
            LOG_WARN("Failed to apply RK performance affinity for {}, ret={}", scope, ret);
            return;
        }

        LOG_INFO("Applied RK performance affinity for {}, cores={}", scope, HlieUtil::join_devices(cores));
    }
#endif
} // namespace

bool HLIERkPerformance::is_enabled() {
    const auto& config_service = Singleton<HLIEConfigService>::instance();
    return config_service->get_bool(RK_PERF_SECTION, RK_PERF_KEY, false, true);
}

void HLIERkPerformance::apply_for_process() {
    if (!is_enabled()) {
        return;
    }

    HlieUtil::set_env(TCIM_SYNC_MODE_ENV, "1");

#ifdef _WIN32
    LOG_WARN("RK performance mode is enabled but affinity is only supported on Linux");
#else
    if (!is_rk3588_platform()) {
        LOG_WARN("RK performance mode is enabled, but current platform is not detected as RK3588; skip CPU affinity");
        return;
    }

    apply_affinity_to_current_thread(get_rk_perf_cores(), "process main thread");
#endif
}

void HLIERkPerformance::apply_for_current_thread() {
    if (!is_enabled()) {
        return;
    }

    HlieUtil::set_env(TCIM_SYNC_MODE_ENV, "1");

#ifdef _WIN32
    return;
#else
    if (!is_rk3588_platform()) {
        return;
    }

    apply_affinity_to_current_thread(get_rk_perf_cores(), "current thread");
#endif
}

HLIE_NAMESPACE_END

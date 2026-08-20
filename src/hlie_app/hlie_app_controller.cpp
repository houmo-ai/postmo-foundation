#include "hlie_app/hlie_app_controller.h"

#include <filesystem>
#include <unordered_map>

#include "hlie_app/hlie_model_info.h"
#include "hlie_app/hlie_config_bundle_validator.h"
#include "hlie_app/hlie_process_manager.h"
#include "hlie_app/hlie_replica_config.h"
#include "hlie_app/runtime/hlie_worker_port_allocator.h"
#include "hlie_app/hlie_web_service.h"
#include "hlie_app/hlie_workflow.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_rk_performance.h"
#include "hlie_base/hlie_util.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

HLIE_NAMESPACE_BEGIN

constexpr auto HLIECpp = R"(
    _/    _/  _/        _/_/_/  _/_/_/_/    _/_/_/
   _/    _/  _/          _/    _/        _/        _/_/_/    _/_/_/
  _/_/_/_/  _/          _/    _/_/_/    _/        _/    _/  _/    _/
 _/    _/  _/          _/    _/        _/        _/    _/  _/    _/
_/    _/  _/_/_/_/  _/_/_/  _/_/_/_/    _/_/_/  _/_/_/    _/_/_/
                                               _/        _/
                                              _/        _/
)";

namespace {
    constexpr int DEFAULT_LOG_RETENTION_DAYS = 7;
    constexpr int DEFAULT_LOG_MAX_FILES = 5;
    constexpr std::uintmax_t DEFAULT_LOG_FILE_SIZE_BYTES = 10U * 1024U * 1024U;
    constexpr std::uintmax_t MIN_LOG_FILE_SIZE_BYTES = 64U * 1024U;
    constexpr std::uintmax_t MAX_LOG_FILE_SIZE_BYTES = 1024U * 1024U * 1024U;
    constexpr bool DEFAULT_LOG_COMPRESS_ROTATED_FILES = true;
    constexpr int DEFAULT_LOG_COMPRESSION_LEVEL = 1;

    struct BootstrapLoggerContext {
        std::shared_ptr<spdlog::logger> logger;
        std::filesystem::path root;
        bool temporary = false;
    };

    std::uint64_t current_process_id() {
#ifdef _WIN32
        return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
        return static_cast<std::uint64_t>(getpid());
#endif
    }

    bool is_standalone_config_mode(const std::unordered_map<std::string, std::string>& args) {
#ifdef _WIN32
        return std::getenv("HLIECPP_STANDALONE_MODE") != nullptr;
#else
        return args.find("--config") != args.end();
#endif
    }

    bool is_sensitive_option(const std::string& key) {
        std::string lowered = key;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](const unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        return lowered.find("api-key") != std::string::npos || lowered.find("token") != std::string::npos ||
            lowered.find("password") != std::string::npos || lowered.find("secret") != std::string::npos;
    }

    std::string loggable_argument_value(const std::string& key, const std::string& value) {
        return is_sensitive_option(key) ? "<redacted>" : value;
    }

    BootstrapLoggerContext create_bootstrap_logger(const std::filesystem::path& platform_default_root,
                                                    const std::string& instance_id,
                                                    const bool standalone) {
        BootstrapLoggerContext context;
        context.root = platform_default_root;
        context.temporary = standalone;
        if (standalone) {
            std::error_code error;
            const auto temporary_root = std::filesystem::temp_directory_path(error);
            if (!error) {
                context.root = temporary_root / "HLIECpp" / "boot_logs";
            } else {
                std::cerr << "Cannot resolve temporary directory for standalone bootstrap logs: " << error.message() << std::endl;
                context.root.clear();
            }
        }

        const std::string file_name = "service_boot-" + std::to_string(current_process_id()) + "-" + instance_id + ".log";
        if (!context.root.empty()) {
            try {
                context.logger = HLIELogger::create_managed_logger(context.root.string(), file_name, HLIE_BOOT, "debug",
                                                                   DEFAULT_LOG_FILE_SIZE_BYTES, standalone ? 2 : DEFAULT_LOG_RETENTION_DAYS);
            } catch (const std::exception& exception) {
                std::cerr << "Cannot create bootstrap file logger at " << context.root.string()
                          << ": " << exception.what() << std::endl;
            }
        }
        if (!context.logger) {
            context.logger = HLIELogger::create_console_logger(HLIE_BOOT, "debug");
            context.root.clear();
        }
        return context;
    }

    bool finalize_and_validate_config_bundle(HLIEConfigService* config_service,
                                             const std::shared_ptr<spdlog::logger>& bootstrap_logger) {
        config_service->finalize_config_bundle(
            config_service->get_string("model", "llm_preset_catalog", "model_presets.json", true),
            config_service->get_string("model", "llm_override_catalog", "model_overrides.json", true));
        const auto validation = HLIEConfigBundleValidator::validate(*config_service);
        for (const auto& note : validation.notes) {
            bootstrap_logger->info("Config bundle: {}", note);
        }
        for (const auto& warning : validation.warnings) {
            bootstrap_logger->warn("Config bundle: {}", warning);
        }
        if (!validation.valid) {
            bootstrap_logger->error("Config bundle validation failed: {}", validation.error);
            return false;
        }
        return true;
    }

    void log_config_bundle(const std::shared_ptr<spdlog::logger>& logger,
                           const HLIEConfigBundle& bundle,
                           const BootstrapLoggerContext& bootstrap) {
        logger->info("Startup mode: {}", bundle.explicit_override ? "explicit-config" : "platform-default");
        logger->info("Config bundle root: {}", bundle.root.string());
        logger->info("Config bundle sys_config: {}", bundle.sys_config.string());
        logger->info("Config bundle config_schema: {}", bundle.config_schema.string());
        logger->info("Config bundle memory_estimates: {}", bundle.memory_estimates.string());
        logger->info("Config bundle model_presets: {}", bundle.model_presets.string());
        logger->info("Config bundle model_overrides: {}", bundle.model_overrides.string());
        logger->info("Bootstrap logging: mode={}, root={}", bootstrap.temporary ? "temporary" : "platform-default",
                     bootstrap.root.empty() ? "console" : bootstrap.root.string());
    }

    HLIEManagedLogConfig load_managed_log_config(const HLIEConfigService* config_services) {
        HLIEManagedLogConfig config;
        config.retention_days = config_services->get_int("log", "retention_days", DEFAULT_LOG_RETENTION_DAYS);
        config.max_files = config_services->get_int("log", "max_files", DEFAULT_LOG_MAX_FILES);
        const auto max_file_size = config_services->get_string("log", "max_file_size", "10M");
        config.compress_rotated_files = config_services->get_bool(
            "log", "compress_rotated_files", DEFAULT_LOG_COMPRESS_ROTATED_FILES);
        config.compression_level = config_services->get_int("log", "compression_level", DEFAULT_LOG_COMPRESSION_LEVEL);
        if (config.retention_days < 0) {
            LOG_WARN("log.retention_days={} is invalid; use default {}", config.retention_days, DEFAULT_LOG_RETENTION_DAYS);
            config.retention_days = DEFAULT_LOG_RETENTION_DAYS;
        }
        if (config.max_files < 0) {
            LOG_WARN("log.max_files={} is invalid; disable file-count limit", config.max_files);
            config.max_files = 0;
        }
        if (!HLIELogger::parse_file_size(max_file_size, config.max_file_size_bytes) ||
            config.max_file_size_bytes < MIN_LOG_FILE_SIZE_BYTES ||
            config.max_file_size_bytes > MAX_LOG_FILE_SIZE_BYTES) {
            LOG_WARN("log.max_file_size='{}' is invalid or outside [64K, 1G]; use default 10M", max_file_size);
            config.max_file_size_bytes = DEFAULT_LOG_FILE_SIZE_BYTES;
        }
        if (config.compression_level < 1 || config.compression_level > 9) {
            LOG_WARN("log.compression_level={} is invalid; use default {}", config.compression_level,
                     DEFAULT_LOG_COMPRESSION_LEVEL);
            config.compression_level = DEFAULT_LOG_COMPRESSION_LEVEL;
        }
        return config;
    }

    struct ReplicaPlacement {
        int instance_id = 0;
        std::vector<int> devices;
    };

    int estimate_replica_device_count(const std::vector<std::string>& model_names) {
        int count = 1;
        for (const auto& model_name : model_names) {
            if (model_name.empty() || model_name == HLIEModelType::SD3) {
                continue;
            }
            count = std::max(count, HLIEModelInfo::estimate_device_num(model_name));
        }
        return count;
    }

    bool build_replica_placements(const std::vector<int>& configured_devices,
                                  const HLIEReplicaConfig& replica_config,
                                  int per_instance_device_count,
                                  std::vector<ReplicaPlacement>& placements) {
        placements.clear();
        if (configured_devices.empty()) {
            LOG_ERROR("replica placement failed: no configured devices");
            return false;
        }

        int max_instances_per_device =
            replica_config.device_policy == "exclusive_card" ? 1 : replica_config.max_instances_per_device;
        if (replica_config.enabled() && replica_config.device_policy != "exclusive_card" &&
            configured_devices.size() == 1 && max_instances_per_device < 2) {
            LOG_WARN("replica.max_instances_per_device={} is too small for single-card shared replica mode; use 2 instead. "
                     "Set replica.device_policy=exclusive_card to force one instance per device.",
                     max_instances_per_device);
            max_instances_per_device = 2;
        }
        std::map<int, int> usage;
        for (const auto device : configured_devices) {
            usage[device] = 0;
        }

        auto sorted_candidates = [&]() {
            std::vector<int> candidates = configured_devices;
            std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
                if (usage[a] == usage[b]) {
                    return a < b;
                }
                if (replica_config.device_policy == "prefer_used_card") {
                    return usage[a] > usage[b];
                }
                return usage[a] < usage[b];
            });
            return candidates;
        };

        for (int instance_id = 0; instance_id < replica_config.count; ++instance_id) {
            ReplicaPlacement placement;
            placement.instance_id = instance_id;
            for (const auto device : sorted_candidates()) {
                if (usage[device] >= max_instances_per_device) {
                    continue;
                }
                placement.devices.push_back(device);
                if (static_cast<int>(placement.devices.size()) == per_instance_device_count) {
                    break;
                }
            }

            if (static_cast<int>(placement.devices.size()) != per_instance_device_count) {
                LOG_ERROR("replica placement failed: count={}, instance_id={}, required_devices={}, max_instances_per_device={}, policy={}",
                          replica_config.count, instance_id, per_instance_device_count,
                          max_instances_per_device, replica_config.device_policy);
                placements.clear();
                return false;
            }

            for (const auto device : placement.devices) {
                ++usage[device];
            }
            LOG_INFO("replica placement: instance_id={}, devices={}", instance_id, HlieUtil::join_devices(placement.devices));
            placements.push_back(std::move(placement));
        }

        return true;
    }

    std::string instance_id_from_args(const std::unordered_map<std::string, std::string>& args) {
        if (const auto it = args.find("--instance_id"); it != args.end()) {
            return it->second;
        }
        return "0";
    }

#ifndef _WIN32
    std::string current_executable_path() {
        std::error_code error;
        const auto path = std::filesystem::read_symlink("/proc/self/exe", error);
        if (!error && path.is_absolute()) {
            return path.string();
        }
        return "./bin/HLIECpp";
    }

    std::string resolve_linux_log_location(
        const HLIEConfigService* config_services,
        const std::shared_ptr<spdlog::logger>& bootstrap_logger) {
        const std::string requested = config_services->get_string(
            "log", "location", "/var/log/HLIECpp", true);
        const auto is_writable_directory = [](const std::filesystem::path& path) {
            std::error_code error;
            std::filesystem::create_directories(path, error);
            return !error && access(path.string().c_str(), W_OK) == 0;
        };
        if (!requested.empty() && is_writable_directory(requested)) {
            return requested;
        }

        std::filesystem::path fallback;
        if (const char* state_home = std::getenv("XDG_STATE_HOME"); state_home && *state_home) {
            fallback = std::filesystem::path(state_home) / "HLIECpp" / "logs";
        } else if (const char* home = std::getenv("HOME"); home && *home) {
            fallback = std::filesystem::path(home) / ".local" / "state" / "HLIECpp" / "logs";
        } else {
            std::error_code error;
            fallback = std::filesystem::temp_directory_path(error) / "HLIECpp" / "logs";
        }
        if (!is_writable_directory(fallback)) {
            bootstrap_logger->error(
                "Neither configured nor per-user log directory is writable: configured={}, fallback={}",
                requested, fallback.string());
            return requested;
        }
        bootstrap_logger->warn(
            "Configured log directory is not writable; using per-user fallback: configured={}, fallback={}, uid={}",
            requested, fallback.string(), static_cast<unsigned long long>(geteuid()));
        HlieUtil::set_env("HLIE__LOG__LOCATION", fallback.string());
        return fallback.string();
    }
#endif

    bool preflight_dynamic_worker_port(const HLIEConfigService* config_services,
                                       const std::unordered_map<std::string, std::string>& args) {
        if (args.find("--launch_generation") == args.end()) {
            return true;
        }
        const auto instance_it = args.find("--instance_id");
        if (instance_it == args.end()) {
            LOG_ERROR("Dynamic worker launch is missing --instance_id");
            return false;
        }

        const int instance_id = HlieUtil::parse_natural_num(instance_it->second, false);
        if (instance_id <= 0) {
            LOG_ERROR("Dynamic worker instance_id is invalid: {}", instance_it->second);
            return false;
        }
        int master_port = config_services->get_int("server", "port", -1, true);
        if (master_port < 0) {
            const auto port_it = args.find("--port");
            try {
                master_port = port_it == args.end() ? 7901 : std::stoi(port_it->second);
            } catch (const std::exception&) {
                master_port = 7901;
            }
        }
        const auto worker_port = static_cast<long long>(master_port) + instance_id;
        if (worker_port <= 0 || worker_port > 65535 ||
            !HLIEWorkerPortAllocator::is_tcp_port_available(static_cast<int>(worker_port))) {
            LOG_ERROR("Dynamic worker port preflight failed: master_port={}, instance_id={}, port={}",
                      master_port, instance_id, worker_port);
            return false;
        }
        LOG_INFO("Dynamic worker port preflight passed: instance_id={}, port={}",
                 instance_id, worker_port);
        return true;
    }

    void set_process_devices_override(const std::string& devices) {
        HlieUtil::set_env("HLIE__SERVER__DEVICES", devices);
    }

    /**
     * Init the multi-instances cases and some common logics
     *
     * @param program_path windows need the absolute path, linux need "./app_name"
     * @return the error code
     */
    int initialize_ex(const std::string& program_path) {
        const auto& config_services = Singleton<HLIEConfigService>::instance();
        auto devices = config_services->get_string("server", "devices", "0", true);
        const auto& args = config_services->get_args();
        const auto replica_config = HLIEReplicaConfig::load();
        if (args.find("--device_id") != args.end()) {
            devices = args.at("--device_id");
            set_process_devices_override(devices);
            if (args.find("--instance_id") == args.end()) {
                LOG_ERROR("Instance Replica must specify all the --instance_id and --device_id params");
                return -1;
            }
        }

        // Master process plans and starts replica workers.
        if (args.find("--instance_id") == args.end()) {
            if (replica_config.enabled()) {
                const auto default_model = config_services->get_string("model", "default","", true);
                const auto enable_sd3 = config_services->get_bool("model", "enable_sd3", false);
                std::vector<std::string> model_names;
                if (default_model.empty() && !enable_sd3) {
                    LOG_ERROR("default_model is empty && enable_sd3 is false in Multi Copies Case");
                    return -1;
                }

                if (!default_model.empty()) {
                    model_names.push_back(default_model);
                }
                if (enable_sd3) {
                    model_names.emplace_back(HLIEModelType::SD3);
                }

                std::vector<int> configured_devices;
                if (HlieUtil::parse_device(devices, configured_devices) < 0 || configured_devices.empty()) {
                    LOG_ERROR("cannot parse replica devices from {}", devices);
                    return -1;
                }
                std::sort(configured_devices.begin(), configured_devices.end());
                configured_devices.erase(std::unique(configured_devices.begin(), configured_devices.end()), configured_devices.end());

                const int per_instance_device_count = estimate_replica_device_count(model_names);
                std::vector<ReplicaPlacement> placements;
                if (!build_replica_placements(configured_devices, replica_config, per_instance_device_count, placements)) {
                    return -1;
                }

                const auto master_devices = HlieUtil::join_devices(placements.front().devices);
                set_process_devices_override(master_devices);
                devices = master_devices;

                LOG_INFO("Main process start to pull new replicas[{}], master devices={}, policy={}, configured_max_instances_per_device={}",
                         replica_config.count - 1, master_devices, replica_config.device_policy,
                         replica_config.max_instances_per_device);

                for (int i = 1; i < replica_config.count; ++i) {
                    const std::string devices_for_sub = HlieUtil::join_devices(placements.at(i).devices);
                    LOG_INFO("acquired device {} for instance-{}", devices_for_sub, i);

                    // Worker instance arguments.
                    std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(i), std::string("--device_id"), devices_for_sub};
                    // Copy llama-params
                    for (const auto& [key, val] : args) {
                        args_for_sub_process.push_back(key);
                        args_for_sub_process.push_back(val);
                    }

                    if (auto pid = Singleton<HLIEProcessManager>::instance()->create_process(program_path, args_for_sub_process, {model_names}, i);
                        pid == 0 || pid == -1) {
                        LOG_ERROR("Failed to start process for instance_id[{}]", i);
                    } else {
                        LOG_INFO("Started process PID[{}] instance_id[{}]", pid, i);
                    }
                }
            }
        } else {
            if (args.find("--device_id") == args.end()) {
                LOG_ERROR("Instance Replica must specify all the --instance_id and --device_id params");
                return -1;
            }

            const auto instance_id_str = args.at("--instance_id");
            const auto instance_id = HlieUtil::parse_natural_num(instance_id_str, false);
            if (instance_id <= 0) {
                LOG_ERROR("Cannot parse the instance_id, must be a positive integer");
                return -1;
            }

            if (replica_config.enabled() && replica_config.memory_saving) {
                LOG_INFO("multi instance mode and memory_saving is true, start to wait the others for {}s.",
                         instance_id * replica_config.startup_delay_sec);
                std::this_thread::sleep_for(std::chrono::seconds(instance_id * replica_config.startup_delay_sec));
            }
        }

        const auto tcim_log_level = config_services->get_int("log", "tcim_level", 2, true);
        HlieUtil::set_env("TCIM_LOG_LEVEL", std::to_string(tcim_log_level));
        HlieUtil::set_env("TCIM_BACKEND", "Xh2HalBackend");
        HLIERkPerformance::apply_for_process();

        return 0;
    }

} // namespace

int HLIEAppController::run() {
    HLIE_CHECK_NULL_RETURN(workflow_, -1);
    if (stop_requested_.load(std::memory_order_acquire)) {
        LOG_DEBUG("Application run skipped because shutdown was requested");
        return 0;
    }
    if (zero_load_) {
        LOG_WARN("Service starting in Zero-Load Model.");
    }
    // Zero-load still needs workflow-level background services such as model
    // discovery and residency management. Individual model services are
    // guarded inside HLIEWorkflow::run().
    if (!workflow_->run()) {
        const bool shutdown_requested =
            stop_requested_.load(std::memory_order_acquire);
        if (shutdown_requested) {
            LOG_DEBUG("Workflow startup was superseded by shutdown");
        } else {
            LOG_ERROR("Workflow failed to enter running state");
        }
        stop();
        return shutdown_requested ? 0 : -1;
    }

    if (!web_service_) {
        LOG_ERROR("Web service is unavailable");
        stop();
        return -1;
    }

    // Lifecycle holds the Running-to-Serving gate while these externally
    // visible resources are published. A concurrent stop either wins before
    // this callback or waits until all three publications are coherent.
    const bool serving_published = workflow_->publish_serving([this] {
        web_service_->register_routers();
        if (web_service_->run() < 0) {
            LOG_ERROR("Web service failed to run");
            return false;
        }
        balancer_client_.start(
            HLIEBalancerConfig::load(), web_service_->get_port(),
            instance_id_from_args(
                Singleton<HLIEConfigService>::instance()->get_args()));
        is_running_.store(true, std::memory_order_release);
        if (stop_requested_.load(std::memory_order_acquire)) {
            is_running_.store(false, std::memory_order_release);
            return false;
        }
        return true;
    });
    if (!serving_published) {
        const bool shutdown_requested =
            stop_requested_.load(std::memory_order_acquire);
        if (shutdown_requested) {
            LOG_DEBUG("Serving publication was superseded by shutdown");
        } else {
            LOG_ERROR("Application failed to publish serving state");
        }
        stop();
        return shutdown_requested ? 0 : -1;
    }

    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] {
        return !is_running_.load(std::memory_order_acquire);
    });
    lock.unlock();

    if (worker_shutdown_requested_.load() && !stop_started_.load()) {
        const int response_grace_ms = std::clamp(
            Singleton<HLIEConfigService>::instance()->get_int(
                "worker_supervisor", "shutdown_response_grace_ms", 200, true),
            0, 2000);
        std::this_thread::sleep_for(std::chrono::milliseconds(response_grace_ms));
    }
    stop();
    return 0;
}

void HLIEAppController::stop() {
    stop_requested_.store(true, std::memory_order_release);
    if (stop_started_.exchange(true)) {
        std::unique_lock lock(stop_completion_mutex_);
        stop_completion_cv_.wait(lock, [this] {
            return stop_completed_.load(std::memory_order_acquire);
        });
        return;
    }

    if (workflow_) {
        workflow_->begin_draining();
    }

    // Draining waits for any in-flight serving publication. Publish the local
    // stopped state only after that gate closes so run() cannot overwrite it.
    is_running_.store(false, std::memory_order_release);
    cv_.notify_all();

    balancer_client_.stop();

    if (web_service_) {
        web_service_->stop();
    }

    if (workflow_) {
        workflow_->stop();
    }

    Singleton<HLIEProcessManager>::instance()->terminate_all();

    stop_completed_.store(true, std::memory_order_release);
    stop_completion_cv_.notify_all();
}

void HLIEAppController::request_stop_after_response() {
    worker_shutdown_requested_.store(true);
    stop_requested_.store(true, std::memory_order_release);
    is_running_.store(false, std::memory_order_release);
    cv_.notify_all();
}


#ifdef _WIN32

int HLIEWinAppController::initialize() {
    char exePath[260];
    GetModuleFileName(nullptr, exePath, MAX_PATH); // 获取exe绝对路径
    const auto program_path = std::string(exePath);
    const auto program_dir = std::string(exePath).substr(0, std::string(exePath).find_last_of('\\') + 1);
    const auto program_name = std::string(exePath).substr(std::string(exePath).find_last_of('\\') + 1);

    const auto& config_services = Singleton<HLIEConfigService>::instance();

    const auto& args = config_services->get_args();
    std::string instance_id{"0"};
    if (args.find("--instance_id") != args.end()) {
        // 子进程ID
        instance_id = args.at("--instance_id");
    }

    const bool standalone_mode = is_standalone_config_mode(args);
    const auto bootstrap = create_bootstrap_logger(program_dir + "\\boot_log", instance_id, standalone_mode);
    const auto& bootstrap_logger = bootstrap.logger;
    HLIE_CHECK_NULL_RETURN(bootstrap_logger, -6);
    bootstrap_logger->info("HLIEWinAppController initialize: mode={}, instance_id={}, pid={}, program_path={}, program_dir={}",
                           standalone_mode ? "standalone" : "service", instance_id, GetCurrentProcessId(), program_path, program_dir);
    for (const auto& [key, val] : args) {
        bootstrap_logger->info("program arg: {}={}", key, loggable_argument_value(key, val));
    }

    std::string config_error;
    if (!config_services->prepare_config_bundle(program_dir + "..\\config\\sys.cfg", &config_error)) {
        bootstrap_logger->error("Cannot resolve configuration directory: {}", config_error);
        return -5;
    }
    const auto config_path = config_services->get_config_bundle().sys_config;
    if (config_services->parse(config_path.string()) < 0) {
        bootstrap_logger->error("sys_config parsed failed: {}", config_path.string());
        return -5;
    }
    if (!finalize_and_validate_config_bundle(config_services, bootstrap_logger)) {
        return -5;
    }
    bootstrap_logger->info("sys_config parsed successfully: {}", config_services->get_config_path());

    const std::string log_location = config_services->get_string("log", "location");
    if (log_location.empty()) {
        bootstrap_logger->error("Cannot found log location config");
        return -1;
    }

    try {
        if (std::filesystem::create_directories(log_location)) {
            bootstrap_logger->info("Log Dir created successfully.");
        } else {
            bootstrap_logger->info("Log Dir already exist.");
        }
    } catch (const std::filesystem::filesystem_error& e) {
        bootstrap_logger->error("Log Dir created failed, exception: {}", e.what());
        return -1;
    }

    const auto managed_log_config = load_managed_log_config(config_services);
    const auto boot_logger = HLIELogger::create_managed_logger(log_location, "service_boot-" + instance_id + ".log",
                                                               HLIE_BOOT, "debug", managed_log_config);
    HLIE_CHECK_NULL_RETURN(boot_logger, -6);

    const auto datetime = HlieUtil::get_datetime_str();
    HlieUtil::set_env("TCIM_LOG_LEVEL", std::to_string(config_services->get_int("log", "tcim_level", 2, true)));
    const auto tcim_fallback_directory = managed_log_config.retention_days == 0
                                             ? std::filesystem::path(log_location)
                                             : HLIELogger::current_date_directory(log_location);
    const auto tcim_log_file = HlieUtil::join_paths(tcim_fallback_directory.string(),
                                                    "tcim-" + datetime + "-" + instance_id + ".log");
    HlieUtil::set_env("TCIM_LOG_FILE", tcim_log_file);
    boot_logger->info("TCIM_LOG_FILE={}", tcim_log_file);
    if (!HLIELogger::register_tcim_log_handler(log_location, "tcim-" + instance_id + ".log", managed_log_config)) {
        boot_logger->warn("TCIM managed log handler registration failed; use TCIM_LOG_FILE fallback");
    }

    const std::string log_file_name = "hlie-llm-" + instance_id + ".log";
    const auto log_level = config_services->get_string("log", "level", "debug", true);
    const auto logger = HLIELogger::init_managed_default_logger(log_location, log_file_name, "HLIE", log_level,
                                                                managed_log_config);
    HLIE_CHECK_NULL_RETURN(logger, -1);
    spdlog::set_default_logger(logger);
    log_config_bundle(logger, config_services->get_config_bundle(), bootstrap);

    LOG_INFO(HLIECpp);
    LOG_INFO("HLIEWinAppController initialized logger: instance_id={}, pid={}, log_file={}, boot_log_file={}",
             instance_id, GetCurrentProcessId(), log_file_name, "service_boot-" + instance_id + ".log");
    for (const auto& [key, val] : args) {
        LOG_INFO("program arg: {}={}", key, loggable_argument_value(key, val));
    }

    const auto access_logger = HLIELogger::create_managed_logger(log_location, "access-" + instance_id + ".log",
                                                                 HLIE_ACCESS, "debug", managed_log_config);
    HLIE_CHECK_NULL_RETURN(access_logger, -6);

    spdlog::flush_every(std::chrono::seconds(10));

    // 为啥方便手动扩展，支持配置环境变量HLIE__MODEL__CUR_MODEL用于指定独立的模型目录, 这种场景必须指定--device_id
    const std::string cur_model = config_services->get_string("model", "cur_model", "", instance_id != "0");
    if (cur_model.empty()) {
        boot_logger->error("WINSERVICE cannot found model, cur_model is empty");
        LOG_ERROR("WINSERVICE cannot found model, cur_model is empty, instance_id={}", instance_id);
        return -2;
    }
    LOG_INFO("HLIEWinAppController cur_model={}, instance_id={}", cur_model, instance_id);

    if (initialize_ex(program_path) < 0) {
        LOG_ERROR("initialize_ex failed: instance_id={}, program_path={}", instance_id, program_path);
        return -3;
    }
    if (!preflight_dynamic_worker_port(config_services, args)) {
        return -4;
    }

    workflow_ = std::make_shared<HLIEWorkflow>();
    workflow_->set_process_shutdown_callback([this] { request_stop_after_response(); });
    if (auto res = workflow_->initialize(cur_model, program_path); res < 0) {
        LOG_ERROR("workflow initialize failed: instance_id={}, result={}", instance_id, res);
        return -1;
    } else {
        zero_load_ = 8 == res;
        LOG_INFO("workflow initialize succeeded: instance_id={}, result={}, zero_load={}", instance_id, res, zero_load_);
    }

    web_service_ = std::make_shared<HLIEWebService>(
        std::static_pointer_cast<IHLIEHttpRouteProvider>(workflow_),
        std::static_pointer_cast<IHLIEWebServiceStateProvider>(workflow_),
        std::static_pointer_cast<IHLIEAuthPolicy>(workflow_));

    const auto n_threads_http = workflow_->calculate_http_thread_count();
    web_service_->init(n_threads_http);

    return 0;
}

#else

int HLIELinuxAppController::initialize() {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    const auto& args = config_services->get_args();
    std::string instance_id{"0"};
    if (args.find("--instance_id") != args.end()) {
        // 子进程ID
        instance_id = args.at("--instance_id");
    }

    const bool standalone_mode = is_standalone_config_mode(args);
    const auto bootstrap = create_bootstrap_logger("./logs", instance_id, standalone_mode);
    const auto& bootstrap_logger = bootstrap.logger;
    HLIE_CHECK_NULL_RETURN(bootstrap_logger, -6);
    bootstrap_logger->info("HLIELinuxAppController initialize: mode={}, instance_id={}, pid={}",
                           standalone_mode ? "standalone" : "service", instance_id, current_process_id());
    for (const auto& [key, val] : args) {
        bootstrap_logger->info("program arg: {}={}", key, loggable_argument_value(key, val));
    }

    std::string config_error;
    if (!config_services->prepare_config_bundle("./config/sys.cfg", &config_error)) {
        bootstrap_logger->error("Cannot resolve configuration directory: {}", config_error);
        return -5;
    }
    const auto config_path = config_services->get_config_bundle().sys_config;
    if (config_services->parse(config_path.string()) < 0) {
        if (args.find("--config") != args.end()) {
            bootstrap_logger->error("Explicit sys_config parsed failed: {}", config_path.string());
            return -5;
        }
        if (config_services->get_string("model", "cur_model").empty()) {
            LOG_ERROR("sys_config parsed failed and no env[HLIE__MODEL__CUR_MODEL]");
            return -5;
        }

        LOG_WARN("sys_config parsed failed and will use env[HLIE__MODEL__CUR_MODEL] to continue");
    }
    if (!finalize_and_validate_config_bundle(config_services, bootstrap_logger)) {
        return -5;
    }

    const std::string log_location = resolve_linux_log_location(config_services, bootstrap_logger);
    const auto managed_log_config = load_managed_log_config(config_services);
    const auto boot_logger = HLIELogger::create_managed_logger(log_location, "service_boot-" + instance_id + ".log",
                                                               HLIE_BOOT, "debug", managed_log_config);
    HLIE_CHECK_NULL_RETURN(boot_logger, -6);

    const auto datetime = HlieUtil::get_datetime_str();
    HlieUtil::set_env("TCIM_LOG_LEVEL", std::to_string(config_services->get_int("log", "tcim_level", 2, true)));
    const auto tcim_fallback_directory = managed_log_config.retention_days == 0
                                             ? std::filesystem::path(log_location)
                                             : HLIELogger::current_date_directory(log_location);
    const auto tcim_log_file = HlieUtil::join_paths(tcim_fallback_directory.string(),
                                                    "tcim-" + datetime + "-" + instance_id + ".log");
    HlieUtil::set_env("TCIM_LOG_FILE", tcim_log_file);
    boot_logger->info("TCIM_LOG_FILE={}", tcim_log_file);
    if (!HLIELogger::register_tcim_log_handler(log_location, "tcim-" + instance_id + ".log", managed_log_config)) {
        boot_logger->warn("TCIM managed log handler registration failed; use TCIM_LOG_FILE fallback");
    }

    const std::string log_file_name = "hlie-llm-" + instance_id + ".log";
    const auto log_level = config_services->get_string("log", "level", "debug", true);
    const auto logger = HLIELogger::init_managed_default_logger(log_location, log_file_name, "HLIE", log_level,
                                                                managed_log_config);
    HLIE_CHECK_NULL_RETURN(logger, -1);
    spdlog::set_default_logger(logger);
    log_config_bundle(logger, config_services->get_config_bundle(), bootstrap);

    LOG_INFO(HLIECpp);

    const auto access_logger = HLIELogger::create_managed_logger(log_location, "access-" + instance_id + ".log",
                                                                 HLIE_ACCESS, "debug", managed_log_config);
    HLIE_CHECK_NULL_RETURN(access_logger, -6);

    spdlog::flush_every(std::chrono::seconds(10));
    const auto program_path = current_executable_path();
    boot_logger->info("Linux executable path: {}", program_path);
    if (initialize_ex( program_path) < 0) {
        return -3;
    }
    if (!preflight_dynamic_worker_port(config_services, args)) {
        return -4;
    }

    const std::string cur_model = config_services->get_string("model", "cur_model", "", instance_id != "0");
    if (cur_model.empty()) {
        LOG_ERROR("cannot found model, cur_model is empty");
        return -2;
    }

    workflow_ = std::make_shared<HLIEWorkflow>();
    workflow_->set_process_shutdown_callback([this] { request_stop_after_response(); });
    if (const auto res = workflow_->initialize(cur_model, program_path); res < 0) {
        return -1;
    } else {
        zero_load_ = 8 == res;
    }

    web_service_ = std::make_shared<HLIEWebService>(
        std::static_pointer_cast<IHLIEHttpRouteProvider>(workflow_),
        std::static_pointer_cast<IHLIEWebServiceStateProvider>(workflow_),
        std::static_pointer_cast<IHLIEAuthPolicy>(workflow_));

    const auto n_threads_http = workflow_->calculate_http_thread_count();
    web_service_->init(n_threads_http);

    return 0;
}

#endif

HLIE_NAMESPACE_END

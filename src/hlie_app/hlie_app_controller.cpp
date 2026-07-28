#include "hlie_app/hlie_app_controller.h"

#include <filesystem>
#include <unordered_map>

#include "hlie_app/hlie_model_info.h"
#include "hlie_app/hlie_process_manager.h"
#include "hlie_app/hlie_replica_config.h"
#include "hlie_app/hlie_web_service.h"
#include "hlie_app/hlie_workflow.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_rk_performance.h"
#include "hlie_base/hlie_util.h"

#include <algorithm>
#include <map>
#include <vector>

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
    constexpr int DEFAULT_LOG_FILE_SIZE_MB = 10;

    struct ManagedLogConfig {
        int retention_days = DEFAULT_LOG_RETENTION_DAYS;
        int max_file_size_mb = DEFAULT_LOG_FILE_SIZE_MB;

        std::uintmax_t max_file_size_bytes() const {
            return static_cast<std::uintmax_t>(max_file_size_mb) * 1024U * 1024U;
        }
    };

    ManagedLogConfig load_managed_log_config(const HLIEConfigService* config_services) {
        ManagedLogConfig config;
        const auto legacy_max_files = config_services->get_int("log", "max_files", DEFAULT_LOG_RETENTION_DAYS);
        config.retention_days = config_services->get_int("log", "retention_days", legacy_max_files);
        config.max_file_size_mb = config_services->get_int("log", "max_file_size_mb", DEFAULT_LOG_FILE_SIZE_MB);
        if (config.retention_days <= 0) {
            LOG_WARN("log.retention_days={} is invalid; use default {}", config.retention_days, DEFAULT_LOG_RETENTION_DAYS);
            config.retention_days = DEFAULT_LOG_RETENTION_DAYS;
        }
        if (config.max_file_size_mb <= 0) {
            LOG_WARN("log.max_file_size_mb={} is invalid; use default {}", config.max_file_size_mb, DEFAULT_LOG_FILE_SIZE_MB);
            config.max_file_size_mb = DEFAULT_LOG_FILE_SIZE_MB;
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

        const auto tcim_log_level = config_services->get_int("log", "tcim_level", 2);
        HlieUtil::set_env("TCIM_LOG_LEVEL", std::to_string(tcim_log_level));
        HlieUtil::set_env("TCIM_BACKEND", "Xh2HalBackend");
        HLIERkPerformance::apply_for_process();

        return 0;
    }

} // namespace

int HLIEAppController::run() {
    HLIE_CHECK_NULL_RETURN(workflow_, -1);
    if (zero_load_) {
        LOG_WARN("Service starting in Zero-Load Model.");
    }
    // Zero-load still needs workflow-level background services such as model
    // discovery and residency management. Individual model services are
    // guarded inside HLIEWorkflow::run().
    workflow_->run();

    HLIE_CHECK_NULL_RETURN(web_service_, -1);
    web_service_->register_routers();
    if (web_service_->run() < 0) {
        LOG_ERROR("web service failed to run");
        return -1;
    }
    balancer_client_.start(HLIEBalancerConfig::load(), web_service_->get_port(),
                           instance_id_from_args(Singleton<HLIEConfigService>::instance()->get_args()));

    is_running_.store(true);

    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return !is_running_.load(); });
    return 0;
}

void HLIEAppController::stop() {
    balancer_client_.stop();

    if (workflow_) {
        workflow_->stop();
    }

    if (web_service_) {
        web_service_->stop();
    }

    Singleton<HLIEProcessManager>::instance()->terminate_all();
    is_running_.store(false);
    cv_.notify_one();
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

    const auto bootstrap_logger = HLIELogger::create_managed_logger(program_dir + "\\boot_log",
                                                                    "service_boot-" + instance_id + ".log",
                                                                    HLIE_BOOT, "debug",
                                                                    static_cast<std::uintmax_t>(DEFAULT_LOG_FILE_SIZE_MB) * 1024U * 1024U,
                                                                    DEFAULT_LOG_RETENTION_DAYS);
    HLIE_CHECK_NULL_RETURN(bootstrap_logger, -6);
    bootstrap_logger->info("HLIEWinAppController initialize: instance_id={}, pid={}, program_path={}, program_dir={}",
                           instance_id, GetCurrentProcessId(), program_path, program_dir);
    for (const auto& [key, val] : args) {
        bootstrap_logger->info("program arg: {}={}", key, val);
    }

    if (const auto config_path = program_dir + "..\\config\\sys.cfg"; config_services->parse(config_path) < 0) {
        bootstrap_logger->error("sys_config parsed failed: {}", config_path);
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
                                                               HLIE_BOOT, "debug", managed_log_config.max_file_size_bytes(),
                                                               managed_log_config.retention_days);
    HLIE_CHECK_NULL_RETURN(boot_logger, -6);

    const auto datetime = HlieUtil::get_datetime_str();
    HlieUtil::set_env("TCIM_LOG_LEVEL", std::to_string(config_services->get_int("log", "tcim_level", 2)));
    const auto tcim_log_file = HlieUtil::join_paths(HLIELogger::current_date_directory(log_location).string(),
                                                    "tcim-" + datetime + "-" + instance_id + ".log");
    HlieUtil::set_env("TCIM_LOG_FILE", tcim_log_file);
    boot_logger->info("TCIM_LOG_FILE={}", tcim_log_file);
    if (!HLIELogger::register_tcim_log_handler(log_location, "tcim-" + instance_id + ".log",
                                               managed_log_config.max_file_size_bytes(),
                                               managed_log_config.retention_days)) {
        boot_logger->warn("TCIM managed log handler registration failed; use TCIM_LOG_FILE fallback");
    }

    const std::string log_file_name = "hlie-llm-" + instance_id + ".log";
    const auto log_level = config_services->get_string("log", "level", "debug");
    const auto logger = HLIELogger::init_managed_default_logger(log_location, log_file_name, "HLIE", log_level,
                                                                managed_log_config.max_file_size_bytes(),
                                                                managed_log_config.retention_days);
    HLIE_CHECK_NULL_RETURN(logger, -1);
    spdlog::set_default_logger(logger);

    LOG_INFO(HLIECpp);
    LOG_INFO("HLIEWinAppController initialized logger: instance_id={}, pid={}, log_file={}, boot_log_file={}",
             instance_id, GetCurrentProcessId(), log_file_name, "service_boot-" + instance_id + ".log");
    for (const auto& [key, val] : args) {
        LOG_INFO("program arg: {}={}", key, val);
    }

    const auto access_logger = HLIELogger::create_managed_logger(log_location, "access-" + instance_id + ".log",
                                                                 HLIE_ACCESS, "debug", managed_log_config.max_file_size_bytes(),
                                                                 managed_log_config.retention_days);
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

    workflow_ = std::make_shared<HLIEWorkflow>();
    if (auto res = workflow_->initialize(cur_model, program_path); res < 0) {
        LOG_ERROR("workflow initialize failed: instance_id={}, result={}", instance_id, res);
        return -1;
    } else {
        zero_load_ = 8 == res;
        LOG_INFO("workflow initialize succeeded: instance_id={}, result={}, zero_load={}", instance_id, res, zero_load_);
    }

    web_service_ = std::make_shared<HLIEWebService>(workflow_);

    const auto n_threads_http = workflow_->cacl_thread_num();
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

    const auto bootstrap_logger = HLIELogger::create_managed_logger("./logs", "service_boot-" + instance_id + ".log",
                                                                    HLIE_BOOT, "debug",
                                                                    static_cast<std::uintmax_t>(DEFAULT_LOG_FILE_SIZE_MB) * 1024U * 1024U,
                                                                    DEFAULT_LOG_RETENTION_DAYS);
    HLIE_CHECK_NULL_RETURN(bootstrap_logger, -6);

    if (const std::string config_path = "./config/sys.cfg"; config_services->parse(config_path) < 0) {
        if (config_services->get_string("model", "cur_model").empty()) {
            LOG_ERROR("sys_config parsed failed and no env[HLIE__MODEL__CUR_MODEL]");
            return -5;
        }

        LOG_WARN("sys_config parsed failed and will use env[HLIE__MODEL__CUR_MODEL] to continue");
    }

    const std::string log_location = config_services->get_string("log", "location", "/var/log/HLIECpp");
    const auto managed_log_config = load_managed_log_config(config_services);
    const auto boot_logger = HLIELogger::create_managed_logger(log_location, "service_boot-" + instance_id + ".log",
                                                               HLIE_BOOT, "debug", managed_log_config.max_file_size_bytes(),
                                                               managed_log_config.retention_days);
    HLIE_CHECK_NULL_RETURN(boot_logger, -6);

    const auto datetime = HlieUtil::get_datetime_str();
    HlieUtil::set_env("TCIM_LOG_LEVEL", std::to_string(config_services->get_int("log", "tcim_level", 2)));
    const auto tcim_log_file = HlieUtil::join_paths(HLIELogger::current_date_directory(log_location).string(),
                                                    "tcim-" + datetime + "-" + instance_id + ".log");
    HlieUtil::set_env("TCIM_LOG_FILE", tcim_log_file);
    boot_logger->info("TCIM_LOG_FILE={}", tcim_log_file);
    if (!HLIELogger::register_tcim_log_handler(log_location, "tcim-" + instance_id + ".log",
                                               managed_log_config.max_file_size_bytes(),
                                               managed_log_config.retention_days)) {
        boot_logger->warn("TCIM managed log handler registration failed; use TCIM_LOG_FILE fallback");
    }

    const std::string log_file_name = "hlie-llm-" + instance_id + ".log";
    const auto log_level = config_services->get_string("log", "level", "debug");
    const auto logger = HLIELogger::init_managed_default_logger(log_location, log_file_name, "HLIE", log_level,
                                                                managed_log_config.max_file_size_bytes(),
                                                                managed_log_config.retention_days);
    HLIE_CHECK_NULL_RETURN(logger, -1);
    spdlog::set_default_logger(logger);

    LOG_INFO(HLIECpp);

    const auto access_logger = HLIELogger::create_managed_logger(log_location, "access-" + instance_id + ".log",
                                                                 HLIE_ACCESS, "debug", managed_log_config.max_file_size_bytes(),
                                                                 managed_log_config.retention_days);
    HLIE_CHECK_NULL_RETURN(access_logger, -6);

    spdlog::flush_every(std::chrono::seconds(10));
    const auto program_path = "./bin/HLIECpp";
    if (initialize_ex( program_path) < 0) {
        return -3;
    }

    const std::string cur_model = config_services->get_string("model", "cur_model", "", instance_id != "0");
    if (cur_model.empty()) {
        LOG_ERROR("cannot found model, cur_model is empty");
        return -2;
    }

    workflow_ = std::make_shared<HLIEWorkflow>();
    if (const auto res = workflow_->initialize(cur_model, program_path); res < 0) {
        return -1;
    } else {
        zero_load_ = 8 == res;
    }

    web_service_ = std::make_shared<HLIEWebService>(workflow_);

    const auto n_threads_http = workflow_->cacl_thread_num();
    web_service_->init(n_threads_http);

    return 0;
}

#endif

HLIE_NAMESPACE_END

#include "hlie_app/hlie_app_controller.h"

#include <filesystem>
#include <unordered_map>

#include "hlie_app/hlie_process_manager.h"
#include "hlie_app/hlie_web_service.h"
#include "hlie_app/hlie_workflow.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

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
        if (args.find("--device_id") != args.end()) {
            devices = args.at("--device_id");
            if (args.find("--instance_id") == args.end()) {
                LOG_ERROR("Instance Replica must specify all the --instance_id and --device_id params");
                return -1;
            }
        }

        // 多实例负载均衡场景判断与初始化
        const auto replicas = config_services->get_int("server", "replicas", 1);
        if (args.find("--instance_id") == args.end()) {
            if (replicas > 1) {
                const auto& device_helper = Singleton<HLIEDeviceHelper>::instance();
                // 主要用于多实例工作场景，单模型独占单IPU
                device_helper->init(devices);
                const int device_count = device_helper->get_device_count();
                if (replicas > device_count) {
                    LOG_ERROR("Copies[{}] is bigger than devices count[]", replicas, device_count);
                    return -1;
                }

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

                LOG_INFO("Main process start to pull new replicas[{}]", replicas - 1);

                int acquires_count = device_count / replicas; // 每个实例获取的NPU数量, 因为每个实例可以同时运行生文和生图
                LOG_INFO("Each process may have {} available devices", acquires_count);
                for (int i = 1; i < replicas; ++i) {
                    std::string devices_for_sub;
                    for (int j = 0; j < acquires_count; ++j) {
                        const std::optional<int> val =
                            device_helper->acquire_device("instance-" + std::to_string(i) + "-" + std::to_string(j));
                        if (!val) {
                            LOG_ERROR("instance_id-[{}] cannot acquire available device", i);
                            return -1;
                        }

                        if (!devices_for_sub.empty()) {
                            devices_for_sub += ",";
                        }

                        devices_for_sub += std::to_string(*val);
                    }

                    LOG_INFO("acquired device {} for instance-{}", devices_for_sub, i);

                    // 子进程实例参数
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
            Singleton<HLIEDeviceHelper>::instance()->init(devices);
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

            if (replicas > 1 && config_services->get_bool("server", "memory_saving", false)) {
                const auto delay_sec = config_services->get_int("server", "replica_delay_sec", 45);
                LOG_INFO("multi instance mode and memory_saving is true, start to wait the others for {}s.", instance_id * delay_sec);
                std::this_thread::sleep_for(std::chrono::seconds(instance_id * delay_sec));
            }
        }

        const auto tcim_log_level = config_services->get_int("log", "tcim_level", 2);
        HlieUtil::set_env("TCIM_LOG_LEVEL", std::to_string(tcim_log_level));
        HlieUtil::set_env("TCIM_BACKEND", "Xh2HalBackend");

        return 0;
    }

} // namespace

int HLIEAppController::run() {
    HLIE_CHECK_NULL_RETURN(workflow_, -1);
    if (!zero_load_) {
        workflow_->run();
    } else {
        LOG_WARN("Service starting in Zero-Load Model.");
    }

    HLIE_CHECK_NULL_RETURN(web_service_, -1);
    web_service_->register_routers();
    if (web_service_->run() < 0) {
        LOG_ERROR("web service failed to run");
        return -1;
    }

    is_running_.store(true);

    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return !is_running_.load(); });
    return 0;
}

void HLIEAppController::stop() {
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

    const auto boot_log_file = program_dir + "\\boot_log\\service_boot-" + instance_id + ".log";
    const auto boot_logger = HLIELogger::create_logger(boot_log_file, HLIE_BOOT, "debug");
    HLIE_CHECK_NULL_RETURN(boot_logger, -6);

    const auto datetime = HlieUtil::get_datetime_str();
    HlieUtil::set_env("TCIM_LOG_FILE", program_dir + "\\boot_log\\tcim-" + datetime + "-" + instance_id + ".log");

    if (const auto config_path = program_dir + "..\\config\\sys.cfg"; config_services->parse(config_path) < 0) {
        boot_logger->error("sys_config parsed failed");
        return -5;
    }

    const std::string log_location = config_services->get_string("log", "location");
    if (log_location.empty()) {
        LOG_ERROR("Cannot found log location config");
        return -1;
    }

    try {
        if (std::filesystem::create_directories(log_location)) {
            boot_logger->info("Log Dir created successfully.");
        } else {
            boot_logger->info("Log Dir already exist.");
        }
    } catch (const std::filesystem::filesystem_error& e) {
        boot_logger->error("Log Dir created failed, exception: {}", e.what());
        return -1;
    }

    const std::string log_file_name = "houmo-llm-" + instance_id + ".log";

    const std::string log_file = HlieUtil::join_paths(log_location, log_file_name);
    const auto log_level = config_services->get_string("log", "level", "debug");
    const auto max_files = config_services->get_int("log", "max_files", 7);
    const auto logger = HLIELogger::init_default_logger(log_file, "HLIE", log_level, max_files);
    HLIE_CHECK_NULL_RETURN(logger, -1);
    spdlog::set_default_logger(logger);

    LOG_INFO(HLIECpp);

    const auto access_log = HlieUtil::join_paths(log_location, "access-" + instance_id + ".log");
    const auto access_logger = HLIELogger::create_logger(access_log, HLIE_ACCESS, "debug");
    HLIE_CHECK_NULL_RETURN(access_logger, -6);

    spdlog::flush_every(std::chrono::seconds(10));

    // 为啥方便手动扩展，支持配置环境变量HLIE__MODEL__CUR_MODEL用于指定独立的模型目录, 这种场景必须指定--device_id
    const std::string cur_model = config_services->get_string("model", "cur_model", "", instance_id != "0");
    if (cur_model.empty()) {
        boot_logger->error("WINSERVICE cannot found model, cur_model is empty");
        return -2;
    }

    if (initialize_ex(program_path) < 0) {
        return -3;
    }

    workflow_ = std::make_shared<HLIEWorkflow>();
    if (auto res = workflow_->initialize(cur_model, program_path); res < 0) {
        return -1;
    } else {
        zero_load_ = 8 == res;
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

    const auto boot_log_file = "./logs/service_boot-" + instance_id + ".log";
    const auto boot_logger = HLIELogger::create_logger(boot_log_file, HLIE_BOOT, "debug");
    HLIE_CHECK_NULL_RETURN(boot_logger, -6);

    if (const std::string config_path = "./config/sys.cfg"; config_services->parse(config_path) < 0) {
        if (config_services->get_string("model", "cur_model").empty()) {
            LOG_ERROR("sys_config parsed failed and no env[HLIE__MODEL__CUR_MODEL]");
            return -5;
        }

        LOG_WARN("sys_config parsed failed and will use env[HLIE__MODEL__CUR_MODEL] to continue");
    }

    const auto datetime = HlieUtil::get_datetime_str();
    HlieUtil::set_env("TCIM_LOG_FILE", "/var/log/houmo/HLIECpp/tcim-" + datetime + "-" + instance_id + ".log");

    const std::string log_location = config_services->get_string("log", "location", "/var/log/houmo/HLIECpp");
    const std::string log_file_name = HlieUtil::join_paths(log_location, "houmo-llm-" + instance_id + ".log");
    const auto log_level = config_services->get_string("log", "level", "debug");
    const auto max_files = config_services->get_int("log", "max_files", 7);
    const auto logger = HLIELogger::init_default_logger(log_file_name, "HLIE", log_level, max_files);
    HLIE_CHECK_NULL_RETURN(logger, -1);
    spdlog::set_default_logger(logger);

    LOG_INFO(HLIECpp);

    const std::string access_log = HlieUtil::join_paths(log_location, "access-" + instance_id + ".log");
    const auto access_logger = HLIELogger::create_logger(access_log, HLIE_ACCESS, "debug");
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

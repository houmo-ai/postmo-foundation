#include "hlie_app/hlie_app_controller.h"

#include <filesystem>
#include <unordered_map>
#include "hlie_app/hlie_web_service.h"
#include "hlie_app/hlie_workflow.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

HLIE_NAMESPACE_BEGIN

int HLIEAppController::run() {
    HLIE_CHECK_NULL_RETURN(workflow_, -1);
    workflow_->run();
    web_service_->register_routers();
    if (web_service_->run("0.0.0.0", 8808) < 0) {
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
    is_running_.store(false);
    cv_.notify_one();
}


#ifdef _WIN32

int HLIEWinAppController::initialize() {
    char exePath[260];
    GetModuleFileName(nullptr, exePath, MAX_PATH); // 获取exe绝对路径
    const auto programPath = std::string(exePath);
    const auto program_dir = std::string(exePath).substr(0, std::string(exePath).find_last_of('\\') + 1);
    const auto program_name = std::string(exePath).substr(std::string(exePath).find_last_of('\\') + 1);

    const auto boot_log_file = program_dir + "\\boot_log\\service_boot.log";
    const auto boot_logger = HLIELogger::create_logger(boot_log_file, "HLIE-BOOT", "debug");
    HLIE_CHECK_NULL_RETURN(boot_logger, -6);

    const auto config_path = program_dir + "..\\config\\sys.cfg";

    const auto& config_services = Singleton<HLIEConfigService>::instance();
    if (config_services->parse(config_path) < 0) {
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

    const std::string log_file_name = "houmo-llm.log";

    const std::string log_file = HlieUtil::join_paths(log_location, log_file_name);
    const auto log_level = config_services->get_string("log", "level", "debug");
    const auto max_files = config_services->get_int("log", "max_files", 7);
    const auto logger = HLIELogger::init_default_logger(log_file, "HLIE", log_level, max_files);
    HLIE_CHECK_NULL_RETURN(logger, -1);
    spdlog::set_default_logger(logger);

    const auto tcim_log_level =  config_services->get_int("log", "tcim_level", 2);
    HlieUtil::set_env("TCIM_LOG_LEVEL", std::to_string(tcim_log_level));

    const auto access_log = HlieUtil::join_paths(log_location, "access.log");
    const auto access_logger = HLIELogger::create_logger(access_log, "HLIE-ACCESS", "debug");
    HLIE_CHECK_NULL_RETURN(access_logger, -6);

    const auto devices = config_services->get_string("server", "devices", "0", true);
    Singleton<HLIEDeviceHelper>::instance()->init(devices);

    const std::string cur_model = config_services->get_string("model", "cur_model");
    if (cur_model.empty()) {
        boot_logger->error("WINSERVICE cannot found model, cur_model is empty");
        return -2;
    }

    workflow_ = std::make_shared<HLIEWorkflow>();
    if (workflow_->initialize(cur_model) < 0) {
        return -1;
    }

    web_service_ = std::make_shared<HLIEWebService>(workflow_);

    const auto n_threads_http = workflow_->cacl_thread_num();
    web_service_->init(n_threads_http);

    return 0;
}

#else

int HLIELinuxAppController::initialize() {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    if (const std::string config_path = "/opt/hicpp/config/sys.cfg"; config_services->parse(config_path) < 0) {
        LOG_ERROR("sys_config parsed failed");
        return -5;
    }

    const int log_level = config_services->get_int("log", "level", 0);

    const auto log_location = "/opt/hicpp/logs/houmo-llm.log";
    const auto logger = HLIELogger::init_default_logger(log_location, "HLIE", static_cast<spdlog::level::level_enum>(log_level));
    HLIE_CHECK_NULL_RETURN(logger, -1);
    set_default_logger(logger);

    const std::string cur_model = config_services->get_string("model", "cur_model");
    if (cur_model.empty()) {
        LOG_ERROR("cannot found model, cur_model is empty");
        return -2;
    }

    workflow_ = std::make_shared<HLIEWorkflow>();
    if (workflow_->initialize(cur_model) < 0) {
        return -1;
    }

    web_service_ = std::make_shared<HLIEWebService>(workflow_);

    const auto n_threads_http = workflow_->cacl_thread_num();
    web_service_->init(n_threads_http);

    return 0;
}

#endif

HLIE_NAMESPACE_END

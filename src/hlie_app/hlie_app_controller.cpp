#include "hlie_app/hlie_app_controller.h"

#include <filesystem>

#include "hlie_app/hlie_web_service.h"
#include "hlie_app/hlie_workflow.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

/**
 * 初始化Workflow等相关
 *
 * @return 0: Normal
 */
int HLIEWinAppController::initialize() {
#ifndef _WIN32
    return 0;
#endif

    char exePath[260];
    GetModuleFileName(nullptr, exePath, MAX_PATH); // 获取exe绝对路径
    const auto programPath = std::string(exePath);
    const auto program_dir = std::string(exePath).substr(0, std::string(exePath).find_last_of('\\') + 1);
    const auto program_name = std::string(exePath).substr(std::string(exePath).find_last_of('\\') + 1);

    const auto boot_log_file = program_dir + "\\boot_log\\service_boot.log";
    const auto boot_logger = HLIELogger::create_logger(boot_log_file, "HLIE-BOOT", spdlog::level::debug);
    HLIE_CHECK_NULL_RETURN(boot_logger, -6);

    const auto configPath = program_dir + "..\\config\\sys.cfg";
    sys_config_ = std::make_shared<HLIEConfigService>(configPath);
    if (sys_config_->parse() < 0) {
        boot_logger->error("sys_config parsed failed");
        return -5;
    }

    const std::string log_location = sys_config_->get_string("log", "location");
    if (log_location.empty()) {
        LOG_ERROR("Cannot found log location config");
        return -1;
    }

    const int log_level = sys_config_->get_int("log", "level", 0);

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

    const std::string log_file = log_location + log_file_name;
    const auto logger = HLIELogger::init_default_logger(log_file, "HLIE", static_cast<spdlog::level::level_enum>(log_level));
    HLIE_CHECK_NULL_RETURN(logger, -1);
    spdlog::set_default_logger(logger);

    const std::string cur_model = sys_config_->get_string("model", "cur_model");
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

int HLIEWinAppController::run() {
#ifndef  _WIN32
    return;
#endif

    HLIE_CHECK_NULL_RETURN(workflow_, -1);
    workflow_->run();
    web_service_->register_routers();
    HLIE_CHECK_NULL_RETURN(sys_config_, -1);
    if (const auto port = sys_config_->get_int("server", "port", 8808); web_service_->run("0.0.0.0", port) < 0) {
        LOG_ERROR("web service failed to run");
        return -1;
    }

    is_running_.store(true);

    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return !is_running_.load(); });
    return 0;
}

HLIE_NAMESPACE_END

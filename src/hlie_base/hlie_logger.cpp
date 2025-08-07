#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_base.h"

#include <iostream>

#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

HLIE_NAMESPACE_BEGIN

std::shared_ptr<spdlog::logger> HLIELogger::init_default_logger(const std::string& log_file_name, const std::string& module_name,
                                                                const spdlog::level::level_enum level) {
    std::cout << "Init Default logger: " << module_name << ", log_file: " << log_file_name << std::endl;
    const auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    const auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file_name, 0, 0, false);
    std::vector<spdlog::sink_ptr> sinks {file_sink, console_sink};
    auto logger = std::make_shared<spdlog::logger>(module_name, sinks.begin(), sinks.end());
    logger->set_level(level);
    spdlog::flush_every(std::chrono::seconds(10));
    return std::move(logger);
}

std::shared_ptr<spdlog::logger> HLIELogger::create_logger(const std::string& log_file_name, const std::string& module_name,
                                                          const spdlog::level::level_enum level) {
    std::cout << "Creating logger: " << module_name << ", log_file: " << log_file_name << std::endl;
    auto logger = spdlog::daily_logger_mt(module_name, log_file_name, 0, 0, false, 7);
    logger->set_level(level);
    return std::move(logger);
}

std::shared_ptr<spdlog::logger> HLIELogger::get_logger(const std::string& module_name) {
    return spdlog::get(module_name);
}

HLIE_NAMESPACE_END

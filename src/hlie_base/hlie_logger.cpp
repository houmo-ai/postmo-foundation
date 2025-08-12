#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_base.h"

#include <iostream>

#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

HLIE_NAMESPACE_BEGIN

namespace {
    using level_enum = spdlog::level::level_enum;
    level_enum from_string(const std::string& level_str) {
        static const std::unordered_map<std::string, level_enum> level_map = {
            {"trace", level_enum::trace},
            {"debug", level_enum::debug},
            {"info", level_enum::info},
            {"warn", level_enum::warn},
            {"warning", level_enum::warn},
            {"error", level_enum::err},
            {"critical", level_enum::critical},
            {"off", level_enum::off}
        };

        if (const auto it = level_map.find(level_str); it != level_map.end()) {
            return it->second;
        }

        return level_enum::warn;
    }
}

std::shared_ptr<spdlog::logger> HLIELogger::init_default_logger(const std::string& log_file_name, const std::string& module_name,
                                                                const std::string& level, const int max_files) {
    std::cout << "Init Default logger: " << module_name << ", log_file: " << log_file_name << std::endl;
    const auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    const auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file_name, 0, 0, false, max_files);
    std::vector<spdlog::sink_ptr> sinks{file_sink, console_sink};
    auto logger = std::make_shared<spdlog::logger>(module_name, sinks.begin(), sinks.end());
    logger->set_level(from_string(level));
    spdlog::flush_every(std::chrono::seconds(10));
    return std::move(logger);
}

std::shared_ptr<spdlog::logger> HLIELogger::create_logger(const std::string& log_file_name, const std::string& module_name,
                                                          const std::string& level, const int max_files) {
    std::cout << "Creating logger: " << module_name << ", log_file: " << log_file_name << std::endl;
    auto logger = spdlog::daily_logger_mt(module_name, log_file_name, 0, 0, false, max_files);
    logger->set_level(from_string(level));
    return std::move(logger);
}

std::shared_ptr<spdlog::logger> HLIELogger::get_logger(const std::string& module_name) {
    return spdlog::get(module_name);
}

HLIE_NAMESPACE_END

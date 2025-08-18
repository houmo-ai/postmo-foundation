#ifndef HLIE_LOGGER_H
#define HLIE_LOGGER_H

#include <spdlog/spdlog.h>

#include "hlie_base.h"

#define LOG_TRACE(...) SPDLOG_LOGGER_TRACE(spdlog::default_logger_raw(), __VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(spdlog::default_logger_raw(), __VA_ARGS__)
#define LOG_INFO(...) SPDLOG_LOGGER_INFO(spdlog::default_logger_raw(), __VA_ARGS__)
#define LOG_WARN(...) SPDLOG_LOGGER_WARN(spdlog::default_logger_raw(), __VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_LOGGER_ERROR(spdlog::default_logger_raw(), __VA_ARGS__)
#define LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(spdlog::default_logger_raw(), __VA_ARGS__)

constexpr auto HLIE_BOOT = "HLIE-BOOT";
constexpr auto HLIE_ACCESS = "HLIE-ACCESS";

#define HLIE_CHECK_NULL(ptr)                                                                                                               \
    do {                                                                                                                                   \
        if ((ptr) == nullptr) {                                                                                                            \
            LOG_ERROR("NULL pointer: {} at {}:{} in function {}", #ptr, __FILE__, __LINE__, __FUNCTION__);                                 \
            return;                                                                                                                        \
        }                                                                                                                                  \
    } while (0)

#define HLIE_CHECK_NULL_RETURN(ptr, ret)                                                                                                   \
    do {                                                                                                                                   \
        if ((ptr) == nullptr) {                                                                                                            \
            LOG_ERROR("NULL pointer: {} at {}:{} in function {}", #ptr, __FILE__, __LINE__, __FUNCTION__);                                 \
            return ret;                                                                                                                    \
        }                                                                                                                                  \
    } while (0)

#define HLIE_CHECK_NULL_MSG(ptr, msg)                                                                                                      \
    do {                                                                                                                                   \
        if ((ptr) == nullptr) {                                                                                                            \
            LOG_ERROR("NULL pointer: {} at {}:{} in function {} - {}", #ptr, __FILE__, __LINE__, __FUNCTION__, msg);                       \
            return;                                                                                                                        \
        }                                                                                                                                  \
    } while (0)

#define HLIE_CHECK_NULL_MSG_RETURN(ptr, ret, msg)                                                                                          \
    do {                                                                                                                                   \
        if ((ptr) == nullptr) {                                                                                                            \
            LOG_ERROR("NULL pointer: {} at {}:{} in function {} - {}", #ptr, __FILE__, __LINE__, __FUNCTION__, msg);                       \
            return ret;                                                                                                                    \
        }                                                                                                                                  \
    } while (0)

HLIE_NAMESPACE_BEGIN

class HLIE_API HLIELogger {
public:
    /**
     * Create the default logger with console and daily sink
     *
     * @param log_file_name the log-file-path
     * @param module_name  the module name
     * @param level the level: most used is info, warn, debug
     * @param max_files the max-count of log files
     * @return the logger ptr
     */
    static std::shared_ptr<spdlog::logger> init_default_logger(const std::string& log_file_name, const std::string& module_name,
                                                               const std::string& level, int max_files = 7);

    /**
     * Just create a logger for filed daily sink
     *
     * @param log_file_name the log-file-path
     * @param module_name  the module name
     * @param level the level: most used is info, warn, debug
     * @param max_files the max-count of log files
     * @return the logger ptr
     */
    static std::shared_ptr<spdlog::logger> create_logger(const std::string& log_file_name, const std::string& module_name,
                                                         const std::string& level, int max_files = 7);

    /**
     * Get the logger with module-name which you created with
     *
     * @param module_name the module-name which you created with
     * @return
     */
    static std::shared_ptr<spdlog::logger> get_logger(const std::string& module_name = "");

private:
    inline static std::shared_ptr<spdlog::logger> default_logger_  = nullptr;;
};

HLIE_NAMESPACE_END

#endif // HLIE_LOGGER_H

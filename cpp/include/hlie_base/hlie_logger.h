#ifndef HLIE_LOGGER_H
#define HLIE_LOGGER_H

#include <cstdint>
#include <filesystem>

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

/**
 * Managed log rotation policy. retention_days > 0 uses YYYY-MM-DD directories;
 * retention_days == 0 uses flat, count-based rotation with dates in filenames.
 * max_files == 0 disables the per-logger part-count limit.
 */
struct HLIE_API HLIEManagedLogConfig {
    std::uintmax_t max_file_size_bytes = 10U * 1024U * 1024U;
    int retention_days = 7;
    int max_files = 0;
    bool compress_rotated_files = false;
    int compression_level = 1;
};

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
     * Create a logger whose files are stored under YYYY-MM-DD directories and
     * split into numbered parts when max_file_size_bytes is reached.
     */
    static std::shared_ptr<spdlog::logger> init_managed_default_logger(const std::string& log_root,
                                                                       const std::string& log_file_name,
                                                                       const std::string& module_name,
                                                                       const std::string& level,
                                                                       std::uintmax_t max_file_size_bytes,
                                                                       int retention_days = 7);

    static std::shared_ptr<spdlog::logger> init_managed_default_logger(const std::string& log_root,
                                                                       const std::string& log_file_name,
                                                                       const std::string& module_name,
                                                                       const std::string& level,
                                                                       const HLIEManagedLogConfig& config);

    static std::shared_ptr<spdlog::logger> create_managed_logger(const std::string& log_root,
                                                                 const std::string& log_file_name,
                                                                 const std::string& module_name,
                                                                 const std::string& level,
                                                                 std::uintmax_t max_file_size_bytes,
                                                                 int retention_days = 7);

    static std::shared_ptr<spdlog::logger> create_managed_logger(const std::string& log_root,
                                                                 const std::string& log_file_name,
                                                                 const std::string& module_name,
                                                                 const std::string& level,
                                                                 const HLIEManagedLogConfig& config);

    /** Create a console-only logger for messages emitted before sys.cfg is available. */
    static std::shared_ptr<spdlog::logger> create_console_logger(const std::string& module_name,
                                                                 const std::string& level);

    /** Route TCIM logs through the managed date/size sink. */
    static bool register_tcim_log_handler(const std::string& log_root,
                                          const std::string& log_file_name,
                                          std::uintmax_t max_file_size_bytes,
                                          int retention_days = 7);

    static bool register_tcim_log_handler(const std::string& log_root,
                                          const std::string& log_file_name,
                                          const HLIEManagedLogConfig& config);

    /** Return log_root/YYYY-MM-DD for the current local date and create it if needed. */
    static std::filesystem::path current_date_directory(const std::string& log_root);

    /** Delete expired managed YYYY-MM-DD directories. Unmarked directories are never removed. */
    static void cleanup_old_log_directories(const std::string& log_root, int retention_days);

    /** Wait until already queued background log compression work has completed (primarily for tests/tools). */
    static void wait_for_compression();

    /** Parse an integer K/M/G size (binary units) with overflow checking. */
    static bool parse_file_size(const std::string& value, std::uintmax_t& bytes);

    /**
     * Get the logger with module-name which you created with
     *
     * @param module_name the module-name which you created with
     * @return
     */
    static std::shared_ptr<spdlog::logger> get_logger(const std::string& module_name = "");

private:
    inline static std::shared_ptr<spdlog::logger> default_logger_  = nullptr;
};

HLIE_NAMESPACE_END

#endif // HLIE_LOGGER_H

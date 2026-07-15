#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_base.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string_view>
#include <system_error>

#include <tcim/tcim_runtime.h>

#include "spdlog/details/file_helper.h"
#include "spdlog/details/os.h"
#include "spdlog/sinks/base_sink.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

HLIE_NAMESPACE_BEGIN

namespace {
    namespace fs = std::filesystem;
    using level_enum = spdlog::level::level_enum;
    constexpr auto MANAGED_DIRECTORY_MARKER = ".hlie-log-directory";

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

    std::string date_string(const spdlog::log_clock::time_point time_point) {
        const auto time = spdlog::log_clock::to_time_t(time_point);
        const auto local = spdlog::details::os::localtime(time);
        return fmt::format("{:04d}-{:02d}-{:02d}", local.tm_year + 1900, local.tm_mon + 1, local.tm_mday);
    }

    bool is_known_managed_log_entry(const fs::directory_entry& entry) {
        std::error_code error;
        if (!entry.is_regular_file(error) || error) {
            return false;
        }
        const auto name = entry.path().filename().string();
        if (name == MANAGED_DIRECTORY_MARKER) {
            return true;
        }
        if (name.size() < 4 || name.compare(name.size() - 4, 4, ".log") != 0) {
            return false;
        }
        return name.rfind("houmo-llm-", 0) == 0 || name.rfind("access-", 0) == 0 ||
               name.rfind("service_boot-", 0) == 0 || name.rfind("tcim-", 0) == 0;
    }

    bool can_adopt_existing_date_directory(const fs::path& directory) {
        std::error_code error;
        for (fs::directory_iterator it(directory, error), end; !error && it != end; it.increment(error)) {
            if (!is_known_managed_log_entry(*it)) {
                return false;
            }
        }
        return !error;
    }

    fs::path ensure_managed_date_directory(const fs::path& log_root, const std::string& date) {
        const auto directory = log_root / date;
        std::error_code error;
        const bool directory_existed = fs::exists(directory, error);
        if (error) {
            throw spdlog::spdlog_ex("failed checking log directory " + directory.string() + ": " + error.message());
        }
        fs::create_directories(directory, error);
        if (error) {
            throw spdlog::spdlog_ex("failed creating log directory " + directory.string() + ": " + error.message());
        }

        const auto marker = directory / MANAGED_DIRECTORY_MARKER;
        if (directory_existed && !fs::exists(marker, error)) {
            if (error) {
                throw spdlog::spdlog_ex("failed checking managed log directory marker " + marker.string() + ": " + error.message());
            }
            if (!can_adopt_existing_date_directory(directory)) {
                throw spdlog::spdlog_ex("refusing to manage non-log date directory " + directory.string());
            }
        }
        std::ofstream marker_stream(marker, std::ios::app);
        if (!marker_stream) {
            throw spdlog::spdlog_ex("failed creating managed log directory marker " + marker.string());
        }
        return directory;
    }

    bool is_valid_date_directory_name(const std::string& name) {
        if (name.size() != 10 || name[4] != '-' || name[7] != '-') {
            return false;
        }
        for (size_t i = 0; i < name.size(); ++i) {
            if (i != 4 && i != 7 && !std::isdigit(static_cast<unsigned char>(name[i]))) {
                return false;
            }
        }

        std::tm value{};
        value.tm_year = std::stoi(name.substr(0, 4)) - 1900;
        value.tm_mon = std::stoi(name.substr(5, 2)) - 1;
        value.tm_mday = std::stoi(name.substr(8, 2));
        value.tm_hour = 12;
        value.tm_isdst = -1;
        const auto normalized_time = std::mktime(&value);
        return normalized_time != static_cast<std::time_t>(-1) &&
               date_string(spdlog::log_clock::from_time_t(normalized_time)) == name;
    }

    std::string retention_cutoff_date(const int retention_days) {
        auto now = spdlog::log_clock::to_time_t(spdlog::log_clock::now());
        auto local = spdlog::details::os::localtime(now);
        local.tm_hour = 12;
        local.tm_min = 0;
        local.tm_sec = 0;
        local.tm_mday -= std::max(1, retention_days) - 1;
        local.tm_isdst = -1;
        return date_string(spdlog::log_clock::from_time_t(std::mktime(&local)));
    }

    template <typename Mutex>
    class date_size_file_sink final : public spdlog::sinks::base_sink<Mutex> {
    public:
        date_size_file_sink(fs::path log_root, fs::path log_file_name,
                            const std::uintmax_t max_file_size_bytes, const int retention_days)
            : log_root_(std::move(log_root)),
              log_file_name_(log_file_name.filename()),
              max_file_size_bytes_(max_file_size_bytes),
              retention_days_(std::max(1, retention_days)) {
            if (log_root_.empty() || log_file_name_.empty()) {
                throw spdlog::spdlog_ex("managed log root and file name must not be empty");
            }
            open_for_date_(date_string(spdlog::log_clock::now()));
        }

    protected:
        void sink_it_(const spdlog::details::log_msg& msg) override {
            const auto message_date = date_string(msg.time);
            if (message_date != current_date_) {
                open_for_date_(message_date);
            }

            spdlog::memory_buf_t formatted;
            spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
            if (max_file_size_bytes_ > 0 && current_size_ > 0 &&
                formatted.size() > max_file_size_bytes_ - std::min(current_size_, max_file_size_bytes_)) {
                ++part_number_;
                open_part_();
            }
            file_helper_.write(formatted);
            current_size_ += formatted.size();
        }

        void flush_() override {
            file_helper_.flush();
        }

    private:
        fs::path part_path_(const std::uint64_t part_number) const {
            const auto stem = log_file_name_.stem().string();
            const auto extension = log_file_name_.extension().string();
            return current_directory_ / fmt::format("{}.{:04d}{}", stem, part_number, extension);
        }

        std::uint64_t find_last_part_() const {
            const auto prefix = log_file_name_.stem().string() + ".";
            const auto extension = log_file_name_.extension().string();
            std::uint64_t last_part = 0;
            std::error_code error;
            for (fs::directory_iterator it(current_directory_, error), end; !error && it != end; it.increment(error)) {
                if (!it->is_regular_file(error)) {
                    continue;
                }
                const auto name = it->path().filename().string();
                if (name.size() <= prefix.size() + extension.size() || name.compare(0, prefix.size(), prefix) != 0 ||
                    name.compare(name.size() - extension.size(), extension.size(), extension) != 0) {
                    continue;
                }
                const auto number = name.substr(prefix.size(), name.size() - prefix.size() - extension.size());
                if (number.empty() || !std::all_of(number.begin(), number.end(), [](const unsigned char c) { return std::isdigit(c); })) {
                    continue;
                }
                try {
                    last_part = std::max(last_part, static_cast<std::uint64_t>(std::stoull(number)));
                } catch (const std::exception&) {
                    // Ignore files whose numeric suffix is outside uint64_t.
                }
            }
            return last_part;
        }

        void open_for_date_(const std::string& date) {
            current_date_ = date;
            current_directory_ = ensure_managed_date_directory(log_root_, current_date_);
            HLIELogger::cleanup_old_log_directories(log_root_.string(), retention_days_);
            part_number_ = std::max<std::uint64_t>(1, find_last_part_());
            open_part_();
            if (max_file_size_bytes_ > 0 && current_size_ >= max_file_size_bytes_) {
                ++part_number_;
                open_part_();
            }
        }

        void open_part_() {
            const auto path = part_path_(part_number_);
            file_helper_.open(path.string(), false);
            std::error_code error;
            current_size_ = fs::file_size(path, error);
            if (error) {
                current_size_ = 0;
            }
        }

        fs::path log_root_;
        fs::path log_file_name_;
        fs::path current_directory_;
        std::uintmax_t max_file_size_bytes_;
        int retention_days_;
        std::string current_date_;
        std::uint64_t part_number_ = 1;
        std::uintmax_t current_size_ = 0;
        spdlog::details::file_helper file_helper_;
    };

    using date_size_file_sink_mt = date_size_file_sink<std::mutex>;

    class TcimManagedLogHandler final : public tcim::LogHandler {
    public:
        TcimManagedLogHandler(const std::string& log_root, const std::string& log_file_name,
                              const std::uintmax_t max_file_size_bytes, const int retention_days) {
            const auto file_sink = std::make_shared<date_size_file_sink_mt>(log_root, log_file_name,
                                                                            max_file_size_bytes, retention_days);
            logger_ = std::make_shared<spdlog::logger>("HLIE-TCIM", file_sink);
            logger_->set_level(spdlog::level::trace);
            logger_->set_pattern("%v");
        }

        void OnLog(const std::string_view msg, const int level) override {
            const auto bounded_level = std::clamp(level, static_cast<int>(spdlog::level::trace),
                                                   static_cast<int>(spdlog::level::critical));
            logger_->log(static_cast<spdlog::level::level_enum>(bounded_level), "{}", msg);
        }

        void Flush() override {
            logger_->flush();
        }

    private:
        std::shared_ptr<spdlog::logger> logger_;
    };
}

std::shared_ptr<spdlog::logger> HLIELogger::init_default_logger(const std::string& log_file_name, const std::string& module_name,
                                                                const std::string& level, const int max_files) {
    std::cout << "Init Default logger: " << module_name << ", log_file: " << log_file_name << std::endl;
    const auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    const auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file_name, 0, 0, false, max_files);
    std::vector<spdlog::sink_ptr> sinks{file_sink, console_sink};
    default_logger_ = std::make_shared<spdlog::logger>(module_name, sinks.begin(), sinks.end());
    default_logger_->set_level(from_string(level));
    spdlog::set_default_logger(default_logger_);
    return default_logger_;
}

std::shared_ptr<spdlog::logger> HLIELogger::create_logger(const std::string& log_file_name, const std::string& module_name,
                                                          const std::string& level, const int max_files) {
    std::cout << "Creating logger: " << module_name << ", log_file: " << log_file_name << std::endl;
    auto logger = spdlog::daily_logger_mt(module_name, log_file_name, 0, 0, false, max_files);
    logger->set_level(from_string(level));
    return std::move(logger);
}

std::shared_ptr<spdlog::logger> HLIELogger::init_managed_default_logger(const std::string& log_root,
                                                                        const std::string& log_file_name,
                                                                        const std::string& module_name,
                                                                        const std::string& level,
                                                                        const std::uintmax_t max_file_size_bytes,
                                                                        const int retention_days) {
    std::cout << "Init managed default logger: " << module_name << ", log_root: " << log_root << std::endl;
    const auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    const auto file_sink = std::make_shared<date_size_file_sink_mt>(log_root, log_file_name, max_file_size_bytes, retention_days);
    std::vector<spdlog::sink_ptr> sinks{file_sink, console_sink};
    default_logger_ = std::make_shared<spdlog::logger>(module_name, sinks.begin(), sinks.end());
    default_logger_->set_level(from_string(level));
    spdlog::set_default_logger(default_logger_);
    return default_logger_;
}

std::shared_ptr<spdlog::logger> HLIELogger::create_managed_logger(const std::string& log_root,
                                                                  const std::string& log_file_name,
                                                                  const std::string& module_name,
                                                                  const std::string& level,
                                                                  const std::uintmax_t max_file_size_bytes,
                                                                  const int retention_days) {
    std::cout << "Creating managed logger: " << module_name << ", log_root: " << log_root << std::endl;
    const auto file_sink = std::make_shared<date_size_file_sink_mt>(log_root, log_file_name, max_file_size_bytes, retention_days);
    auto logger = std::make_shared<spdlog::logger>(module_name, file_sink);
    logger->set_level(from_string(level));
    // Bootstrap and managed loggers may intentionally use the same module name.
    // Replace inside hlie-base so Windows DLL-local spdlog registries are handled correctly.
    spdlog::register_or_replace(logger);
    return logger;
}

std::shared_ptr<spdlog::logger> HLIELogger::create_console_logger(const std::string& module_name, const std::string& level) {
    const auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>(module_name, console_sink);
    logger->set_level(from_string(level));
    spdlog::register_or_replace(logger);
    return logger;
}

bool HLIELogger::register_tcim_log_handler(const std::string& log_root, const std::string& log_file_name,
                                           const std::uintmax_t max_file_size_bytes, const int retention_days) {
    try {
        auto handler = std::make_unique<TcimManagedLogHandler>(log_root, log_file_name, max_file_size_bytes, retention_days);
        const auto status = tcim::RegisterLogHandler(std::move(handler));
        if (status != tcim::Status::OK) {
            std::cerr << "Failed registering TCIM managed log handler, status=" << static_cast<int>(status) << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        std::cerr << "Failed registering TCIM managed log handler: " << exception.what() << std::endl;
        return false;
    }
}

std::filesystem::path HLIELogger::current_date_directory(const std::string& log_root) {
    return ensure_managed_date_directory(log_root, date_string(spdlog::log_clock::now()));
}

void HLIELogger::cleanup_old_log_directories(const std::string& log_root, const int retention_days) {
    const auto cutoff = retention_cutoff_date(retention_days);
    std::error_code error;
    for (fs::directory_iterator it(log_root, error), end; !error && it != end; it.increment(error)) {
        std::error_code entry_error;
        if (it->is_directory(entry_error) && !entry_error) {
            const auto name = it->path().filename().string();
            if (!is_valid_date_directory_name(name) || name >= cutoff ||
                !fs::exists(it->path() / MANAGED_DIRECTORY_MARKER, entry_error) || entry_error) {
                continue;
            }
            fs::remove_all(it->path(), entry_error);
        } else {
            continue;
        }
        if (entry_error && entry_error != std::errc::no_such_file_or_directory) {
            std::cerr << "Failed removing expired log entry " << it->path() << ": " << entry_error.message() << std::endl;
        }
    }
}

std::shared_ptr<spdlog::logger> HLIELogger::get_logger(const std::string& module_name) {
    if (module_name.empty()) {
        if (default_logger_) {
            return default_logger_;
        }

        if (spdlog::get(HLIE_BOOT)) {
            std::cout << "No default logger found, using boot logger" << std::endl;
            return spdlog::get(HLIE_BOOT);
        }

        throw std::runtime_error("No default or boot logger found");
    }

    return spdlog::get(module_name);
}

HLIE_NAMESPACE_END

#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_base.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <cerrno>
#include <csignal>
#include <unistd.h>
#endif

#include <tcim/tcim_runtime.h>

#include "spdlog/details/file_helper.h"
#include "spdlog/details/os.h"
#include "spdlog/sinks/base_sink.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "hlie_base/hlie_gzip.h"

HLIE_NAMESPACE_BEGIN

namespace {
    namespace fs = std::filesystem;
    using level_enum = spdlog::level::level_enum;
    constexpr auto MANAGED_DIRECTORY_MARKER = ".hlie-log-directory";
    constexpr std::size_t MAX_PENDING_COMPRESSIONS_PER_LOGGER = 128;
    constexpr auto ABANDONED_COMPRESSION_AGE = std::chrono::hours(24);

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
        if (entry.is_symlink(error) || error || !entry.is_regular_file(error) || error) {
            return false;
        }
        const auto name = entry.path().filename().string();
        if (name == MANAGED_DIRECTORY_MARKER) {
            return true;
        }
        const bool is_log = name.size() >= 4 && name.compare(name.size() - 4, 4, ".log") == 0;
        const bool is_gzip_log = name.size() >= 7 && name.compare(name.size() - 7, 7, ".log.gz") == 0;
        if (!is_log && !is_gzip_log) {
            return false;
        }
        return name.rfind("hlie-llm-", 0) == 0 || name.rfind("access-", 0) == 0 ||
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

    std::optional<std::uint64_t> owner_process_id(const std::string& name, const std::string_view marker) {
        const auto marker_position = name.find(marker);
        if (marker_position == std::string::npos) {
            return std::nullopt;
        }
        const auto first = marker_position + marker.size();
        const auto last = name.find('.', first);
        if (first == last || last == std::string::npos) {
            return std::nullopt;
        }
        const auto value = name.substr(first, last - first);
        if (!std::all_of(value.begin(), value.end(), [](const unsigned char c) { return std::isdigit(c); })) {
            return std::nullopt;
        }
        try {
            return std::stoull(value);
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

    bool process_is_alive(const std::uint64_t process_id) {
        if (process_id == 0) {
            return false;
        }
#ifdef _WIN32
        if (process_id > std::numeric_limits<DWORD>::max()) {
            return false;
        }
        const auto handle = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                                        static_cast<DWORD>(process_id));
        if (handle == nullptr) {
            return GetLastError() == ERROR_ACCESS_DENIED;
        }
        DWORD exit_code = 0;
        const bool alive = GetExitCodeProcess(handle, &exit_code) != FALSE && exit_code == STILL_ACTIVE;
        CloseHandle(handle);
        return alive;
#else
        if (process_id > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
            return false;
        }
        if (::kill(static_cast<pid_t>(process_id), 0) == 0) {
            return true;
        }
        return errno == EPERM;
#endif
    }

    bool compression_artifact_is_abandoned(const fs::directory_entry& entry, const std::string& name,
                                            const std::string_view marker) {
        const auto owner = owner_process_id(name, marker);
        if (!owner) {
            return false;
        }
        if (!process_is_alive(*owner)) {
            return true;
        }
        std::error_code error;
        const auto modified = entry.last_write_time(error);
        return !error && fs::file_time_type::clock::now() - modified > ABANDONED_COMPRESSION_AGE;
    }

    class LogFileOperationCoordinator {
    public:
        static LogFileOperationCoordinator& instance() {
            static LogFileOperationCoordinator coordinator;
            return coordinator;
        }

        bool begin_compression(const fs::path& path) {
            std::lock_guard lock(mutex_);
            const auto key = operation_key_(path);
            if (deleting_.count(key) != 0) {
                return false;
            }
            cancelled_.erase(key);
            compressing_.insert(key);
            return true;
        }

        bool compression_cancelled(const fs::path& path) {
            std::lock_guard lock(mutex_);
            return cancelled_.count(operation_key_(path)) != 0;
        }

        bool compression_in_progress(const fs::path& path) {
            std::lock_guard lock(mutex_);
            return compressing_.count(operation_key_(path)) != 0;
        }

        void end_compression(const fs::path& path) {
            std::lock_guard lock(mutex_);
            const auto key = operation_key_(path);
            compressing_.erase(key);
            cancelled_.erase(key);
        }

        bool try_remove(const fs::path& path, std::error_code& error, bool& busy) {
            const auto key = operation_key_(path);
            {
                std::lock_guard lock(mutex_);
                if (deleting_.count(key) != 0) {
                    busy = true;
                    return false;
                }
                if (compressing_.count(key) != 0) {
                    busy = true;
                    return false;
                }
                deleting_.insert(key);
            }
            busy = false;
            const bool removed = fs::remove(path, error);
            {
                std::lock_guard lock(mutex_);
                deleting_.erase(key);
            }
            return removed;
        }

    private:
        static std::string operation_key_(fs::path path) {
            if (path.extension() == ".gz") {
                path.replace_extension();
            }
            return path.lexically_normal().string();
        }

        std::mutex mutex_;
        std::unordered_set<std::string> compressing_;
        std::unordered_set<std::string> cancelled_;
        std::unordered_set<std::string> deleting_;
    };

    struct CompressionWorkItem {
        fs::path path;
        int level = 1;
    };

    struct CompressionGroupState {
        std::mutex mutex;
        std::deque<CompressionWorkItem> pending;
        std::unordered_set<std::string> pending_paths;
        fs::path log_file_name;
        fs::path protected_path;
        int max_files = 0;
        bool count_mode = false;
        bool scheduled = false;
        bool overflow_reported = false;
    };

    void enforce_group_file_limit(const std::shared_ptr<CompressionGroupState>& group,
                                  const fs::path& directory);

    class LogCompressionWorker {
    public:
        static LogCompressionWorker& instance() {
            static LogCompressionWorker worker;
            return worker;
        }

        void enqueue(const std::shared_ptr<CompressionGroupState>& group, const fs::path& path, const int level) {
            const auto key = path.lexically_normal().string();
            bool schedule_group = false;
            {
                std::lock_guard group_lock(group->mutex);
                if (group->pending_paths.count(key) != 0) {
                    return;
                }
                if (group->pending.size() >= MAX_PENDING_COMPRESSIONS_PER_LOGGER) {
                    if (!group->overflow_reported) {
                        std::cerr << "Log compression backlog reached " << MAX_PENDING_COMPRESSIONS_PER_LOGGER
                                  << " files; retaining additional files uncompressed" << std::endl;
                        group->overflow_reported = true;
                    }
                    return;
                }
                group->pending_paths.insert(key);
                group->pending.push_back({path, std::clamp(level, 1, 9)});
                if (!group->scheduled) {
                    group->scheduled = true;
                    schedule_group = true;
                }
            }
            if (schedule_group) {
                std::lock_guard lock(mutex_);
                if (!stopping_) {
                    queue_.push_back(group);
                    condition_.notify_one();
                }
            }
        }

        void cancel(const std::shared_ptr<CompressionGroupState>& group, const fs::path& path) {
            const auto key = path.lexically_normal().string();
            std::lock_guard lock(group->mutex);
            group->pending_paths.erase(key);
            group->pending.erase(std::remove_if(group->pending.begin(), group->pending.end(), [&](const auto& item) {
                return item.path.lexically_normal().string() == key;
            }), group->pending.end());
        }

        void wait() {
            std::unique_lock lock(mutex_);
            drained_.wait(lock, [this] { return queue_.empty() && !active_; });
        }

        ~LogCompressionWorker() {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
                queue_.clear();
            }
            condition_.notify_one();
            if (thread_.joinable()) {
                thread_.join();
            }
        }

    private:
        LogCompressionWorker() {
            (void)LogFileOperationCoordinator::instance();
            thread_ = std::thread([this] { run_(); });
        }

        void run_() {
            for (;;) {
                std::shared_ptr<CompressionGroupState> group;
                {
                    std::unique_lock lock(mutex_);
                    condition_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                    if (stopping_) {
                        break;
                    }
                    group = std::move(queue_.front());
                    queue_.pop_front();
                    active_ = true;
                }
                CompressionWorkItem item;
                {
                    std::lock_guard group_lock(group->mutex);
                    if (!group->pending.empty()) {
                        item = std::move(group->pending.front());
                        group->pending.pop_front();
                        group->pending_paths.erase(item.path.lexically_normal().string());
                        if (group->pending.size() < MAX_PENDING_COMPRESSIONS_PER_LOGGER / 2) {
                            group->overflow_reported = false;
                        }
                    }
                }
                const bool can_compress = LogFileOperationCoordinator::instance().begin_compression(item.path);
                const bool compressed = can_compress && detail::gzip_log_file(
                    item.path, item.level, [&item] {
                        return LogFileOperationCoordinator::instance().compression_cancelled(item.path);
                    });
                if (can_compress) {
                    LogFileOperationCoordinator::instance().end_compression(item.path);
                }
                if (can_compress && !compressed) {
                    std::error_code error;
                    if (fs::exists(item.path, error) && !error) {
                        std::cerr << "Failed compressing rotated log " << item.path << std::endl;
                    }
                }
                if (!item.path.empty()) {
                    enforce_group_file_limit(group, item.path.parent_path());
                }
                {
                    std::lock_guard lock(mutex_);
                    bool requeue = false;
                    {
                        std::lock_guard group_lock(group->mutex);
                        requeue = !group->pending.empty();
                        if (!requeue) {
                            group->scheduled = false;
                        }
                    }
                    if (requeue && !stopping_) {
                        queue_.push_back(group);
                        condition_.notify_one();
                    }
                    active_ = false;
                    if (queue_.empty()) {
                        drained_.notify_all();
                    }
                }
            }
        }

        std::mutex mutex_;
        std::condition_variable condition_;
        std::condition_variable drained_;
        std::deque<std::shared_ptr<CompressionGroupState>> queue_;
        bool active_ = false;
        bool stopping_ = false;
        std::thread thread_;
    };

    struct ManagedLogPart {
        fs::path path;
        std::string date;
        std::uint64_t number = 0;
        bool compressed = false;
    };

    bool parse_unsigned(const std::string& value, std::uint64_t& number) {
        if (value.empty() || !std::all_of(value.begin(), value.end(), [](const unsigned char c) { return std::isdigit(c); })) {
            return false;
        }
        try {
            number = std::stoull(value);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    std::vector<ManagedLogPart> list_log_parts(const fs::path& directory, const fs::path& log_file_name,
                                               const bool count_mode) {
        std::vector<ManagedLogPart> parts;
        const auto stem = log_file_name.stem().string();
        const auto extension = log_file_name.extension().string();
        const auto prefix = stem + ".";
        std::error_code error;
        for (fs::directory_iterator it(directory, error), end; !error && it != end; it.increment(error)) {
            std::error_code entry_error;
            if (it->is_symlink(entry_error) || entry_error || !it->is_regular_file(entry_error) || entry_error) {
                continue;
            }
            auto name = it->path().filename().string();
            bool compressed = false;
            if (name.size() >= 3 && name.compare(name.size() - 3, 3, ".gz") == 0) {
                compressed = true;
                name.resize(name.size() - 3);
            }
            if (name.size() <= prefix.size() + extension.size() || name.compare(0, prefix.size(), prefix) != 0 ||
                name.compare(name.size() - extension.size(), extension.size(), extension) != 0) {
                continue;
            }
            const auto middle = name.substr(prefix.size(), name.size() - prefix.size() - extension.size());
            ManagedLogPart part{it->path(), {}, 0, compressed};
            if (count_mode) {
                if (middle.size() <= 11 || middle[10] != '.' || !is_valid_date_directory_name(middle.substr(0, 10)) ||
                    !parse_unsigned(middle.substr(11), part.number)) {
                    continue;
                }
                part.date = middle.substr(0, 10);
            } else if (!parse_unsigned(middle, part.number)) {
                continue;
            }
            parts.push_back(std::move(part));
        }
        std::sort(parts.begin(), parts.end(), [](const ManagedLogPart& left, const ManagedLogPart& right) {
            if (left.number != right.number) {
                return left.number < right.number;
            }
            return left.path.string() < right.path.string();
        });
        return parts;
    }

    void enforce_group_file_limit(const std::shared_ptr<CompressionGroupState>& group,
                                  const fs::path& directory) {
        fs::path log_file_name;
        fs::path protected_path;
        int max_files = 0;
        bool count_mode = false;
        {
            std::lock_guard group_lock(group->mutex);
            log_file_name = group->log_file_name;
            protected_path = group->protected_path;
            max_files = group->max_files;
            count_mode = group->count_mode;
        }
        if (max_files == 0) {
            return;
        }
        if (protected_path.parent_path() != directory) {
            protected_path.clear();
        }

        auto parts = list_log_parts(directory, log_file_name, count_mode);
        std::unordered_map<std::string, unsigned> representations;
        for (const auto& part : parts) {
            const auto identity = part.date + ":" + std::to_string(part.number);
            representations[identity] |= part.compressed ? 2U : 1U;
        }
        std::size_t remaining = parts.size();
        for (const auto& part : parts) {
            if (remaining <= static_cast<std::size_t>(max_files)) {
                break;
            }
            if (!protected_path.empty() && part.path == protected_path) {
                continue;
            }
            const auto identity = part.date + ":" + std::to_string(part.number);
            if (representations[identity] == 3U) {
                auto source_path = part.path;
                if (source_path.extension() == ".gz") {
                    source_path.replace_extension();
                }
                if (LogFileOperationCoordinator::instance().compression_in_progress(source_path)) {
                    break;
                }
                auto compressed_path = source_path;
                compressed_path += ".gz";
                std::error_code source_error;
                std::error_code compressed_error;
                const bool source_exists = fs::exists(source_path, source_error) && !source_error;
                const bool compressed_exists = fs::exists(compressed_path, compressed_error) && !compressed_error;
                if (source_error || compressed_error || (source_exists && compressed_exists)) {
                    std::cerr << "Refusing automatic retention for conflicting .log/.log.gz pair "
                              << part.path.parent_path() << " identity=" << identity << std::endl;
                    break;
                }
                representations[identity] = (source_exists ? 1U : 0U) | (compressed_exists ? 2U : 0U);
                std::error_code part_error;
                if (!fs::exists(part.path, part_error) && !part_error) {
                    --remaining;
                    continue;
                }
            }
            std::error_code error;
            bool busy = false;
            const bool removed = LogFileOperationCoordinator::instance().try_remove(part.path, error, busy);
            if (busy) {
                break;
            }
            std::error_code exists_error;
            const bool already_missing = !error && !fs::exists(part.path, exists_error) && !exists_error;
            if (removed || already_missing || error == std::errc::no_such_file_or_directory) {
                LogCompressionWorker::instance().cancel(group, part.path);
                --remaining;
            } else if (error) {
                std::cerr << "Failed removing excess rotated log " << part.path << ": " << error.message() << std::endl;
            }
        }
    }

    template <typename Mutex>
    class date_size_file_sink final : public spdlog::sinks::base_sink<Mutex> {
    public:
        date_size_file_sink(fs::path log_root, fs::path log_file_name, HLIEManagedLogConfig config)
            : log_root_(std::move(log_root)),
              log_file_name_(log_file_name.filename()),
              config_(config),
              count_mode_(config.retention_days == 0) {
            if (log_root_.empty() || log_file_name_.empty()) {
                throw spdlog::spdlog_ex("managed log root and file name must not be empty");
            }
            if (config_.retention_days < 0 || config_.max_files < 0 || config_.max_file_size_bytes == 0 ||
                config_.compression_level < 1 || config_.compression_level > 9) {
                throw spdlog::spdlog_ex("invalid managed log rotation configuration");
            }
            {
                std::lock_guard group_lock(compression_state_->mutex);
                compression_state_->log_file_name = log_file_name_;
                compression_state_->max_files = config_.max_files;
                compression_state_->count_mode = count_mode_;
            }
            open_for_date_(date_string(spdlog::log_clock::now()), true);
            schedule_historical_compression_();
        }

    protected:
        void sink_it_(const spdlog::details::log_msg& msg) override {
            const auto message_date = date_string(msg.time);
            if (message_date != current_date_) {
                open_for_date_(message_date, false);
            }

            spdlog::memory_buf_t formatted;
            spdlog::sinks::base_sink<Mutex>::formatter_->format(msg, formatted);
            if (current_size_ > 0 &&
                formatted.size() > config_.max_file_size_bytes - std::min(current_size_, config_.max_file_size_bytes)) {
                rotate_part_();
            }
            file_helper_.write(formatted);
            current_size_ += formatted.size();
        }

        void flush_() override { file_helper_.flush(); }

    private:
        fs::path part_path_(const std::uint64_t part_number) const {
            const auto stem = log_file_name_.stem().string();
            const auto extension = log_file_name_.extension().string();
            if (count_mode_) {
                return log_root_ / fmt::format("{}.{}.{:06d}{}", stem, current_date_, part_number, extension);
            }
            return current_directory_ / fmt::format("{}.{:04d}{}", stem, part_number, extension);
        }

        std::vector<ManagedLogPart> current_parts_() const {
            return list_log_parts(count_mode_ ? log_root_ : current_directory_, log_file_name_, count_mode_);
        }

        void open_for_date_(const std::string& date, const bool initial) {
            const auto closed_path = initial ? fs::path{} : current_path_;
            current_date_ = date;
            if (count_mode_) {
                std::error_code error;
                fs::create_directories(log_root_, error);
                if (error) {
                    throw spdlog::spdlog_ex("failed creating log root " + log_root_.string() + ": " + error.message());
                }
                current_directory_ = log_root_;
            } else {
                current_directory_ = ensure_managed_date_directory(log_root_, current_date_);
                HLIELogger::cleanup_old_log_directories(log_root_.string(), config_.retention_days);
            }

            recover_incomplete_compressions_(current_directory_);
            const auto parts = current_parts_();
            part_number_ = parts.empty() ? 1 : parts.back().number;
            bool can_append = !parts.empty() && !parts.back().compressed &&
                              (!count_mode_ || parts.back().date == current_date_);
            if (!can_append) {
                part_number_ = parts.empty() ? 1 : parts.back().number + 1;
            }
            open_part_();
            if (current_size_ >= config_.max_file_size_bytes) {
                rotate_part_();
            }
            enforce_file_limit_();
            finish_closed_part_(closed_path);
        }

        void rotate_part_() {
            const auto closed_path = current_path_;
            ++part_number_;
            open_part_();
            enforce_file_limit_();
            finish_closed_part_(closed_path);
        }

        void open_part_() {
            current_path_ = part_path_(part_number_);
            file_helper_.open(current_path_.string(), false);
            {
                std::lock_guard group_lock(compression_state_->mutex);
                compression_state_->protected_path = current_path_;
            }
            std::error_code error;
            current_size_ = fs::file_size(current_path_, error);
            if (error) {
                current_size_ = 0;
            }
        }

        void finish_closed_part_(const fs::path& path) const {
            if (!path.empty() && path != current_path_ && config_.compress_rotated_files) {
                LogCompressionWorker::instance().enqueue(compression_state_, path, config_.compression_level);
            }
        }

        void enforce_file_limit_() const {
            enforce_group_file_limit(compression_state_, count_mode_ ? log_root_ : current_directory_);
        }

        void schedule_historical_compression_() const {
            auto schedule_directory = [this](const fs::path& directory, const bool count_mode) {
                recover_incomplete_compressions_(directory);
                enforce_group_file_limit(compression_state_, directory);
                for (const auto& part : list_log_parts(directory, log_file_name_, count_mode)) {
                    if (config_.compress_rotated_files && !part.compressed && part.path != current_path_) {
                        LogCompressionWorker::instance().enqueue(compression_state_, part.path, config_.compression_level);
                    }
                }
            };
            if (count_mode_) {
                schedule_directory(log_root_, true);
                return;
            }
            std::error_code error;
            for (fs::directory_iterator it(log_root_, error), end; !error && it != end; it.increment(error)) {
                std::error_code entry_error;
                if (it->is_directory(entry_error) && !entry_error &&
                    is_valid_date_directory_name(it->path().filename().string()) &&
                    fs::exists(it->path() / MANAGED_DIRECTORY_MARKER, entry_error) && !entry_error) {
                    schedule_directory(it->path(), false);
                }
            }
        }

        void recover_incomplete_compressions_(const fs::path& directory) const {
            const auto prefix = log_file_name_.stem().string() + ".";
            std::error_code error;
            for (fs::directory_iterator it(directory, error), end; !error && it != end; it.increment(error)) {
                std::error_code entry_error;
                if (it->is_symlink(entry_error) || entry_error || !it->is_regular_file(entry_error) || entry_error) {
                    continue;
                }
                const auto name = it->path().filename().string();
                if (name.rfind(prefix, 0) != 0) {
                    continue;
                }

                constexpr std::string_view CLAIM_MARKER = ".log.compressing.";
                constexpr std::string_view TEMPORARY_MARKER = ".log.gz.tmp.";
                if (const auto marker = name.find(CLAIM_MARKER); marker != std::string::npos &&
                    compression_artifact_is_abandoned(*it, name, CLAIM_MARKER)) {
                    const auto source = it->path().parent_path() / name.substr(0, marker + 4);
                    entry_error.clear();
                    fs::create_hard_link(it->path(), source, entry_error);
                    if (!entry_error) {
                        fs::remove(it->path(), entry_error);
                    }
                    if (entry_error) {
                        std::cerr << "Preserving unrecoverable claimed log " << it->path()
                                  << ": " << entry_error.message() << std::endl;
                    }
                } else if (name.find(TEMPORARY_MARKER) != std::string::npos &&
                           compression_artifact_is_abandoned(*it, name, TEMPORARY_MARKER)) {
                    fs::remove(it->path(), entry_error);
                    if (entry_error) {
                        std::cerr << "Failed removing abandoned gzip temporary file " << it->path()
                                  << ": " << entry_error.message() << std::endl;
                    }
                }
            }
        }

        fs::path log_root_;
        fs::path log_file_name_;
        fs::path current_directory_;
        fs::path current_path_;
        HLIEManagedLogConfig config_;
        std::shared_ptr<CompressionGroupState> compression_state_ = std::make_shared<CompressionGroupState>();
        bool count_mode_ = false;
        std::string current_date_;
        std::uint64_t part_number_ = 1;
        std::uintmax_t current_size_ = 0;
        spdlog::details::file_helper file_helper_;
    };

    using date_size_file_sink_mt = date_size_file_sink<std::mutex>;

    class TcimManagedLogHandler final : public tcim::LogHandler {
    public:
        TcimManagedLogHandler(const std::string& log_root, const std::string& log_file_name,
                              const HLIEManagedLogConfig& config) {
            const auto file_sink = std::make_shared<date_size_file_sink_mt>(log_root, log_file_name, config);
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
    HLIEManagedLogConfig config;
    config.max_file_size_bytes = max_file_size_bytes;
    config.retention_days = retention_days;
    return init_managed_default_logger(log_root, log_file_name, module_name, level, config);
}

std::shared_ptr<spdlog::logger> HLIELogger::init_managed_default_logger(const std::string& log_root,
                                                                        const std::string& log_file_name,
                                                                        const std::string& module_name,
                                                                        const std::string& level,
                                                                        const HLIEManagedLogConfig& config) {
    std::cout << "Init managed default logger: " << module_name << ", log_root: " << log_root << std::endl;
    const auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    const auto file_sink = std::make_shared<date_size_file_sink_mt>(log_root, log_file_name, config);
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
    HLIEManagedLogConfig config;
    config.max_file_size_bytes = max_file_size_bytes;
    config.retention_days = retention_days;
    return create_managed_logger(log_root, log_file_name, module_name, level, config);
}

std::shared_ptr<spdlog::logger> HLIELogger::create_managed_logger(const std::string& log_root,
                                                                  const std::string& log_file_name,
                                                                  const std::string& module_name,
                                                                  const std::string& level,
                                                                  const HLIEManagedLogConfig& config) {
    std::cout << "Creating managed logger: " << module_name << ", log_root: " << log_root << std::endl;
    const auto file_sink = std::make_shared<date_size_file_sink_mt>(log_root, log_file_name, config);
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
    HLIEManagedLogConfig config;
    config.max_file_size_bytes = max_file_size_bytes;
    config.retention_days = retention_days;
    return register_tcim_log_handler(log_root, log_file_name, config);
}

bool HLIELogger::register_tcim_log_handler(const std::string& log_root, const std::string& log_file_name,
                                           const HLIEManagedLogConfig& config) {
    try {
        auto handler = std::make_unique<TcimManagedLogHandler>(log_root, log_file_name, config);
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
    if (retention_days <= 0) {
        return;
    }
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

void HLIELogger::wait_for_compression() {
    LogCompressionWorker::instance().wait();
}

bool HLIELogger::parse_file_size(const std::string& value, std::uintmax_t& bytes) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return false;
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    const auto normalized = value.substr(first, last - first + 1);
    if (normalized.size() < 2) {
        return false;
    }

    std::uintmax_t multiplier = 0;
    switch (static_cast<char>(std::toupper(static_cast<unsigned char>(normalized.back())))) {
        case 'K': multiplier = 1024U; break;
        case 'M': multiplier = 1024U * 1024U; break;
        case 'G': multiplier = 1024U * 1024U * 1024U; break;
        default: return false;
    }
    const auto number = normalized.substr(0, normalized.size() - 1);
    if (number.empty() || number.front() == '0' ||
        !std::all_of(number.begin(), number.end(), [](const unsigned char c) { return std::isdigit(c); })) {
        return false;
    }
    try {
        const auto parsed = static_cast<std::uintmax_t>(std::stoull(number));
        if (parsed > std::numeric_limits<std::uintmax_t>::max() / multiplier) {
            return false;
        }
        bytes = parsed * multiplier;
        return true;
    } catch (const std::exception&) {
        return false;
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

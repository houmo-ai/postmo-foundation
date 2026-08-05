#include "hlie_gzip.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <system_error>

#include <zlib.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1U << 0)
#endif
#endif
#endif

namespace hm_HLIE::detail {
namespace {
    namespace fs = std::filesystem;
    constexpr std::size_t BUFFER_SIZE = 256U * 1024U;
    constexpr int UNIQUE_PATH_ATTEMPTS = 16;
    std::atomic<std::uint64_t> temporary_sequence{0};

    struct FileIdentity {
        std::uint64_t device = 0;
        std::uint64_t file = 0;
        std::uint64_t size = 0;
        std::uint64_t write_time_high = 0;
        std::uint64_t write_time_low = 0;

        bool operator==(const FileIdentity& other) const {
            return device == other.device && file == other.file && size == other.size &&
                   write_time_high == other.write_time_high && write_time_low == other.write_time_low;
        }
    };

    class SourceReader {
    public:
        explicit SourceReader(const fs::path& path) {
#ifdef _WIN32
            handle_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                                  nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
            if (handle_ != INVALID_HANDLE_VALUE) {
                BY_HANDLE_FILE_INFORMATION information{};
                if (!GetFileInformationByHandle(handle_, &information) ||
                    (information.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) != 0) {
                    close();
                }
            }
#else
            handle_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
            struct stat status{};
            if (handle_ >= 0 && (::fstat(handle_, &status) != 0 || !S_ISREG(status.st_mode))) {
                close();
            }
#endif
        }

        ~SourceReader() { close(); }

        void close() {
#ifdef _WIN32
            if (handle_ != INVALID_HANDLE_VALUE) {
                CloseHandle(handle_);
                handle_ = INVALID_HANDLE_VALUE;
            }
#else
            if (handle_ >= 0) {
                ::close(handle_);
                handle_ = -1;
            }
#endif
        }

        [[nodiscard]] bool valid() const {
#ifdef _WIN32
            return handle_ != INVALID_HANDLE_VALUE;
#else
            return handle_ >= 0;
#endif
        }

        bool identity(FileIdentity& identity) const {
#ifdef _WIN32
            BY_HANDLE_FILE_INFORMATION information{};
            if (!valid() || !GetFileInformationByHandle(handle_, &information)) {
                return false;
            }
            identity.device = information.dwVolumeSerialNumber;
            identity.file = (static_cast<std::uint64_t>(information.nFileIndexHigh) << 32U) |
                            information.nFileIndexLow;
            identity.size = (static_cast<std::uint64_t>(information.nFileSizeHigh) << 32U) |
                            information.nFileSizeLow;
            identity.write_time_high = information.ftLastWriteTime.dwHighDateTime;
            identity.write_time_low = information.ftLastWriteTime.dwLowDateTime;
            return true;
#else
            struct stat status{};
            if (!valid() || ::fstat(handle_, &status) != 0) {
                return false;
            }
            identity.device = static_cast<std::uint64_t>(status.st_dev);
            identity.file = static_cast<std::uint64_t>(status.st_ino);
            identity.size = static_cast<std::uint64_t>(status.st_size);
#ifdef __APPLE__
            identity.write_time_high = static_cast<std::uint64_t>(status.st_mtimespec.tv_sec);
            identity.write_time_low = static_cast<std::uint64_t>(status.st_mtimespec.tv_nsec);
#else
            identity.write_time_high = static_cast<std::uint64_t>(status.st_mtim.tv_sec);
            identity.write_time_low = static_cast<std::uint64_t>(status.st_mtim.tv_nsec);
#endif
            return true;
#endif
        }

        bool read(unsigned char* buffer, const std::size_t capacity, std::size_t& bytes_read, bool& eof) {
#ifdef _WIN32
            DWORD read_count = 0;
            if (!ReadFile(handle_, buffer, static_cast<DWORD>(capacity), &read_count, nullptr)) {
                return false;
            }
            bytes_read = read_count;
#else
            ssize_t read_count = 0;
            do {
                read_count = ::read(handle_, buffer, capacity);
            } while (read_count < 0 && errno == EINTR);
            if (read_count < 0) {
                return false;
            }
            bytes_read = static_cast<std::size_t>(read_count);
#endif
            eof = bytes_read == 0;
            return true;
        }

    private:
#ifdef _WIN32
        HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
        int handle_ = -1;
#endif
    };

    class OutputWriter {
    public:
        explicit OutputWriter(const fs::path& path) {
#ifdef _WIN32
            handle_ = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                                  nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
#else
            handle_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
#endif
        }

        ~OutputWriter() { close(); }

        [[nodiscard]] bool valid() const {
#ifdef _WIN32
            return handle_ != INVALID_HANDLE_VALUE;
#else
            return handle_ >= 0;
#endif
        }

        bool write(const unsigned char* data, std::size_t size) {
            while (size > 0) {
#ifdef _WIN32
                DWORD written = 0;
                const auto chunk = static_cast<DWORD>(std::min<std::size_t>(size, std::numeric_limits<DWORD>::max()));
                if (!WriteFile(handle_, data, chunk, &written, nullptr) || written == 0) {
                    return false;
                }
#else
                ssize_t written = 0;
                do {
                    written = ::write(handle_, data, size);
                } while (written < 0 && errno == EINTR);
                if (written <= 0) {
                    return false;
                }
#endif
                data += written;
                size -= static_cast<std::size_t>(written);
            }
            return true;
        }

        bool finish() {
#ifdef _WIN32
            return valid() && FlushFileBuffers(handle_) != FALSE;
#else
            return valid() && ::fsync(handle_) == 0;
#endif
        }

        void close() {
#ifdef _WIN32
            if (handle_ != INVALID_HANDLE_VALUE) {
                CloseHandle(handle_);
                handle_ = INVALID_HANDLE_VALUE;
            }
#else
            if (handle_ >= 0) {
                ::close(handle_);
                handle_ = -1;
            }
#endif
        }

    private:
#ifdef _WIN32
        HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
        int handle_ = -1;
#endif
    };

    std::uint64_t process_id() {
#ifdef _WIN32
        return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
        return static_cast<std::uint64_t>(getpid());
#endif
    }

    fs::path unique_path_for(const fs::path& base, const char* marker) {
        auto path = base;
        path += marker + std::to_string(process_id()) + "." +
                std::to_string(temporary_sequence.fetch_add(1, std::memory_order_relaxed));
        return path;
    }

    bool move_no_replace(const fs::path& source, const fs::path& destination, std::error_code& error) {
#ifdef _WIN32
        if (MoveFileExW(source.c_str(), destination.c_str(), MOVEFILE_WRITE_THROUGH) != FALSE) {
            error.clear();
            return true;
        }
        error = std::error_code(static_cast<int>(GetLastError()), std::system_category());
        return false;
#else
#ifdef __linux__
        if (::syscall(SYS_renameat2, AT_FDCWD, source.c_str(), AT_FDCWD, destination.c_str(), RENAME_NOREPLACE) == 0) {
            error.clear();
            return true;
        }
        if (errno != ENOSYS && errno != EINVAL) {
            error = std::error_code(errno, std::generic_category());
            return false;
        }
#endif
        if (::link(source.c_str(), destination.c_str()) != 0) {
            error = std::error_code(errno, std::generic_category());
            return false;
        }
        if (::unlink(source.c_str()) == 0) {
            error.clear();
            return true;
        }
        const auto unlink_error = errno;
        ::unlink(destination.c_str());
        error = std::error_code(unlink_error, std::generic_category());
        return false;
#endif
    }

    bool claim_source(const fs::path& source, fs::path& claimed) {
        for (int attempt = 0; attempt < UNIQUE_PATH_ATTEMPTS; ++attempt) {
            claimed = unique_path_for(source, ".compressing.");
            std::error_code error;
            if (move_no_replace(source, claimed, error)) {
                return true;
            }
            if (error != std::errc::file_exists) {
                return false;
            }
        }
        return false;
    }

    void restore_claim(const fs::path& claimed, const fs::path& source) {
        std::error_code error;
        if (!fs::exists(source, error) && !error) {
            move_no_replace(claimed, source, error);
        }
    }
} // namespace

bool gzip_log_file(const fs::path& source, const int compression_level,
                   const std::function<bool()>& cancelled) noexcept {
    fs::path claimed;
    fs::path temporary;
    try {
        std::error_code error;
        if (source.extension() != ".log" || fs::is_symlink(source, error) || error ||
            !fs::is_regular_file(source, error) || error) {
            return false;
        }

        auto destination = source;
        destination += ".gz";
        if (fs::exists(destination, error) || error || !claim_source(source, claimed)) {
            return false;
        }

        SourceReader reader(claimed);
        FileIdentity original_identity{};
        if (!reader.valid() || !reader.identity(original_identity)) {
            reader.close();
            restore_claim(claimed, source);
            return false;
        }
        const auto original_permissions = fs::status(claimed, error).permissions();
        error.clear();

        std::unique_ptr<OutputWriter> output;
        for (int attempt = 0; attempt < UNIQUE_PATH_ATTEMPTS && !output; ++attempt) {
            temporary = unique_path_for(destination, ".tmp.");
            auto candidate = std::make_unique<OutputWriter>(temporary);
            if (candidate->valid()) {
                output = std::move(candidate);
            }
        }
        if (!output) {
            reader.close();
            restore_claim(claimed, source);
            return false;
        }

        z_stream stream{};
        if (deflateInit2(&stream, std::clamp(compression_level, 1, 9), Z_DEFLATED,
                         15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
            output->close();
            fs::remove(temporary, error);
            reader.close();
            restore_claim(claimed, source);
            return false;
        }

        std::array<unsigned char, BUFFER_SIZE> input_buffer{};
        std::array<unsigned char, BUFFER_SIZE> output_buffer{};
        bool success = true;
        bool eof = false;
        int result = Z_OK;
        while (!eof && success) {
            if (cancelled && cancelled()) {
                success = false;
                break;
            }
            std::size_t bytes_read = 0;
            if (!reader.read(input_buffer.data(), input_buffer.size(), bytes_read, eof)) {
                success = false;
                break;
            }
            stream.next_in = input_buffer.data();
            stream.avail_in = static_cast<uInt>(bytes_read);
            const int flush = eof ? Z_FINISH : Z_NO_FLUSH;
            do {
                stream.next_out = output_buffer.data();
                stream.avail_out = static_cast<uInt>(output_buffer.size());
                result = deflate(&stream, flush);
                if (result != Z_OK && result != Z_STREAM_END) {
                    success = false;
                    break;
                }
                const auto produced = output_buffer.size() - stream.avail_out;
                if (!output->write(output_buffer.data(), produced)) {
                    success = false;
                    break;
                }
            } while (stream.avail_out == 0);
        }
        const int end_result = deflateEnd(&stream);
        FileIdentity final_identity{};
        success = success && result == Z_STREAM_END && end_result == Z_OK && output->finish() &&
                  reader.identity(final_identity) && final_identity == original_identity &&
                  !(cancelled && cancelled());
        output->close();

        if (!success) {
            fs::remove(temporary, error);
            reader.close();
            restore_claim(claimed, source);
            return false;
        }

        error.clear();
        if (!move_no_replace(temporary, destination, error)) {
            fs::remove(temporary, error);
            reader.close();
            restore_claim(claimed, source);
            return false;
        }

        FileIdentity committed_identity{};
        if (!reader.identity(committed_identity) || !(committed_identity == original_identity) ||
            (cancelled && cancelled())) {
            reader.close();
            fs::remove(destination, error);
            restore_claim(claimed, source);
            return false;
        }

        fs::permissions(destination, original_permissions, fs::perm_options::replace, error);
        error.clear();
        const auto original_time = fs::last_write_time(claimed, error);
        if (!error) {
            fs::last_write_time(destination, original_time, error);
        }
        error.clear();
        reader.close();
        const bool removed_claim = fs::remove(claimed, error);
        return removed_claim && !error;
    } catch (...) {
        std::error_code error;
        if (!temporary.empty()) {
            fs::remove(temporary, error);
        }
        if (!claimed.empty()) {
            restore_claim(claimed, source);
        }
        return false;
    }
}

} // namespace hm_HLIE::detail

#include "mirage_process.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

std::string require_executable() {
    const char* value = std::getenv("HLIE_MIRAGE_EXECUTABLE");
    std::string executable = value == nullptr ? std::string{} : std::string{value};
#ifdef HLIE_MIRAGE_EXECUTABLE_DEFAULT
    if (executable.empty()) {
        executable = HLIE_MIRAGE_EXECUTABLE_DEFAULT;
    }
#endif
    if (executable.empty()) {
        throw std::runtime_error("HLIE_MIRAGE_EXECUTABLE is not set");
    }
    if (!std::filesystem::is_regular_file(executable)) {
        throw std::runtime_error("HLAW Mirage executable does not exist: " + executable);
    }
    return executable;
}

std::string unique_ready_file() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return (std::filesystem::temp_directory_path() /
            ("hlaw-mirage-ready-" + std::to_string(stamp) + ".json")).string();
}

std::string extract_json_string(const std::string& json, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    const auto key_pos = json.find(marker);
    const auto colon_pos = key_pos == std::string::npos ? key_pos : json.find(':', key_pos + marker.size());
    const auto quote_begin = colon_pos == std::string::npos ? colon_pos : json.find('"', colon_pos + 1);
    const auto quote_end = quote_begin == std::string::npos ? quote_begin : json.find('"', quote_begin + 1);
    if (quote_end == std::string::npos) {
        throw std::runtime_error("invalid Mirage ready file: missing " + key);
    }
    return json.substr(quote_begin + 1, quote_end - quote_begin - 1);
}

void parse_control_url(const std::string& url, std::string& host, int& port) {
    constexpr auto http_prefix = "http://";
    if (url.rfind(http_prefix, 0) != 0) {
        throw std::runtime_error("unsupported Mirage control URL: " + url);
    }
    const std::string authority = url.substr(std::char_traits<char>::length(http_prefix));
    const auto colon = authority.rfind(':');
    if (colon == std::string::npos) {
        throw std::runtime_error("Mirage control URL has no port: " + url);
    }
    host = authority.substr(0, colon);
    try {
        port = std::stoi(authority.substr(colon + 1));
    } catch (const std::exception&) {
        throw std::runtime_error("Mirage control URL has an invalid port: " + url);
    }
    if (host.empty() || port <= 0 || port > 65535) {
        throw std::runtime_error("Mirage control URL has an invalid authority: " + url);
    }
}

#ifdef _WIN32

std::wstring utf8_to_wide(const std::string& value) {
    if (value.empty()) {
        return {};
    }
    const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                            static_cast<int>(value.size()), nullptr, 0);
    if (length <= 0) {
        throw std::runtime_error("failed to convert process argument to UTF-16");
    }
    std::wstring result(static_cast<std::size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()),
                        result.data(), length);
    return result;
}

std::wstring quote_windows_argument(const std::string& value) {
    const std::wstring wide = utf8_to_wide(value);
    std::wstring result = L"\"";
    std::size_t backslashes = 0;
    for (const wchar_t ch : wide) {
        if (ch == L'\\') {
            ++backslashes;
            continue;
        }
        if (ch == L'\"') {
            result.append(backslashes * 2 + 1, L'\\');
            result.push_back(ch);
        } else {
            result.append(backslashes, L'\\');
            result.push_back(ch);
        }
        backslashes = 0;
    }
    result.append(backslashes * 2, L'\\');
    result.push_back(L'\"');
    return result;
}

#endif

} // namespace

struct MirageProcess::Impl {
#ifdef _WIN32
    HANDLE process = nullptr;
    HANDLE job = nullptr;
#else
    pid_t pid = -1;
#endif
};

MirageProcess::MirageProcess()
    : impl_(std::make_unique<Impl>()), ready_file_(unique_ready_file()) {
    try {
        launch(require_executable());
        wait_until_ready();
    } catch (...) {
        stop();
        std::error_code ignored;
        std::filesystem::remove(ready_file_, ignored);
        throw;
    }
}

MirageProcess::~MirageProcess() {
    stop();
    std::error_code ignored;
    std::filesystem::remove(ready_file_, ignored);
}

void MirageProcess::launch(const std::string& executable) {
#ifdef _WIN32
    std::wstring command = quote_windows_argument(executable) +
        L" serve --listen 127.0.0.1:0 --ready-file " + quote_windows_argument(ready_file_);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process_info{};
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                        nullptr, nullptr, &startup, &process_info)) {
        throw std::runtime_error("failed to start HLAW Mirage, Win32 error " +
                                 std::to_string(GetLastError()));
    }
    CloseHandle(process_info.hThread);
    impl_->process = process_info.hProcess;

    impl_->job = CreateJobObjectW(nullptr, nullptr);
    if (impl_->job == nullptr) {
        throw std::runtime_error("failed to create Mirage process job");
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!SetInformationJobObject(impl_->job, JobObjectExtendedLimitInformation,
                                 &limits, sizeof(limits)) ||
        !AssignProcessToJobObject(impl_->job, impl_->process)) {
        throw std::runtime_error("failed to assign Mirage process to cleanup job");
    }
#else
    impl_->pid = fork();
    if (impl_->pid < 0) {
        throw std::runtime_error("failed to fork HLAW Mirage process");
    }
    if (impl_->pid == 0) {
        execl(executable.c_str(), executable.c_str(), "serve", "--listen", "127.0.0.1:0",
              "--ready-file", ready_file_.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
#endif
}

void MirageProcess::wait_until_ready() {
    constexpr auto timeout = std::chrono::seconds(10);
    constexpr auto poll_interval = std::chrono::milliseconds(25);
    const auto deadline = std::chrono::steady_clock::now() + timeout;

    while (std::chrono::steady_clock::now() < deadline) {
        std::ifstream ready(ready_file_);
        if (ready) {
            std::ostringstream contents;
            contents << ready.rdbuf();
            const std::string json = contents.str();
            if (!json.empty()) {
                base_url_ = extract_json_string(json, "base_url");
                control_url_ = extract_json_string(json, "control_url");
                parse_control_url(control_url_, host_, port_);
                return;
            }
        }

#ifdef _WIN32
        if (WaitForSingleObject(impl_->process, 0) == WAIT_OBJECT_0) {
            throw std::runtime_error("HLAW Mirage exited before becoming ready");
        }
#else
        int status = 0;
        if (waitpid(impl_->pid, &status, WNOHANG) == impl_->pid) {
            impl_->pid = -1;
            throw std::runtime_error("HLAW Mirage exited before becoming ready");
        }
#endif
        std::this_thread::sleep_for(poll_interval);
    }
    throw std::runtime_error("timed out waiting for HLAW Mirage readiness");
}

void MirageProcess::stop() noexcept {
    if (!impl_) {
        return;
    }
#ifdef _WIN32
    if (impl_->job != nullptr) {
        CloseHandle(impl_->job);
        impl_->job = nullptr;
    }
    if (impl_->process != nullptr) {
        if (WaitForSingleObject(impl_->process, 3000) == WAIT_TIMEOUT) {
            // Closing the job normally terminates the child. Keep this fallback
            // for partial launch failures where the process was never assigned.
            TerminateProcess(impl_->process, 1);
            WaitForSingleObject(impl_->process, 3000);
        }
        CloseHandle(impl_->process);
        impl_->process = nullptr;
    }
#else
    if (impl_->pid > 0) {
        kill(impl_->pid, SIGTERM);
        for (int attempt = 0; attempt < 30; ++attempt) {
            int status = 0;
            if (waitpid(impl_->pid, &status, WNOHANG) == impl_->pid) {
                impl_->pid = -1;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (impl_->pid > 0) {
            kill(impl_->pid, SIGKILL);
            waitpid(impl_->pid, nullptr, 0);
            impl_->pid = -1;
        }
    }
#endif
}

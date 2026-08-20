#include "hlie_app/hlie_process_manager.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

#ifndef _WIN32
extern char** environ;
#endif

#ifdef _WIN32
#pragma comment(lib, "Advapi32.lib")
#endif

HLIE_NAMESPACE_BEGIN

namespace {
    bool is_sensitive_process_option(const std::string& option) {
        std::string lowered = option;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](const unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        return lowered.find("token") != std::string::npos ||
               lowered.find("api-key") != std::string::npos ||
               lowered.find("password") != std::string::npos ||
               lowered.find("secret") != std::string::npos;
    }

    std::vector<int> parse_process_device_ids(const std::vector<std::string>& args) {
        for (size_t i = 0; i + 1 < args.size(); ++i) {
            if (args[i] != "--device_id") {
                continue;
            }

            std::vector<int> device_ids;
            if (HlieUtil::parse_device(args[i + 1], device_ids) == 0) {
                return device_ids;
            }
            break;
        }
        return {};
    }

#ifdef _WIN32
    constexpr DWORD INVALID_PID = 0;

    std::string quote_windows_argument(const std::string& argument) {
        if (argument.empty()) {
            return "\"\"";
        }
        if (argument.find_first_of(" \t\n\v\"") == std::string::npos) {
            return argument;
        }

        std::string quoted{"\""};
        size_t backslash_count = 0;
        for (const char ch : argument) {
            if (ch == '\\') {
                ++backslash_count;
                continue;
            }
            if (ch == '"') {
                quoted.append(backslash_count * 2 + 1, '\\');
                quoted.push_back('"');
            } else {
                quoted.append(backslash_count, '\\');
                quoted.push_back(ch);
            }
            backslash_count = 0;
        }
        quoted.append(backslash_count * 2, '\\');
        quoted.push_back('"');
        return quoted;
    }

    std::string windows_error_message(const DWORD error) {
        char* message = nullptr;
        const DWORD size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                          nullptr, error, 0, reinterpret_cast<char*>(&message), 0, nullptr);
        std::string result = size && message ? std::string(message, size) : "unknown error";
        if (message) {
            LocalFree(message);
        }
        while (!result.empty() && (result.back() == '\r' || result.back() == '\n')) {
            result.pop_back();
        }
        return result;
    }
#else
    constexpr pid_t INVALID_PID = static_cast<pid_t>(-1);
#endif

} // namespace

HLIEProcessManager::~HLIEProcessManager() {
    // 友好地尝试结束所有子进程（先 TERM 再 KILL 的逻辑只在 POSIX，Windows 直接 TerminateProcess）
    terminate_all();
#ifdef _WIN32
    // 释放仍然持有的句柄（如果有的话）
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : processes_) {
        auto& info = kv.second;
        if (info.is_running && info.process_handle) {
            // 不主动杀，交给上面的 terminate_all 处理；这里只做句柄清理
        }
        if (info.process_handle) {
            CloseHandle(info.process_handle);
            info.process_handle = nullptr;
        }
    }
#endif
}

ProcessID HLIEProcessManager::create_process(const std::string& exe_path,
    const std::vector<std::string>& args, const std::vector<std::string>& model_names,
    const int instance_id, const HLIEModelLaunchTicket& launch_ticket) {
    std::lock_guard lock(mutex_);
#ifdef _WIN32
    // Windows: 构造命令行，程序路径用双引号包起来，防止路径里有空格
    std::stringstream ss;
    std::stringstream display_ss;
    ss << quote_windows_argument(exe_path);
    display_ss << quote_windows_argument(exe_path);
    bool redact_next = false;
    for (const auto& arg : args) {
        ss << " " << quote_windows_argument(arg);
        display_ss << " " << quote_windows_argument(redact_next ? "<redacted>" : arg);
        if (redact_next) {
            redact_next = false;
        } else {
            redact_next = is_sensitive_process_option(arg);
        }
    }
    std::string cmd_line = ss.str();
    const std::string display_cmd_line = display_ss.str();
    const auto exe_directory = std::filesystem::path(exe_path).parent_path().string();
    char inherited_working_directory[MAX_PATH] = {};
    if (const DWORD size = GetEnvironmentVariableA("HLIECPP_WORKING_DIR", inherited_working_directory, MAX_PATH);
        size == 0 || size >= MAX_PATH || GetFileAttributesA(inherited_working_directory) == INVALID_FILE_ATTRIBUTES) {
        if (GetCurrentDirectoryA(MAX_PATH, inherited_working_directory) == 0 ||
            GetFileAttributesA(inherited_working_directory) == INVALID_FILE_ATTRIBUTES) {
            strncpy_s(inherited_working_directory, exe_directory.c_str(), _TRUNCATE);
        }
    }
    const std::string working_directory = inherited_working_directory;
    const DWORD original_path_size = GetEnvironmentVariableA("PATH", nullptr, 0);
    std::string original_path;
    if (original_path_size > 0) {
        original_path.resize(original_path_size);
        GetEnvironmentVariableA("PATH", original_path.data(), original_path_size);
        if (!original_path.empty() && original_path.back() == '\0') {
            original_path.pop_back();
        }
    }
    SetEnvironmentVariableA("PATH", (working_directory + ";" + original_path).c_str());

    STARTUPINFOA si = { sizeof(si) };
    PROCESS_INFORMATION pi{};

    const BOOL ok = CreateProcessA(
        exe_path.c_str(),        // lpApplicationName
        cmd_line.data(),         // lpCommandLine
        nullptr,                 // lpProcessAttrs
        nullptr,                 // lpThreadAttrs
        FALSE,                   // bInheritHandles
        CREATE_NEW_PROCESS_GROUP, // dwCreationFlags
        nullptr,                 // lpEnvironment
        working_directory.empty() ? nullptr : working_directory.c_str(), // lpCurrentDir
        &si,
        &pi
    );
    if (original_path_size > 0) {
        SetEnvironmentVariableA("PATH", original_path.c_str());
    } else {
        SetEnvironmentVariableA("PATH", nullptr);
    }

    if (!ok) {
        const DWORD error = GetLastError();
        LOG_ERROR("Failed to CreateProcessA with cmd {}, error {} ({})", display_cmd_line, error, windows_error_message(error));
        return 0;
    }

    CloseHandle(pi.hThread);

    if (WaitForSingleObject(pi.hProcess, 100) == WAIT_OBJECT_0) {
        DWORD exit_code = 0;
        if (!GetExitCodeProcess(pi.hProcess, &exit_code)) {
            exit_code = GetLastError();
        }
        LOG_ERROR("Child process PID[{}] exited immediately with code {}, cmd {}", pi.dwProcessId, exit_code, display_cmd_line);
        CloseHandle(pi.hProcess);
        return 0;
    }

    LOG_INFO("Created child process PID[{}] instance_id[{}] cwd={} exe_dir={} cmd={}",
             pi.dwProcessId, instance_id, working_directory, exe_directory, display_cmd_line);

    ProcessInfo info;
    info.uuid = HlieUtil::uuid();
    info.process_handle = pi.hProcess;
    info.process_id = pi.dwProcessId;
    info.params = display_cmd_line;
    info.model_names = model_names;
    info.device_ids = parse_process_device_ids(args);
    info.instance_id = instance_id;
    info.start_time = time(nullptr);
    info.is_running = true;
    info.launch_ticket = launch_ticket;

    processes_[info.process_id] = info;

    return info.process_id;

#else
    // Build argv entirely in the parent and use posix_spawnp. Running logging
    // or allocator-backed C++ code in a post-fork child can deadlock when the
    // parent process has multiple threads.
    std::vector<char*> argv;
    argv.reserve(args.size() + 2);
    argv.push_back(const_cast<char*>(exe_path.c_str()));
    bool redact_next = false;
    for (const auto& arg : args) {
        LOG_DEBUG("child arg: {}", redact_next ? "<redacted>" : arg);
        if (redact_next) {
            redact_next = false;
        } else {
            redact_next = is_sensitive_process_option(arg);
        }
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = INVALID_PID;
    const int spawn_error = posix_spawnp(
        &pid, exe_path.c_str(), nullptr, nullptr, argv.data(), environ);
    if (spawn_error != 0) {
        LOG_ERROR("Failed to posix_spawnp Worker path={}, error={} ({})",
                  exe_path, spawn_error, std::strerror(spawn_error));
        return 0;
    }

    ProcessInfo info;
    info.pid = pid;
    info.uuid = HlieUtil::uuid();
    info.process_id = pid;
    info.model_names = model_names;
    info.device_ids = parse_process_device_ids(args);
    info.instance_id = instance_id;
    info.start_time = time(nullptr);
    info.is_running = true;
    info.launch_ticket = launch_ticket;

    processes_[pid] = info;
    LOG_INFO("Created child process PID[{}] instance_id[{}] path={}",
             pid, instance_id, exe_path);
    return pid;
#endif
}

bool HLIEProcessManager::terminate_process(ProcessID pid) {
    std::lock_guard lock(mutex_);
    auto it = processes_.find(pid);
    if (it == processes_.end())
        return false;
    auto& info = it->second;

#ifdef _WIN32
    if (!info.process_handle)
        return false;

    // 尝试优雅结束：先发送 CTRL_BREAK（如果创建了新进程组且支持控制台）
    // 由于是否有控制台/附加控制台不确定，直接使用 TerminateProcess 简化处理。
    // 如需更优雅的退出，可在子进程实现自定义的 IPC 退出协议。
    if (!TerminateProcess(info.process_handle, 0)) {
        // TerminateProcess 失败，仍尝试 GetExitCodeProcess 更新状态
        DWORD code = STILL_ACTIVE;
        GetExitCodeProcess(info.process_handle, &code);
        info.is_running = (code == STILL_ACTIVE);
        return false;
    }

    // Do not remove the process record until termination is confirmed.
    const DWORD wait_result = WaitForSingleObject(info.process_handle, 5000 /*ms*/);
    if (wait_result != WAIT_OBJECT_0) {
        LOG_ERROR("Timed out or failed waiting for terminated process PID[{}], wait_result={}, error={}",
                  info.process_id, wait_result, GetLastError());
        info.is_running = true;
        return false;
    }

    // 更新状态并清理句柄
    info.is_running = false;
    CloseHandle(info.process_handle);
    info.process_handle = nullptr;

    // 从表中移除
    processes_.erase(it);
    process_cv_.notify_all();
    return true;

#else
    const auto wait_nointr = [](const pid_t child, int* status, const int options) {
        pid_t result;
        do {
            result = waitpid(child, status, options);
        } while (result < 0 && errno == EINTR);
        return result;
    };
    const auto confirmed_absent = [](const pid_t child) {
        if (kill(child, 0) == 0) {
            return false;
        }
        return errno == ESRCH;
    };

    // Send SIGTERM first, then fall back to SIGKILL. The process record and
    // reservation owner must remain live unless exit is positively confirmed.
    bool success = false;

    if (kill(info.pid, SIGTERM) == 0 || errno == ESRCH) {
        // 非阻塞等待
        for (int i = 0; i < 50; ++i) { // 总计 ~500ms
            int status = 0;
            const pid_t r = wait_nointr(info.pid, &status, WNOHANG);
            if (r == info.pid) {
                success = true;
                break;
            }
            if (r < 0) {
                if (errno == ECHILD && confirmed_absent(info.pid)) {
                    success = true;
                }
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    if (!success) {
        if (kill(info.pid, SIGKILL) != 0 && errno != ESRCH) {
            LOG_ERROR("Failed to SIGKILL child process PID[{}], error={} ({})",
                      info.pid, errno, std::strerror(errno));
            return false;
        }
        int status = 0;
        const pid_t r = wait_nointr(info.pid, &status, 0);
        success = r == info.pid ||
                  (r < 0 && errno == ECHILD && confirmed_absent(info.pid));
        if (!success) {
            LOG_ERROR("Failed to confirm child process exit PID[{}], waitpid_result={}, error={} ({})",
                      info.pid, r, errno, std::strerror(errno));
            return false;
        }
    }

    info.is_running = false;
    processes_.erase(it);
    process_cv_.notify_all();
    return success;
#endif
}

std::optional<HLIEProcessManager::ProcessInfo> HLIEProcessManager::get_process(
    const ProcessID pid) {
    monitor_processes();
    std::lock_guard lock(mutex_);
    const auto it = processes_.find(pid);
    return it == processes_.end() ? std::nullopt
                                  : std::optional<ProcessInfo>{it->second};
}

bool HLIEProcessManager::mark_expected_exit(const ProcessID pid, const bool expected) {
    std::lock_guard lock(mutex_);
    const auto it = processes_.find(pid);
    if (it == processes_.end()) {
        return false;
    }
    it->second.expected_exit = expected;
    return true;
}

bool HLIEProcessManager::wait_for_exit(const ProcessID pid,
                                       const std::chrono::milliseconds timeout) {
    const auto non_negative_timeout = timeout < std::chrono::milliseconds::zero()
        ? std::chrono::milliseconds::zero()
        : timeout;
    const auto deadline = std::chrono::steady_clock::now() + non_negative_timeout;
    while (true) {
        monitor_processes();
        std::unique_lock lock(mutex_);
        if (processes_.find(pid) == processes_.end()) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        process_cv_.wait_for(lock, std::chrono::milliseconds(25));
    }
}

bool HLIEProcessManager::terminate_all() {
    std::vector<ProcessID> pids;
    {
        std::lock_guard lock(mutex_);
        pids.reserve(processes_.size());
        for (auto const& kv : processes_) {
            pids.push_back(kv.first);
        }
    }

    bool all_ok = true;
    for (const auto pid : pids) {
        const bool ok = terminate_process(pid);
        all_ok = all_ok && ok;
    }

    return all_ok;
}

std::vector<HLIEProcessManager::ProcessInfo> HLIEProcessManager::get_running_processes() {
    monitor_processes(); // 更新状态后再返回
    std::vector<ProcessInfo> result;
    {
        std::lock_guard lock(mutex_);
        result.reserve(processes_.size());
        for (const auto& [pid, process] : processes_) {
            if (process.is_running) {
                result.push_back(process);
            }
        }
    }

    return result;
}

void HLIEProcessManager::erase_model(const std::string& model_name) {
    monitor_processes(); // 更新状态后再返回

    std::lock_guard lock(mutex_);

    for (auto& [pid, process] : processes_) {
        if (!process.is_running) {
            continue;
        }

        auto it = std::find(process.model_names.begin(),
                            process.model_names.end(),
                            model_name);

        if (it == process.model_names.end()) {
            continue;
        }

        process.model_names.erase(it);

        LOG_INFO("model {} removed from process {}", model_name, pid);

        // Todo:: 如果进程已经没有模型了，可以考虑关闭
        if (process.model_names.empty()) {
            LOG_DEBUG("process {} has no registered models", pid);
        }

        break;
    }
}

std::vector<std::string> HLIEProcessManager::get_working_models(const int instance_id) {
    monitor_processes();

    std::lock_guard lock(mutex_);

    for (const auto& [pid, process] : processes_) {
        if (!process.is_running || process.instance_id != instance_id) {
            continue;
        }

        return std::vector(
            process.model_names.begin(),
            process.model_names.end()
        );
    }

    return {};
}

void HLIEProcessManager::add_model(const std::string& model_name, const int instance_id) {
    monitor_processes(); // 更新状态后再返回

    std::lock_guard lock(mutex_);

    for (auto& [pid, process] : processes_) {
        if (!process.is_running || process.instance_id != instance_id) {
            continue;
        }

        auto it = std::find(process.model_names.begin(),
                            process.model_names.end(),
                            model_name);

        if (it != process.model_names.end()) {
            continue;
        }

        process.model_names.push_back(model_name);

        LOG_INFO("model {} add into process {}", model_name, pid);

        break;
    }
}

void HLIEProcessManager::monitor_processes() {
    std::lock_guard lock(mutex_);

#ifdef _WIN32
    std::vector<ProcessID> to_remove;
    for (auto& kv : processes_) {
        auto& info = kv.second;
        if (!info.process_handle) {
            info.is_running = false;
            exit_events_.push_back({info, -1});
            to_remove.push_back(kv.first);
            continue;
        }
        DWORD code = STILL_ACTIVE;
        if (!GetExitCodeProcess(info.process_handle, &code)) {
            const DWORD error = GetLastError();
            LOG_ERROR("Failed to query child process PID[{}] instance_id[{}], error {} ({})",
                      info.process_id, info.instance_id, error, windows_error_message(error));
            // 无法获取，认为已结束
            if (WaitForSingleObject(info.process_handle, 0) == WAIT_OBJECT_0) {
                info.is_running = false;
                CloseHandle(info.process_handle);
                info.process_handle = nullptr;
                exit_events_.push_back({info, -static_cast<int>(error)});
                to_remove.push_back(kv.first);
            } else {
                info.is_running = true;
            }
        } else {
            info.is_running = (code == STILL_ACTIVE);
            if (!info.is_running) {
                if (info.expected_exit) {
                    LOG_INFO("Child process PID[{}] instance_id[{}] completed expected exit with code {}",
                             info.process_id, info.instance_id, code);
                } else {
                    LOG_WARN("Child process PID[{}] instance_id[{}] exited unexpectedly with code {}, cmd {}",
                             info.process_id, info.instance_id, code, info.params);
                }
                CloseHandle(info.process_handle);
                info.process_handle = nullptr;
                exit_events_.push_back({info, static_cast<int>(code)});
                to_remove.push_back(kv.first);
            }
        }
    }
    for (auto pid : to_remove) {
        processes_.erase(pid);
    }
    if (!to_remove.empty()) {
        process_cv_.notify_all();
    }
#else
    std::vector<ProcessID> to_remove;
    for (auto& kv : processes_) {
        auto& info = kv.second;
        int status = 0;
        pid_t r;
        do {
            r = waitpid(info.pid, &status, WNOHANG);
        } while (r < 0 && errno == EINTR);
        if (r == 0) {
            // 仍在运行
            info.is_running = true;
        } else if (r == info.pid) {
            // 已退出
            info.is_running = false;
            const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status)
                : WIFSIGNALED(status) ? 128 + WTERMSIG(status) : -1;
            exit_events_.push_back({info, exit_code});
            to_remove.push_back(kv.first);
        } else {
            const int wait_error = errno;
            bool confirmed_exit = false;
            if (wait_error == ECHILD) {
                confirmed_exit = kill(info.pid, 0) != 0 && errno == ESRCH;
            }
            if (confirmed_exit) {
                info.is_running = false;
                exit_events_.push_back({info, -1});
                to_remove.push_back(kv.first);
            } else {
                info.is_running = true;
                LOG_ERROR("waitpid failed for live child PID[{}], error={} ({})",
                          info.pid, wait_error, std::strerror(wait_error));
            }
        }
    }
    for (auto pid : to_remove) {
        processes_.erase(pid);
    }
    if (!to_remove.empty()) {
        process_cv_.notify_all();
    }
#endif
}

std::vector<HLIEProcessManager::ProcessExitEvent> HLIEProcessManager::drain_exit_events() {
    std::lock_guard lock(mutex_);
    std::vector<ProcessExitEvent> events;
    events.reserve(exit_events_.size());
    while (!exit_events_.empty()) {
        events.push_back(std::move(exit_events_.front()));
        exit_events_.pop_front();
    }
    return events;
}

void HLIEProcessManager::cleanup_terminated_processes() {
    std::lock_guard lock(mutex_);
    std::vector<ProcessID> to_remove;
    for (auto const& kv : processes_) {
        if (!kv.second.is_running) {
            to_remove.push_back(kv.first);
        }
    }
#ifdef _WIN32
    for (auto pid : to_remove) {
        auto it = processes_.find(pid);
        if (it != processes_.end()) {
            if (it->second.process_handle) {
                CloseHandle(it->second.process_handle);
                it->second.process_handle = nullptr;
            }
            processes_.erase(it);
        }
    }
#else
    for (auto pid : to_remove) {
        processes_.erase(pid);
    }
#endif
}

HLIE_NAMESPACE_END

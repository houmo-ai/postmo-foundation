#include "hlie_app/hlie_process_manager.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

#ifdef _WIN32
#pragma comment(lib, "Advapi32.lib")
#endif

HLIE_NAMESPACE_BEGIN

namespace {
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
    const std::vector<std::string>& args, const std::vector<std::string>& model_names, const int instance_id) {
    std::lock_guard lock(mutex_);

#ifdef _WIN32
    // Windows: 构造命令行，程序路径用双引号包起来，防止路径里有空格
    std::stringstream ss;
    ss << quote_windows_argument(exe_path);
    for (const auto& arg : args) {
        ss << " " << quote_windows_argument(arg);
    }
    std::string cmd_line = ss.str();
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
        LOG_ERROR("Failed to CreateProcessA with cmd {}, error {} ({})", cmd_line, error, windows_error_message(error));
        return 0;
    }

    CloseHandle(pi.hThread);

    if (WaitForSingleObject(pi.hProcess, 100) == WAIT_OBJECT_0) {
        DWORD exit_code = 0;
        if (!GetExitCodeProcess(pi.hProcess, &exit_code)) {
            exit_code = GetLastError();
        }
        LOG_ERROR("Child process PID[{}] exited immediately with code {}, cmd {}", pi.dwProcessId, exit_code, cmd_line);
        CloseHandle(pi.hProcess);
        return 0;
    }

    LOG_INFO("Created child process PID[{}] instance_id[{}] cwd={} exe_dir={} cmd={}",
             pi.dwProcessId, instance_id, working_directory, exe_directory, cmd_line);

    ProcessInfo info;
    info.uuid = HlieUtil::uuid();
    info.process_handle = pi.hProcess;
    info.process_id = pi.dwProcessId;
    info.params = cmd_line;
    info.model_names = model_names;
    info.instance_id = instance_id;
    info.start_time = time(nullptr);
    info.is_running = true;

    processes_[info.process_id] = info;

    return info.process_id;

#else
    // Linux: fork + execvp
    pid_t pid = fork();
    if (pid < 0) {
        return 0;
    } else if (pid == 0) {
        // 子进程
        LOG_INFO("Child process, ready to run with path: {}", exe_path);
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(exe_path.c_str()));
        for (auto& arg : args) {
            LOG_INFO("arg: {}", arg);
            argv.push_back(const_cast<char*>(arg.c_str()));
        }

        argv.push_back(nullptr);
        const int error_code = execvp(exe_path.c_str(), argv.data());
        perror("execvp failed");
        if (error_code != 0) {
            LOG_ERROR("Child process, failed to run with path: {}", exe_path);
        }
        _exit(error_code);
    } else {
        // 父进程
        ProcessInfo info;
        info.pid = pid;
        info.uuid = HlieUtil::uuid();
        info.process_id = pid;
        info.model_names = model_names;
        info.instance_id = instance_id;
        info.start_time = time(nullptr);
        info.is_running = true;

        processes_[pid] = info;

        return pid;
    }
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

    // 等待子进程退出
    WaitForSingleObject(info.process_handle, 5000 /*ms*/);

    // 更新状态并清理句柄
    info.is_running = false;
    CloseHandle(info.process_handle);
    info.process_handle = nullptr;

    // 从表中移除
    processes_.erase(it);
    return true;

#else
    // 先发 SIGTERM，再等待一会；不行再 SIGKILL
    bool success = false;

    if (kill(info.pid, SIGTERM) == 0) {
        // 非阻塞等待
        for (int i = 0; i < 50; ++i) { // 总计 ~500ms
            int status = 0;
            pid_t r = waitpid(info.pid, &status, WNOHANG);
            if (r == info.pid) {
                success = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    if (!success) {
        // 强制杀
        kill(info.pid, SIGKILL);
        // 等到退出
        int status = 0;
        (void)waitpid(info.pid, &status, 0);
        success = true;
    }

    info.is_running = false;
    processes_.erase(it);
    return success;
#endif
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
            LOG_INFO("process {} has no models, consider terminating", pid);
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
            to_remove.push_back(kv.first);
            continue;
        }
        DWORD code = STILL_ACTIVE;
        if (!GetExitCodeProcess(info.process_handle, &code)) {
            const DWORD error = GetLastError();
            LOG_ERROR("Failed to query child process PID[{}] instance_id[{}], error {} ({})",
                      info.process_id, info.instance_id, error, windows_error_message(error));
            // 无法获取，认为已结束
            info.is_running = false;
            CloseHandle(info.process_handle);
            info.process_handle = nullptr;
            to_remove.push_back(kv.first);
        } else {
            info.is_running = (code == STILL_ACTIVE);
            if (!info.is_running) {
                LOG_ERROR("Child process PID[{}] instance_id[{}] exited with code {}, cmd {}",
                          info.process_id, info.instance_id, code, info.params);
                CloseHandle(info.process_handle);
                info.process_handle = nullptr;
                to_remove.push_back(kv.first);
            }
        }
    }
    for (auto pid : to_remove) {
        processes_.erase(pid);
    }
#else
    std::vector<ProcessID> to_remove;
    for (auto& kv : processes_) {
        auto& info = kv.second;
        int status = 0;
        pid_t r = waitpid(info.pid, &status, WNOHANG);
        if (r == 0) {
            // 仍在运行
            info.is_running = true;
        } else if (r == info.pid) {
            // 已退出
            info.is_running = false;
            to_remove.push_back(kv.first);
        } else {
            // waitpid 出错，认为已终止
            info.is_running = false;
            to_remove.push_back(kv.first);
        }
    }
    for (auto pid : to_remove) {
        processes_.erase(pid);
    }
#endif
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

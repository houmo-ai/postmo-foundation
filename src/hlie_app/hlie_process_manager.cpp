#include "hlie_app/hlie_process_manager.h"

#include <atomic>
#include <chrono>
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
    const std::vector<std::string>& args) {
    std::lock_guard lock(mutex_);

#ifdef _WIN32
    // Windows: 构造命令行，程序路径用双引号包起来，防止路径里有空格
    std::stringstream ss;
    ss << "\"" << exe_path << "\"";
    for (const auto& arg : args) {
        ss << " " << arg;
    }
    std::string cmd_line = ss.str();

    STARTUPINFOA si = { sizeof(si) };
    PROCESS_INFORMATION pi;

    const BOOL ok = CreateProcessA(
        nullptr,                 // lpApplicationName
        cmd_line.data(),         // lpCommandLine
        nullptr,                 // lpProcessAttrs
        nullptr,                 // lpThreadAttrs
        FALSE,                   // bInheritHandles
        CREATE_NEW_PROCESS_GROUP, // dwCreationFlags
        nullptr,                 // lpEnvironment
        nullptr,                 // lpCurrentDir
        &si,
        &pi
    );

    if (!ok) {
        LOG_ERROR("Failed to CreateProcessA with cmd {}", cmd_line);
        return 0;
    }

    CloseHandle(pi.hThread);

    ProcessInfo info;
    info.instance_id = HlieUtil::uuid();
    info.process_handle = pi.hProcess;
    info.process_id = pi.dwProcessId;
    info.params = cmd_line;
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
        info.process_id = pid;
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
            // 无法获取，认为已结束
            info.is_running = false;
            CloseHandle(info.process_handle);
            info.process_handle = nullptr;
            to_remove.push_back(kv.first);
        } else {
            info.is_running = (code == STILL_ACTIVE);
            if (!info.is_running) {
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

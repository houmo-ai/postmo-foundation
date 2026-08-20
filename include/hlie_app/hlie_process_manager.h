#ifndef HLIE_PROCESS_MANAGER_H
#define HLIE_PROCESS_MANAGER_H

#include <deque>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <spawn.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#endif

#include "hlie_base/hlie_base.h"
#include "hlie_core/domain/hlie_model_launch_ticket.h"

HLIE_NAMESPACE_BEGIN

#ifdef _WIN32
using ProcessID = DWORD;
#else
using ProcessID = pid_t;
#endif


class HLIEProcessManager {
public:
    struct ProcessInfo {
#ifdef _WIN32
        HANDLE process_handle;
#else
        pid_t pid;
#endif

        ProcessID process_id;
        std::string uuid;
        std::vector<std::string> model_names;
        std::vector<int> device_ids;
        int instance_id;
        std::string params;
        time_t start_time;
        bool is_running;
        bool expected_exit = false;
        HLIEModelLaunchTicket launch_ticket;
    };

    struct ProcessExitEvent {
        ProcessInfo process;
        int exit_code = -1;
    };

    ~HLIEProcessManager();

    ProcessID create_process(const std::string& exe_path, const std::vector<std::string>& args,
                             const std::vector<std::string>& model_names, int instance_id,
                             const HLIEModelLaunchTicket& launch_ticket = {});
    bool terminate_process(ProcessID pid);
    bool terminate_all();
    [[nodiscard]] std::optional<ProcessInfo> get_process(ProcessID pid);
    bool mark_expected_exit(ProcessID pid, bool expected = true);
    bool wait_for_exit(ProcessID pid, std::chrono::milliseconds timeout);
    std::vector<ProcessInfo> get_running_processes();
    void erase_model(const std::string& model_name);
    std::vector<std::string> get_working_models(int instance_id);
    void add_model(const std::string& model_name, int instance_id);
    void monitor_processes();
    std::vector<ProcessExitEvent> drain_exit_events();

private:
    std::unordered_map<ProcessID, ProcessInfo> processes_;
    std::deque<ProcessExitEvent> exit_events_;
    std::mutex mutex_;
    std::condition_variable process_cv_;

    void cleanup_terminated_processes();
};

HLIE_NAMESPACE_END

#endif // HLIE_PROCESS_MANAGER_H

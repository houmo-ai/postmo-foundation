#ifndef HLIE_PROCESS_MANAGER_H
#define HLIE_PROCESS_MANAGER_H

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include "httplib.h"
#include <windows.h>
#else
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#endif

#include "hlie_base/hlie_base.h"

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
        int instance_id;
        std::string params;
        time_t start_time;
        bool is_running;
    };

    ~HLIEProcessManager();

    ProcessID create_process(const std::string& exe_path, const std::vector<std::string>& args, const std::vector<std::string>& model_names, int instance_id);
    bool terminate_process(ProcessID pid);
    bool terminate_all();
    std::vector<ProcessInfo> get_running_processes();
    void monitor_processes();

private:
    std::unordered_map<ProcessID, ProcessInfo> processes_;
    std::mutex mutex_;

    void cleanup_terminated_processes();
};

HLIE_NAMESPACE_END

#endif // HLIE_PROCESS_MANAGER_H

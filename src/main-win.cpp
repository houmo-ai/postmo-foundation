#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

#include <windows.h>
#include <DbgHelp.h>
#include <shlobj.h>
#include "hlie_app/hlie_app_controller.h"
#include "hlie_base/hlie_config_service.h"


#pragma comment(lib, "dbghelp.lib")

namespace minidump {

    static std::string g_dumpDir;
    static std::atomic<bool> g_handling{false};

    // -------------------- Utility: Create directory recursively --------------------
    static void EnsureDirectory(const std::string& path) {
        SHCreateDirectoryExA(nullptr, path.c_str(), nullptr);
    }

    // -------------------- Utility: Generate timestamped filename --------------------
    static std::string GenerateDumpPath() {
        SYSTEMTIME st;
        GetLocalTime(&st);

        char buffer[MAX_PATH];
        snprintf(buffer, MAX_PATH, "%s\\crash_%04d%02d%02d_%02d%02d%02d_%lu.dmp", g_dumpDir.c_str(), st.wYear, st.wMonth, st.wDay, st.wHour,
                 st.wMinute, st.wSecond, GetCurrentProcessId());

        return std::string(buffer);
    }

    // -------------------- Core dump writer --------------------
    static void WriteDump(EXCEPTION_POINTERS* exceptionInfo) {
        if (g_handling.exchange(true)) {
            return; // prevent re-entry
        }

        std::string filePath = GenerateDumpPath();

        HANDLE hFile =
            CreateFileA(filePath.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

        if (hFile == INVALID_HANDLE_VALUE) {
            return;
        }

        MINIDUMP_EXCEPTION_INFORMATION mei;
        mei.ThreadId = GetCurrentThreadId();
        mei.ExceptionPointers = exceptionInfo;
        mei.ClientPointers = TRUE;

        MINIDUMP_TYPE dumpType = (MINIDUMP_TYPE)(MiniDumpWithDataSegs | MiniDumpWithHandleData | MiniDumpWithThreadInfo |
                                                 MiniDumpWithIndirectlyReferencedMemory);

        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, dumpType, exceptionInfo ? &mei : nullptr, nullptr, nullptr);

        FlushFileBuffers(hFile);
        CloseHandle(hFile);
    }

    // -------------------- Handlers --------------------
    static LONG WINAPI UnhandledHandler(EXCEPTION_POINTERS* exceptionInfo) {
        WriteDump(exceptionInfo);
        return EXCEPTION_EXECUTE_HANDLER;
    }

    static LONG CALLBACK VectoredHandler(PEXCEPTION_POINTERS exceptionInfo) {
        WriteDump(exceptionInfo);
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // -------------------- Initialization --------------------
    void Initialize(const std::string& customDir = "") {
        char path[MAX_PATH] = {0};

        if (!customDir.empty()) {
            g_dumpDir = customDir;
        } else {
            PWSTR wpath = nullptr;
            if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &wpath))) {
                char tmp[MAX_PATH];
                WideCharToMultiByte(CP_ACP, 0, wpath, -1, tmp, MAX_PATH, nullptr, nullptr);
                g_dumpDir = std::string(tmp) + "\\HLIECpp\\dumps";
                CoTaskMemFree(wpath);
            } else {
                GetTempPathA(MAX_PATH, path);
                g_dumpDir = std::string(path) + "HLIECpp\\dumps";
            }
        }

        EnsureDirectory(g_dumpDir);

        AddVectoredExceptionHandler(1, VectoredHandler);
        SetUnhandledExceptionFilter(UnhandledHandler);
    }

} // namespace minidump


// Windows SCM 全局状态句柄
SERVICE_STATUS_HANDLE g_StatusHandle = nullptr;
SERVICE_STATUS g_ServiceStatus = {0};

std::thread srv_t;

void ReportSvcStatus(const DWORD currentState, const DWORD win32ExitCode, const DWORD waitHint) {
    static DWORD checkPoint = 1;
    g_ServiceStatus.dwCurrentState = currentState;
    g_ServiceStatus.dwWin32ExitCode = win32ExitCode;
    g_ServiceStatus.dwWaitHint = waitHint;
    g_ServiceStatus.dwCheckPoint = (currentState == SERVICE_START_PENDING || currentState == SERVICE_STOP_PENDING) ? checkPoint++ : 0;
    SetServiceStatus(g_StatusHandle, &g_ServiceStatus);
}

hm_HLIE::HLIEWinAppController appController;

void WINAPI Handler(const DWORD controlCode) {
    switch (controlCode) {
    case SERVICE_CONTROL_STOP:
        ReportSvcStatus(SERVICE_STOP_PENDING, NO_ERROR, 0);
        appController.stop();
        break;
    default:
        break;
    }
}

int run() {
    // 1、初始化配置服务，保存参数
    int ret = appController.initialize();
    if (ret < 0) {
        std::cout << "ServiceWorkerThread appController failed to initialize:" << ret << std::endl;
        return -1;
    }

    ret = appController.run();
    std::cout << "ServiceWorkerThread appController run, exit code:" << ret << std::endl;
    return ret;
}

void ServiceWorkerThread() {
    run();
    ReportSvcStatus(SERVICE_STOPPED, NO_ERROR, 0);
}

void WINAPI ServiceMain(DWORD argc, LPTSTR* argv) {
    // 1. 注册控制处理函数
    g_StatusHandle = RegisterServiceCtrlHandler(TEXT("HoumoHI"), Handler);

    // 2. 初始化 SERVICE_STATUS
    g_ServiceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    g_ServiceStatus.dwControlsAccepted = SERVICE_ACCEPT_STOP;
    g_ServiceStatus.dwWin32ExitCode = NO_ERROR;
    g_ServiceStatus.dwServiceSpecificExitCode = 0;

    // 3. 报告“启动中”
    ReportSvcStatus(SERVICE_START_PENDING, NO_ERROR, 5000); // 最多等待 5 秒

    // 4. 启动后台线程
    srv_t = std::thread(ServiceWorkerThread);
    srv_t.detach();

    // 5. 报告“运行中”
    ReportSvcStatus(SERVICE_RUNNING, NO_ERROR, 0);
    // ServiceMain 返回，线程接管工作
}

int main(const int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "C");
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    _putenv("TCIM_LOG_TAGS=all");
    _putenv("TCIM_BACKEND=Xh1HdiBackend");
    _putenv("HM_PLATFORM=ASIC");

    minidump::Initialize();

    const auto& config_service = hm_HLIE::Singleton<hm_HLIE::HLIEConfigService>::instance();
    config_service->set_args(argc, argv);

    char exePath[MAX_PATH];
    GetModuleFileName(nullptr, exePath, MAX_PATH); // 获取exe绝对路径
    const auto programPath = std::string(exePath);
    const auto program_dir = std::string(exePath).substr(0, std::string(exePath).find_last_of('\\') + 1);
    _putenv(("TCIM_RUNTIME_PATH=" + program_dir + "/hal").c_str());

    if (std::getenv("HLIECPP_WIN_D") || config_service->get_args().find("--instance_id") != config_service->get_args().end()) {
        printf("========HLIECpp Instance & COPY Mode======== \n");
        return run();
    }

    char serviceName[] = "HLIECpp";
    SERVICE_TABLE_ENTRY ServiceTable[] = {{serviceName, ServiceMain}, {NULL, NULL}};
    if (!StartServiceCtrlDispatcher(ServiceTable)) {
        // 此处处理错误（如记录日志）
        const DWORD err = GetLastError();
        std::cerr << "HoumoHI WIN32 Service StartServiceCtrlDispatcher failed, ErrCode: " << err;

        return 8;
    }

    return 0;
}

#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

#include "hlie_base/hlie_config_service.h"
#include "hlie_app/hlie_app_controller.h"
#include <Windows.h>


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
    switch (controlCode)
    {
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
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    _putenv("TCIM_LOG_TAGS=all");
    _putenv("TCIM_BACKEND=Xh1HdiBackend");
    _putenv("HM_PLATFORM=ASIC");

    const auto& config_service = hm_HLIE::Singleton<hm_HLIE::HLIEConfigService>::instance();
    config_service->set_args(argc, argv);

#ifdef HOUMO_M50
    _putenv("TCIM_BACKEND=Xh2HalBackend");

    char exePath[MAX_PATH];
    GetModuleFileName(nullptr, exePath, MAX_PATH); // 获取exe绝对路径
    const auto programPath = std::string(exePath);
    const auto program_dir = std::string(exePath).substr(0, std::string(exePath).find_last_of('\\') + 1);
    _putenv(("TCIM_RUNTIME_PATH=" + program_dir + "/hal").c_str());
#endif


    if (std::getenv("HICPP_WIN_D") || config_service->get_args().find("--instance_id") != config_service->get_args().end()) {
        printf("========HLIECpp Instance & COPY Mode======== \n");
        return run();
    }

    char serviceName[] = "HLIECpp";
    SERVICE_TABLE_ENTRY ServiceTable[] = {{serviceName, ServiceMain}, {NULL, NULL}};
    if (!StartServiceCtrlDispatcher(ServiceTable))
    {
        // 此处处理错误（如记录日志）
        const DWORD err = GetLastError();
        std::cerr << "HoumoHI WIN32 Service StartServiceCtrlDispatcher failed, ErrCode: " << err;

        return 8;
    }

    return 0;
}

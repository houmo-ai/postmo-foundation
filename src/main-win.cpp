#include <cstdio>
#include <iostream>
#include "houmo-llm.h"

#ifdef _WIN32
#include <Windows.h>
#endif


#ifdef _WIN32

#include <thread>

// 全局状态句柄
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

void WINAPI Handler(const DWORD controlCode) {
    switch (controlCode)
    {
    case SERVICE_CONTROL_STOP:
        ReportSvcStatus(SERVICE_STOP_PENDING, NO_ERROR, 0);
        stop_server();
        break;
    default:
        break;
    }
}

void ServiceWorkerThread() {
    run_server();
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


#endif

int main(int argc, char** argv) {
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    #endif

#ifdef WINSERVICE_MODE
    // LOG(Info) << "WIN32 Service Mode";

    char serviceName[] = "HoumoHI";

    SERVICE_TABLE_ENTRY ServiceTable[] = {{serviceName, ServiceMain}, {NULL, NULL}};
    if (!StartServiceCtrlDispatcher(ServiceTable))
    {
        // 此处处理错误（如记录日志）
        const DWORD err = GetLastError();
        // LOG(Error) << "HoumoHI WIN32 Service StartServiceCtrlDispatcher failed, ErrCode: " << err;
    }

#else
    HANDLE instance = create_instance("./models/hmm_deepseek_14b_256_8k_4cores_20250514_26l/", nullptr);
    if (instance == nullptr) {
        printf("instance is null\n");
    }
    printf("instance 3 = %p\n", instance);

    int input;
    std::cin >> input;

    int ret = destroy_instance(instance);
    printf("ret = %d\n", ret);
    std::cin >> input;
    printf("exit = %d!\n", input);

#endif

    return 0;
}

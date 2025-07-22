#include <cstdio>
#include <iostream>
#include "houmo-llm.h"
#include <chrono>
#include <thread>
#ifdef _WIN32
#include <Windows.h>
#endif


#ifdef _WIN32


// 全局状态句柄
SERVICE_STATUS_HANDLE g_StatusHandle = nullptr;
SERVICE_STATUS g_ServiceStatus = {0};
#pragma comment(lib,"houmo-llm.lib")
extern  __declspec(dllimport) int global_argc;
extern  __declspec(dllimport) char**global_argv;

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
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    _putenv("TCIM_LOG_TAGS=all");
    _putenv("TCIM_BACKEND=Xh1HdiBackend");
    _putenv("HM_PLATFORM=ASIC");
    // 保存到全局变量
    global_argc = argc;
    global_argv = argv;

#ifdef HOUMO_M50
    _putenv("TCIM_BACKEND=Xh2HalBackend");
#endif


    const char* debug_mode = std::getenv("HICPP_WIN_D");
    if (debug_mode) {
        _putenv("TCIM_RUNTIME_PATH=./hal");
        printf(" _   _ ___   ____ ____  ____    ____       _                    ____                      _   \n"
               "| | | |_ _| / ___|  _ \\|  _ \\  |  _ \\  ___| |__  _   _  __ _   / ___|___  _ __  ___  ___ | | ___ \n"
               "| |_| || | | |   | |_) | |_) | | | | |/ _ \\ '_ \\| | | |/ _` | | |   / _ \\| '_ \\/ __|/ _ \\| |/ _ \n"
               "|  _  || | | |___|  __/|  __/  | |_| |  __/ |_) | |_| | (_| | | |__| (_) | | | \\__ \\ (_) | |  __/\n"
               "|_| |_|___(_)____|_|   |_|     |____/ \\___|_.__/ \\__,_|\\__, |  \\____\\___/|_| |_|___/\\___/|_|\\___|\n"
               "                                                       |___/                                     \n");
        run_server();
        return 0;
    }

#ifdef WINSERVICE_MODE
    // LOG(Info) << "WIN32 Service Mode";

#ifdef HOUMO_M50
    char exePath[MAX_PATH];
    GetModuleFileName(nullptr, exePath, MAX_PATH); // 获取exe绝对路径
    const auto programPath = std::string(exePath);
    const auto program_dir = std::string(exePath).substr(0, std::string(exePath).find_last_of('\\') + 1);
    _putenv(("TCIM_RUNTIME_PATH=" + program_dir + "/hal").c_str());
#endif


    char serviceName[] = "HoumoHI";
    SERVICE_TABLE_ENTRY ServiceTable[] = {{serviceName, ServiceMain}, {NULL, NULL}};
    if (!StartServiceCtrlDispatcher(ServiceTable))
    {
        // 此处处理错误（如记录日志）
        const DWORD err = GetLastError();
        // LOG(Error) << "HoumoHI WIN32 Service StartServiceCtrlDispatcher failed, ErrCode: " << err;
    }

#else
    const char* hi_modelpath = std::getenv("HICPP_MODEL_PATH");
    std::string hPath;
    if(hi_modelpath){
      hPath=hi_modelpath;
    } else {
        printf("No Env HICPP_MODEL_PATH Found, read default path:./models/qwen3-8b-1batch/4core_8k");
        hPath="./models/qwen3-8b-1batch/4core_8k";
    }
    std::cout << "hPath is: " << hPath << std::endl;
    HANDLE instance = create_instance(hPath, nullptr);
    if (instance == nullptr) {
        printf("instance is null\n");
    }
    printf("instance 3 = %p\n", instance);

    int input;
    while(true){
      std::cin >> input;
      if(input==12315) {
        printf("+++get exit code:%d\r\n",input);
        break;
      }
      else{
        printf("### Error get false exit code,input is:%d\r\n",input);
        // 睡眠 2 秒（精确到毫秒级）
        std::this_thread::sleep_for(std::chrono::seconds(2));
        continue;
      }
    }

    int ret = destroy_instance(instance);
    printf("ret = %d\n", ret);
    std::cin >> input;
    printf("exit = %d!\n", input);

#endif

    return 0;
}

#include <chrono>
#include <cstdio>
#include "hlie_app/hlie_app_controller.h"
#include "hlie_base/hlie_config_service.h"

#include <iostream>
#include <thread>
#include <unistd.h>
// #include "houmo-llm.h"

// extern char* hi_modelpath;
// extern char** global_argv;
// extern int global_argc;
int main(int argc, char** argv) {
    // 保存到全局变量
    // global_argc = argc;
    // global_argv = argv;
    if (geteuid() != 0) {
        std::cerr << "Error: Not running as root!" << std::endl;
        return 1;
    }

    std::cout << "Running as root." << std::endl;

    const auto& config_service = hm_HLIE::Singleton<hm_HLIE::HLIEConfigService>::instance();
    config_service->set_args(argc, argv);

    hm_HLIE::HLIELinuxAppController appController;
    std::cout << "11111." << std::endl;
    int ret = appController.initialize();
    std::cout << "22222." << std::endl;
    if (ret < 0) {
        std::cout << "HLIECpp appController failed to initialize:" << ret << std::endl;
        return ret;
    }

    ret = appController.run();
    std::cout << "HLIECpp appController run, exit code:" << ret << std::endl;
    return ret;


    /*

    //const char* hi_modelpath = std::getenv("HICPP_MODEL_PATH");
    std::string hPath;
    if(hi_modelpath){
      hPath=hi_modelpath;
      printf("Find cfg HICPP_MODEL_PATH:%s\r\n",hPath.c_str());
    } else {
        printf("No HICPP_MODEL_PATH cfg Found, read default path:/opt/hicpp/models/DeepSeek-R1-Distill-Qwen-7B-1Batch/4core_8k");
        hPath="/opt/hicpp/models/DeepSeek-R1-Distill-Qwen-7B-1Batch/4core_8k";
    }
    std::cout << "env HICPP_MODEL_PATH or default: " << hPath << std::endl;
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

    // int ret = destroy_instance(instance);
    printf("ret = %d\n", ret);
    std::cin >> input;
    printf("exit = %d!\n", input);*/
    return 0;
}

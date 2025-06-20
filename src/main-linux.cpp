#include <cstdio>
#include <iostream>
#include "houmo-llm.h"
#include <chrono>
#include <thread>


int main(int argc, char** argv) {
    const char* hi_modelpath = std::getenv("HICPP_MODEL_PATH");
    std::string hPath;
    if(hi_modelpath){
      hPath=hi_modelpath;
    } else {
        printf("No Env HICPP_MODEL_PATH Found, read default path:/opt/hicpp/models/DeepSeek-R1-Distill-Qwen-7B-1Batch/4core_8k");
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

    int ret = destroy_instance(instance);
    printf("ret = %d\n", ret);
    std::cin >> input;
    printf("exit = %d!\n", input);
    return 0;
}

#include "houmo-llm.h"
#include "houmo-llm.hpp"
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <iostream>


int main(int argc, char ** argv) {

    //signal(SIGSEGV, handler);   // install our handler
    //signal(SIGABRT, handler);   // install our handler
    
    //HANDLE instance = create_instance("./models/hmm_qwen2_256_4096_4cores_20250317/");
//
    //
//
    //if (instance == nullptr) {
    //    printf("instance is null\n");
    //}
    //int ret = generate(instance);
    //ret = generate_stream(instance);
    //ret = generate_stream_next_token_all(instance);
//
    //printf("instance 1 = %p\n", instance);
    //ret = destroy_instance(instance);
    //printf("ret = %d\n", ret);
//

    //instance = create_instance("./models/hmm_qwen2_256_4096_4cores_20250317/");
    HANDLE instance = create_instance("./models/hmm_deepseek_256_4096_4cores_20250317/");
    if (instance == nullptr) {
        printf("instance is null\n");
    }
    printf("instance 3 = %p\n", instance);

    //ret = generate(instance);
//
    //ret = generate_stream(instance);
    //while (true) {
    //    std::string result_json_str;
    //    ret = generate_stream_next_token_string(instance, result_json_str);
    //    if (ret == 1) {
    //        printf("1 stop found str: %s\n", result_json_str.c_str());
    //        break;
    //    } else if (ret == -1) {
    //        printf("1 Error: %d\n", ret);
    //        break;
    //    } else {
    //        printf("1 content str: %s\n", result_json_str.c_str());
    //    }
    //}
    //
//
    //ret = generate_stream(instance);
    //while (true) {
    //    size_t size;
    //    char * result;
    //    printf("new round: %d\n", ret);
    //    ret = generate_stream_next_token(instance, &result, &size);
    //    printf("ret 2: %d\n", ret);
    //    if (ret == 1) {
    //        printf("2 stop found str: %s\n", result);
    //        break;
    //    } else if (ret == -1) {
    //        printf("2 Error: %d\n", ret);
    //        break;
    //    } else {
    //        printf("2 content str: %s\n", result);
    //    }
    //}
//
    //ret = generate_stream(instance);
    //while (true) {
    //    std::string result_json_str;
    //    ret = generate_stream_next_token_string(instance, result_json_str);
    //    if (ret == 1) {
    //        printf("3 stop found str: %s\n", result_json_str.c_str());
    //        break;
    //    } else if (ret == -1) {
    //        printf("3 Error: %d\n", ret);
    //        break;
    //    } else {
    //        printf("3 content str: %s\n", result_json_str.c_str());
    //    }
    //}
//
    //ret = generate_stream(instance);
    //while (true) {
    //    size_t size;
    //    char * result;
    //    ret = generate_stream_next_token(instance, &result, &size);
    //    printf("ret 4: %d\n", ret);
    //    if (ret == 1) {
    //        printf("4 c stop found str: %s\n", result);
    //        break;
    //    } else if (ret == -1) {
    //        printf("4 c Error: %d\n", ret);
    //        break;
    //    } else {
    //        printf("4 c content str: %s\n", result);
    //    }
    //}

    int input;
    std::cin >> input;
    
    int ret = destroy_instance(instance);
    printf("ret = %d\n", ret);
    std::cin >> input;
    printf("exit = %d!\n", input);

    return 0;
}

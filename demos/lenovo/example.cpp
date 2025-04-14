
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <http_server_c_interface_3.h>
#include <iostream>
#include <regex>
#include <algorithm>

#ifdef WIN32
#include <windows.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

#include <limits>
#include <ios>

void SetStdinEcho(bool enable = true)
{
#ifdef WIN32
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE); 
    DWORD mode;
    GetConsoleMode(hStdin, &mode);

    if( !enable )
        mode &= ~ENABLE_ECHO_INPUT;
    else
        mode |= ENABLE_ECHO_INPUT;

    SetConsoleMode(hStdin, mode );

#else
    struct termios tty;
    tcgetattr(STDIN_FILENO, &tty);
    if( !enable )
        tty.c_lflag &= ~ECHO;
    else
        tty.c_lflag |= ECHO;

    (void) tcsetattr(STDIN_FILENO, TCSANOW, &tty);
#endif
}

int main(int argc, char ** argv) {
    params params;
    params.n_predict = 300;
    params.n_ctx = 4096;
    size_t params_bytes = 0;
    HANDLE instance = create_api_instance(&params, params_bytes);

    if (instance == nullptr) {
        printf("instance is null\n");
    }

    const char* model_path = "./qwen2-7b.gguf";
    const char* model_names[1] = {"qwen2-7b"};
    size_t model_name_count = 1;
    hardware_info hwinfo;
    int status = api_status(instance);
    printf("api_status: %d\n", status);

    bool state = api_loadmodel_by_path(instance, model_path, model_names, model_name_count, hwinfo);
    printf("api_token_num = %d\n", state);
    status = api_status(instance);
    printf("api_status: %d\n", status);

    const char* token_fullpath = "./qwen2-7b.gguf";

    state = api_loadtoken(instance, token_fullpath);
    printf("api_loadtoken = %d\n", state);
    
    int ret = api_token_num(instance, "规整化后的文本要忠于原始会议记录，保留会议记录中全部的有用信息；");
    printf("api_token_num = %d\n", ret);


    char* prompt = "你在做啥";
    char* result;
    size_t result_size;

    status = api_Generate_status(instance);
    printf("api_Generate_status: %d\n", status);


    ret = api_Generate(instance, prompt, &params, params_bytes, &result, &result_size);
    printf("api_Generate = %d\n", ret);
    
    printf("result: %s\n", result);
    printf("result_size: %d\n", result_size);

    status = api_Generate_status(instance);
    printf("api_Generate_status: %d\n", status);

    status = api_status(instance);
    printf("api_status: %d\n", status);
    
    char* prompt2 = "在北京生活有什么要注意的？";
    char* result2;
    size_t result_size2;
    ret = api_Generate_stream(instance, prompt2, &params, params_bytes);
    printf("api_Generate_stream = %d\n", ret);

    status = api_Generate_status(instance);
    printf("api_Generate_status: %d\n", status);

    status = api_status(instance);
    printf("api_status: %d\n", status);

    int count = 0;
    if (ret) {
        while (true) {
            if (count > 10) {
                api_stop(instance);
                status = api_Generate_status(instance);
                printf("api_Generate_status: %d\n", status);
                status = api_status(instance);
                printf("api_status: %d\n", status);
            }
            ret = api_Get_Generate_token(instance, &result2, &result_size2);
            status = api_Generate_status(instance);
            printf("api_Generate_status: %d\n", status);
            if (ret == 1) {
                printf("stop found str: %s\n", result2);
                break;
            } else if (ret == -1) {
                printf("Error: %d\n", ret);
                break;
            } else {
                printf("content str: %s\n", result2);
            }
            count = count + 1;
        }
    }
    int input;
    //std::cin >> input;

    //ret = api_Generate(instance, prompt, &params, params_bytes, &result, &result_size);
    //printf("api_Generate 2  = %d\n", ret);
    //
    //printf("result 2: %s\n", result);
    //printf("result_size 2: %d\n", result_size);
//
    //status = api_Generate_status(instance);
    //printf("api_Generate_status 2: %d\n", status);
    status = api_status(instance);
    printf("api_status: %d\n", status);

    api_unloadmodel(instance);
    status = api_status(instance);
    printf("api_status: %d\n", status);
    {

        bool state = api_loadmodel_by_path(instance, model_path, model_names, model_name_count, hwinfo);
        printf("api_token_num = %d\n", state);

        const char* token_fullpath = "./qwen2-7b.gguf";

        state = api_loadtoken(instance, token_fullpath);
        printf("api_loadtoken = %d\n", state);
        
        int ret = api_token_num(instance, "规整化后的文本要忠于原始会议记录，保留会议记录中全部的有用信息；");
        printf("api_token_num = %d\n", ret);


        char* prompt = "有什么南京特产推荐么";
        char* result;
        size_t result_size;

        int status = api_Generate_status(instance);
        printf("api_Generate_status: %d\n", status);


        ret = api_Generate(instance, prompt, &params, params_bytes, &result, &result_size);
        printf("api_Generate = %d\n", ret);
        
        printf("result: %s\n", result);
        printf("result_size: %d\n", result_size);

        status = api_Generate_status(instance);
        printf("api_Generate_status: %d\n", status);
        
        char* prompt2 = "为啥说三人行必有我师";
        char* result2;
        size_t result_size2;
        ret = api_Generate_stream(instance, prompt2, &params, params_bytes);
        printf("api_Generate_stream = %d\n", ret);

        status = api_Generate_status(instance);
        printf("api_Generate_status: %d\n", status);
        status = api_status(instance);
        printf("api_status: %d\n", status);
        int count = 0;
        if (ret) {
            while (true) {
                ret = api_Get_Generate_token(instance, &result2, &result_size2);
                status = api_Generate_status(instance);
                printf("api_Generate_status: %d\n", status);
                if (ret == 1) {
                    printf("stop found str: %s\n", result2);
                    break;
                } else if (ret == -1) {
                    printf("Error: %d\n", ret);
                    break;
                } else {
                    printf("content str: %s\n", result2);
                }
                count = count + 1;
            }
        }
        status = api_status(instance);
        printf("api_status: %d\n", status);
    }
    SetStdinEcho(true);
    while (true) {
        std::cin.clear();
        std::cin.sync();
        //std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        std::string prompt;
        std::cout << std::endl;
        std::cout << "输入问题(CTRL + C退出): ";
        //std::cin >> prompt;
        getline(std::cin,prompt);
        printf("\n你的问题: %s\n", prompt.c_str());
        char* result2;
        size_t result_size2;
        ret = api_Generate_stream(instance, prompt.c_str(), &params, params_bytes);
        //printf("\napi_Generate_stream = %d\n", ret);

        //status = api_Generate_status(instance);
        //printf("api_Generate_status: %d\n", status);
        //status = api_status(instance);
        //printf("api_status: %d\n", status);
        std::cout << "回答: " << std::endl;
        if (ret) {
            while (true) {
                ret = api_Get_Generate_token(instance, &result2, &result_size2);
                //status = api_Generate_status(instance);
                //printf("api_Generate_status: %d\n", status);
                if (ret == 1) {
                    //printf("stop found str: %s\n", result2);
                    break;
                } else if (ret == -1) {
                    printf("Error: %d\n", ret);
                    break;
                } else {
                    //printf("content str: %s\n", result2);
                    std::smatch results;
                    std::string result = result2;
                    if (std::regex_search(result, results, std::regex(".*\"content\":\"(.*?)\".*"))) {
                        std::string content = results[1].str();
                        content = std::regex_replace(content, std::regex(R"(\\n)"), "\n");
                        //std::string::size_type index = 0;
                        //while ((index = content.find("\\n", index)) != std::string::npos) {
                        //    content.replace(index, 2, "\n");
                        //    ++index;
                        //}
                        std::cout << content << std::flush;
                    }
                }
            }
            std::cout << std::endl;
        }
        //status = api_status(instance);
        //printf("api_status: %d\n", status);
    }

    api_unloadmodel(instance);
    status = api_status(instance);
    printf("api_status: %d\n", status);


    destroy_api_instance(instance);

    printf("exit!\n");

    return 0;
}

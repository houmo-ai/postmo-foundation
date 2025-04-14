
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
#include <chrono>
#include <thread>
#endif

#include <functional>
#include <chrono>
#include <future>
#include <cstdio>
#include <limits>
#include <ios>

#include "utils/include/utf8_console.h"

void sleep_seconds(int time) {
#ifdef WIN32
    Sleep(time);
#else
    std::this_thread::sleep_for(std::chrono::milliseconds(time));
#endif
}

class CallBackTimer
{
public:
    CallBackTimer(HANDLE handle)
    :_execute(false),instance(handle)
    {
        printf("instance 1: %p\n", instance);
    }

    CallBackTimer& operator=(const CallBackTimer& c)
	{
		instance = c.instance;
		return *this;
	}

    ~CallBackTimer() {
        if( _execute.load(std::memory_order_acquire) ) {
            stop();
        };
    }

    void stop()
    {
        _execute.store(false, std::memory_order_release);
        if( _thd.joinable() )
            _thd.join();
    }

    void start(int interval, std::function<int32 HTTP_SERVER_API (HANDLE handle, char** result, size_t* result_size)> func)
    {
        if( _execute.load(std::memory_order_acquire) ) {
            stop();
        };
        _execute.store(true, std::memory_order_release);
        _thd = std::thread([this, interval, func]()
        {
            int ret = 0;
            int status = 0;
            int count = 0;
            size_t result_len = 0;
            while (_execute.load(std::memory_order_acquire)) {
                count = count + 1;
                printf("count: %d\n", count);
                printf("instance point: %p\n", this->instance);
                status = api_Generate_status(this->instance);
                printf("timer api_Generate_status: %d\n", status);
                if (status == 0) {
                    printf("stop found str: %s\n", result3);
                    return;
                } 
                do {
                    ret = api_Get_Generate_token(this->instance, nullptr, &result_len);
                    printf("ret: %d\n", ret);
                    if (ret != E_BUFFER_NOT_ENOUGH) {
                        sleep_seconds(100);
                    }
                } while(ret != E_BUFFER_NOT_ENOUGH);
                char * result_c = new char [result_len];
                ret = api_Get_Generate_token(this->instance, &result_c, &result_len);
                printf("ret: %d\n", ret);
                if (ret != E_OK) {
                    printf("ret wrong: %d\n", ret);
                    return;
                }
                if (ret == 2) {
                    
                    printf("timer data not ready: %d\n", ret);
                } else if (ret == -1) {
                    printf("Error: %d\n", ret);
                    return;
                } else {
                    printf("timer content str: %s\n", result_c);
                }                 
                std::this_thread::sleep_for(
                std::chrono::milliseconds(interval));
            }
        });
    }

    bool is_running() const noexcept {
        return ( _execute.load(std::memory_order_acquire) && 
                 _thd.joinable() );
    }

private:
    HANDLE instance;

    std::atomic<bool> _execute;
    std::thread _thd;
    char* result3 = nullptr;
    size_t result_size3 = 0;
};

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
    initialize_utf8_console();
    params params;
    params.n_predict = 300;
    params.n_ctx = 4096;
    size_t params_bytes = 0;
    HANDLE instance = create_api_instance(&params, params_bytes);

    if (instance == nullptr) {
        printf("instance is null\n");
    }

    const char* model_path = "./models/hmm_qwen2_256_4096_4cores_20250317/";
    //const char* token_fullpath = "./models/hmm_qwen2_256_4096_4cores_20250317/qwen2-1_5b-instruct-q4_0.gguf";

    const char* model_names[1] = {"qwen2-7b"};
    size_t model_name_count = 1;
    hardware_info hwinfo;
    int status = api_status(instance);
    printf("api_status: %d\n", status);

    bool state = api_loadmodel_by_path(instance, model_path, model_names, model_name_count, hwinfo);
    printf("api_token_num = %d\n", state);
    status = api_status(instance);
    printf("api_status: %d\n", status);

    //state = api_loadtoken(instance, token_fullpath);
    //printf("api_loadtoken = %d\n", state);
    
    int ret = api_token_num(instance, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n${history}<|im_start|>user\nThe following information is from your knowledgebase, it might be helpful to answer the question:\n---\n\n---\nPlease remember the above knowledgebase info.<|im_end|>\n<|im_start|>assistant\nYes, I have remembered these knowledgebase information, is there anything I can do for you?<|im_end|>\n${history}<|im_start|>user\nQuestion: hello\nYou MUST answer the question ONLY using the relevant information in Knowledgebase. Avoid generating hallucinations.<|im_end|>\n<|im_start|>assistant\n");
    printf("api_token_num = %d\n", ret);

    char* prompt2 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n${history}<|im_start|>user\nThe following information is from your knowledgebase, it might be helpful to answer the question:\n---\n\n---\nPlease remember the above knowledgebase info.<|im_end|>\n<|im_start|>assistant\nYes, I have remembered these knowledgebase information, is there anything I can do for you?<|im_end|>\n${history}<|im_start|>user\nQuestion: hello\nYou MUST answer the question ONLY using the relevant information in Knowledgebase. Avoid generating hallucinations.<|im_end|>\n<|im_start|>assistant\n";
    char * * result2 = nullptr;
    size_t result_size2 = 0;
    ret = api_Generate_stream(instance, prompt2, &params, params_bytes);
    printf("api_Generate_stream = %d\n", ret);

    status = api_Generate_status(instance);
    printf("api_Generate_status: %d\n", status);
    status = api_status(instance);
    printf("api_status: %d\n", status);
    int count = 0;
    if (ret) {
        while (true) {
            
             count = count + 1;
            //if (count > 10) {
            //    break;
            //}
            ret = api_Get_Generate_token(instance, result2, &result_size2);
            //printf("result_size2: %d\n", result_size2);
            
            status = api_Generate_status(instance);
            //printf("api_Generate_status: %d\n", status);
            if (status == 0) {
                //printf("stop found str: %s\n", result2);
                break;
            } else if (ret == 2) {
                sleep_seconds(10);
                printf("data not ready: %d\n", ret);
                continue;
            } else if (ret == -1) {
                //printf("Error: %d\n", ret);
                break;
            } else if (ret == 1) {
                printf("not enough: %d\n", result_size2);
                if (result2 && *result2) {
                    delete *result2;
                }
                
                char * result = new char [result_size2];
                result2 = &result;
                continue;
            } else {
                printf("content str: %s\n", *result2);
            }
        }
    }
    status = api_status(instance);
    printf("api_status: %d\n", status);

    char* prompt3 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n以下信息来自你的知识库，可能对回答某些问题有帮助：\n\n请记住以上知识库的信息。<|im_end|>\n<|im_start|>assistant\n好的，我已经记住了这些知识库的信息，请问有什么我可以帮助你解答的问题吗？<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nHello! How can I assist you today?<|im_end|>\n<|im_start|>user\n问题:给我详细讲讲红楼梦吧\n请根据相关资料中有用的信息作为依据回答问题。\n回答一定要忠于原文，不要创造答案; 回答简洁而口语化，语义流畅而连贯。<|im_end|>\n<|im_start|>assistant\n";
    char* result3 = nullptr;
    size_t result_size3 = 0;
    ret = api_Generate_stream(instance, prompt3, &params, params_bytes);
    printf("api_Generate_stream = %d\n", ret);

    status = api_Generate_status(instance);
    printf("api_Generate_status: %d\n", status);
    status = api_status(instance);
    printf("api_status: %d\n", status);
    count = 0;
    
    if (ret) {
        //bad:
        //CallBackTimer t = new CallBackTimer(instance);
        //t.start(100, api_Get_Generate_token);

        //CallBackTimer * t = new CallBackTimer(instance);
        //t->start(100, api_Get_Generate_token);

        CallBackTimer t(instance);
        t.start(100, api_Get_Generate_token);
        printf("CallBackTimer started: %d\n", 1);
        int input;
        std::cin >> input;
        printf("CallBackTimer input: %d\n", input);
        //while (true) {
        //    sleep_milliseconds(1);
        //    count = count + 1;
        //    //if (count > 10) {
        //    //    break;
        //    //}
        //    ret = api_Get_Generate_token(instance, &result3, &result_size3);
        //    //printf("result_size3: %d\n", result_size3);
        //    
        //    status = api_Generate_status(instance);
        //    printf("api_Generate_status: %d\n", status);
        //    if (status == 0) {
        //        //printf("stop found str: %s\n", result3);
        //        break;
        //    } else if (ret == 2) {
        //        
        //        printf("data not ready: %d\n", ret);
        //        continue;
        //    } else if (ret == -1) {
        //        //printf("Error: %d\n", ret);
        //        break;
        //    } else if (ret == 1) {
        //        printf("not enough: %d\n", result_size3);
        //        if (result3) {
        //            delete result3;
        //        }
        //        
        //        result3 = new char [result_size3];
        //        continue;
        //    } else {
        //        printf("content str: %s\n", result3);
        //    }
        //   
        //    
        //}
    }
    status = api_status(instance);
    printf("api_status: %d\n", status);

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
        char* result2 = new char [1024];
        size_t result_size2 = 1024;
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
                status = api_Generate_status(instance);
                //printf("api_Generate_status: %d\n", status);
                if (status == 0) {
                    //printf("stop found str: %s\n", result2);
                    break;
                } else if (ret == 2) {
                    sleep_seconds(100);
                    //printf("data not ready: %d\n", ret);
                    continue;
                } else if (ret == -1) {
                    //printf("Error: %d\n", ret);
                    break;
                } else if (ret == 1) {
                    //printf("not enough: %d\n", result_size2);
                    if (result2) {
                        delete result2;
                    }
                    result2 = new char [result_size2];
                    continue;
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


#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <http_server_c_interface_3.h>
#include <iostream>
#include <regex>
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>


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
                    printf("stop found\n");
                    return;
                }
                do {
                    ret = api_Get_Generate_token(this->instance, nullptr, &result_len);
                    printf("ret: %d\n", ret);
                    if (ret != E_BUFFER_NOT_ENOUGH) {
                        sleep_seconds(100);
                    }
                } while(ret != E_BUFFER_NOT_ENOUGH);
                //char * result_c = new char [result_len + 1];
                //result_c[result_len] = '\0';
                char * result_c = new char [result_len];
                printf("result_c: %p\n", result_c);
                printf("&result_c: %p\n", &result_c);
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
                    delete result_c;
                    result_c = nullptr;
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

int test_qwen_template_revert();

template <typename T>
void save_raw_data_1(const T* data, size_t count, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        throw std::runtime_error("无法打开文件: " + filename);
    }
    uint64_t file_size= count * sizeof(T);
    printf("++data addr is %p , write file_size is %llu\r\n",data,file_size);
    outfile.write(reinterpret_cast<const char*>(data), file_size);
    outfile.close();
}


std::ifstream::pos_type filesize(const char* filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

std::vector<char>  read_whole_file(const std::string& filename,uint64_t & modelfile_size) {
    namespace fs = std::filesystem;

    std::filesystem::path p{filename.c_str()};

    std::cout << "The size of " << p.u8string() << " is " <<
    std::filesystem::file_size(p) << " bytes.\n";

    //uintmax_t file_size = fs::file_size(filename);
    uintmax_t file_size=std::filesystem::file_size(p);
    printf("+++modelfile:%s  size is: %llu\r\n",filename.c_str(),file_size);
    if (file_size > SIZE_MAX) {
        throw std::runtime_error("文件太大，无法一次性读取");
    }

    modelfile_size=file_size;

    std::vector<char> buffer(file_size);
    std::ifstream file(filename, std::ios::binary);

    if (!file.read(buffer.data(), file_size)) {
        throw std::runtime_error("读取文件失败");
    }
    printf("++buffer.data() is: %p \r\n",buffer.data());
    return std::move(buffer);
    // 处理完整文件数据...
    // process_data(buffer.data(), buffer.size());
}

int main(int argc, char ** argv) {

    //test_qwen_template_revert();
    //printf("++Attention:only test_qwen_template_revert\r\n");
    //return 0;
    int testcase_id=0;
    if (argc != 2) {
        std::cerr << "Error:Missing param. No inter testcase_id specified!" << std::endl;
        std::cerr << "Usage: " << argv[0] << " testcase_id" << std::endl;
        std::cerr << "testcase_id should be following  number:" << std::endl;
        std::cerr << "1: test qwen3 8b" << std::endl;
        std::cerr << "2: test qwen2 7B" << std::endl;
        std::cerr << "3: test qwen2 14B" << std::endl;
        std::cerr << "4: test load llm model from membuffer:" << std::endl;
        return -1;
    } else{
        std::string arg = argv[1];
        testcase_id = std::stoi(arg);
    }
    std::cout<<"++testcase_id is:"<<testcase_id<<std::endl;
    initialize_utf8_console();
    params params;
    params.n_predict = 300;
    params.n_ctx = 4096;
    //后采样参数，建议参考默认值，在多样性和正确性之间保持平衡
    params.top_k = 40;
    params.top_p = 0.9;
    params.temperature = 1.1f;
    params.repeat_penalty = 1.05;
    params.repeat_last_n = 60;

    size_t params_bytes = 0;



    //const char* token_fullpath = "./models/hmm_qwen2_256_4096_4cores_20250317/qwen2-1_5b-instruct-q4_0.gguf";

    const char* model_names[1] = {"qwen2-7b"};
    size_t model_name_count = 1;
    hardware_info hwinfo;
    HANDLE instance = create_api_instance(&params, params_bytes);

    if (instance == nullptr) {
        printf("instance is null\n");
    }

    int status = api_status(instance);
    printf("api_status: %d\n", status);
    bool state=false;
    char* model_path=nullptr;
    switch(testcase_id) {
        case 1: //测试qwen3 从文件加载模型
            model_path = "./models/qwen3-8b-1batch/4core_8k";
            state = api_loadmodel_by_path(instance, model_path, model_names, model_name_count, hwinfo);
            printf("api_loadmodel_by_path = %d\n", state);
            break;
        case 2: //测试14B 模型
            model_path = "./models/hmm_deepseek_14b_256_8k_4cores_20250514_26l";
            state = api_loadmodel_by_path(instance, model_path, model_names, model_name_count, hwinfo);
            printf("api_loadmodel_by_path = %d\n", state);
            break;
        case 3: //测试qwen2 从文件加载模型
            model_path = "./models/hmm_deepseek_256_4096_4cores_20250317/";
            state = api_loadmodel_by_path(instance, model_path, model_names, model_name_count, hwinfo);
            printf("api_loadmodel_by_path = %d\n", state);
            break;
        case 4:  //测试从mem加载模型
            {
            uint8_t *buffers[2]={nullptr};
            constexpr size_t buffers_count=2;
            uint64_t model_filesizes[buffers_count]={0};
            std::string prefill_model_path="./models/hmm_deepseek_256_4096_4cores_20250317/deepseek_prefill.hmm";
            //std::string prefill_model_path="./models/hmm_deepseek_256_4096_4cores_20250317/yolov5m.hmm";
            std::vector<char> prefill_model=read_whole_file(prefill_model_path,model_filesizes[0]);
            std::string decode_model_path="./models/hmm_deepseek_256_4096_4cores_20250317/deepseek_decode.hmm";
            std::vector<char> decode_model=read_whole_file(decode_model_path,model_filesizes[1]);

            buffers[0]=(uint8_t *)prefill_model.data();
            buffers[1]=(uint8_t *)decode_model.data();
            printf("++buffers[0] is %p\r\n",buffers[0]);
            printf("++buffers[1] is %p\r\n",buffers[1]);

            //如果需要保存模型文件对比原始文件可以用下面的方法
            //save_raw_data_1(buffers[0],model_filesizes[0],"prefill_from_vectordata.hmm");
            for(int i=0;i<buffers_count;i++) {
            printf("++model index:%d model_filesizes is %llu\r\n",i,model_filesizes[i]);
            }
            //return 0;
            state = api_loadmodel_by_buffer(instance, buffers, model_filesizes, buffers_count, hwinfo);
            printf("api_loadmodel_by_buffer = %d\n", state);
            break;
            }
        default:
            std::cerr <<"++Error not supported testcase_id"<<std::endl;
            exit(-1);
    }
    status = api_status(instance);
    printf("-api_status: %d\n", status);

    //state = api_loadtoken(instance, token_fullpath);
    //printf("api_loadtoken = %d\n", state);

    int ret = api_token_num(instance, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n以下信息来自你的知识库，可能对回答某些问题有帮助：\n\n请记住以上知识库的信息。<|im_end|>\n<|im_start|>assistant\n好的，我已经记住了这些知识库的信息，请问有什么我可以帮助你解答的问题吗？<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\nAlright, the user greeted me with \"hello.\" I need to respond politely. Since the knowledgebase is empty, I can't provide specific information. I should acknowledge their greeting and let them know I'm here to help. Keeping it friendly and open-ended makes sense.\n</think>\n\nHello! How can I assist you today?<|im_end|>\n<|im_start|>user\n问题:你好\n请根据相关资料中有用的信息作为依据回答问题。\n回答一定要忠于原文，不要创造答案; 回答简洁而口语化，语义流畅而连贯。<|im_end|>\n<|im_start|>assistant\n");
    printf("api_token_num = %d\n", ret);
    status = api_status(instance);
    printf("api_status: %d\n", status);

    ret = api_token_num(instance, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n${history}<|im_start|>user\nThe following information is from your knowledgebase, it might be helpful to answer the question:\n---\n\n---\nPlease remember the above knowledgebase info.<|im_end|>\n<|im_start|>assistant\nYes, I have remembered these knowledgebase information, is there anything I can do for you?<|im_end|>\n${history}<|im_start|>user\nQuestion: hello\nYou MUST answer the question ONLY using the relevant information in Knowledgebase. Avoid generating hallucinations.<|im_end|>\n<|im_start|>assistant\n");
    printf("api_token_num = %d\n", ret);
    status = api_status(instance);
    printf("api_status: %d\n", status);

    char* prompt2 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n${history}<|im_start|>user\nThe following information is from your knowledgebase, it might  //helpful to answer the question:\n---\n\n---\nPlease remember the above knowledgebase info.<|im_end|>\n<|im_start|>assistant\nYes, I have remembered these owledgebase //information, is there anything I can do for you?<|im_end|>\n${history}<|im_start|>user\nQuestion: hello\nYou MUST answer the question ONLY using the levant information //in Knowledgebase. Avoid generating hallucinations.<|im_end|>\n<|im_start|>assistant\n";

    ret = api_Generate_stream(instance, prompt2, &params, params_bytes);
    printf("api_Generate_stream = %d\n", ret);
//
    status = api_Generate_status(instance);
    printf("api_Generate_status: %d\n", status);
    status = api_status(instance);
    printf("api_status: %d\n", status);
    int count = 0;

    #if 1
    if (ret) {
        char * * result2 = nullptr;
        size_t result_size2 = 0;
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
                printf("stop found str: %s\n", result2);
                break;
            } else if (ret == 2) {
                sleep_seconds(10);
                printf("data not ready: %d\n", ret);
                continue;
            } else if (ret == -1) {
                printf("Error: %d\n", ret);
                break;
            } else if (ret == 1) {
                printf("not enough: %d\n", result_size2);
                if (result2 && *result2) {
                    delete *result2;
                    result2 = nullptr;
                }
                result_size2 = 0;

                char * result = new char [result_size2];
                printf("result: %p\n", result);
                result2 = &result;
                printf("result2: %p\n", result2);
                continue;
            } else {
                printf("content str: %s\n", *result2);
                if (result2 && *result2) {
                    delete *result2;
                    result2 = nullptr;
                }
                result_size2 = 0;
            }
        }
    }
    #endif
//
    status = api_Generate_status(instance);
    printf("api_Generate_status 1: %d\n", status);
//
    status = api_Generate_status(instance);
    printf("api_Generate_status 2: %d\n", status);
//
//
    status = api_status(instance);
    printf("api_status 1: %d\n", status);
//
    status = api_status(instance);
    printf("api_status 2: %d\n", status);
//
    status = api_status(instance);
    printf("api_status 3: %d\n", status);
//
    status = api_status(instance);
    printf("api_status 4: %d\n", status);
//
    ret = api_token_num(instance, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n${history}<|im_start|>user\nThe following information is from your //owledgebase, it might be helpful to answer the question:\n---\n\n---\nPlease remember the above knowledgebase info.<|im_end|>\n<|im_start|>assistant\nYes, I have //membered these knowledgebase information, is there anything I can do for you?<|im_end|>\n${history}<|im_start|>user\nQuestion: hello\nYou MUST answer the question ONLY //ing the relevant information in Knowledgebase. Avoid generating hallucinations.<|im_end|>\n<|im_start|>assistant\n");
    printf("api_token_num = %d\n", ret);
    status = api_status(instance);
    printf("api_status: %d\n", status);
//
    ret = api_token_num(instance, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n以下信息来自你的知识库，可能对回答某些问题有帮助：\n\n请记住以上知识库的信息。<|im_end|>\n<|im_start|>assistant\n好的，我已经记住了这些知识库的信息，请问有什么我可以帮助你解答的问题吗？<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\nAlright, the user greeted me with \"hello.\" I need to respond politely. Since the knowledgebase is empty, I can't provide specific information. I should acknowledge their greeting and let them know I'm here to help. Keeping it friendly and open-ended makes sense.\n</think>\n\nHello! How can I assist you today?<|im_end|>\n<|im_start|>user\n问题:你好\n请根据相关资料中有用的信息作为依据回答问题。\n回答一定要忠于原文，不要创造答案; 回答简洁而口语化，语义流畅而连贯。<|im_end|>\n<|im_start|>assistant\n");
    printf("api_token_num = %d\n", ret);
    status = api_status(instance);
    printf("api_status: %d\n", status);

    int input;
    #if 0
    std::cin >> input;
    api_unloadmodel(instance);
    status = api_status(instance);
    printf("api_unloadmodel api_status: %d\n", status);


    destroy_api_instance(instance);
    input;
    std::cin >> input;
    instance = create_api_instance(&params, params_bytes);

    if (instance == nullptr) {
        printf("instance is null\n");
    }

    status = api_status(instance);
    printf("api_status: %d\n", status);


    state = api_loadmodel_by_path(instance, model_path, model_names, model_name_count, hwinfo);
    printf("api_loadmodel_by_path = %d\n", state);
    #endif

    status = api_status(instance);
    printf("api_status: %d\n", status);

    //char* prompt3 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n以下信息来自你的知识库，可能对回答某些问题有帮助：\n\n请记住以上知识库的信息。//im_end|>\n<|im_start|>assistant\n好的，我已经记住了这些知识库的信息，请问有什么我可以帮助你解答的问题吗？<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|//ssistant\nHello! How can I assist you today?<|im_end|>\n<|im_start|>user\n问题:给我讲讲三国演义\n请根据相关资料中有用的信息作为依据回答问题。\n回答一定要忠于原文，不要创造//案; 回答简洁而口语化，语义流畅而连贯。<|im_end|>\n<|im_start|>assistant\n";
    //联想提供的prompt
    //char* prompt3 = "<｜begin▁of▁sentence｜><｜User｜><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n以下信息来自你的知识库，可能对回答某些问题有帮助：\n\n请记住以上知识库的信息。<|im_end|>\n<|im_start|>assistant\n好的，我已经记住了这些知识库的信息，请问有什么我可以帮助你解答的问题吗？<|im_end|>\n<|im_start|>user\n给我讲讲红楼梦<|im_end|>\n<|im_start|>assistant\n<think>\n嗯，我现在要回答用户的问题：“给我讲讲红楼梦”。首先，我需要回想一下自己之前学过的关于《红楼梦》的知识。我记得这是一部中国古典四大名著之一，讲述了四个主要人物的发展，特别是贾宝玉和林黛玉的爱情故事。\n\n我记得故事分为上中下三部曲，大约是按时间顺序写的。上篇可能讲述了贾府的衰落，中篇是林黛玉的悲剧，下篇则是贾宝玉的结局。贾宝玉和黛玉的爱情线贯穿了整个故事，非常感人。\n\n还有，书中有很多细节描写，比如贾府的奢华生活，还有那些所谓的“1304句”，这些都是描写细腻的地方，展现了当时的社会风貌。不过，我还不太确定这些细节的具体内容，可能需要再回忆一下。\n\n另外，书中还有一些复杂的结局，比如贾宝玉的早逝，黛玉的悲惨离世，以及紫玉的牺牲，这些都是故事的重要转折点。可能还有一些象征意义的东西，比如“三生三世”之类的主题，但我不太确定这些细节。\n\n总的来说，我需要确保回答准确，不添加虚构的内容。我要按照用户的要求，忠于原文，简洁口语化，语义流畅。可能需要分段描述各个部分，让用户更容易理解。\n\n嗯，可能还需要提到一些关键人物的性格特点，比如贾宝玉的傲慢，黛玉的温柔，还有其他如史玉儿、薛宝钗等的性格特点。这些可以帮助用户更好地理解人物之间的关系和发展。\n\n另外，我应该提到一些重要的情节转折，比如贾母去世后的变化，贾赦的堕落，黛玉的葬礼，以及贾宝玉的续uation。这些都是故事的关键部分。\n\n不过，我不太确定关于续写部分的具体内容，可能需要确认一下。我记得续写可能有不同的版本，但原著到这里就已经结束了，可能用户想知道更多细节的话，我可以稍微提及一下。\n\n总之，我需要组织好语言，确保信息准确，结构清晰，让用户能够轻松理解《红楼梦》的故事和主题。同时，避免使用过于学术化的词汇，保持口语化的表达，让回答更贴近用户的需求。\n</think>\n\n当然可以！《红楼梦》是中国古典四大名著之一，讲述了贾府两个主人公贾宝玉和林黛玉的爱情故事，以及他们如何在封建家族制度和社会压力下挣扎求生。故事分为上中下三部，讲述了贾府从奢华到衰败的过程，展现了封建社会的压迫和人性的光辉。书中充满了细腻的人物描写和深刻的社会批判，尤其是贾宝玉和黛玉的爱情故事，被认为是文学史上的经典之作。总的来说，《红楼梦》不仅是一部爱情小说，更是一部深刻探讨人性和社会的著作。<|im_end|>\n<|im_start|>user\n问题:给我讲讲三国演义\n请根据相关资料中有用的信息作为依据回答问题。\n回答一定要忠于原文，不要创造答案; 回答简洁而口语化，语义流畅而连贯。<|im_end|>\n<|im_start|>assistant\n<｜Assistant｜>";

    //内容可以回答的挺多的提问
    //char* prompt3 ="<｜begin▁of▁sentence｜>Please reason step by step, and put your final answer within ``.<｜User｜>给我讲讲三国演义<｜Assistant｜>";

    //回答的还可以，不太多
    //char* prompt3 ="<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>以下信息来自你的知识库，可能对回答某些问题有帮助：\n\n请记住以上知识库的信息。<｜Assistant｜>好的，我已经记住了这些知识库的信息，请问有什么我可以帮助你解答的问题吗？<｜User｜>给我讲讲三国演义\n请根据相关资料中有用的信息作为依据回答问题。\n回答一定要忠于原文，不要创造答案; 回答简洁而口语化，语义流畅而连贯。<｜Assistant｜><｜end▁of▁sentence｜><｜Assistant｜>";

    char* prompt3 ="<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>以下信息来自你的知识库，可能对回答某些问题有帮助：\n\n请记住以上知识库的信息。<｜Assistant｜><｜end▁of▁sentence｜><｜Assistant｜>好的，我已经记住了这些知识库的信息，请问有什么我可以帮助你解答的问题吗？<｜begin▁of▁sentence｜><｜User｜>给我讲讲三国演义\n请根据相关资料中有用的信息作为依据回答问题。\n回答一定要忠于原文，不要创造答案; 回答简洁而口语化，语义流畅而连贯。<｜Assistant｜><｜end▁of▁sentence｜><｜Assistant｜>";


    params.n_predict = 3000;
    char* result3 = nullptr;
    size_t result_size3 = 0;
    ret = api_Generate_stream(instance, prompt3, &params, params_bytes);
    printf("api_Generate_stream = %d\n", ret);
//
    status = api_Generate_status(instance);
    printf("api_Generate_status: %d\n", status);
    status = api_status(instance);
    printf("api_status: %d\n", status);
    count = 0;

    if (ret) {
        //bad:
        //CallBackTimer t = new CallBackTimer(instance);
        //t.start(100, api_Get_Generate_token);
//
        //CallBackTimer * t = new CallBackTimer(instance);
        //t->start(100, api_Get_Generate_token);
//
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
    std::cin >> input;

    status = api_Generate_status(instance);
    printf("api_Generate_status 1: %d\n", status);

    status = api_Generate_status(instance);
    printf("api_Generate_status 2: %d\n", status);


    status = api_status(instance);
    printf("api_status 1: %d\n", status);

    status = api_status(instance);
    printf("api_status 2: %d\n", status);

    status = api_status(instance);
    printf("api_status 3: %d\n", status);

    status = api_status(instance);
    printf("api_status 4: %d\n", status);

    ret = api_token_num(instance, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n${history}<|im_start|>user\nThe following information is from your knowledgebase, it might be helpful to answer the question:\n---\n\n---\nPlease remember the above knowledgebase info.<|im_end|>\n<|im_start|>assistant\nYes, I have remembered these knowledgebase information, is there anything I can do for you?<|im_end|>\n${history}<|im_start|>user\nQuestion: hello\nYou MUST answer the question ONLY using the relevant information in Knowledgebase. Avoid generating hallucinations.<|im_end|>\n<|im_start|>assistant\n");
    printf("api_token_num = %d\n", ret);
    status = api_status(instance);
    printf("api_status: %d\n", status);

    ret = api_token_num(instance, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n${history}<|im_start|>user\nThe following information is from your knowledgebase, it might be helpful to answer the question:\n---\n\n---\nPlease remember the above knowledgebase info.<|im_end|>\n<|im_start|>assistant\nYes, I have remembered these knowledgebase information, is there anything I can do for you?<|im_end|>\n${history}<|im_start|>user\nQuestion: hello\nYou MUST answer the question ONLY using the relevant information in Knowledgebase. Avoid generating hallucinations.<|im_end|>\n<|im_start|>assistant\n");
    printf("api_token_num = %d\n", ret);
    status = api_status(instance);
    printf("api_status: %d\n", status);
    std::cin >> input;

//
    //SetStdinEcho(true);
    //while (true) {
    //    std::cin.clear();
    //    std::cin.sync();
    //    //std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    //    std::string prompt;
    //    std::cout << std::endl;
    //    std::cout << "输入问题(CTRL + C退出): ";
    //    //std::cin >> prompt;
    //    getline(std::cin,prompt);
    //    printf("\n你的问题: %s\n", prompt.c_str());
    //    char* result2 = new char [1024];
    //    size_t result_size2 = 1024;
    //    ret = api_Generate_stream(instance, prompt.c_str(), &params, params_bytes);
    //    //printf("\napi_Generate_stream = %d\n", ret);
//
    //    //status = api_Generate_status(instance);
    //    //printf("api_Generate_status: %d\n", status);
    //    //status = api_status(instance);
    //    //printf("api_status: %d\n", status);
    //    std::cout << "回答: " << std::endl;
    //    if (ret) {
    //        while (true) {
    //            ret = api_Get_Generate_token(instance, &result2, &result_size2);
    //            status = api_Generate_status(instance);
    //            //printf("api_Generate_status: %d\n", status);
    //            if (status == 0) {
    //                //printf("stop found str: %s\n", result2);
    //                break;
    //            } else if (ret == 2) {
    //                sleep_seconds(100);
    //                //printf("data not ready: %d\n", ret);
    //                continue;
    //            } else if (ret == -1) {
    //                //printf("Error: %d\n", ret);
    //                break;
    //            } else if (ret == 1) {
    //                //printf("not enough: %d\n", result_size2);
    //                if (result2) {
    //                    delete result2;
    //                }
    //                result2 = new char [result_size2];
    //                continue;
    //            } else {
    //                //printf("content str: %s\n", result2);
    //                std::smatch results;
    //                std::string result = result2;
    //                if (std::regex_search(result, results, std::regex(".*\"content\":\"(.*?)\".*"))) {
    //                    std::string content = results[1].str();
    //                    content = std::regex_replace(content, std::regex(R"(\\n)"), "\n");
    //                    //std::string::size_type index = 0;
    //                    //while ((index = content.find("\\n", index)) != std::string::npos) {
    //                    //    content.replace(index, 2, "\n");
    //                    //    ++index;
    //                    //}
    //                    std::cout << content << std::flush;
    //                }
    //            }
    //        }
    //        std::cout << std::endl;
    //    }
    //    //status = api_status(instance);
    //    //printf("api_status: %d\n", status);
    //}

    api_unloadmodel(instance);
    status = api_status(instance);
    printf("api_status: %d\n", status);


    destroy_api_instance(instance);

    printf("exit!\n");

    return 0;
}



struct ChatMessage {
    std::string role;     // "system" 或 "user"
    std::string content;  // 消息内容
};


// 从Qwen模板输出中提取消息
std::vector<ChatMessage> extractMessagesFromQwenTemplate(const std::string& templateOutput) {
    std::vector<ChatMessage> messages;

    // Qwen模板格式示例:
    // <|im_start|>system\nYou are a helpful assistant<|im_end|>
    // <|im_start|>user\n你好<|im_end|>
    // <|im_start|>assistant\n你好！有什么我可以帮忙的吗？<|im_end|>

    /*
    这里使用了 C++11 引入的原始字符串字面量（Raw string literals），以 R"( 开头，以 )" 结尾，这样可以避免对反斜杠 \ 进行额外的转义，增强正则表达式的可读性。下面来详细分析这个正则表达式：
      <\|im_start\|>：匹配文本 <|im_start|>。这里的 \| 是对竖线字符 | 进行转义，因为在正则表达式里，竖线有特殊含义（表示或运算），所以要使用反斜杠进行转义，使其表示普通的竖线字符。
      ([^\n]+)：
      () 是捕获组，用于提取匹配到的内容。
      [^\n] 是一个字符类，表示匹配除换行符 \n 之外的任意字符。
      + 表示前面的字符类（即除换行符之外的字符）出现一次或多次。所以这个捕获组会匹配一行中除换行符之外的所有字符。
      \n：匹配一个换行符。
      ([^<]+)：
      同样是一个捕获组。
      [^<] 表示匹配除小于号 < 之外的任意字符。
      + 表示前面的字符类（即除小于号之外的字符）出现一次或多次。这个捕获组会匹配从换行符之后到下一个 < 符号之前的所有字符。
      <\|im_end\|>：匹配文本 <|im_end|>，同样对竖线进行了转义。
    */
    //std::regex pattern(R"(<\|im_start\|>([^\n]+)\n([^<]+)<\|im_end\|>)");
    std::regex pattern(R"(<\|im_start\|>([^\n]+)\n([^<]+)<\|im_end\|>)");
    std::sregex_iterator it(templateOutput.begin(), templateOutput.end(), pattern);
    std::sregex_iterator end;

    while (it != end) {
        std::smatch match = *it;
        if (match.size() == 3) {  // 整个匹配 + 2个捕获组(role和content)
            ChatMessage msg;
            msg.role = match[1].str();
            msg.content = match[2].str();

            // 去除content首尾空白字符
            msg.content.erase(0, msg.content.find_first_not_of(" \t\n\r"));
            msg.content.erase(msg.content.find_last_not_of(" \t\n\r") + 1);

            messages.push_back(msg);
        }
        ++it;
    }

    return messages;
}

// 使用示例:
int test_qwen_template_revert() {
    std::string qwenOutput = R"(
        <|im_start|>system
        你是一个有帮助的AI助手<|im_end|>
        <|im_start|>user
        你好，今天天气怎么样？<|im_end|>
        <|im_start|>assistant
        你好！今天天气晴朗，适合外出。<|im_end|>
    )";

    qwenOutput="<｜begin▁of▁sentence｜><｜User｜><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n以下信息来自你的知识库，可能对回答某些问题有帮助：\n\n请记住以上知识库的信息。<|im_end|>\n<|im_start|>assistant\n好的，我已经记住了这些知识库的信息，请问有什么我可以帮助你解答的问题吗？<|im_end|>\n<|im_start|>user\n给我讲讲红楼梦<|im_end|>\n<|im_start|>assistant\n<think>\n嗯，我现在要回答用户的问题：“给我讲讲红楼梦”。首先，我需要回想一下自己之前学过的关于《红楼梦》的知识。我记得这是一部中国古典四大名著之一，讲述了四个主要人物的发展，特别是贾宝玉和林黛玉的爱情故事。\n\n我记得故事分为上中下三部曲，大约是按时间顺序写的。上篇可能讲述了贾府的衰落，中篇是林黛玉的悲剧，下篇则是贾宝玉的结局。贾宝玉和黛玉的爱情线贯穿了整个故事，非常感人。\n\n还有，书中有很多细节描写，比如贾府的奢华生活，还有那些所谓的“1304句”，这些都是描写细腻的地方，展现了当时的社会风貌。不过，我还不太确定这些细节的具体内容，可能需要再回忆一下。\n\n另外，书中还有一些复杂的结局，比如贾宝玉的早逝，黛玉的悲惨离世，以及紫玉的牺牲，这些都是故事的重要转折点。可能还有一些象征意义的东西，比如“三生三世”之类的主题，但我不太确定这些细节。\n\n总的来说，我需要确保回答准确，不添加虚构的内容。我要按照用户的要求，忠于原文，简洁口语化，语义流畅。可能需要分段描述各个部分，让用户更容易理解。\n\n嗯，可能还需要提到一些关键人物的性格特点，比如贾宝玉的傲慢，黛玉的温柔，还有其他如史玉儿、薛宝钗等的性格特点。这些可以帮助用户更好地理解人物之间的关系和发展。\n\n另外，我应该提到一些重要的情节转折，比如贾母去世后的变化，贾赦的堕落，黛玉的葬礼，以及贾宝玉的续uation。这些都是故事的关键部分。\n\n不过，我不太确定关于续写部分的具体内容，可能需要确认一下。我记得续写可能有不同的版本，但原著到这里就已经结束了，可能用户想知道更多细节的话，我可以稍微提及一下。\n\n总之，我需要组织好语言，确保信息准确，结构清晰，让用户能够轻松理解《红楼梦》的故事和主题。同时，避免使用过于学术化的词汇，保持口语化的表达，让回答更贴近用户的需求。\n</think>\n\n当然可以！《红楼梦》是中国古典四大名著之一，讲述了贾府两个主人公贾宝玉和林黛玉的爱情故事，以及他们如何在封建家族制度和社会压力下挣扎求生。故事分为上中下三部，讲述了贾府从奢华到衰败的过程，展现了封建社会的压迫和人性的光辉。书中充满了细腻的人物描写和深刻的社会批判，尤其是贾宝玉和黛玉的爱情故事，被认为是文学史上的经典之作。总的来说，《红楼梦》不仅是一部爱情小说，更是一部深刻探讨人性和社会的著作。<|im_end|>\n<|im_start|>user\n问题:给我讲讲三国演义\n请根据相关资料中有用的信息作为依据回答问题。\n回答一定要忠于原文，不要创造答案; 回答简洁而口语化，语义流畅而连贯。<|im_end|>\n<|im_start|>assistant\n<｜Assistant｜>";

    auto messages = extractMessagesFromQwenTemplate(qwenOutput);

    for (const auto& msg : messages) {
        if (msg.role == "system") {
            std::cout << "System: " << msg.content << std::endl;
        } else if (msg.role == "user") {
            std::cout << "User Question: " << msg.content << std::endl;
        } else if (msg.role == "assistant") {
            std::cout << "Assistant Response: " << msg.content << std::endl;
        }
    }

    return 0;
}
//*/


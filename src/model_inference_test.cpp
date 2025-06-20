/*
基于后摩大道api，加载模型进行推理
用于测试模型、代码功能等
重命名成此文件-以支持更多模型
*/
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <torch/torch.h>

#include "tcim/tcim_runtime.h"
#if (__GNUC__ < 8 && !defined(_MSC_VER))
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#define SPLIT_COUNT (26)

using namespace std::chrono;

enum model_scale_type {
  MODEL_1DOT5B=1,
  MODEL_3B,
  MODEL_7B,
  MODEL_8B,
  MODEL_14B,
  MODEL_32B,
  MODEL_72B,
};

class HmQwen_14B;

static void int16toint16(int16_t *src, int16_t *dst, size_t size) {
  for (int i = 0; i < size; i++) {
    src[i] = dst[i];
  }
}

class TensorConverter {
public:
    // 将torch::Tensor转换为tcim::Tensor
    //static tcim::Tensor FromTorch(const torch::Tensor& torch_tensor, HmQwen *hmqwen, const std::string &tcim_tensorname,int which_part) ;
    static void FromTorch(const torch::Tensor& torch_tensor, tcim::Tensor &tcim_tensor);

    // 将tcim::Tensor转换为torch::Tensor
    static torch::Tensor ToTorch(const tcim::Tensor& tcim_tensor);

    // 将vector<float>转换为tcim::Tensor
    //template<typename T>
    //static tcim::Tensor FromVector(const std::vector<int16_t>& vec, const std::vector<uint64_t>& shape, HmQwen *hmqwen, const std::string &tcim_tensorname,int which_part);

    static void FromVector(const std::vector<int16_t>& vec, tcim::Tensor &tcim_tensor);

    // 将tcim::Tensor转换为vector<T>
    //template<typename T>
    static std::vector<int16_t> ToVector(const tcim::Tensor& tcim_tensor);
};


// 颜色输出宏
#define COLOR_BLUE "\033[0;34m"
#define COLOR_PURPLE "\033[0;35m"
#define COLOR_RESET "\033[0m"

// 判断字符是否有效
bool is_valid_char(int cp) {
    return ((cp >= 0x4E00 && cp <= 0x9FFF) ||
            (cp >= 0x3400 && cp <= 0x4DBF) ||
            (cp >= 0x20000 && cp <= 0x2A6DF) ||
            (cp >= 0x2A700 && cp <= 0x2B73F) ||
            (cp >= 0x2B740 && cp <= 0x2B81F) ||
            (cp >= 0x2B820 && cp <= 0x2CEAF) ||
            (cp >= 0xF900 && cp <= 0xFAFF) ||
            (cp >= 0x2F800 && cp <= 0x2FA1F) ||
            (0x0041 <= cp && cp <= 0x005A) ||
            (0x0061 <= cp && cp <= 0x007A));
}

// 解析命令行参数
struct Args {
    //24layer model path
    // std::string tokenizer_path = "./models/DeepSeek-R1-Distill-Qwen-14B-1Batch/4cores_8k/hmquant/deepseek14b_traced_model.pt";
    // std::string model_dir = "./models/DeepSeek-R1-Distill-Qwen-14B-1Batch/4cores_8k/";

    //26layer model path hmm_deepseek_14b_256_8k_4cores_20250514_26l
    std::string tokenizer_path = "./models/hmm_deepseek_14b_256_8k_4cores_20250514_26l/hmquant/deepseek14b_traced_model.pt";
    std::string model_dir = "./models/hmm_deepseek_14b_256_8k_4cores_20250514_26l/";

    int prefill_length = 256;
    int decode_length = 300;
    int nblocks = 48;
    int ncore =4;
    int batch =4;
    model_scale_type hm_model_scale_type=MODEL_7B;  //根据加载的模型，修改成对应的宏
};

Args get_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tokenizer_path" && i + 1 < argc) {
            args.tokenizer_path = argv[++i];
        } else if (arg == "--model_dir" && i + 1 < argc) {
            args.model_dir = argv[++i];
        } else if (arg == "--prefill" && i + 1 < argc) {
            args.prefill_length = std::stoi(argv[++i]);
        } else if (arg == "--decode" && i + 1 < argc) {
            args.decode_length = std::stoi(argv[++i]);
        } else if (arg == "--nblocks" && i + 1 < argc) {
            args.nblocks = std::stoi(argv[++i]);
        } else if (arg == "--model_scale_type" && i + 1 < argc) {
            args.hm_model_scale_type = static_cast<model_scale_type>(std::stoi(argv[++i]));
        } else if (arg == "--ncore" && i + 1 < argc) {
            args.ncore = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            args.batch = std::stoi(argv[++i]);
        }

    }
    return args;
}

// 线程安全的管道实现
class Pipe {
public:
    void send(const std::string& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(data);
        cv_.notify_one();
    }

    std::string recv() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]{ return !queue_.empty(); });
        std::string data = queue_.front();
        queue_.pop();
        return data;
    }

private:
    std::queue<std::string> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

// 线程安全的队列实现
template<typename T>
class SafeQueue {
public:
    void put(const T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(value);
        cv_.notify_one();
    }

    T get() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]{ return !queue_.empty(); });
        T value = queue_.front();
        queue_.pop();
        return value;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

// 原子标志
class AtomicFlag {
public:
    AtomicFlag(bool initial_value = false) : flag_(initial_value) {}
    void operator=(bool value) { flag_.store(value); }
    operator bool() const { return flag_.load(); }

private:
    std::atomic<bool> flag_;
};

template <typename T>
void printVector(const std::vector<T>& vec) {
    std::cout<<"===vector value is:"<<std::endl;
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    std::cout<<"===vector size is:"<<vec.size();
    std::cout << std::endl;
}


void print_tcimtensor(const tcim::Tensor& tcim_tensor) {

        std::cout<<"tcim_tensor value:";
        int16_t* data = (int16_t*)tcim_tensor.Data();
        for(int i=0;i<tcim_tensor.Info().MemSize()/2;i++){
           if(i%512==0)
           printf("%hd, ",*(data+i));
        }
        printf("\r\n");

        auto info = tcim_tensor.Info();
        auto shape = info.Shape();
        std::cout<<"tcim_tensor shape:";
        for(auto shape_i:shape){
            std::cout<<shape_i<<" ,";
        }
        std::cout<<std::endl;

}

void print_tensor(const torch::Tensor& tensor) {

    std::cout << std::endl;
    std::cout << "Tensor shape: ";
    auto sizes = tensor.sizes();
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << sizes[i] << " ";
    }
    std::cout << std::endl;
    // std::cout << "Tensor values:" << std::endl;

    // for (int64_t i = 0; i < tensor.size(0); ++i) { // 遍历第一维
    //     for (int64_t j = 0; j < tensor.size(1); ++j) { // 遍历第二维
    //         //std::cout << std::fixed << std::setprecision(4) << tensor[i][j].item<float>() << " "; // 设置浮点数格式并打印元素值
    //         std::cout  << tensor[i][j].item<int>() << " , "; // 设置浮点数格式并打印元素值
    //     }
    //     std::cout << std::endl;
    // }

    std::cout << "Tensor values::\n" << tensor << std::endl;

}

// 模型类
class HmQwen_14B {
public:
    struct Config {
        std::string tokenizer_path;
        std::string model_dir;
        int prefill_length;
        int decode_length;
        int nblocks;
    };

    HmQwen_14B(const Config& config, tcim::Module::WeightManager* wm0, tcim::Module::WeightManager* wm1, int process_id, int batch = 1)
        : config_(config), wm0_(wm0), wm1_(wm1), process_id_(process_id), batch_(batch) {
        init();
    }

    void chat(const std::string& question, Pipe* pipe, SafeQueue<std::vector<double>>* queue, bool initializer, AtomicFlag* is_end) {
        std::cout << "question:" << std::endl;
        std::cout << "\033[1;95m" << question << "\033[0m" << std::endl;
        auto start_time = high_resolution_clock::now();

        // 准备聊天模板
        std::vector<std::pair<std::string, std::string>> messages = {
            {"system", "You are a helpful assistant."},
            {"user", question}
        };

        //TODO:szj change tokenizer to em
        //std::string text = tokenizer_.apply_chat_template(messages, false, true);
        //auto inputs = tokenizer_.encode(text, false, false);
        //你好，你是谁？用qwen 14B encode 的输出
        //std::vector<int64_t> all_input_ids = {108386,   3837, 105043, 100165,  11319};//inputs.ids;
        //qwen2.5 14B tokenier 输出的token，不对
        // std::vector<int64_t> all_input_ids = {151644,   8948,    198,  56568, 101909, 100032, 104653, 110498,   3837,
        //                                       102104,  86119, 110485,  30858,  34187,   1773, 151645,    198, 151644,
        //                                          872,    198, 108386,   3837, 105043, 100165,  11319, 151645,    198};

        //我手动运行python脚本的输出
        // std::vector<int32_t> all_input_ids = {151646,  56568, 101909, 100032, 104653, 110498,   3837, 102104,
        //                                       86119, 110485,  30858,  34187,   1773, 151644, 108386,   3837, 105043,
        //                                       100165,  11319, 151645};

        //下面是huatongpython脚本每次输出都一样的inputid
        std::vector<int32_t> all_input_ids = {151646,   2610,    525,    264,  10950,  17847,     13, 151644, 108386,
           3837, 105043, 100165,  11319, 151645, 151648,    198};

        int input_echo_len = all_input_ids.size();
        std::cout << "input_echo_len: " << input_echo_len << std::endl;

        if (input_echo_len >= config_.decode_length) {
            std::cerr << "Question long than " << config_.decode_length << ", please shorten it!" << std::endl;
            return;
        }

        // 初始化KVCache历史
        for (int i = 0; i < SPLIT_COUNT; ++i) {
            //auto kcache_history_sum = prefill_model_part1_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum");
            //std::vector<float> zeros(kcache_history_sum.info().size(), 0.0f);
            auto input_data = prefill_part1_tensormap.at("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum");
             auto kcache_history_sum_init = torch::zeros(
              {1, (int16_t)(input_data.MemSize() / 2)}, torch::kInt16);
            int16toint16((int16_t *)input_data.Data(),
                       (int16_t *)kcache_history_sum_init.data_ptr<int16_t>(),
                       kcache_history_sum_init.numel());
            prefill_model_part1_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum", input_data);
        }
        for (int i = SPLIT_COUNT; i < config_.nblocks; ++i) {
            //auto kcache_history_sum = prefill_model_part2_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum");
            //std::vector<float> zeros(kcache_history_sum.info().size(), 0.0f);

            auto input_data = prefill_part2_tensormap.at("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum");
            auto kcache_history_sum_init = torch::zeros(
              {1, (int16_t)(input_data.MemSize() / 2)}, torch::kInt16);
            int16toint16((int16_t *)input_data.Data(),
                       (int16_t *)kcache_history_sum_init.data_ptr<int16_t>(),
                       kcache_history_sum_init.numel());

            prefill_model_part2_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum", input_data);
        }
        printf("@@1+with kvcache clear\r\n");
        // 预填充阶段
        int prefill_loop_round = std::ceil(static_cast<double>(input_echo_len) / config_.prefill_length);
        printf("++prefill_loop_round is:%d\r\n",prefill_loop_round);
        for (int round = 0; round < prefill_loop_round; ++round) {
            int16_t valid_length = round * config_.prefill_length;
            int16_t current_length;
            std::vector<int32_t> input_ids;

            if (round == prefill_loop_round - 1) {
                current_length = input_echo_len - round * config_.prefill_length;
                input_ids.assign(all_input_ids.begin() + round * config_.prefill_length, all_input_ids.end());
            } else {
                current_length = config_.prefill_length;
                input_ids.assign(all_input_ids.begin() + round * config_.prefill_length,
                                all_input_ids.begin() + (round + 1) * config_.prefill_length);
            }
            printf("@@2\r\n");
            // 准备输入嵌入
            auto opts = torch::TensorOptions().dtype(torch::kInt);
            auto inp = torch::from_blob(input_ids.data(),
                                    {int64_t(input_ids.size())}, opts)
                       .clone();
            printf("@@2.1 after round input_ids.size is:%d\r\n",input_ids.size());
            torch::Tensor inputs_embeds = emb->forward(inp);//torch::embedding(torch::tensor(input_ids), embedding_weight_);
            printf("@@2.1.1 \r\n");
            int effective_length = input_ids.size();
            torch::Tensor pad_embeds = torch::zeros({config_.prefill_length - effective_length, inputs_embeds.size(-1)}, inputs_embeds.options());
            printf("@@2.1.2 effective_length=%d inputs_embeds.size(-1)=%d\r\n",effective_length,inputs_embeds.size(-1));

            std::cout << "inputs_embeds shape: " << inputs_embeds.sizes() << std::endl;
            std::cout << "pad_embeds shape: " << pad_embeds.sizes() << std::endl;
            //打印torch tensor
            //print_tensor(inputs_embeds);
            // 检查是否有空张量
            if (inputs_embeds.numel() == 0 ) {
                std::cerr << "Empty inputs_embeds tensor detected!" << std::endl;
                //return;
            }
            if (pad_embeds.numel() == 0) {
                std::cerr << "Empty pad_embeds tensor detected!" << std::endl;
                //return;
            }

            //torch::Tensor input_data = torch::cat({inputs_embeds, pad_embeds}, 0).reshape({4, config_.prefill_length / 4, 5120});
            torch::Tensor input_data = torch::cat({inputs_embeds, pad_embeds});
            input_data=input_data.view({4, config_.prefill_length / 4, 5120}).to(torch::kInt16);
            //printf("----input_data after .view to int16:\r\n");
            //print_tensor(input_data);

            //reshape({4, config_.prefill_length / 4, 5120});
            printf("@@--2.2\r\n");
            // 设置并运行模型
            std::vector<int16_t> valid_length_data = {valid_length};
            std::vector<int16_t> current_length_data = {current_length};
            // printf("+++valid_length_data info:\n");
            // printVector(valid_length_data);

            // printf("+++current_length_data info:\n");
            // printVector(current_length_data);


            printf("@@3 valid_length=%d  current_length=%d\r\n",valid_length,current_length);
            #if 1
            TensorConverter::FromTorch(input_data.to(torch::kInt16),prefill_part1_tensormap.at("input_1"));
            prefill_model_part1_.SetInput("input_1",prefill_part1_tensormap.at("input_1"));

            TensorConverter::FromVector(valid_length_data, prefill_part1_tensormap.at("valid_length"));
            prefill_model_part1_.SetInput("valid_length",  prefill_part1_tensormap.at("valid_length"));

            TensorConverter::FromVector(current_length_data,prefill_part1_tensormap.at("current_length"));
            prefill_model_part1_.SetInput("current_length", prefill_part1_tensormap.at("current_length"));
            #else
              //input_1
              auto prefill_part1_input_data = prefill_part1_tensormap.at("input_1");
              int16toint16((int16_t *)prefill_part1_input_data.Data(),
                           (int16_t *)input_data.data_ptr<int16_t>(),
                           input_data.numel());
              //valid_length
              std::vector<int16_t> valid_length_input = {valid_length};
              auto valid_length_input_tensor_opts =
                  torch::TensorOptions().dtype(torch::kInt16);
              auto valid_length_input_tensor =
                  torch::from_blob(valid_length_input.data(),
                                   {int64_t(valid_length_input.size())},
                                   valid_length_input_tensor_opts)
                      .clone();

              auto prefill_valid_length = prefill_part1_tensormap.at("valid_length");
              int16toint16((int16_t *)prefill_valid_length.Data(),
                           (int16_t *)valid_length_input_tensor.data_ptr<int16_t>(),
                           valid_length_input_tensor.numel());


              //current_length
              std::vector<int16_t> current_length_input = {current_length};
              auto current_length_input_tensor_opts =
                  torch::TensorOptions().dtype(torch::kInt16);
              auto current_length_input_tensor =
                  torch::from_blob(current_length_input.data(),
                                   {int64_t(current_length_input.size())},
                                   current_length_input_tensor_opts)
                      .clone();

              // std::cout << "current_length_input_tensor: " <<
              // current_length_input_tensor << std::endl;
              auto prefill_current_length = prefill_part1_tensormap.at("current_length");
              int16toint16((int16_t *)prefill_current_length.Data(),
                     (int16_t *)current_length_input_tensor.data_ptr<int16_t>(),
              current_length_input_tensor.numel());

              prefill_model_part1_.SetInput("input_1",prefill_part1_input_data);
              prefill_model_part1_.SetInput("valid_length",  prefill_valid_length);
              prefill_model_part1_.SetInput("current_length", prefill_current_length);



            #endif
            if (initializer) {
                pipe->send("OK");
            }

            std::string pipe_data = is_end->operator bool() ? "OK" : pipe->recv();
            if (!is_end->operator bool()) {
                pipe->send("OK");
            }
            printf("@@4\r\n");
            if (!pipe_data.empty()) {
                printf("+++call prefill part1 run+sync\r\n");
                prefill_model_part1_.Run();
                prefill_model_part1_.Sync();
                printf("@@4.1\r\n");
                tcim::Tensor prefill_part1_output = prefill_model_part1_.GetOutput("model_layers_25_resadd2");
                //打印prefill1输出-每隔512个打印输出
                //print_tcimtensor(prefill_part1_output);
                printf("@@4.2\r\n");
                prefill_model_part2_.SetInput("model_layers_25_resadd2", prefill_part1_output);
                printf("@@4.3\r\n");

                TensorConverter::FromVector(valid_length_data, prefill_part2_tensormap.at("valid_length"));
                prefill_model_part2_.SetInput("valid_length",  prefill_part2_tensormap.at("valid_length"));
                printf("@@4.4\r\n");
                TensorConverter::FromVector(current_length_data,prefill_part2_tensormap.at("current_length"));
                prefill_model_part2_.SetInput("current_length", prefill_part2_tensormap.at("current_length"));
                printf("@@4.5\r\n");
                pipe_data = is_end->operator bool() ? "OK" : pipe->recv();
                if (!is_end->operator bool()) {
                    pipe->send("OK");
                }
                printf("@@4.6\r\n");
                if (!pipe_data.empty()) {
                    printf("+++call prefill part2 run+sync\r\n");
                    prefill_model_part2_.Run();
                    prefill_model_part2_.Sync();
                }
            }
        }
        printf("@@5\r\n");
        // 获取预填充结果
        tcim::Tensor output = prefill_model_part2_.GetOutput("Output_lm_head_add_list_1");
        //std::vector<float> output_data(output.data(), output.data() + output.size());
        auto output_data=TensorConverter::ToVector(output);

        //下面打印prefill2的输出
        // for(auto int16_v:output_data) {
        //     //std::cout<<static_cast<int16_t>(int16_v)<<", ";
        //     printf("%hd,",int16_v);
        // }
        std::cout<<std::endl;
        printf("++output_data size is:%d\r\n",output_data.size());
        std::cout<<std::endl;
        int32_t next_id=0;
        #if USE_STDDISTANCE
            next_id = std::distance(output_data.begin(), std::max_element(output_data.begin(), output_data.end()));
        #else
        auto prefill_output_opts = torch::TensorOptions().dtype(torch::kInt16);
        auto prefill_output_tensor =
            torch::from_blob(output_data.data(),
                            {int64_t(output_data.size())},
                            prefill_output_opts)
                .clone();

        auto selected_output =
            torch::argmax(prefill_output_tensor).to(torch::kInt32);
        // std::cout << "selected_output: " << selected_output << std::endl;
        // 下面打印的selected_output是乱码，index
        // int值，调下面的common_token_to_piece转成字符串
        printf("++selected_output(tensor addr not selected index) is:%u\r\n", selected_output);
        next_id = *(int32_t *)selected_output.data_ptr<int32_t>();
        printf("++selected tensor index is:%d\r\n",next_id);
        #endif

        //TODO:szj
        std::string prefill_response = "remem to change to real txt";//tokenizer_.decode({next_id});
        auto prefill_time = duration_cast<duration<double>>(high_resolution_clock::now() - start_time).count();
        printf("@@6\r\n");
        // 初始化聊天历史
        std::vector<int32_t> chat_history_ids = all_input_ids;
        chat_history_ids.push_back(next_id);
        printf("@@9 prefill output:next_id=%d\r\n",next_id);

        // 准备解码输入
        torch::Tensor next_id_tensor = torch::tensor({next_id});
        torch::Tensor decode_input = emb->forward(next_id_tensor.to(torch::kInt));//.reshape({1});//torch::embedding(next_id_tensor, embedding_weight_).reshape({1, 1, -1});
        decode_input = decode_input.view({1, 1, -1}).to(torch::kInt16);
        std::string all_response = prefill_response;
        int context_length = input_echo_len;
        int decode_count = 0;
        int skip_tokens = 0;
        int slide_len = 10;
        //TODO:szj
        std::string last_response = "fixed last remember to change to real txt";//tokenizer_.decode(std::vector<int64_t>(chat_history_ids.end() - slide_len, chat_history_ids.end()));

        // 解码阶段
        auto decode_start_time = high_resolution_clock::now();
        while (true) {
            if (context_length >= config_.decode_length) {
                std::cout << "context length greater than " << config_.decode_length << ", break!" << std::endl;
                break;
            }
            else {
                // 获取数据指针
                printf("++6.1\r\n");
                int16_t* data = decode_input.data_ptr<int16_t>();
                printf("++6.2\r\n");
                // 打印每个元素
                int size = decode_input.numel();  // 元素总数
                for (int i = 0; i < size; ++i) {
                    //std::cout << "decode_input[" << i << "] = " << data[i] << std::endl;
                }
            }
            printf("@@7\r\n");
            // 设置并运行解码模型
            TensorConverter::FromTorch(decode_input.to(torch::kInt16),decode_part1_tensormap.at("input_1"));
            decode_model_part1_.SetInput("input_1", decode_part1_tensormap.at("input_1"));
            std::vector<int16_t> valid_length_data = {static_cast<int16_t>(context_length)};
            TensorConverter::FromVector(valid_length_data,decode_part1_tensormap.at("valid_length"));
            decode_model_part1_.SetInput("valid_length",  decode_part1_tensormap.at("valid_length"));

            std::string pipe_data = is_end->operator bool() ? "OK" : pipe->recv();
            if (!is_end->operator bool()) {
                pipe->send("OK");
            }
            printf("@@8\r\n");
            if (!pipe_data.empty()) {
                decode_model_part1_.Run();
                decode_model_part1_.Sync();
                tcim::Tensor decode_part1_output = decode_model_part1_.GetOutput("model_layers_25_resadd2");

                decode_model_part2_.SetInput("model_layers_25_resadd2", decode_part1_output);
                TensorConverter::FromVector(valid_length_data,decode_part2_tensormap.at("valid_length"));
                decode_model_part2_.SetInput("valid_length",  decode_part2_tensormap.at("valid_length"));

                pipe_data = is_end->operator bool() ? "OK" : pipe->recv();
                if (!is_end->operator bool()) {
                    pipe->send("OK");
                }

                if (!pipe_data.empty()) {
                    decode_model_part2_.Run();
                    decode_model_part2_.Sync();
                    tcim::Tensor decode_output = decode_model_part2_.GetOutput("Output_lm_head_add_list_1");
                    //output_data.assign(decode_output.data(), decode_output.data() + decode_output.size());
                    output_data=TensorConverter::ToVector(decode_output);
                    auto info=decode_output.Info().GetQuantInfo();
                    if(info){
                        auto quant_scales=info->scales;
                        if(quant_scales.size()>0) {
                            for(auto quant_scale:quant_scales) {
                            printf("++quant_scale is:%f \r\n",quant_scale);
                            }
                        } else {
                            printf("+++Error quant_scale size is:%d\r\n",quant_scales.size());
                        }
                    }
                    else {
                        printf("++Error info is null\r\n");
                    }

                }
            }

            ++decode_count;
            #if USE_STDDISTANCE
                next_id = std::distance(output_data.begin(), std::max_element(output_data.begin(), output_data.end()));
            #else
                auto decode_output_opts = torch::TensorOptions().dtype(torch::kInt16);
                auto decode_output_tensor =
                torch::from_blob(output_data.data(),
                            {int64_t(output_data.size())}, decode_output_opts)
                .clone();
                auto selected_output =
                torch::argmax(decode_output_tensor).to(torch::kInt32);
                next_id = *(int32_t *)selected_output.data_ptr<int32_t>();
            #endif

            //TODO:szj change to real eos id
            //if (next_id == tokenizer_.eos_token_id()) {
            if (next_id ==151643) {
                printf("++++++++++Get EOS token just break\r\n");
                break;
            }
            printf("@@9 after decode2 argmux:next_id=%d\r\n",next_id);
            chat_history_ids.push_back(next_id);

            // 解码并处理响应
            //TODO:szj change to real
            std::string decode_response = "demo fixed string szj|";//tokenizer_.decode(
            //     std::vector<int64_t>(chat_history_ids.end() - (slide_len + 1 + skip_tokens), chat_history_ids.end())
            // ).substr(last_response.length());

            if (!decode_response.empty() && is_valid_char(static_cast<int>(decode_response.back()))) {
                all_response += decode_response;
                //TODO:szj change to real
                last_response = "|fixed last string szj";//tokenizer_.decode(
                //    std::vector<int64_t>(chat_history_ids.end() - slide_len, chat_history_ids.end())
                //);
                skip_tokens = 0;
            } else {
                ++skip_tokens;
            }
            printf("@@10\r\n");
            // 准备下一次迭代的输入
            next_id_tensor = torch::tensor({next_id});
            decode_input = emb->forward(next_id_tensor.to(torch::kInt));//.reshape({1});//torch::embedding(next_id_tensor, embedding_weight_).reshape({1, 1, -1});
            decode_input = decode_input.view({1, 1, -1}).to(torch::kInt16);
            ++context_length;
        }

        auto decode_end_time = high_resolution_clock::now();
        std::cout << "response: " << std::endl;
        std::cout << "\033[1;95m" << all_response << "\033[0m" << std::endl;
        printf("========decode output ids is:\r\n");
        for (const auto& num : chat_history_ids) {
        std::cout << num << " , ";
        }
        std::cout << std::endl;
        std::cout <<"final all token count is:"<<chat_history_ids.size()<< std::endl;
        // 计算性能指标并放入队列
        double decode_time = duration_cast<duration<double>>(decode_end_time - decode_start_time).count();
        decode_count=chat_history_ids.size()-1;
        queue->put({
            static_cast<double>(input_echo_len),
            static_cast<double>(decode_count + 1),
            prefill_time,
            static_cast<double>(duration_cast<milliseconds>(decode_start_time - start_time).count()) / 1000.0,
            static_cast<double>(duration_cast<milliseconds>(decode_end_time - start_time).count()) / 1000.0
        });

        *is_end = true;
    }
public:
    std::map<std::string, tcim::Tensor> prefill_part1_tensormap;
    std::map<std::string, tcim::Tensor> prefill_part2_tensormap;
    std::map<std::string, tcim::Tensor> decode_part1_tensormap;
    std::map<std::string, tcim::Tensor> decode_part2_tensormap;
private:
    Config config_;
    tcim::Module::WeightManager* wm0_;
    tcim::Module::WeightManager* wm1_;
    int process_id_;
    int batch_;

    tcim::Stream stream;
    tcim::Stream stream2;
    tcim::Module prefill_model_part1_;
    tcim::Module prefill_model_part2_;
    tcim::Module decode_model_part1_;
    tcim::Module decode_model_part2_;


    //transformers::Tokenizer tokenizer_;
    torch::nn::Embedding emb = nullptr;
    torch::Tensor embedding_weight_;

    void init() {
        std::cout << "Initializing model on process " << process_id_ << std::endl;

        // 准备模型路径
        fs::path tokenizer_path(config_.tokenizer_path);
        fs::path model_dir(config_.model_dir);
        fs::path embedding_path = config_.tokenizer_path;
        fs::path hmm_model_path = model_dir.string();

        // 创建模型选项
        tcim::Module::Option option1_0(*wm0_);
        tcim::Module::Option option1_1(*wm1_);
        tcim::Module::Option option2_0(*wm0_);
        tcim::Module::Option option2_1(*wm1_);

        // 设置dummy tensors
        std::vector<std::string> dummy_tensor_names;
        for (int i = 0; i < config_.nblocks; ++i) {
            dummy_tensor_names.push_back("model_layers_" + std::to_string(i) + "_self_attn_kcache_input");
            dummy_tensor_names.push_back("model_layers_" + std::to_string(i) + "_self_attn_vcache_input");
            dummy_tensor_names.push_back("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum");
        }
        option2_0.SetDummyTensors(dummy_tensor_names);
        option2_1.SetDummyTensors(dummy_tensor_names);

        stream = tcim::Stream(true);
        stream2 = tcim::Stream(true);

        // 加载模型
        prefill_model_part1_ = tcim::Module::LoadFromFile((hmm_model_path / "deepseek_prefill_part1.hmm").string(), option1_0);
        std::cout << "Process " << process_id_ << " loaded prefill part1" << std::endl;

        prefill_model_part2_ = tcim::Module::LoadFromFile((hmm_model_path / "deepseek_prefill_part2.hmm").string(), option1_1);
        std::cout << "Process " << process_id_ << " loaded prefill part2" << std::endl;

        decode_model_part1_ = tcim::Module::LoadFromFile((hmm_model_path / "deepseek_decode_part1.hmm").string(), option2_0);
        std::cout << "Process " << process_id_ << " loaded decoder part1" << std::endl;

        decode_model_part2_ = tcim::Module::LoadFromFile((hmm_model_path / "deepseek_decode_part2.hmm").string(), option2_1);
        std::cout << "Process " << process_id_ << " loaded decoder part2" << std::endl;

        prefill_model_part1_.SetStream(stream);
        prefill_model_part2_.SetStream(stream2);
        decode_model_part1_.SetStream(stream);
        decode_model_part2_.SetStream(stream2);

        fillinfomap_from_modelinfo();
        printf("++1\r\n");
        // 设置KVCache输入
        for (int i = 0; i < SPLIT_COUNT; ++i) {
            //printf("++i =%d\r\n",i);
            auto kcache = prefill_model_part1_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_input");
            //printf("##1\r\n");
            decode_model_part1_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_input", kcache);
            //printf("##2\r\n");
            auto vcache = prefill_model_part1_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_vcache_input");
            //printf("##3\r\n");
            decode_model_part1_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_vcache_input", vcache);
            //printf("##4\r\n");
            auto kcache_history_sum = prefill_model_part1_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum");
            //printf("##5\r\n");
            decode_model_part1_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum", kcache_history_sum);
            //printf("##6\r\n");
        }
        printf("++2\r\n");
        for (int i = SPLIT_COUNT; i < config_.nblocks; ++i) {
            auto kcache = prefill_model_part2_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_input");
            decode_model_part2_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_input", kcache);

            auto vcache = prefill_model_part2_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_vcache_input");
            decode_model_part2_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_vcache_input", vcache);

            auto kcache_history_sum = prefill_model_part2_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum");
            decode_model_part2_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum", kcache_history_sum);
        }
        printf("++3\r\n");
        // 设置decode输入
        std::vector<int16_t> current_length_input_1 = {1};
        //TODO:szj 确认 shape 是1维的1 还是二维的1
        TensorConverter::FromVector(current_length_input_1, decode_part1_tensormap.at("current_length"));
        decode_model_part1_.SetInput("current_length", decode_part1_tensormap.at("current_length"));
        TensorConverter::FromVector(current_length_input_1, decode_part2_tensormap.at("current_length"));
        decode_model_part2_.SetInput("current_length",  decode_part2_tensormap.at("current_length"));

        // 加载tokenizer和embedding weight
        //tokenizer_ = transformers::Tokenizer::from_pretrained(tokenizer_path.string(), true);
        printf("++4 embedding_path.string() is:%s\r\n",embedding_path.string().c_str());
        torch::load(embedding_weight_,embedding_path.string());
        //torch::Tensor input;
        // std::string input_path = "./qembedding.pt";
        //std::string input_path = model_path + "hmquant/quant_qembedding.pt";

        try {
          //torch::load(input, input_path);
          embedding_weight_ = embedding_weight_.view({-1, 5120});
          // std::cout << "embedding_weight_ options: " << embedding_weight_.options() << "\n";
          // std::cout << "embedding_weight_ dim: " << embedding_weight_.dim() << "\n";
          // std::cout << "embedding_weight_ size: " << embedding_weight_.size(0) << "\n";
          // std::cout << "embedding content: " << std::endl;
          // for (int i = 0; i < 20; i++) {
          //     std::cout << embedding_weight_[0][i] << std::endl;
          // }
          // std::cout << std::endl;
          emb = torch::nn::Embedding::from_pretrained(embedding_weight_);
        } catch (const c10::Error &e) {
          std::cerr << "error loading the model: " << e.what_without_backtrace()
                    << std::endl;
          return;
        }

        printf("%s\n", "emb loaded");
        }
    void fillinfomap_from_modelinfo() {
        // 1. get prefillpart1 input info
        int input_num = prefill_model_part1_.GetInputNum();
        // std::cout << "Count of Input: " << input_num << std::endl;
        for (int idx = 0; idx < input_num; idx++) {
          auto input_name = prefill_model_part1_.GetInputName(idx);
          auto input_info =
              prefill_model_part1_.GetInputInfo(input_name).AsContiguous();
          // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
          auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
          prefill_part1_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 1. get prefillpart1 info
        int output_num = prefill_model_part1_.GetOutputNum();
        // std::cout << "Count of Output: " << output_num << std::endl;
        for (int idx = 0; idx < output_num; idx++) {
          auto output_name = prefill_model_part1_.GetOutputName(idx);
          auto output_info =
              prefill_model_part1_.GetOutputInfo(output_name).AsContiguous();
          // std::cout << "Output[" << output_name << "] " << output_info <<
          // std::endl;
          auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
          prefill_part1_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 2. get prefillpart2 input info
        input_num = prefill_model_part2_.GetInputNum();
        // std::cout << "Count of Input: " << input_num << std::endl;
        for (int idx = 0; idx < input_num; idx++) {
          auto input_name = prefill_model_part2_.GetInputName(idx);
          auto input_info =
              prefill_model_part2_.GetInputInfo(input_name).AsContiguous();
          // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
          auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
          prefill_part2_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 2. get prefillpart2 info
        output_num = prefill_model_part2_.GetOutputNum();
        // std::cout << "Count of Output: " << output_num << std::endl;
        for (int idx = 0; idx < output_num; idx++) {
          auto output_name = prefill_model_part2_.GetOutputName(idx);
          auto output_info =
              prefill_model_part2_.GetOutputInfo(output_name).AsContiguous();
          // std::cout << "Output[" << output_name << "] " << output_info <<
          // std::endl;
          auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
          prefill_part2_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 3. get decodepart1 input info
        input_num = decode_model_part1_.GetInputNum();
        // std::cout << "Count of Input: " << input_num << std::endl;
        for (int idx = 0; idx < input_num; idx++) {
          auto input_name = decode_model_part1_.GetInputName(idx);
          auto input_info =
              decode_model_part1_.GetInputInfo(input_name).AsContiguous();
          // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
          auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
          decode_part1_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 3. get decodepart1 info
        output_num = decode_model_part1_.GetOutputNum();
        // std::cout << "Count of Output: " << output_num << std::endl;
        for (int idx = 0; idx < output_num; idx++) {
          auto output_name = decode_model_part1_.GetOutputName(idx);
          auto output_info =
              decode_model_part1_.GetOutputInfo(output_name).AsContiguous();
          // std::cout << "Output[" << output_name << "] " << output_info <<
          // std::endl;
          auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
          decode_part1_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 4. get decodepart2 input info
        input_num = decode_model_part2_.GetInputNum();
        // std::cout << "Count of Input: " << input_num << std::endl;
        for (int idx = 0; idx < input_num; idx++) {
          auto input_name = decode_model_part2_.GetInputName(idx);
          auto input_info =
              decode_model_part2_.GetInputInfo(input_name).AsContiguous();
          // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
          auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
          decode_part2_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get decodepart2 info
        output_num = decode_model_part2_.GetOutputNum();
        // std::cout << "Count of Output: " << output_num << std::endl;
        for (int idx = 0; idx < output_num; idx++) {
          auto output_name = decode_model_part2_.GetOutputName(idx);
          auto output_info =
              decode_model_part2_.GetOutputInfo(output_name).AsContiguous();
          // std::cout << "Output[" << output_name << "] " << output_info <<
          // std::endl;
          auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
          decode_part2_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }
    }
};

// 假设的分词器类
class AutoTokenizer {
public:
    AutoTokenizer() {
        // 初始化分词器
    }

    std::string apply_chat_template(const std::vector<std::pair<std::string, std::string>>& messages, bool tokenize, bool add_generation_prompt) {
        // 实现聊天模板应用逻辑
        return "";
    }

    std::vector<int> tokenize(const std::string& text) {
        // 实现分词逻辑
        return {};
    }

    std::string decode(const std::vector<int>& ids) {
        // 实现解码逻辑
        return "";
    }

    std::string eos_token;
};

// 假设的HmQwen类
class HmQwen3_8B {
public:
    HmQwen3_8B(const Args& args)
        : batch(args.batch), prefill_length(args.prefill_length), decode_length(args.decode_length),nblocks(args.nblocks),ncore(args.ncore) {
        printf("nblocks=%d,args.nblocks=%d \r\n",nblocks,args.nblocks);
        std::string HMM_MODEL_PATH = args.model_dir;
        tcim::Module::WeightManager weight_manager=tcim::Module::WeightManager::CreateWeightManager(0);
        tcim::Module::Option option1(weight_manager);
        tcim::Module::Option option2(weight_manager);
        std::vector<std::string> dummy_tensor_names;
        for (int i = 0; i < args.nblocks; ++i) {
            dummy_tensor_names.push_back("model_layers_" + std::to_string(i) + "_self_attn_kcache_input");
            dummy_tensor_names.push_back("model_layers_" + std::to_string(i) + "_self_attn_vcache_input");
        }
        option2.SetDummyTensors(dummy_tensor_names);
        prefill_model = tcim::Module::LoadFromFile((fs::path(HMM_MODEL_PATH) / "prefill.hmm").string(),option1);
        printf("++1 prefill model loaded\r\n");
        decode_model = tcim::Module::LoadFromFile((fs::path(HMM_MODEL_PATH) / "decoder.hmm").string(),option2);
        printf("++2 decode model loaded\r\n");
        fillinfomap_from_modelinfo();
        printf("++2.1 after fillmap\r\n");

        // 设置KVCache输入
        for (int i = 0; i < args.nblocks; ++i) {
            std::string kcache_input_name = "model_layers_" + std::to_string(i) + "_self_attn_kcache_input";
            std::string vcache_input_name = "model_layers_" + std::to_string(i) + "_self_attn_vcache_input";
            auto kcache = prefill_model.GetInput(kcache_input_name);
            decode_model.SetInput(kcache_input_name, kcache);
            auto vcache = prefill_model.GetInput(vcache_input_name);
            decode_model.SetInput(vcache_input_name, vcache);
        }

        // 设置decode输入
        std::vector<int16_t> current_length = {1};
        printf("++3\r\n");
        //TODO:szj 确认 shape 是1维的1 还是二维的1
        TensorConverter::FromVector(current_length, decode_input_map.at("current_length"));
        decode_model.SetInput("current_length", decode_input_map.at("current_length"));

        // 加载embedding权重
        // 这里应该实现加载embedding权重的逻辑
        printf("++4 EMBEDDING_PATH.string() is:%s\r\n",(HMM_MODEL_PATH+"/quant_embedding_traced.pt").c_str());
        torch::load(embedding_weight_,(HMM_MODEL_PATH+"/quant_embedding_traced.pt").c_str());
        try {
          //torch::load(input, input_path);
          embedding_weight_ = embedding_weight_.view({-1, 4096});
          // std::cout << "embedding_weight_ options: " << embedding_weight_.options() << "\n";
          // std::cout << "embedding_weight_ dim: " << embedding_weight_.dim() << "\n";
          // std::cout << "embedding_weight_ size: " << embedding_weight_.size(0) << "\n";
          // std::cout << "embedding content: " << std::endl;
          // for (int i = 0; i < 20; i++) {
          //     std::cout << embedding_weight_[0][i] << std::endl;
          // }
          // std::cout << std::endl;
          emb = torch::nn::Embedding::from_pretrained(embedding_weight_);
        } catch (const c10::Error &e) {
          std::cerr << "error loading the model: " << e.what_without_backtrace()
                    << std::endl;
          return;
        }
        printf("++line:%d %s\n",__LINE__ ,"emb loaded");
    }

    void fillinfomap_from_modelinfo() {
        // 2. get input info
        int input_num = prefill_model.GetInputNum();
        printf("input_num is：%d\r\n",input_num);
        // std::cout << "Count of Input: " << input_num << std::endl;
        for (int idx = 0; idx < input_num; idx++) {
        auto input_name = prefill_model.GetInputName(idx);
        auto input_info =
            prefill_model.GetInputInfo(input_name).AsContiguous();
        // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        prefill_input_map.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get output info
        int output_num = prefill_model.GetOutputNum();
        printf("output_num is：%d\r\n",output_num);
        // std::cout << "Count of Output: " << output_num << std::endl;
        for (int idx = 0; idx < output_num; idx++) {
        auto output_name = prefill_model.GetOutputName(idx);
        auto output_info =
            prefill_model.GetOutputInfo(output_name).AsContiguous();
        // std::cout << "Output[" << output_name << "] " << output_info <<
        // std::endl;
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        prefill_input_map.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 2. get input info
        int input_num_decode = decode_model.GetInputNum();
        printf("input_num_decode is：%d\r\n",input_num_decode);
        // std::cout << "Count of Input: " << input_num_decode << std::endl;
        for (int idx = 0; idx < input_num_decode; idx++) {
        auto input_name = decode_model.GetInputName(idx);
        auto input_info =
            decode_model.GetInputInfo(input_name).AsContiguous();
        // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        decode_input_map.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get output info
        int output_num_decode = decode_model.GetOutputNum();
        printf("output_num_decode is：%d\r\n",output_num_decode);
        // std::cout << "Count of Output: " << output_num_decode << std::endl;
        for (int idx = 0; idx < output_num_decode; idx++) {
        auto output_name = decode_model.GetOutputName(idx);
        auto output_info =
            decode_model.GetOutputInfo(output_name).AsContiguous();
        // std::cout << "Output[" << output_name << "] " << output_info <<
        // std::endl;
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        decode_input_map.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

    }

    std::tuple<std::string, int, double, double> chat(const std::string& question) {
        std::cout << "Question: " << question << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<int> all_input_ids ={151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198, 108386,   3837, 105043, 100165,
          11319, 151645,    198, 151644,  77091,    198};// tokenizer->tokenize(question);
        int input_echo_len = all_input_ids.size();
        if (input_echo_len >= decode_length) {
            std::cout << "Question long than " << decode_length << ", please shorten it!" << std::endl;
            return {"Question long than " + std::to_string(decode_length) + ", please shorten it!", 0, 0.0, 0.0};
        }

        int prefill_loop_round = std::ceil(static_cast<double>(input_echo_len) / prefill_length);
        for (int round = 0; round < prefill_loop_round; ++round) {
            int16_t valid_length = round * prefill_length;
            int16_t current_length;
            std::vector<int> input_ids;
            if (round == prefill_loop_round - 1) {
                current_length = input_echo_len - round * prefill_length;
                input_ids = std::vector<int>(all_input_ids.begin() + round * prefill_length, all_input_ids.end());
            } else {
                current_length = prefill_length;
                input_ids = std::vector<int>(all_input_ids.begin() + round * prefill_length, all_input_ids.begin() + (round + 1) * prefill_length);
            }

            // 这里应该实现embedding逻辑
            // 准备输入嵌入
            auto opts = torch::TensorOptions().dtype(torch::kInt);
            auto inp = torch::from_blob(input_ids.data(),
                                    {int64_t(input_ids.size())}, opts)
                    .clone();
            printf("@@2.1 after round input_ids.size is:%d\r\n",input_ids.size());
            torch::Tensor inputs_embeds = emb->forward(inp);//torch::embedding(torch::tensor(input_ids), embedding_weight_);
            printf("@@2.1.1 \r\n");
            int effective_length = input_ids.size();
            torch::Tensor pad_embeds = torch::zeros({prefill_length - effective_length, inputs_embeds.size(-1)}, inputs_embeds.options());
            printf("@@2.1.2 effective_length=%d inputs_embeds.size(-1)=%d\r\n",effective_length,inputs_embeds.size(-1));

            std::cout << "inputs_embeds shape: " << inputs_embeds.sizes() << std::endl;
            std::cout << "pad_embeds shape: " << pad_embeds.sizes() << std::endl;
            //打印torch tensor
            //print_tensor(inputs_embeds);
            // 检查是否有空张量
            if (inputs_embeds.numel() == 0 ) {
                std::cerr << "Empty inputs_embeds tensor detected!" << std::endl;
                //return;
            }
            if (pad_embeds.numel() == 0) {
                std::cerr << "Empty pad_embeds tensor detected!" << std::endl;
                //return;
            }

            //torch::Tensor input_data = torch::cat({inputs_embeds, pad_embeds}, 0).reshape({4, prefill_length / 4, 4096});
            torch::Tensor input_data = torch::cat({inputs_embeds, pad_embeds});
            input_data=input_data.view({4, prefill_length / 4, 4096}).to(torch::kInt16);

            printf("@@--2.2\r\n");
            // 设置并运行模型
            std::vector<int16_t> valid_length_data = {valid_length};
            std::vector<int16_t> current_length_data = {current_length};

            printf("@@3 valid_length=%d  current_length=%d\r\n",valid_length,current_length);

            TensorConverter::FromTorch(input_data.to(torch::kInt16),prefill_input_map.at("input_1"));
            TensorConverter::FromVector(valid_length_data, prefill_input_map.at("valid_length"));
            TensorConverter::FromVector(current_length_data,prefill_input_map.at("current_length"));
            prefill_model.SetInput("input_1", prefill_input_map.at("input_1"));
            prefill_model.SetInput("valid_length", prefill_input_map.at("valid_length"));
            prefill_model.SetInput("current_length", prefill_input_map.at("current_length"));
            prefill_model.Run();
            prefill_model.Sync();
        }

        input_echo_len = static_cast<int>(all_input_ids.size());

        auto prefill_output_tcimtensor = prefill_model.GetOutput("Output_lm_head_add_list_0");
        auto prefill_output_tensor=TensorConverter::ToTorch(prefill_output_tcimtensor);

        auto selected_output =
        torch::argmax(prefill_output_tensor).to(torch::kInt32);
        // std::cout << "selected_output: " << selected_output << std::endl;
        // 下面打印的selected_output是乱码，index
        // int值，调下面的common_token_to_piece转成字符串
        printf("++selected_output(tensor addr not selected index) is:%u\r\n", selected_output);
        int next_id = *(int32_t *)selected_output.data_ptr<int32_t>();
        printf("++prefill selected tensor index is:%d\r\n",next_id);

        //all_responses[b] = tokenizer.decode({next_ids[b]});
        //准备decode的输入
        torch::Tensor next_id_tensor = torch::tensor({next_id});
        torch::Tensor decode_input = emb->forward(next_id_tensor.to(torch::kInt));//.reshape({1});//torch::embedding(next_id_tensor, embedding_weight_).reshape({1, 1, -1});
        decode_input = decode_input.view({1, 1, -1}).to(torch::kInt16);

        auto prefill_end_time = std::chrono::high_resolution_clock::now();
        double prefill_time = std::chrono::duration<double>(prefill_end_time - start_time).count();

        std::vector<int> chat_history_ids = all_input_ids;
        chat_history_ids.push_back(next_id);

        int context_length = input_echo_len;
        int decode_count = 0;
        int skip_tokens = 0;
        int slide_len = 10;
        int decode_loopcount=0;
        auto decode_start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            if (context_length >= decode_length) {
                std::cout << "context length greater than " << decode_length << ", break!" << std::endl;
                break;
            }
            printf("======decode_loopcount:%d context_length=%d\r\n",decode_loopcount,context_length);
            TensorConverter::FromTorch(decode_input.to(torch::kInt16),decode_input_map.at("input_1"));
            decode_model.SetInput("input_1", decode_input_map.at("input_1"));
            std::vector<int16_t> valid_length_data = {static_cast<int16_t>(context_length-1)};
            TensorConverter::FromVector(valid_length_data,decode_input_map.at("valid_length"));
            decode_model.SetInput("valid_length", decode_input_map.at("valid_length"));
            decode_model.Run();
            decode_model.Sync();

            auto decode_output_tcimtensor = decode_model.GetOutput("Output_lm_head_add_list_0");
            auto decode_output_tensor=TensorConverter::ToTorch(decode_output_tcimtensor);
            std::cout << "decode_output_tensor Shape: " << decode_output_tensor.sizes() << std::endl;

            selected_output =
            torch::argmax(decode_output_tensor).to(torch::kInt32);
            // std::cout << "selected_output: " << selected_output << std::endl;
            // 下面打印的selected_output是乱码，index
            // int值，调下面的common_token_to_piece转成字符串
            printf("++selected_output(tensor addr not selected index) is:%u\r\n", selected_output);
            next_id = *(int32_t *)selected_output.data_ptr<int32_t>();
            printf("--decode selected tensor index is:%d\r\n",next_id);
            chat_history_ids.push_back(next_id);
            //TODO:szj 改成实际的id-<|im_end|>
            if (next_id==151645) {
                std::cout<<" end with eostokenid:"<<next_id<<std::endl;
                //TODO:确认eos的batch，是否还要送回模型decode
                break;
            } else {
                context_length++;
                decode_count++;
                decode_loopcount++;
            }
            // 实现嵌入逻辑
            // std::vector<float> input_data = ...
            std::cout<<"decode emb forward nextid:"<<next_id<<std::endl;
            next_id_tensor = torch::tensor({next_id});
            decode_input = emb->forward(next_id_tensor.to(torch::kInt));//.reshape({1});//torch::embedding(next_id_tensor, embedding_weight_).reshape({1, 1, -1});
            decode_input = decode_input.view({1, 1, -1}).to(torch::kInt16);

            // std::string decode_response = "onlyfortest";
            // if (!decode_response.empty() && is_valid_char(decode_response.back())) {
            //     std::cout << decode_response<<std::endl;
            //     skip_tokens = 0;
            // } else {
            //     printf("++invalid outtokens\r\n");
            //     skip_tokens++;
            // }

        }

        auto decode_end_time = std::chrono::high_resolution_clock::now();
        double decode_time = std::chrono::duration<double>(decode_end_time - decode_start_time).count();

        printf("========decode output ids is:\r\n");
        for (const auto& num : chat_history_ids) {
        std::cout << num << " , ";
        }
        std::cout << std::endl;
        std::cout <<"final all token count is:"<<chat_history_ids.size()<< std::endl;

        return {"only for test", decode_count + 1, prefill_time, decode_time};
    }

private:
    int batch;
    int prefill_length;
    int decode_length;
    int nblocks;
    int ncore;
    tcim::Module prefill_model;

    tcim::Module decode_model;
    torch::nn::Embedding emb = nullptr;
    torch::Tensor embedding_weight_;
    AutoTokenizer tokenizer;
    std::map<std::string, tcim::Tensor> prefill_input_map;
    std::map<std::string, tcim::Tensor> decode_input_map;
};

// 假设的HmQwen类
class HmQwen {
public:
    HmQwen(const Args& args)
        : batch(args.batch), prefill_length(args.prefill_length), decoder_length(args.decode_length), output("Output_lm_head_add_list_0") {
        // 初始化权重管理器和选项
        tcim::Module::WeightManager weight_manager=tcim::Module::WeightManager::CreateWeightManager(0);
        tcim::Module::Option option1(weight_manager);
        tcim::Module::Option option2(weight_manager);
        std::vector<std::string> dummy_tensor_names;
        for (int i = 0; i < args.nblocks; ++i) {
            dummy_tensor_names.push_back("model_layers_" + std::to_string(i) + "_self_attn_kcache_input");
            dummy_tensor_names.push_back("model_layers_" + std::to_string(i) + "_self_attn_vcache_input");
        }
        option2.SetDummyTensors(dummy_tensor_names);

        //std::string HMM_MODEL_PATH = args.model_dir.substr(3) + "_" + std::to_string(args.ncore) + "core"+ "_" + std::to_string(args.args.batch) + "args.batch";
        std::string HMM_MODEL_PATH = args.model_dir;
        //std::string EMBEDDING_PATH = args.model_dir + "/quant_embedding.pt";
        std::string EMBEDDING_PATH = args.tokenizer_path;
        printf("###--0 args.nblocks=%d args.batch=%d\r\n",args.nblocks,args.batch);
        for (int b = 0; b < args.batch; ++b) {
            //std::string prefill_modelname="models\\hmm_xh2_qwen2.5_256_8k_2cores_20250610\\prefill.hmm";
            std::string prefill_modelname=(fs::path(HMM_MODEL_PATH) / "prefill.hmm").string();
            printf("++b=%d load prefill model=%s\r\n",b,prefill_modelname.c_str());
            prefill_models.emplace_back(tcim::Module::LoadFromFile(prefill_modelname,option1));
        }
        std::string decode_modelname=(fs::path(HMM_MODEL_PATH) / "decoder.hmm").string();
        printf("before load decode model path:%s\r\n",decode_modelname.c_str());
        decode_model = tcim::Module::LoadFromFile(decode_modelname,option2);
        printf("after load decode model\r\n");
        printf("++1\r\n");
        fillinfomap_from_modelinfo();
        printf("++2\r\n");
        exit(-1);
        for (int b = 0; b < args.batch; ++b) {
            for (int i = 0; i < args.nblocks; ++i) {
                auto kcache = prefill_models[b].GetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_input");
                decode_model.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_input_batch" + std::to_string(b), kcache);
                auto vcache = prefill_models[b].GetInput("model_layers_" + std::to_string(i) + "_self_attn_vcache_input");
                decode_model.SetInput("model_layers_" + std::to_string(i) + "_self_attn_vcache_input_batch" + std::to_string(b), vcache);
            }
        }
        //std::vector<int> current_length = {1, 1, 1, 1};
        #if 0//(4==args.batch)
            std::vector<int16_t> current_length = {1, 1, 1, 1};
        #else
            std::vector<int16_t> current_length = {1};
        #endif
        printf("++3\r\n");
        //TODO:szj 确认 shape 是1维的1 还是二维的1
        TensorConverter::FromVector(current_length, decode_input_map.at("current_length"));
        decode_model.SetInput("current_length", decode_input_map.at("current_length"));

        // 加载tokenizer和embedding weight
        //tokenizer_ = transformers::Tokenizer::from_pretrained(tokenizer_path.string(), true);
        printf("++4 EMBEDDING_PATH.string() is:%s\r\n",EMBEDDING_PATH.c_str());
        torch::load(embedding_weight_,EMBEDDING_PATH);
        //torch::Tensor input;
        // std::string input_path = "./qembedding.pt";
        //std::string input_path = model_path + "hmquant/quant_qembedding.pt";

        try {
          //torch::load(input, input_path);
          embedding_weight_ = embedding_weight_.view({-1, 3584});
          // std::cout << "embedding_weight_ options: " << embedding_weight_.options() << "\n";
          // std::cout << "embedding_weight_ dim: " << embedding_weight_.dim() << "\n";
          // std::cout << "embedding_weight_ size: " << embedding_weight_.size(0) << "\n";
          // std::cout << "embedding content: " << std::endl;
          // for (int i = 0; i < 20; i++) {
          //     std::cout << embedding_weight_[0][i] << std::endl;
          // }
          // std::cout << std::endl;
          emb = torch::nn::Embedding::from_pretrained(embedding_weight_);
        } catch (const c10::Error &e) {
          std::cerr << "error loading the model: " << e.what_without_backtrace()
                    << std::endl;
          return;
        }

        printf("++line:%d %s\n",__LINE__ ,"emb loaded");
    }
    void fillinfomap_from_modelinfo() {
        // 2. get input info
        int input_num = prefill_models[0].GetInputNum();
        // std::cout << "Count of Input: " << input_num << std::endl;
        for (int idx = 0; idx < input_num; idx++) {
        auto input_name = prefill_models[0].GetInputName(idx);
        auto input_info =
            prefill_models[0].GetInputInfo(input_name).AsContiguous();
        std::cout << "PrefillModel Input[" <<idx<<"] withname:["<< input_name << "] " << input_info << std::endl;
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        prefill_input_map.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get output info
        int output_num = prefill_models[0].GetOutputNum();
        std::cout << "Count of Output: " << output_num << std::endl;
        for (int idx = 0; idx < output_num; idx++) {
        auto output_name = prefill_models[0].GetOutputName(idx);
        auto output_info =
            prefill_models[0].GetOutputInfo(output_name).AsContiguous();
        std::cout << "PrefillModel Output[" <<idx<<"] withname:["<< output_name << "] " << output_info <<
        std::endl;
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        prefill_input_map.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 2. get input info
        int input_num_decode = decode_model.GetInputNum();
        // std::cout << "Count of Input: " << input_num_decode << std::endl;
        for (int idx = 0; idx < input_num_decode; idx++) {
        auto input_name = decode_model.GetInputName(idx);
        auto input_info =
            decode_model.GetInputInfo(input_name).AsContiguous();
        std::cout << "DecodeModel Input[" <<idx<<"] withname:["<<  input_name << "] " << input_info << std::endl;
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        decode_input_map.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get output info
        int output_num_decode = decode_model.GetOutputNum();
        // std::cout << "Count of Output: " << output_num_decode << std::endl;
        for (int idx = 0; idx < output_num_decode; idx++) {
        auto output_name = decode_model.GetOutputName(idx);
        auto output_info =
            decode_model.GetOutputInfo(output_name).AsContiguous();
        std::cout << "DecodeModel Output[" <<idx<<"] withname:["<<  output_name << "] " << output_info <<
        std::endl;
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        decode_input_map.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

    }

    std::tuple<std::vector<std::string>, int, int, double, double> chat(const std::vector<std::string>& questions, int nblocks = 28) {
        std::cout << "questions: \r\n";
        for (const auto& q : questions) {
            std::cout <<"question:" <<q << " "<<std::endl;
        }
        std::cout << std::endl;

        std::vector<std::vector<std::pair<std::string, std::string>>> messages(batch);
        for (int b = 0; b < batch; ++b) {
            messages[b] = {{"system", "You are a helpful assistant."}, {"user", questions[b]}};
        }

        std::vector<std::string> texts(batch);
        for (int b = 0; b < batch; ++b) {
            texts[b] = tokenizer.apply_chat_template(messages[b], false, true);
        }

        //std::vector<std::vector<int>> all_input_ids(batch);
        //14B tokenizer 的输出
        //std::vector<std::vector<int>> all_input_ids = {
        //       {151646,   2610,    525,    264,  10950,  17847,     13, 151644, 108386,
        //       3837, 105043, 100165,  11319, 151645, 151648,    198},
        //       {151646,   2610,    525,    264,  10950,  17847,     13, 151644, 108386,
        //       3837, 105043, 100165,  11319, 151645, 151648,    198},
        //       {151646,   2610,    525,    264,  10950,  17847,     13, 151644, 108386,
        //       3837, 105043, 100165,  11319, 151645, 151648,    198},
        //       {151646,   2610,    525,    264,  10950,  17847,     13, 151644, 108386,
        //       3837, 105043, 100165,  11319, 151645, 151648,    198}
        //   };
        //下面的inputid 对应的token是: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n
        std::vector<std::vector<int>> all_input_ids = {
            {151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
             151645,    198, 151644,    872,    198, 105043, 100165,  11319, 151645,
                198, 151644,  77091,    198},
            {151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
             151645,    198, 151644,    872,    198, 105043, 100165,  11319, 151645,
                198, 151644,  77091,    198},
            {151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
             151645,    198, 151644,    872,    198, 105043, 100165,  11319, 151645,
                198, 151644,  77091,    198},
            {151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
             151645,    198, 151644,    872,    198, 105043, 100165,  11319, 151645,
                198, 151644,  77091,    198},
        };
        std::vector<int16_t> input_echo_lens(batch);
        for (int b = 0; b < batch; ++b) {
            //all_input_ids[b] = tokenizer.tokenize(texts[b]);
            input_echo_lens[b] = all_input_ids[b].size();
            if (input_echo_lens[b] >= decoder_length) {
                std::cerr << "batch: " << b << " Question too long, please shorten it!" << std::endl;
                return {};
            } else {
                std::cout << "batch: " << b <<" lenth is:"<<input_echo_lens[b] << std::endl;
            }
        }

        std::vector<std::string> all_responses(batch);
        std::vector<int> next_ids(batch);
        std::vector<double> start_times(batch);
        std::vector<double> prefill_times(batch);
        std::vector<torch::Tensor> decode_inputs;
        for (int b = 0; b < batch; ++b) {
            int prefill_loop_round = std::ceil(static_cast<double>(input_echo_lens[b]) / prefill_length);
            start_times[b] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            for (int round = 0; round < prefill_loop_round; ++round) {
                int16_t valid_length = round * prefill_length;
                int16_t current_length;
                std::vector<int> input_ids;
                if (round == prefill_loop_round - 1) {
                    current_length = input_echo_lens[b] - round * prefill_length;
                    input_ids = std::vector<int>(all_input_ids[b].begin() + round * prefill_length, all_input_ids[b].end());
                } else {
                    current_length = prefill_length;
                    input_ids = std::vector<int>(all_input_ids[b].begin() + round * prefill_length, all_input_ids[b].begin() + (round + 1) * prefill_length);
                }

                // 实现嵌入逻辑
                // std::vector<float> inputs_embeds = ...
                // 准备输入嵌入
                auto opts = torch::TensorOptions().dtype(torch::kInt);
                auto inp = torch::from_blob(input_ids.data(),
                                        {int64_t(input_ids.size())}, opts)
                        .clone();
                printf("@@2.1 after round input_ids.size is:%d\r\n",input_ids.size());
                torch::Tensor inputs_embeds = emb->forward(inp);//torch::embedding(torch::tensor(input_ids), embedding_weight_);
                printf("@@2.1.1 \r\n");
                int effective_length = input_ids.size();
                torch::Tensor pad_embeds = torch::zeros({prefill_length - effective_length, inputs_embeds.size(-1)}, inputs_embeds.options());
                printf("@@2.1.2 effective_length=%d inputs_embeds.size(-1)=%d\r\n",effective_length,inputs_embeds.size(-1));

                std::cout << "inputs_embeds shape: " << inputs_embeds.sizes() << std::endl;
                std::cout << "pad_embeds shape: " << pad_embeds.sizes() << std::endl;
                //打印torch tensor
                //print_tensor(inputs_embeds);
                // 检查是否有空张量
                if (inputs_embeds.numel() == 0 ) {
                    std::cerr << "Empty inputs_embeds tensor detected!" << std::endl;
                    //return;
                }
                if (pad_embeds.numel() == 0) {
                    std::cerr << "Empty pad_embeds tensor detected!" << std::endl;
                    //return;
                }

                //torch::Tensor input_data = torch::cat({inputs_embeds, pad_embeds}, 0).reshape({4, prefill_length / 4, 3584});
                torch::Tensor input_data = torch::cat({inputs_embeds, pad_embeds});
                input_data=input_data.view({4, prefill_length / 4, 3584}).to(torch::kInt16);
                //printf("----input_data after .view to int16:\r\n");
                //print_tensor(input_data);

                //reshape({4, prefill_length / 4, 3584});
                printf("@@--2.2\r\n");
                // 设置并运行模型
                std::vector<int16_t> valid_length_data = {valid_length};
                std::vector<int16_t> current_length_data = {current_length};
                // printf("+++valid_length_data info:\n");
                // printVector(valid_length_data);

                // printf("+++current_length_data info:\n");
                // printVector(current_length_data);


                printf("@@3 valid_length=%d  current_length=%d\r\n",valid_length,current_length);
                // 实现嵌入end


                //effective_length = input_ids.size();
                //std::vector<float> _pad_embeds(3584 * (prefill_length - effective_length), 0.0);
                // std::vector<float> input_data = ...

                TensorConverter::FromTorch(input_data.to(torch::kInt16),prefill_input_map.at("input_1"));
                TensorConverter::FromVector(valid_length_data, prefill_input_map.at("valid_length"));
                TensorConverter::FromVector(current_length_data,prefill_input_map.at("current_length"));

                prefill_models[b].SetInput("input_1", prefill_input_map.at("input_1"));
                prefill_models[b].SetInput("valid_length", prefill_input_map.at("valid_length"));
                prefill_models[b].SetInput("current_length", prefill_input_map.at("current_length"));
                prefill_models[b].Run();
                prefill_models[b].Sync();
            }

            auto prefill_output_tcimtensor = prefill_models[b].GetOutput(output);
            auto prefill_output_tensor=TensorConverter::ToTorch(prefill_output_tcimtensor);
            prefill_times[b] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start_times[b];

            auto selected_output =
            torch::argmax(prefill_output_tensor).to(torch::kInt32);
            // std::cout << "selected_output: " << selected_output << std::endl;
            // 下面打印的selected_output是乱码，index
            // int值，调下面的common_token_to_piece转成字符串
            printf("++selected_output(tensor addr not selected index) is:%u\r\n", selected_output);
            int next_id = *(int32_t *)selected_output.data_ptr<int32_t>();
            printf("++prefill[%d] selected tensor index is:%d\r\n",b,next_id);

            next_ids[b] = next_id;
            //all_responses[b] = tokenizer.decode({next_ids[b]});
            //准备decode的输入
            torch::Tensor next_id_tensor = torch::tensor({next_id});
            torch::Tensor decode_input = emb->forward(next_id_tensor.to(torch::kInt));//.reshape({1});//torch::embedding(next_id_tensor, embedding_weight_).reshape({1, 1, -1});
            decode_input = decode_input.view({1, 1, -1}).to(torch::kInt16);
            decode_inputs.push_back(std::move(decode_input));
        }

        std::vector<int16_t> input_datas(batch*3584, 0);
        for (int b = 0; b < batch; ++b) {
            // 组装嵌入的结果
            // input_datas[b] = ...
            int16toint16((int16_t*)input_datas.data()+3584*b,
                (int16_t *)decode_inputs[b].data_ptr<int16_t>(),
                decode_inputs[b].numel());
        }

        std::vector<int16_t> context_lengths = input_echo_lens;
        std::vector<int> decode_counts(batch, 0);
        double decode_times = 0.0;
        std::vector<bool> decode_break(batch, false);
        int decode_loopcount=0;
        while (true) {
            if (*std::max_element(context_lengths.begin(), context_lengths.end()) >= decoder_length) {
                std::cout << "context length greater than " << decoder_length << ", break!" << std::endl;
                break;
            }
            std::cout<<"=============decode loop count："<<decode_loopcount<<std::endl;
            std::vector<int16_t> valid_length_data = context_lengths;
            // std::vector<std::vector<float>> input_datas_combined;
            // for (const auto& data : input_datas) {
            //     input_datas_combined.insert(input_datas_combined.end(), data.begin(), data.end());
            // }

            auto start_time = std::chrono::system_clock::now();
            std::cout<<"decode mode input vector data size is:"<<input_datas.size()<<std::endl;
            TensorConverter::FromVector(input_datas,decode_input_map.at("input_1"));
            TensorConverter::FromVector(valid_length_data, decode_input_map.at("valid_length"));

            decode_model.SetInput("input_1", decode_input_map.at("input_1"));
            decode_model.SetInput("valid_length", decode_input_map.at("valid_length"));
            decode_model.Run();
            decode_model.Sync();

            auto decode_output_tcimtensor = decode_model.GetOutput(output);
            auto decode_output_tensor=TensorConverter::ToTorch(decode_output_tcimtensor);
            std::cout << "decode_output_tensor Shape: " << decode_output_tensor.sizes() << std::endl;

            auto end_time = std::chrono::system_clock::now();
            double decode_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            decode_times += decode_time;
            //std::vector<int> next_ids(batch);
            std::vector<torch::Tensor> decode_output_tensor_rows_split = torch::split(decode_output_tensor, 1, 0);

            for (int b = 0; b < batch; ++b) {
                auto selected_output =
                torch::argmax(decode_output_tensor_rows_split[b]).to(torch::kInt32);
                // std::cout << "selected_output: " << selected_output << std::endl;
                // 下面打印的selected_output是乱码，index
                // int值，调下面的common_token_to_piece转成字符串
                printf("++selected_output(tensor addr not selected index) is:%u\r\n", selected_output);
                int next_id = *(int32_t *)selected_output.data_ptr<int32_t>();
                printf("##decode[%d] selected tensor index is:%d\r\n",b,next_id);
                next_ids[b] = next_id;
            }
            //decode_responses[b] = tokenizer.decode({next_ids[b]});
            //input_datas.clear();
            decode_inputs.clear();
            for (int b = 0; b < batch; ++b) {
                //TODO:szj 改成实际的id
                if (next_ids[b]==151643) {
                    std::cout<<"batch:"<<b<<" end with eostokenid:"<<next_ids[b]<<std::endl;
                    //TODO:确认eos的batch，是否还要送回模型decode
                    decode_break[b] = true;
                } else {
                    context_lengths[b]++;
                    decode_counts[b]++;
                }
                // 实现嵌入逻辑
                // std::vector<float> input_data = ...
                std::cout<<"decode emb forward nextid:"<<next_ids[b]<<std::endl;
                torch::Tensor next_id_tensor = torch::tensor({next_ids[b]});
                torch::Tensor decode_input = emb->forward(next_id_tensor.to(torch::kInt));//.reshape({1});//torch::embedding(next_id_tensor, embedding_weight_).reshape({1, 1, -1});
                decode_input = decode_input.view({1, 1, -1}).to(torch::kInt16);
                decode_inputs.push_back(std::move(decode_input));
                // if (!decode_break[b]) {
                //     all_responses[b] += decode_responses[b];
                // }
            }

            if (std::all_of(decode_break.begin(), decode_break.end(), [](bool b) { return b; })) {
                std::cout<<"All batch decoder end"<<std::endl;
                break;
            }

            for (int b = 0; b < batch; ++b) {
                // 实现嵌入逻辑
                // input_datas[b] = ...
                int16toint16((int16_t*)input_datas.data()+3584*b,
                    (int16_t *)decode_inputs[b].data_ptr<int16_t>(),
                    decode_inputs[b].numel());
            }
            decode_loopcount++;

        }

        int input_tokens = std::accumulate(input_echo_lens.begin(), input_echo_lens.end(), 0);
        int output_tokens = std::accumulate(decode_counts.begin(), decode_counts.end(), 0);
        double prefill_time = std::accumulate(prefill_times.begin(), prefill_times.end(), 0.0);
        all_responses[0]="only for test";
        return {all_responses, input_tokens, output_tokens, prefill_time, decode_times};
    }

private:
    int batch;
    int prefill_length;
    int decoder_length;
    std::string output;
    std::vector<tcim::Module> prefill_models;

    tcim::Module decode_model;
    torch::nn::Embedding emb = nullptr;
    torch::Tensor embedding_weight_;
    AutoTokenizer tokenizer;
    std::map<std::string, tcim::Tensor> prefill_input_map;
    std::map<std::string, tcim::Tensor> decode_input_map;
    // std::vector<float> embedding_weight;
};

// 读取JSON文件
std::vector<std::string> read_example_json(const std::string& file_path) {
    // std::ifstream file(file_path);
    // std::stringstream buffer;
    // buffer << file.rdbuf();
    // std::string json_str = buffer.str();
    // 实现JSON解析逻辑
    return {};
}

// 写入JSON文件
void write_performance_results(const std::string& file_path, const std::string& model_name, const std::string& data) {
    return;
}


//tcim tensor and  torch tensor converter
//将torch::Tensor转换为tcim::Tensor
void TensorConverter::FromTorch(const torch::Tensor& torch_tensor, tcim::Tensor &tcim_tensor) {
    if(0==torch_tensor.numel()){
        printf("+++Error torch_tensor.numel() is 0\r\n");
    }
    int16toint16((int16_t *)tcim_tensor.Data(),
                (int16_t *)torch_tensor.data_ptr<int16_t>(),
                torch_tensor.numel());
}

// 将tcim::Tensor转换为torch::Tensor
torch::Tensor TensorConverter::ToTorch(const tcim::Tensor& tcim_tensor) {
    // 获取tcim tensor的形状
    auto info = tcim_tensor.Info();
    auto shape = info.Shape();

    // 获取tcim tensor的数据类型并转换为torch类型
    torch::ScalarType dtype;
    switch(info.DataType()) {
        case tcim::FLOAT32: dtype = torch::kFloat32; break;
        case tcim::FLOAT16: dtype = torch::kFloat16; break;
        case tcim::INT32: dtype = torch::kInt32; break;
        case tcim::INT16: dtype = torch::kInt16; break;
        case tcim::INT8: dtype = torch::kInt8; break;
        case tcim::UINT8: dtype = torch::kUInt8; break;
        default:
            throw std::runtime_error("Unsupported tensor type");
    }

    // 创建torch tensor
    auto options = torch::TensorOptions().dtype(dtype);
    auto torch_tensor = torch::empty(
        std::vector<int64_t>(shape.begin(), shape.end()),
        options
    );

    // 复制数据
    void* data = tcim_tensor.Data();
    memcpy(torch_tensor.data_ptr(), data, tcim_tensor.Info().MemSize());

    return torch_tensor;
}

// 将vector<float>转换为tcim::Tensor
//template<typename T>
// tcim::Tensor TensorConverter::FromVector(const std::vector<int16_t>& vec, const std::vector<uint64_t>& shape,
//                                HmQwen *hmqwen, const std::string &tcim_tensorname,int which_part) {
void TensorConverter::FromVector(const std::vector<int16_t>& vec, tcim::Tensor &tcim_tensor) {
    if(0==vec.size()){
        printf("!!!Error vector size is 0,something wrong \r\n");
    }
    int16toint16((int16_t *) tcim_tensor.Data(),
                (int16_t*)vec.data(),
                vec.size());
}

// 将tcim::Tensor转换为vector<T>
//template<typename T>
std::vector<int16_t> TensorConverter::ToVector(const tcim::Tensor& tcim_tensor) {
    auto info = tcim_tensor.Info();
    // 检查数据类型是否匹配
    // bool type_matched = false;
    // if (typeid(T) == typeid(float) && info.DataType() == tcim::FLOAT32) {
    //     type_matched = true;
    // } else if (typeid(T) == typeid(int32_t) && info.DataType() == tcim::INT32) {
    //     type_matched = true;
    // } else if (typeid(T) == typeid(int16_t) && info.DataType() == tcim::INT16) {
    //     type_matched = true;
    // } else if (typeid(T) == typeid(int8_t) && info.DataType() == tcim::INT8) {
    //     type_matched = true;
    // }

    // if (!type_matched) {
    //     throw std::runtime_error("Tensor type does not match requested vector type");
    // }

    // 计算元素总数
    size_t element_count = info.MemSize() / sizeof(int16_t);
    std::cout<<"element_count is:"<<element_count<<std::endl;
    std::vector<int16_t> result(element_count);

    // 复制数据
    const void* data = tcim_tensor.Data();
    memcpy(result.data(), data, info.MemSize());

    return std::move(result);
}


// JSON处理辅助函数
void save_performance_data(const std::string& model_dir, const std::vector<double>& data1, const std::vector<double>& data2, double total_time) {
    try {
        // 计算性能指标
        int input_tokens = data1[0] + data2[0];
        int output_tokens = data1[1] + data2[1];
        double prefill_time = std::max(data1[2], data2[2]);
        double decode_start_time = std::min(data1[3], data2[3]);
        double decode_end_time = std::min(data1[4], data2[4]);
        double decode_time = decode_end_time - decode_start_time;
        //int tokens = std::min(data1[1], data2[1]) * 2;
        int tokens = std::min(data1[1], data2[1]);

        double decode_latency = decode_time * 1000 / (tokens - 1);
        double res_latency = total_time * 1000 / (data1[1] + data2[1]);

        // 输出性能指标
        std::cout << COLOR_BLUE << "total: " << tokens << " tokens, cost " << total_time << " s" << COLOR_RESET << std::endl;
        std::cout << COLOR_BLUE << "prefill time: " << prefill_time * 1000 << " ms, " << 1 / prefill_time << " tokens/s" << COLOR_RESET << std::endl;
        std::cout << COLOR_BLUE << "decode average time: " << decode_latency << " ms, " << 1000 / decode_latency << " tokens/s" << COLOR_RESET << std::endl;
        std::cout << COLOR_BLUE << "end2end average time: " << res_latency << " ms, " << 1000 / res_latency << " tokens/s" << COLOR_RESET << std::endl;

        // 保存到JSON文件
        // 注意：实际实现中需要使用真正的JSON库
        std::cout << "Saving performance data to performance_results.json" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving performance data: " << e.what() << std::endl;
    }
}

int print_torchversion() {
    // 打印 PyTorch 版本
    std::cout << "PyTorch 版本: " << TORCH_VERSION_MAJOR << "."
              << TORCH_VERSION_MINOR << "."
              << TORCH_VERSION_PATCH << std::endl;

    // 或者使用 torch::version::cudnn() 打印 CuDNN 版本等更多信息
    std::cout << "CUDA 可用: " << (torch::cuda::is_available() ? "是" : "否") << std::endl;
    //std::cout << "cuDNN 版本: " << torch::cuda::cudnn_version() << std::endl;

    return 0;
}

int main(int argc, char* argv[]) {
    print_torchversion();
    try {
        // 解析命令行参数
        Args args = get_args(argc, argv);
        //启动14B测试程序
        if(model_scale_type::MODEL_14B==args.hm_model_scale_type)
        {
        printf("=============Run MODEL_14B\r\n");
        // 初始化权重管理器
        tcim::Module::WeightManager weight_manager_0=tcim::Module::WeightManager::CreateWeightManager(0);
        tcim::Module::WeightManager weight_manager_1=tcim::Module::WeightManager::CreateWeightManager(1);

        // 创建管道和队列
        Pipe pipe_a, pipe_b;
        SafeQueue<std::vector<double>> queue1, queue2;
        AtomicFlag is_end(false);

        // 初始化模型
        HmQwen_14B::Config config;
        config.tokenizer_path = args.tokenizer_path;
        config.model_dir = args.model_dir;
        config.prefill_length = args.prefill_length;
        config.decode_length = args.decode_length;
        config.nblocks = args.nblocks;
        printf("tokenizer_path is %s\r\n",config.tokenizer_path.c_str());
        printf("model_dir is %s\r\n",config.model_dir.c_str());

        HmQwen_14B hmqwen1(config, &weight_manager_0, &weight_manager_1, 1);
        //HmQwen_14B hmqwen2(config, &weight_manager_0, &weight_manager_1, 2);

        // 定义问题
        std::string question1 = "你好，你是谁？";
        std::string question2 = "请介绍一下联想公司";

        // 创建线程
        auto start_time = high_resolution_clock::now();

        // std::thread t1([&]() {
        //     hmqwen1.chat(question1, &pipe_a, &queue1, true, &is_end);
        // });

        hmqwen1.chat(question1, &pipe_a, &queue1, true, &is_end);

        // std::thread t2([&]() {
        //     hmqwen2.chat(question2, &pipe_b, &queue2, false, &is_end);
        // });

        // 等待线程完成
        //t1.join();
        //t2.join();

        auto total_time = duration_cast<duration<double>>(high_resolution_clock::now() - start_time).count();

        // 获取结果
        printf("+++before queue1 get\r\n");
        auto data1 = queue1.get();
        printf("+++before queue2 get\r\n");
        //auto data2 = queue2.get();

        // 保存性能数据
        printf("+++before save performance data\r\n");
        save_performance_data(args.model_dir, data1, data1, total_time);

        return 0;
        }
        //启动7B测试程序
        else if(model_scale_type::MODEL_7B==args.hm_model_scale_type) {
            printf("=============Run MODEL_7B\r\n");
            HmQwen hmqwen(args);
            std::string model_name = args.model_dir.substr(3) + "_" + std::to_string(args.ncore) + "core";

            std::vector<std::string> example = read_example_json("example.json");

            //for (size_t i = 0; i < example.size(); i += 4) {
            for (size_t i = 0; i < 1; i += 4) {
                std::vector<std::string> questions={"你好，你是谁？","你好，你是谁？","你好，你是谁？","你好，你是谁？"};
                for (size_t j = i; j < i + 4 && j < example.size(); ++j) {
                    questions.push_back(example[j]);
                }

                auto start_time = std::chrono::system_clock::now();
                auto [response, input_tokens, output_tokens, prefill_time, decode_time] = hmqwen.chat(questions, args.nblocks);
                auto end_time = std::chrono::system_clock::now();
                double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

                auto [response2, input_tokens2, output_tokens2, prefill_time2, decode_time2] = hmqwen.chat(questions, args.nblocks);
                int consistency = (response == response2) ? 1 : 0;

                std::cout << "total: " << output_tokens << " tokens, cost " << total_time / 1000.0 << " s" << std::endl;
                std::cout << "prefill time: " << prefill_time << " ms, " << 1.0 / (prefill_time / 1000.0) << " tokens/s" << std::endl;
                double decode_latency = decode_time / (output_tokens - 1);
                std::cout << "decode average time: " << decode_latency << " ms, " << 1000.0 / decode_latency << " tokens/s" << std::endl;
                double res_latency = total_time / output_tokens;
                std::cout << "end2end average time: " << res_latency << " ms, " << 1000.0 / res_latency << " tokens/s" << std::endl;

                std::string data = "{\"input_tokens\": " + std::to_string(input_tokens) +
                                ", \"output_tokens\": " + std::to_string(output_tokens) +
                                ", \"prefill\": \"" + std::to_string(prefill_time) + " ms\", " +
                                "\"decoder\": \"" + std::to_string(1000.0 / decode_latency) + "\", " +
                                "\"e2e\": \"" + std::to_string(1000.0 / res_latency) + "\", " +
                                "\"consistency\": " + std::to_string(consistency) + "}";

                write_performance_results("performance_results.json", model_name, data);
            }

            return 0;
        }
        //启动qwen3 8B测试
        else if(model_scale_type::MODEL_8B) {
            printf("=============Run MODEL_8B\r\n");
            HmQwen3_8B hmqwen(args);
            std::string question = "你好，你是谁？";

            auto start_time = std::chrono::high_resolution_clock::now();
            auto [response, tokens, prefill_time, decode_time] = hmqwen.chat(question);
            auto end_time = std::chrono::high_resolution_clock::now();
            double total_time = std::chrono::duration<double>(end_time - start_time).count();

            std::cout << "Total: " << tokens << " tokens, cost " << total_time << " s" << std::endl;
            std::cout << "Prefill time: " << prefill_time * 1000 << " ms, " << 1 / prefill_time << " tokens/s" << std::endl;
            double decode_latency = decode_time * 1000 / (tokens - 1);
            std::cout << "Decode average time: " << decode_latency << " ms, " << 1000 / decode_latency << " tokens/s" << std::endl;
            double res_latency = total_time * 1000 / tokens;
            std::cout << "End2end average time: " << res_latency << " ms, " << 1000 / res_latency << " tokens/s" << std::endl;

            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
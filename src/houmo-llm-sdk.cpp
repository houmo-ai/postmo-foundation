#include "arg.h"
#include "common.h"
#include "hlie_app/llama_server.hpp"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "utils.hpp"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
// mime type for sending response
#define MIMETYPE_JSON "application/json; charset=utf-8"

#include <csignal>
#include <torch/torch.h>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "http_server_c_interface_3.h"
#include "INIReader.h"
#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <cctype>
#include <locale>
#include "tcim/tcim_runtime.h"

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <tchar.h>
#include <psapi.h>
#include <strsafe.h>
#else
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/memfd.h>
#endif

#include "hlie_base/hlie_config_service.h"

#if (__GNUC__ < 8 && !defined(_MSC_VER))
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

using json = nlohmann::ordered_json;

using namespace hm_HLIE;

// 全局配置变量
static bool model_real_load_flag = false;
static bool open_console_flag = false;


bool fileExists(const std::string& path) {
    const std::ifstream file(path);
    return file.good();  // 若文件打开成功，返回true
}

#ifdef _WIN32

bool IsLoadedFromConsole() {
    // 如果返回非NULL，表示调用进程有控制台窗口
    return GetConsoleWindow() != nullptr;
}

// 获取LastError的详细信息
std::wstring GetLastErrorAsString() {
    DWORD errorCode = GetLastError();
    if (errorCode == 0) return L"No error";

    LPWSTR messageBuffer = nullptr;
    size_t size = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, errorCode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPWSTR)&messageBuffer, 0, NULL
    );

    std::wstring message(messageBuffer, size);
    LocalFree(messageBuffer);
    return message;
}

// 导出函数，用于创建并打开新控制台窗口
extern "C" __declspec(dllexport) bool CreateNewConsole() {

    // 检查是否已有控制台
    if (IsLoadedFromConsole()) {
        std::wcout << "错误: 进程已有控制台，无法分配新的控制台窗口!" << std::endl;
        return false;
    }
    bool freeconsolestatus=FreeConsole();
    wchar_t buffer[256]={0};
    swprintf_s(buffer, L"FreeConsole return %d\r\n", freeconsolestatus);
    OutputDebugStringW(buffer);

    // 尝试分配新的控制台窗口
    if (!AllocConsole()) {
        DWORD error = GetLastError();
        std::wcout << "AllocConsole 失败，错误码: " << error << ", 详情: " << GetLastErrorAsString() << std::endl;
        OutputDebugStringW((L"AllocConsole Failed with reason:" + GetLastErrorAsString()).c_str());
        return false;
    }

    // 分配新的控制台窗口
    // 设置控制台标题
    SetConsoleTitleA("HoumoLLM DLL Debug Console");

    // 重定向标准输入输出流
    FILE* fp;

    // 重定向标准输出
    freopen_s(&fp, "CONOUT$", "w", stdout);
    std::cout.clear();

    // 重定向标准错误
    freopen_s(&fp, "CONOUT$", "w", stderr);
    std::cerr.clear();

    // 重定向标准输入
    freopen_s(&fp, "CONIN$", "r", stdin);
    std::cin.clear();

    SetConsoleOutputCP(CP_UTF8);
    std::cout << "新控制台窗口已创建!" << std::endl;
    std::cout << "DLL 控制台输出在此窗口" << std::endl;
    printf("Hello New window\r\n");
    return true;
}

// 导出函数，用于关闭控制台窗口
extern "C" __declspec(dllexport) void CloseConsole() {
    FreeConsole();
}

#endif


std::string getCurrentTime() {
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();

    // 转换为time_t类型用于获取日期和时间
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    // 计算毫秒部分
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;

    // 格式化输出
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << millis; // 添加毫秒部分
    return ss.str();
}


// trim from start (in place)
inline void ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
inline void rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

template <typename... Args>
std::string hm_string_format(const std::string& format, Args... args) {
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

template <typename T>
static void int16toint16(T* dst, T* src, const size_t size) {
    // 检查类型是否为 TriviallyCopyable（可直接内存复制）
    // printf("src=%p,dst=%p,size=%d\r\n",src,dst,size);
    if constexpr (std::is_trivially_copyable_v<T>) {
        // 使用 memcpy 进行高效内存复制
        memcpy(dst, src, size * sizeof(T));
    } else {
        // 对于不可直接复制的类型，使用循环
        for (size_t i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    }
}


std::string dump_result(const json& data) {
    const std::string str = data.dump(-1, ' ', false, json::error_handler_t::replace);
    // data.dump(-1, ' ', false, json::error_handler_t::replace) +
    //     "\n\n"; // required by RFC 8895 - A message is terminated by a blank
    //     line (two line terminators in a row).
    return str;
}

void log_error(const json& error_data) {
    // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
    const std::string str = error_data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    SRV_ERR("%s\n", str.c_str());
}

std::string getenv_or_default(const std::string& name, const std::string& default_value) {
    const char* value = std::getenv(name.c_str());
    return value ? std::string(value) : default_value;
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


// api_Generate 中调用了这个函数
int LlamaServer::completion_generate(json& data, std::vector<server_task_result_ptr>& results) {
    auto type = SERVER_TASK_TYPE_COMPLETION;
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    std::unique_lock<std::mutex> lock(mutex_context);
    generate_status = 1;
    model_status = 3;
    auto completion_id = gen_chatcmplid();
    std::vector<server_task> tasks;
    oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;

    try {
        const auto& prompt = data.at("prompt");
        std::vector<server_tokens> inputs = tokenize_input_prompts(vocab, mctx, prompt, true, true);
        tasks.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(ctx, params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);
            SRV_INF("++task.id_selected_slot is:%d \r\n", task.id_selected_slot);

            // OAI-compat
            task.params.oaicompat = oaicompat;
            task.params.oaicompat_cmpl_id = completion_id;
            // oaicompat_model is already populated by params_from_json_cmpl

            tasks.push_back(std::move(task));
        }
    } catch (const std::exception& e) {
        log_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return -1;
    }

    queue_results.add_waiting_tasks(tasks);
    queue_tasks.post(std::move(tasks));

    bool stream = json_value(data, "stream", false);
    const auto task_ids = server_task::get_list_id(tasks);
    results.resize(task_ids.size());

    if (!stream) {
        SRV_DBG("mode: %s\n", "no stream");
        for (size_t i = 0; i < task_ids.size(); i++) {
            server_task_result_ptr result = queue_results.recv(task_ids);

            if (result->is_error()) {
                cancel_tasks(task_ids);
                return -1;
            }

            GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr ||
                        dynamic_cast<server_task_result_embd*>(result.get()) != nullptr ||
                        dynamic_cast<server_task_result_rerank*>(result.get()) != nullptr);
            const size_t idx = result->get_index();
            GGML_ASSERT(idx < results.size() && "index out of range");
            results[idx] = std::move(result);
        }

        queue_results.remove_waiting_task_ids(task_ids);
        return 0;
    }

    SRV_ERR("Error: %s\n", "This is not stream api!");
    queue_results.remove_waiting_task_ids(task_ids);
    return -1;
}

int LlamaServer::completion_generate_str(json& data, std::string& result_str) {
    bool stream = json_value(data, "stream", false);
    if (stream) {
        SRV_ERR("Error: %s\n", "This is not stream api!");
        return -1;
    }
    std::unique_lock<std::mutex> lock(mutex_context);
    generate_status = 1;
    model_status = 3;
    auto type = SERVER_TASK_TYPE_COMPLETION;
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);
    auto completion_id = gen_chatcmplid();
    std::vector<server_task> tasks;
    oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;
    try {
        const auto& prompt = data.at("prompt");
        std::vector<server_tokens> inputs = tokenize_input_prompts(vocab, mctx, prompt, true, true);
        tasks.reserve(inputs.size());
        SRV_DBG("++tokenized_prompts.size() is:%d\r\n", inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(ctx, params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.oaicompat = oaicompat;
            task.params.oaicompat_cmpl_id = completion_id;
            // oaicompat_model is already populated by params_from_json_cmpl

            tasks.push_back(std::move(task));
        }
    } catch (const std::exception& e) {
        log_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return -1;
    }
    queue_results.add_waiting_tasks(tasks);
    queue_tasks.post(std::move(tasks));
    const auto task_ids = server_task::get_list_id(tasks);
    std::vector<server_task_result_ptr> results(task_ids.size());
    if (!stream) {
        SRV_DBG("mode: %s\n", "no stream");
        for (size_t i = 0; i < task_ids.size(); i++) {
            server_task_result_ptr result = queue_results.recv(task_ids);

            if (result->is_error()) {
                cancel_tasks(task_ids);
                generate_status = -1;
                return -1;
            }

            GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr ||
                        dynamic_cast<server_task_result_embd*>(result.get()) != nullptr ||
                        dynamic_cast<server_task_result_rerank*>(result.get()) != nullptr);
            const size_t idx = result->get_index();
            GGML_ASSERT(idx < results.size() && "index out of range");
            results[idx] = std::move(result);
        }

        if (results.size() == 1) {
            result_str = dump_result(results[0]->to_json());
        } else {
            SRV_ERR("Error: %s\n", "More than one requests not supported!");
        }
        queue_results.remove_waiting_task_ids(task_ids);
        generate_status = 0;
        model_status = 1;
        return 0;
    }

    generate_status = 0;
    model_status = 1;
    return 0;
}

int LlamaServer::completion_stream_generate(json& data) {
    LOG_DBG("this: %p\n", this);
    bool stream = json_value(data, "stream", false);
    if (!stream) {
        LOG_ERR("Error: %s\n", "This is stream api!");
        return -1;
    }
    std::unique_lock<std::mutex> lock(mutex_context);
    generate_status = 1;
    model_status = 3;

    auto type = SERVER_TASK_TYPE_COMPLETION;
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto completion_id = gen_chatcmplid();

    oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;

    try {
        const auto& prompt = data.at("prompt");
        std::vector<server_tokens> inputs = tokenize_input_prompts(vocab, mctx, prompt, true, true);
        tasks.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(ctx, params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.oaicompat = oaicompat;
            task.params.oaicompat_cmpl_id = completion_id;
            // oaicompat_model is already populated by params_from_json_cmpl
            SRV_DBG("push_back one task i=%d", i);
            tasks.push_back(std::move(task));
        }
    } catch (const std::exception& e) {
        log_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return -1;
    }

    queue_results.add_waiting_tasks(tasks);
    queue_tasks.post(std::move(tasks));

    for (const int task_id : task_ids) {
        std::cout << task_id << " ";
    }

    task_ids = server_task::get_list_id(tasks);

    stop = false;
    SRV_INF("create stop: %p\n", &stop);
    SRV_INF("create stop: %d\n", stop);

    return 0;
}

int LlamaServer::completion_token_count(json& data) const {
    try {
        const auto& prompt = data.at("prompt");
        auto tokenized_prompts = tokenize_input_prompts(vocab, mctx, prompt, true, true);
        if (tokenized_prompts.size() == 1) {
            return tokenized_prompts[0].size();
        }

        SRV_WRN("Warn: %s\n", "Multi prompts not supported yet!");
    } catch (const std::exception& e) {
        log_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return -1;
    }

    return 0;
}

int LlamaServer::completion_stream_generate_next_token_str(std::string& result_str, const bool confirm) {
    SRV_INF("this: %p\n", this);

    SRV_INF("next stop: %p\n", &stop);
    SRV_INF("next stop: %d\n", stop);

    printf("this: %p\n", this);
    printf("next stop: %p\n", &stop);
    printf("next stop: %d\n", stop);
    SRV_INF("confirm: %d\n", confirm);
    if (stop && !confirm) {
        SRV_WRN("%s\n", "stop and not confirm, ignore it--return 1 E_BUFFER_NOT_ENOUGH.");
        generate_status = 0;
        return 1;
    }

    if (task_ids.empty() && !confirm) {
        SRV_WRN("task_ids is empty and not confirm = %d , ignore it.\n", task_ids.size());
        generate_status = 0;
        model_status = 1;
        return -1;
    }
    if (stop) {
        SRV_WRN("%s\n", "stop, ignore it.");
        generate_status = 2;
        return -1;
    }

    if (task_ids.empty()) {
        SRV_WRN("task_ids is empty = %d , ignore it.\n", task_ids.size());
        generate_status = 0;
        model_status = 1;
        return -1;
    }

    server_task_result_ptr result = nullptr;
    server_task_result* result2 = nullptr;
    // szj confirm 和 not confirm 都会进
    if (confirm) {
        SRV_INF("confirm: %d\n", confirm);
        result = queue_results.recv_aync_confirm(task_ids);
        SRV_INF("result confirm: %p\n", result.get());
    } else {
        SRV_INF("confirm: %d\n", confirm);
        result2 = queue_results.recv_aync(task_ids);
        SRV_INF("result not confirm: %p\n", result2);
    }
    if (confirm && !result) {
        // SRV_INF("confirm no data: %p\n", result.get());
        return 2;
    } else if (!confirm && !result2) {
        // SRV_INF("no data: %p\n", result2);
        return 2;
    }

    if (!confirm) {
        if (result2->is_error()) {
            cancel_tasks(task_ids);
            generate_status = -1;
            return -1;
        }

        result_str = dump_result(result2->to_json());
        rtrim(result_str);
        SRV_INF("result_str not confirm: %s\n", result_str.c_str());

        return 1;
    }

    if (result->is_error()) {
        cancel_tasks(task_ids);
        generate_status = -1;
        return -1;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr ||
                dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr);
    result_str = dump_result(result->to_json());

    rtrim(result_str);
    SRV_INF("result_str: %s\n", result_str.c_str());

    if (result->is_stop()) {
        SRV_DBG("result->is_stop n_finished = %d\n", n_finished);
        SRV_DBG("result->is_stop task_ids size  = %d\n", task_ids.size());
        if (++n_finished == task_ids.size()) {
            SRV_DBG("all task finished = %d\n", n_finished);
            queue_results.remove_waiting_task_ids(task_ids);
            task_ids.clear();
            tasks.clear();
            n_finished = 0;
            generate_status = 0;
            model_status = 1;
            return 0;
        }
    }
    if (!confirm) {
        return 1;
    }
    return 0;
}

int LlamaServer::completion_stream_generate_stop() {
    std::unique_lock<std::mutex> lock(mutex_context);

    if (task_ids.empty()) {
        SRV_DBG("task_ids is empty = %d\n", task_ids.size());
        generate_status = 0;
        model_status = 1;
        return 0;
    }

    cancel_tasks(task_ids);

    stop = true;
    SRV_INF("stop stop: %p\n", &stop);
    SRV_INF("stop stop: %d\n", stop);

    queue_results.remove_waiting_task_ids(task_ids);

    task_ids.clear();
    tasks.clear();
    n_finished = 0;
    generate_status = 2;
    model_status = 1;
    return 0;
}

//将torch::Tensor转换为tcim::Tensor
void TensorConverter::FromTorch(const torch::Tensor& torch_tensor, tcim::Tensor &tcim_tensor) {
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

HANDLE completion_create_instance(void* param) {
    auto* ctx_server = new LlamaServer();

    SRV_DBG("ctx_server: %p\n", ctx_server);
    SRV_DBG("ctx_server->model: %p\n", ctx_server->model);
    SRV_DBG("ctx_server->ctx: %p\n", ctx_server->ctx);
    SRV_DBG("ctx_server->params: %p\n", ctx_server->params);

    ctx_server->model_path = getenv_or_default("HM_MODEL_PATH", "./models/hmm_deepseek_256_4096_4cores_20250317/");
    const auto p = static_cast<::params*>(param);
    if (!ctx_server->params) {
        ctx_server->params = new common_params();
        LOG_DBG("++%s \r\n", "new common_params");
    }

    ctx_server->params->sampling.top_k = p->top_k;
    ctx_server->params->sampling.top_p = p->top_p;
    ctx_server->params->sampling.temp = p->temperature;
    ctx_server->params->sampling.penalty_repeat = p->repeat_penalty;
    ctx_server->params->sampling.penalty_last_n = p->repeat_last_n;

    return ctx_server;
}


static std::shared_ptr<spdlog::logger> default_logger_ = nullptr;

int completion_load_model(HANDLE h, std::string model_path) {
    if (h == nullptr) {
        LOG_ERR("HANDLE is: %p\n", h);
        return -1;
    }
    auto* ctx_server = static_cast<LlamaServer*>(h);

    if (model_real_load_flag) {
        LOG_DBG("%s\r\n", "model loaded dont load again");
        ctx_server->model_status = 1;
        return 0;
    }

    if (ctx_server->magic != 12345) {
        LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
        return -1;
    }

    if (model_path.empty()) {
        model_path = "./";
    }

    std::string tok_model_path = model_path;
    LOG_INF("--Loading  HiModel file:%s \r\n", tok_model_path.c_str());
    if (!fileExists(tok_model_path)) {
        LOG_ERR("Error:HiModel file:%s  not exists,Please check model params passed to this function\r\n", tok_model_path);
        exit(-1);
    }

    LOG_DBG("load tok_model_path:%s\n", tok_model_path.c_str());

#ifdef _WIN32
    const char* userProfile = getenv("APPDATA");
    if (userProfile != nullptr) {
        std::cout << "userdata dir: " << userProfile << std::endl;
    } else {
        userProfile = "D:\\";
        std::cout << "get userdata dir failed,write do D:\\,please makesure D exists" << std::endl;
    }
    std::string log_folder(userProfile);
    std::string log_file = log_folder + "\\houmo-llm.log";
#else
    std::string log_file = "houmo-llm.log";
#endif
    std::cout << "final logfile path:" << log_file << std::endl;

    const auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    const auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file, 0, 0, false, 7);
    std::vector<spdlog::sink_ptr> sinks{file_sink, console_sink};
    default_logger_ = std::make_shared<spdlog::logger>("HLIECpp", sinks.begin(), sinks.end());
    default_logger_->set_level(spdlog::level::debug);
    spdlog::set_default_logger(default_logger_);

    auto* p_params = new common_params();
    p_params->model.path = tok_model_path;
    p_params->n_parallel = 1;
    p_params->n_ctx = 640000; // 配置大值让llama根据模型自动设置
    p_params->ctx_shift = false;
    p_params->warmup = false;
    p_params->hostname = "0.0.0.0";
    p_params->port = 7901;
    p_params->verbosity = -888;
    common_log_set_verbosity_thold(-888); // 负数用于同化llama.cpp日志

    common_init();

    p_params->sampling.top_k = ctx_server->params->sampling.top_k;
    p_params->sampling.top_p = ctx_server->params->sampling.top_p;
    p_params->sampling.temp = ctx_server->params->sampling.temp;
    p_params->sampling.penalty_repeat = ctx_server->params->sampling.penalty_repeat;
    p_params->sampling.penalty_last_n = ctx_server->params->sampling.penalty_last_n;

    if (ctx_server->params) {
        LOG_DBG("--%s\r\n", "delete ctx_server->params");
        delete ctx_server->params;
    }

    ctx_server->params = p_params;

    llama_backend_init();

    LOG_DBG("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", ctx_server->params->cpuparams.n_threads,
            ctx_server->params->cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_DBG("\n");
    LOG_DBG("%s\n", common_params_get_system_info(*ctx_server->params).c_str());
    LOG_DBG("\n");

    // Necessary similarity of prompt for slot selection
    ctx_server->slot_prompt_similarity = ctx_server->params->slot_prompt_similarity;

    LOG_DBG("%s: loading model\n", __func__);
    if (!ctx_server->load_model(*ctx_server->params)) {
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return -1;
    }

    ctx_server->model_status = 0;

    ctx_server->initialize();

    LOG_DBG("%s: model loaded\n", __func__);

    ctx_server->queue_tasks.on_new_task(std::bind(&LlamaServer::process_single_task, &*ctx_server, std::placeholders::_1));
    LOG_DBG("%s", "++after on new task\r\n");

    ctx_server->queue_tasks.on_update_slots(std::bind(&LlamaServer::update_slots, &*ctx_server));

    LOG_DBG("%s", "++after on update slots\r\n");

    ctx_server->ctx_t = std::thread([&]() { ctx_server->queue_tasks.start_loop(); });
    std::string req_body = string_format("{\"messages\":[{\"content\":\"你好！\",\"role\":\"user\"}],\"stream\":"
                                         "false,\"model\": \"%s\", \"max_tokens\": 20}",
                                         "fixed_hm_llama_qwen3_8b_szj");
    json data = oaicompat_completion_params_parse(json::parse(req_body), ctx_server->params_base.use_jinja,
                                                  ctx_server->params_base.reasoning_format, ctx_server->chat_templates.get());
    LOG_DBG("%s", "++after oaicompat param parse\r\n");
    int ret = 0;
    const std::string str = data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    LOG_DBG("data: %s\n", str.c_str());

    LOG_DBG("Start str generate222\n");
    LOG_DBG("%s", "++before completion_generate_str\r\n");
    std::string result_json_str;
    ret = ctx_server->completion_generate_str(data, result_json_str);
    if (ret == 0) {
        SRV_INF("completion_generate_str content: %s\n", result_json_str.c_str());
    } else {
        LOG_ERR("Error: %d\n", ret);
    }
    LOG_DBG("%s", "++after completion_generate_str before completion_stream_generate\r\n");
    ret = ctx_server->completion_stream_generate(data);
    if (ret == 0) {
        while (true) {
            std::string result_json_str;
            ret = ctx_server->completion_stream_generate_next_token_str(result_json_str);
            SRV_INF("next_token_str ret: %d\n", ret);
            if (ctx_server->generate_status == 0) {
                SRV_INF("stop found str: %s\n", result_json_str.c_str());
                break;
            } else if (ret == 2) {
                SRV_INF("Data not ready: %d\n", ret);
            } else if (ret == -1) {
                LOG_ERR("Error: %d\n", ret);
            } else {
                SRV_INF("completion_stream_generate_next_token_str str: %s\n", result_json_str.c_str());
            }
        }
    } else {
        LOG_ERR("Error: %d\n", ret);
    }
    LOG_DBG("%s", "###after completion_stream_generate\r\n");
    LOG_DBG("ctx_server: %p\n", ctx_server);
    LOG_DBG("ctx_server->model: %p\n", ctx_server->model);
    LOG_DBG("ctx_server->ctx: %p\n", ctx_server->ctx);
    LOG_DBG("ctx_server->params: %p\n", ctx_server->params);

    return 0;
}

// 构造格式为 "add1_add2_add3.mem" 的字符串
std::string constructMemoryString(void**add1, void* add2, const int add3) {
    std::stringstream ss;

    // 将void**转换为十六进制字符串（通常地址用十六进制表示，保持16位长度）
    ss << std::hex << std::setfill('0') << std::setw(16)
       << reinterpret_cast<uint64_t>(add1) << "_";

    // 将void*转换为十六进制字符串
    ss << std::hex << std::setfill('0') << std::setw(16)
       << reinterpret_cast<uint64_t>(add2) << "_";

    // 将int转换为十进制字符串
    ss << std::dec << add3 << ".mem";

    return ss.str();
}

int completion_load_model_by_buffer(HANDLE h, uint8_t** buffers, size_t* bufferSize, size_t buffers_count, hardware_info hwinfo) {
    if (h == nullptr) {
        LOG_ERR("HANDLE is null: %p\n", h);
        return -1;
    }

    auto* ctx_server = static_cast<LlamaServer*>(h);
    if (ctx_server->magic != 12345) {
        LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
        return -1;
    }

    const auto tok_model_path = constructMemoryString(reinterpret_cast<void**>(buffers), bufferSize, buffers_count);
    std::cout << "++tok_model_path from mem:" << tok_model_path << std::endl;

    const std::string model_path = ctx_server->model_path;

    auto* p_params = new common_params();
    p_params->model.path = tok_model_path;
    p_params->n_parallel = 1;
    p_params->n_ctx = 640000; // 配置大值让llama根据模型自动设置
    p_params->ctx_shift = false;
    p_params->warmup = false;
    p_params->hostname = "0.0.0.0";
    p_params->port = 7901;
    p_params->verbosity = -888;
    common_log_set_verbosity_thold(-888); // 负数用于同化llama.cpp日志

    common_init();

    p_params->sampling.top_k = ctx_server->params->sampling.top_k;
    p_params->sampling.top_p = ctx_server->params->sampling.top_p;
    p_params->sampling.temp = ctx_server->params->sampling.temp;
    p_params->sampling.penalty_repeat = ctx_server->params->sampling.penalty_repeat;
    p_params->sampling.penalty_last_n = ctx_server->params->sampling.penalty_last_n;

    if (ctx_server->params) {
        LOG_DBG("**%s\r\n", "delete ctx_server->params");
        delete ctx_server->params;
    }

    ctx_server->params = p_params;

    llama_backend_init();

    LOG_DBG("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", ctx_server->params->cpuparams.n_threads,
            ctx_server->params->cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_DBG("\n");
    LOG_DBG("%s\n", common_params_get_system_info(*ctx_server->params).c_str());
    LOG_DBG("\n");

    // Necessary similarity of prompt for slot selection
    ctx_server->slot_prompt_similarity = ctx_server->params->slot_prompt_similarity;

    // load the model
    LOG_DBG("%s: loading model\n", __func__);

    if (!ctx_server->load_model(*ctx_server->params)) {
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return -1;
    }

    ctx_server->model_status = 0;
    LOG_INF("%s", "++before call init_by_buffer\r\n");
    ctx_server->initialize();
    LOG_INF("%s", "++after call init_by_buffer\r\n");

    LOG_DBG("%s: model loaded\n", __func__);

    ctx_server->queue_tasks.on_new_task(std::bind(&LlamaServer::process_single_task, &*ctx_server, std::placeholders::_1));

    ctx_server->queue_tasks.on_update_slots(std::bind(&LlamaServer::update_slots, &*ctx_server));

    ctx_server->ctx_t = std::thread([&]() { ctx_server->queue_tasks.start_loop(); });

    std::string req_body = string_format("{\"messages\":[{\"content\":\"你好！\",\"role\":\"user\"}],\"stream\":"
                                         "false,\"model\": \"%s\", \"max_tokens\": 20}",
                                         "fixed_hm_llama_qwen3_8b_szj");

    json data = oaicompat_completion_params_parse(json::parse(req_body), ctx_server->params_base.use_jinja,
                                                  ctx_server->params_base.reasoning_format, ctx_server->chat_templates.get());
    int ret = 0;
    const std::string str = data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    LOG_DBG("data: %s\n", str.c_str());

    LOG_DBG("Start str generate333\n");

    std::string result_json_str;
    ret = ctx_server->completion_generate_str(data, result_json_str);
    if (ret == 0) {
        SRV_INF("completion_generate_str content: %s\n", result_json_str.c_str());
    } else {
        LOG_ERR("Error: %d\n", ret);
    }

    ret = ctx_server->completion_stream_generate(data);
    if (ret == 0) {
        while (true) {
            std::string result_json_str;
            ret = ctx_server->completion_stream_generate_next_token_str(result_json_str);
            SRV_INF("next_token_str ret: %d\n", ret);
            if (ctx_server->generate_status == 0) {
                SRV_INF("stop found str: %s\n", result_json_str.c_str());
                break;
            } else if (ret == 2) {
                SRV_INF("Data not ready: %d\n", ret);
            } else if (ret == -1) {
                LOG_ERR("Error: %d\n", ret);
            } else {
                SRV_INF("completion_stream_generate_next_token_str str: %s\n", result_json_str.c_str());
            }
        }
    } else {
        LOG_ERR("Error: %d\n", ret);
    }

    LOG_DBG("ctx_server: %p\n", ctx_server);
    LOG_DBG("ctx_server->model: %p\n", ctx_server->model);
    LOG_DBG("ctx_server->ctx: %p\n", ctx_server->ctx);

    LOG_DBG("ctx_server->params: %p\n", ctx_server->params);

    return 0;
}

int completion_generate(HANDLE h, json& req, std::string& result) {
    if (h == nullptr) {
        LOG_ERR("HANDLE is: %p\n", h);
        return -1;
    }

    auto* ctx_server = static_cast<LlamaServer*>(h);
    if (ctx_server->magic != 12345) {
        LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
        return -1;
    }

    LOG_DBG("ctx_server: %p\n", ctx_server);

    json data = oaicompat_completion_params_parse(req, ctx_server->params_base.use_jinja, ctx_server->params_base.reasoning_format,
                                                  ctx_server->chat_templates.get());
    int ret = 0;
    const std::string str = data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    LOG_DBG("data: %s\n", str.c_str());

    LOG_DBG("Start str generate 8888\n");

    ret = ctx_server->completion_generate_str(data, result);
    if (ret == 0) {
        LOG_DBG("completion_generate_str content: %s\n", result.c_str());
    } else {
        LOG_ERR("Error: %d\n", ret);
    }

    LOG_DBG("ctx_server: %p\n", ctx_server);
    return 0;
}

int completion_generate_stream(HANDLE h, const json &req) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  auto *ctx_server = static_cast<LlamaServer*>(h);
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return -1;
  }

  json data = oaicompat_completion_params_parse(
      req, ctx_server->params_base.use_jinja,
      ctx_server->params_base.reasoning_format,
      ctx_server->chat_templates.get());
  int ret = 0;
  const std::string str =
      data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
  LOG_DBG("data: %s\n", str.c_str());

  LOG_DBG("Time:%s Start str generate 666\n",getCurrentTime().c_str());

  ret = ctx_server->completion_stream_generate(data);
  if (ret == 0) {
  } else {
    LOG_ERR("Error: %d\n", ret);
  }

  LOG_DBG("ctx_server: %p\n", ctx_server);

  return ret;
}

int generate_token_count(HANDLE h, std::string prompt) {
    if (h == nullptr) {
        LOG_ERR("HANDLE is: %p\n", h);
        return -1;
    }

    auto* ctx_server = static_cast<LlamaServer*>(h);
    if (ctx_server->magic != 12345) {
        LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
        return -1;
    }

    LOG_INF("prompt: %s\n", prompt.c_str());
    prompt = std::regex_replace(prompt, std::regex(R"(\n)"), "\\n");
    prompt = std::regex_replace(prompt, std::regex(R"(\r)"), "\\r");
    prompt = std::regex_replace(prompt, std::regex(R"(\")"), "\\\"");
    SRV_INF("prompt: %s\n", prompt.c_str());

    std::string req_body = string_format("{\"prompt\": \"%s\"}", prompt.c_str());
    LOG_INF("req_body: %s\n", req_body.c_str());
    json data = json::parse(req_body);
    int ret = 0;
    const std::string str = data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
    LOG_INF("data: %s\n", str.c_str());

    ret = ctx_server->completion_token_count(data);
    SRV_INF("generate_token_count ret: %d\n", ret);

    return ret;
}

// api_Get_Generate_token 调用下面的函数
int generate_stream_next_token(HANDLE h, char** result, size_t* result_size) {
    if (h == nullptr) {
        LOG_ERR("HANDLE is: %p\n", h);
        return -1;
    }

    if (result == nullptr) {
        LOG_ERR("result is nullptr\n");
    }

    auto* ctx_server = static_cast<LlamaServer*>(h);
    if (ctx_server->magic != 12345) {
        LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
        return -1;
    }

    std::string result_json_str;
    int ret = 0;
    if (result != nullptr && *result) {
        SRV_INF("%s", "++confirm True:completion_stream_generate_next_token_str\r\n");
        ret = ctx_server->completion_stream_generate_next_token_str(result_json_str);
    } else {
        SRV_INF("%s", "--confirm False:completion_stream_generate_next_token_str\r\n");
        ret = ctx_server->completion_stream_generate_next_token_str(result_json_str, false);
    }
    SRV_INF("+++ret: %d\n", ret);

    if (ret == -1) {
        return ret;
    }

    size_t len = strlen(result_json_str.c_str()) + 1;
    SRV_INF("len: %d\n", len);

    if (len > 0) {
        if (result == nullptr || !*result || *result_size < len) {
            SRV_INF("%s ", "enter 1*result: \n");
            *result_size = len;
            return ret;
        }

        SRV_INF("2 *result: %p\n", *result);
        SRV_INF("result_size: %d\n", *result_size);
        strcpy(*result, result_json_str.c_str());

        SRV_INF("*result length: %d\n", strlen(*result));
        SRV_INF("*result size: %d\n", (sizeof(*result) / sizeof(**result)));
        SRV_INF("result str: %s\n", *result);
    }

  return ret;
}

int generate_stream_stop(HANDLE h) {
    if (h == nullptr) {
        LOG_ERR("HANDLE is: %p\n", h);
        return -1;
    }

    auto* ctx_server = static_cast<LlamaServer*>(h);
    const int ret = ctx_server->completion_stream_generate_stop();
    return ret;
}

int completion_destroy_instance(HANDLE h) {
    if (h == nullptr) {
        LOG_ERR("HANDLE is: %p\n", h);
        return -1;
    }

    auto* ctx_server = static_cast<LlamaServer*>(h);
    ctx_server->queue_tasks.terminate();
    if (ctx_server->ctx_t.joinable()) {
        ctx_server->ctx_t.join();
    }
    SRV_INF("join ctx_t finished: %s\n", "");
    delete ctx_server;
    ctx_server = nullptr;
    SRV_INF("ctx_server: %p\n", ctx_server);
    return 0;
}

// if fail, return nullptr
// HANDLE HTTP_SERVER_API create_api_instance(params& params);
HANDLE HTTP_SERVER_API create_api_instance(void *params, size_t params_bytes) {
  return completion_create_instance(params);
}

void HTTP_SERVER_API destroy_api_instance(HANDLE handle) {
  completion_destroy_instance(handle);
}

// load model with model name ,layer:-1 load all layer.
bool HTTP_SERVER_API api_loadmodel_by_path(HANDLE handle, const char* model_path, const char** model_names, size_t model_name_count,
                                           hardware_info hwinfo) {
    SRV_DBG("model_path: %s\n", model_path);
    SRV_DBG("model_name: %s\n", model_names[model_name_count - 1]);

    common_log_pause(common_log_main());
    std::vector<std::string> model_name_list;
    for (int i = 0; i < model_name_count; i++) {
        model_name_list.emplace_back(model_names[i]);
    }
    const int ret = completion_load_model(handle, model_path);
    common_log_resume(common_log_main());
    if (ret == 0) {
        model_real_load_flag = true;
        return true;
    }
    SRV_ERR("%s\r\n", "load model failed");
    return false;
}

// load model with buffer stream,layer:-1 load all layer.
bool HTTP_SERVER_API api_loadmodel_by_buffer(HANDLE handle, uint8_t** buffers, size_t* bufferSize, size_t buffers_count,
                                             hardware_info hwinfo) {
    SRV_DBG("buffers_count: %d\n", buffers_count);

    const auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    const auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>("houmo-llm.log", 0, 0, false, 7);
    std::vector<spdlog::sink_ptr> sinks{file_sink, console_sink};
    default_logger_ = std::make_shared<spdlog::logger>("HLIECpp", sinks.begin(), sinks.end());
    default_logger_->set_level(spdlog::level::debug);
    spdlog::set_default_logger(default_logger_);

    common_log_pause(common_log_main());
    const int ret = completion_load_model_by_buffer(handle, buffers, bufferSize, buffers_count, hwinfo);
    common_log_resume(common_log_main());

    if (ret == 0) {
        return true;
    }

    SRV_ERR("%s", "++Loadmodel from buffer error, Please Check!\r\n");
    return false;
}

// load token model not must
bool HTTP_SERVER_API api_loadtoken(HANDLE handle, const char *token_fullpath) {
  SRV_DBG("token_fullpath: %s\n", token_fullpath);
  return true;
}

// unload model
bool HTTP_SERVER_API api_unloadmodel(HANDLE handle) {
    const int ret = completion_destroy_instance(handle);
    LOG_DBG("api_unloadmodel: %d\n", ret);
    model_real_load_flag = false;
    if (ret == 0) {
        return true;
    }

    return false;
}

// model status see the enum status
// return enum status
int32 HTTP_SERVER_API api_status(HANDLE handle) {
    LOG_DBG("Call api_status atTime:%s\n", getCurrentTime().c_str());
    int32 status = 0;
    if (handle == nullptr) {
        LOG_ERR("HANDLE is: %p\n", handle);
        return -1;
    }

    auto* ctx_server = static_cast<LlamaServer*>(handle);
    if (ctx_server->magic != 12345) {
        LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
        return -1;
    }
    status = ctx_server->model_status;
    LOG_DBG("Return from api_status atTime:%s\n", getCurrentTime().c_str());
    return status;
}

// Generate
// if result == nullptr or *result_size is not enough, return E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HTTP_SERVER_API api_Generate(HANDLE handle, const char* prompt, const void* gen_params, size_t gen_params_size, char** result,
                                   size_t* result_size) {
    SRV_DBG("prompt: %s\n", prompt);

    std::string pp(prompt);
    std::string input = pp;
    input = std::regex_replace(input, std::regex(R"(\n)"), "\\n");
    input = std::regex_replace(input, std::regex(R"(\r)"), "\\r");
    input = std::regex_replace(input, std::regex(R"(\")"), "\\\"");
    rtrim(input);
    SRV_INF("++input: %s\n", input.c_str());

    auto p = (params *)gen_params;
    if (handle == nullptr) {
        LOG_ERR("HANDLE is: %p\n", handle);
        return -1;
    }
    //printf("++1\r\n");
    //server_context *ctx_server = (server_context *)handle;

    // std::string req_body =
    //     "{\"messages\":[{\"content\":\"%s”\",\"role\":\"user\"}],\"stream\":"
    //     "false,\"model\": \"%s\", \"max_tokens\": %d}";
    std::string req_body =
        "{\"messages\":[{\"content\":\"%s\",\"role\":\"user\"}],\"stream\":"
        "false,\"model\": \"%s\", \"max_tokens\": %d}";
    #if ENABLE_HOUMO_LLAMACPP
    req_body = string_format(req_body.c_str(), input.c_str(),
                            "fixed_hm_llama_qwen3_8b_szj", p->n_predict);
    #else
    //printf("++2 ctx_server=%p ctx_server->hmctx=%p ctx_server->hmctx->model_name.c_str()=%s\r\n",ctx_server,ctx_server->hmctx,ctx_server->hmctx->model_name.c_str());
    req_body = string_format(req_body.c_str(), input.c_str(),
                            "fixed_hm_llama_qwen3_8b_szj", p->n_predict);
    #endif
    //printf("--2.1 req_body.c_str()=%s\r\n",req_body.c_str());
    ///SRV_DBG("req_body",req_body)
    json req = json::parse(req_body);
    std::string decode_result;
    //printf("++3\r\n");
    int ret = completion_generate(handle, req, decode_result);
    //printf("++szj decode_result.size() is:%d\r\n",decode_result.size());
    char *result_ptr =
        new char[decode_result.size() + 1];  // +1 for terminating NUL

    strncpy(result_ptr, decode_result.c_str(), decode_result.size() + 1);
    //printf("++4\r\n");
    *result = result_ptr;
    *result_size = decode_result.size();
    SRV_DBG("decode_result: %s\n", decode_result.c_str());
    SRV_DBG("--result: %s\n", *result);

    return 0;
}

// get prompt token number
int32 HTTP_SERVER_API api_token_num(HANDLE handle, const char *prompt) {
  SRV_DBG("prompt: %s\n", prompt);
  return generate_token_count(handle, prompt);
}


// stream generate
bool HTTP_SERVER_API api_Generate_stream(HANDLE handle, const char* prompt, const void* gen_params, size_t gen_params_size) {
    LOG_DBG("Call api_Generate_stream atTime:%s  get stream prompt: %s\n", getCurrentTime().c_str(), prompt);
    const std::string pp(prompt);
    std::string input = pp;
    input = std::regex_replace(input, std::regex(R"(\n)"), "\\n");
    input = std::regex_replace(input, std::regex(R"(\r)"), "\\r");
    input = std::regex_replace(input, std::regex(R"(\")"), "\\\"");
    SRV_INF("input: %s\n", input.c_str());
    rtrim(input);
    auto p = (params*)gen_params;
    int ret = 0;
    if (handle == nullptr) {
        LOG_ERR("HANDLE is: %p\n", handle);
        return -1;
    }

    try {
        std::string req_body = "{\"messages\":[{\"content\":\"%s\",\"role\":\"user\"}],\"stream\":"
                               "true,\"model\": \"%s\", \"max_tokens\": %d}";

        req_body = string_format(req_body.c_str(), input.c_str(), "fixed_hm_llama_qwen3_8b_szj", p->n_predict);

        const json req = json::parse(req_body);
        ret = completion_generate_stream(handle, req);
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
    }
    if (ret != 0) {
        LOG_DBG("ReturnFalse from api_Generate_stream atTime:%s\n", getCurrentTime().c_str());
        return false;
    }
    LOG_DBG("ReturnTrue from api_Generate_stream atTime:%s\n", getCurrentTime().c_str());
    return true;
}

// the generate token ,use for stream mode
// if result == nullptr or *result_size is not enough, return
// E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HTTP_SERVER_API api_Get_Generate_token(HANDLE handle, char** result, size_t* result_size) {
    LOG_DBG("++++call api_Get_Generate_token atTime:%s\r\n", getCurrentTime().c_str());
    int ret = 0;
    try {
        ret = generate_stream_next_token(handle, result, result_size);
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        ret = -1;
    }
    LOG_DBG("----Return from api_Get_Generate_token atTime:%s\n ", getCurrentTime().c_str());
    return ret;
}

// generate status see generate_status
// return enum generate_status
int32 HTTP_SERVER_API api_Generate_status(HANDLE handle) {
    LOG_DBG("Call api_Generate_status atTime:%s\n", getCurrentTime().c_str());
    int32 status = 0;
    if (handle == nullptr) {
        LOG_ERR("HANDLE is: %p\n", handle);
        return -1;
    }

    auto* ctx_server = static_cast<LlamaServer*>(handle);
    if (ctx_server->magic != 12345) {
        LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
        return -1;
    }

    status = ctx_server->generate_status;
    if (0 == ctx_server->generate_status) {
        ctx_server->model_status = 1;
    }
    LOG_DBG("Return from api_Generate_status atTime:%s ctx_server->model_status=%d after set to 1\n", getCurrentTime().c_str(),
            ctx_server->model_status);

    return status;
}

// stop generate
bool HTTP_SERVER_API api_stop(HANDLE handle) {
    LOG_DBG("Call api_stop atTime:%s\n", getCurrentTime().c_str());

    if (const int ret = generate_stream_stop(handle); ret == 0) {
        LOG_DBG("Return from api_stop atTime:%s  ret=0\n", getCurrentTime().c_str());
        return true;
    }

    LOG_DBG("Return from api_stop atTime:%s\n  ret not 0", getCurrentTime().c_str());
    return false;
}

// warmup not must
bool HTTP_SERVER_API api_warmup(HANDLE handle) {
    return true;
}

// reset model not must
void HTTP_SERVER_API api_Reset(HANDLE handle) {
}


// hi.h api for normal customers

#ifdef _MSC_VER
#define HMINFER_API __declspec(dllexport)
#else
#define HMINFER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// if fail, return nullptr
// HANDLE HMINFER_API create_infer_instance(params& params);
HANDLE HMINFER_API create_infer_instance(void* params, size_t params_bytes) {
    return create_api_instance(params, params_bytes);
}

void HMINFER_API destroy_infer_instance(HANDLE handle) {
    return destroy_api_instance(handle);
}

// load model with model name ,layer:-1 load all layer.
bool HMINFER_API infer_loadmodel_by_path(HANDLE handle, const char* model_path, const char** model_names, size_t model_name_count) {
    hardware_info hwinfo;
    return api_loadmodel_by_path(handle, model_path, model_names, model_name_count, hwinfo);
}

// load model with buffer stream,layer:-1 load all layer.
bool HMINFER_API infer_loadmodel_by_buffer(HANDLE handle, uint8_t** buffers, size_t* bufferSize, size_t buffers_count) {
    hardware_info hwinfo;
    return api_loadmodel_by_buffer(handle, buffers, bufferSize, buffers_count, hwinfo);
}

// load token model not must
bool HMINFER_API infer_loadtoken(HANDLE handle, const char* token_fullpath) {
    return api_loadtoken(handle, token_fullpath);
}

// unload model
bool HMINFER_API infer_unloadmodel(HANDLE handle) {
    return api_unloadmodel(handle);
}

// model status see the enum status
// return enum status
int32 HMINFER_API infer_status(HANDLE handle) {
    return api_status(handle);
}

// Generate
// if result == nullptr or *result_size is not enough, return E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HMINFER_API infer_Generate(HANDLE handle, const char* prompt, const void* params, size_t params_bytes, char** result,
                                 size_t* result_size) {
    return api_Generate(handle, prompt, params, params_bytes, result, result_size);
}

// get prompt token number
int32 HMINFER_API infer_token_num(HANDLE handle, const char* prompt) {
    return api_token_num(handle, prompt);
}

// stream generate
bool HMINFER_API infer_Generate_stream(HANDLE handle, const char* prompt, const void* params, size_t params_size) {
    return api_Generate_stream(handle, prompt, params, params_size);
}

// the generate token ,use for stream mode
// if result == nullptr or *result_size is not enough, return E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HMINFER_API infer_Get_Generate_token(HANDLE handle, char** result, size_t* result_size) {
    return api_Get_Generate_token(handle, result, result_size);
}

// generate status see enum generater_status defined ahead
// return enum generater_status
int32 HMINFER_API infer_Generate_status(HANDLE handle) {
    return api_Generate_status(handle);
}

// stop generate
bool HMINFER_API infer_stop(HANDLE handle) {
    return api_stop(handle);
}

// warmup not must
bool HMINFER_API infer_warmup(HANDLE handle) {
    return api_warmup(handle);
}

// reset model not must
void HMINFER_API infer_Reset(HANDLE handle) {
    return api_Reset(handle);
}

#ifdef __cplusplus
}
#endif

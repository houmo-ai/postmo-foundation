#include "arg.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "utils.hpp"
#include "mtmd.h"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
// mime type for sending response
#define MIMETYPE_JSON "application/json; charset=utf-8"

#include <csignal>
#include <torch/torch.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "http_server_c_interface_3.h"
#include "hlie_base/hlie_std_err_redirector.h"
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

#include "hlie_app/hlie_model_info.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"

#if (__GNUC__ < 8 && !defined(_MSC_VER))
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif
using json = nlohmann::ordered_json;

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
inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
}

// trim from end (in place)
inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

template <typename... Args>
std::string hm_string_format(const std::string &format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
               1;  // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1);  // We don't want the '\0' inside
}

template <typename T>
static void int16toint16(T *src, T *dst, size_t size) {
  // 检查类型是否为 TriviallyCopyable（可直接内存复制）
  // printf("src=%p,dst=%p,size=%d\r\n",src,dst,size);
  if constexpr (std::is_trivially_copyable_v<T>) {
    // 使用 memcpy 进行高效内存复制
    memcpy(src, dst, size * sizeof(T));
  } else {
    // 对于不可直接复制的类型，使用循环
    for (size_t i = 0; i < size; i++) {
      src[i] = dst[i];
    }
  }
}

bool containsChinese(const std::string& str) {
    std::regex pattern("[\u4e00-\u9fff]");
    return std::regex_search(str, pattern);
}

bool containsEnglishLetters(const std::string& str) {
    for (unsigned char c : str) {
        if (std::isalpha(c))  // 检查是否为字母（包含大小写）
            return true;
    }
    return false;
}

bool isPunctuation(const std::string& str) {
    // 空字符串不视为标点
    if (str.empty()) return false;

    // 检查每个字符是否都是标点符号
    return std::all_of(str.begin(), str.end(), [](unsigned char c) {
        return std::ispunct(c);
    });
}

bool isNumeric(const std::string& str) {
    return !str.empty() &&
           std::all_of(str.begin(), str.end(), [](unsigned char c) {
               return std::isdigit(c);
           });
}

std::string dump_result(const json &data) {
  const std::string str =
      data.dump(-1, ' ', false, json::error_handler_t::replace);
  // data.dump(-1, ' ', false, json::error_handler_t::replace) +
  //     "\n\n"; // required by RFC 8895 - A message is terminated by a blank
  //     line (two line terminators in a row).
  return str;
}

void log_error(const json &error_data) {
  const std::string str =
      error_data.dump(-1, ' ', false, json::error_handler_t::replace) +
      "\n\n";  // required by RFC 8895 - A message is terminated by a blank line
               // (two line terminators in a row).

  SRV_ERR("%s\n", str.c_str());
}

std::string getenv_or_default(const std::string& name, const std::string& default_value) {
    const char* value = std::getenv(name.c_str());
    return value ? std::string(value) : default_value;
}

void log_ok(const json &data) {
  const std::string str =
      data.dump(-1, ' ', false, json::error_handler_t::replace) +
      "\n\n";  // required by RFC 8895 - A message is terminated by a blank line
               // (two line terminators in a row).
  SRV_INF("%s\n", str.c_str());
}

constexpr int HTTP_POLLING_SECONDS = 1;

enum stop_type {
  STOP_TYPE_NONE,
  STOP_TYPE_EOS,
  STOP_TYPE_WORD,
  STOP_TYPE_LIMIT,
};

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
  SLOT_STATE_IDLE,
  SLOT_STATE_STARTED,  // TODO: this state is only used for setting up the
                       // initial prompt processing; maybe merge it with
                       // launch_slot_with_task in the future
  SLOT_STATE_PROCESSING_PROMPT,
  SLOT_STATE_DONE_PROMPT,
  SLOT_STATE_GENERATING,
};

enum server_state {
  SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded
                               // yet
  SERVER_STATE_READY,          // Server is ready and model is loaded
};

enum server_task_type {
  SERVER_TASK_TYPE_COMPLETION,
  SERVER_TASK_TYPE_EMBEDDING,
  SERVER_TASK_TYPE_RERANK,
  SERVER_TASK_TYPE_INFILL,
  SERVER_TASK_TYPE_CANCEL,
  SERVER_TASK_TYPE_NEXT_RESPONSE,
  SERVER_TASK_TYPE_METRICS,
  SERVER_TASK_TYPE_SLOT_SAVE,
  SERVER_TASK_TYPE_SLOT_RESTORE,
  SERVER_TASK_TYPE_SLOT_ERASE,
  SERVER_TASK_TYPE_SET_LORA,
};

enum oaicompat_type {
  OAICOMPAT_TYPE_NONE,
  OAICOMPAT_TYPE_CHAT,
  OAICOMPAT_TYPE_COMPLETION,
  OAICOMPAT_TYPE_EMBEDDING,
};

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
  ERROR_TYPE_INVALID_REQUEST,
  ERROR_TYPE_AUTHENTICATION,
  ERROR_TYPE_SERVER,
  ERROR_TYPE_NOT_FOUND,
  ERROR_TYPE_PERMISSION,
  ERROR_TYPE_UNAVAILABLE,    // custom error
  ERROR_TYPE_NOT_SUPPORTED,  // custom error
};

enum model_scale_type {
  MODEL_1DOT5B=1,
  MODEL_3B,
  MODEL_7B,
  MODEL_8B,
  MODEL_14B,
  MODEL_32B,
  MODEL_72B,
};

// 数值稳定的softmax实现
inline void softmax(const float* logits, float* probs, int n_vocab) {
    float max_logit = *std::max_element(logits, logits + n_vocab);

    float sum = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] = expf(logits[i] - max_logit); // 减去最大值保证数值稳定
        sum += probs[i];
    }

    // 归一化
    for (int i = 0; i < n_vocab; ++i) {
        probs[i] /= sum;
    }
}

// 判断两个浮点数是否相等
bool areFloatsEqual(double a, double b) {
    const double tolerance = 1e-9; // 定义容差
    return std::abs(a - b) < tolerance;
}

static bool server_task_type_need_embd(server_task_type task_type) {
  switch (task_type) {
  case SERVER_TASK_TYPE_EMBEDDING:
  case SERVER_TASK_TYPE_RERANK:
    return true;
  default:
    return false;
  }
}

static bool server_task_type_need_logits(server_task_type task_type) {
  switch (task_type) {
  case SERVER_TASK_TYPE_COMPLETION:
  case SERVER_TASK_TYPE_INFILL:
    return true;
  default:
    return false;
  }
}

struct slot_params {
  bool stream = true;
  bool cache_prompt =
      false;  // remember the prompt to avoid reprocessing all prompt
  bool return_tokens = true;

  int32_t n_keep = 0;     // number of tokens to keep from initial prompt
  int32_t n_discard = 0;  // number of tokens after n_keep that may be discarded
                          // when shifting context, 0 defaults to half
  int32_t n_predict = -1;  // new tokens to predict
  int32_t n_indent = 0;    // mininum line indentation for the generated text in
                           // number of whitespace characters

  int64_t t_max_prompt_ms = -1;  // TODO: implement
  int64_t t_max_predict_ms =
      -1;  // if positive, limit the generation phase to this time limit

  std::vector<common_adapter_lora_info> lora;

  std::vector<std::string> antiprompt;
  std::vector<std::string> response_fields;
  bool timings_per_token = false;
  bool post_sampling_probs = false;
  bool ignore_eos = false;

  struct common_params_sampling sampling;
  struct common_params_speculative speculative;

  // OAI-compat fields
  bool verbose = false;
  oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;
  std::string oaicompat_model;
  std::string oaicompat_cmpl_id;
  common_chat_format oaicompat_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
  #if ENABLE_HOUMO_LLAMACPP
  common_chat_syntax           oaicompat_chat_syntax;
  #endif

  json to_json() const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto &sampler : sampling.samplers) {
      samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (size_t i = 0; i < this->lora.size(); ++i) {
      lora.push_back({{"id", i}, {"scale", this->lora[i].scale}});
    }

    auto grammar_triggers = json::array();
    for (const auto &trigger : sampling.grammar_triggers) {
      #if ENABLE_HOUMO_LLAMACPP
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
      #else
        grammar_triggers.push_back(trigger.to_json<json>());
      #endif
    }
    #if ENABLE_HOUMO_LLAMACPP
    return json {
            {"n_predict",                 n_predict},     // Server configured n_predict
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"stop",                      antiprompt},
            {"max_tokens",                n_predict}, // User configured n_predict
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"logit_bias",                format_logit_bias(sampling.logit_bias)},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"grammar",                   sampling.grammar},
            {"grammar_lazy",              sampling.grammar_lazy},
            {"grammar_triggers",          grammar_triggers},
            {"preserved_tokens",          sampling.preserved_tokens},
            {"chat_format",               common_chat_format_name(oaicompat_chat_syntax.format)},
            {"reasoning_format",          common_reasoning_format_name(oaicompat_chat_syntax.reasoning_format)},
            {"reasoning_in_content",      oaicompat_chat_syntax.reasoning_in_content},
            {"thinking_forced_open",      oaicompat_chat_syntax.thinking_forced_open},
            {"samplers",                  samplers},
            {"speculative.n_max",         speculative.n_max},
            {"speculative.n_min",         speculative.n_min},
            {"speculative.p_min",         speculative.p_min},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"lora",                      lora},
        };
    #else
    return json{
        {"n_predict", n_predict},  // Server configured n_predict
        {"seed", sampling.seed},
        {"temperature", sampling.temp},
        {"dynatemp_range", sampling.dynatemp_range},
        {"dynatemp_exponent", sampling.dynatemp_exponent},
        {"top_k", sampling.top_k},
        {"top_p", sampling.top_p},
        {"min_p", sampling.min_p},
        {"xtc_probability", sampling.xtc_probability},
        {"xtc_threshold", sampling.xtc_threshold},
        {"typical_p", sampling.typ_p},
        {"repeat_last_n", sampling.penalty_last_n},
        {"repeat_penalty", sampling.penalty_repeat},
        {"presence_penalty", sampling.penalty_present},
        {"frequency_penalty", sampling.penalty_freq},
        {"dry_multiplier", sampling.dry_multiplier},
        {"dry_base", sampling.dry_base},
        {"dry_allowed_length", sampling.dry_allowed_length},
        {"dry_penalty_last_n", sampling.dry_penalty_last_n},
        {"dry_sequence_breakers", sampling.dry_sequence_breakers},
        {"mirostat", sampling.mirostat},
        {"mirostat_tau", sampling.mirostat_tau},
        {"mirostat_eta", sampling.mirostat_eta},
        {"stop", antiprompt},
        {"max_tokens", n_predict},  // User configured n_predict
        {"n_keep", n_keep},
        {"n_discard", n_discard},
        {"ignore_eos", sampling.ignore_eos},
        {"stream", stream},
        {"logit_bias", format_logit_bias(sampling.logit_bias)},
        {"n_probs", sampling.n_probs},
        {"min_keep", sampling.min_keep},
        {"grammar", sampling.grammar},
        {"grammar_lazy", sampling.grammar_lazy},
        {"grammar_triggers", grammar_triggers},
        {"preserved_tokens", sampling.preserved_tokens},
        {"chat_format", common_chat_format_name(oaicompat_chat_format)},
        {"samplers", samplers},
        {"speculative.n_max", speculative.n_max},
        {"speculative.n_min", speculative.n_min},
        {"speculative.p_min", speculative.p_min},
        {"timings_per_token", timings_per_token},
        {"post_sampling_probs", post_sampling_probs},
        {"lora", lora},
    };
    #endif
  }
};

struct server_context;
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


struct server_task {
  int id = -1;     // to be filled by server_queue
  int index = -1;  // used when there are multiple prompts (batch request)

  server_task_type type;

  // used by SERVER_TASK_TYPE_CANCEL
  int id_target = -1;

  // used by SERVER_TASK_TYPE_INFERENCE
  slot_params params;
  server_tokens prompt_tokens;
  int id_selected_slot = -1;

  // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE,
  // SERVER_TASK_TYPE_SLOT_ERASE
  struct slot_action {
    int slot_id;
    std::string filename;
    std::string filepath;
  };
  slot_action slot_action;

  // used by SERVER_TASK_TYPE_METRICS
  bool metrics_reset_bucket = false;

  // used by SERVER_TASK_TYPE_SET_LORA
  std::vector<common_adapter_lora_info> set_lora;

  server_task(server_task_type type) : type(type) {}

  static slot_params params_from_json_cmpl(const llama_context *ctx,
                                           const common_params &params_base,
                                           const json &data) {
    const llama_model *model = llama_get_model(ctx);
    const llama_vocab *vocab = llama_model_get_vocab(model);

    slot_params params;

    // Sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
    slot_params defaults;
    defaults.sampling = params_base.sampling;
    defaults.speculative = params_base.speculative;
    defaults.n_keep      = params_base.n_keep;

    // enabling this will output extra debug information in the HTTP responses from the server
    params.verbose = params_base.verbosity > 9;
    params.timings_per_token = json_value(data, "timings_per_token", false);

    params.stream = json_value(data, "stream", false);
    params.cache_prompt = json_value(data, "cache_prompt", false);
    params.return_tokens = json_value(data, "return_tokens", true);
    params.n_predict = json_value(data, "n_predict", json_value(data, "max_tokens", defaults.n_predict));
    params.n_indent = json_value(data, "n_indent", defaults.n_indent);
    params.n_keep = json_value(data, "n_keep", defaults.n_keep);
    params.n_discard = json_value(data, "n_discard", defaults.n_discard);
    // params.t_max_prompt_ms  = json_value(data, "t_max_prompt_ms",
    // defaults.t_max_prompt_ms); // TODO: implement
    params.t_max_predict_ms = json_value(data, "t_max_predict_ms", defaults.t_max_predict_ms);
    params.response_fields = json_value(data, "response_fields", std::vector<std::string>());

    params.sampling.top_k = json_value(data, "top_k", defaults.sampling.top_k);
    params.sampling.top_p = json_value(data, "top_p", defaults.sampling.top_p);
    params.sampling.min_p = json_value(data, "min_p", defaults.sampling.min_p);
    params.sampling.top_n_sigma = json_value(data, "top_n_sigma", defaults.sampling.top_n_sigma);
    params.sampling.xtc_probability = json_value(data, "xtc_probability", defaults.sampling.xtc_probability);
    params.sampling.xtc_threshold = json_value(data, "xtc_threshold", defaults.sampling.xtc_threshold);
    params.sampling.typ_p = json_value(data, "typical_p", defaults.sampling.typ_p);
    params.sampling.temp = json_value(data, "temperature", defaults.sampling.temp);
    params.sampling.dynatemp_range = json_value(data, "dynatemp_range", defaults.sampling.dynatemp_range);
    params.sampling.dynatemp_exponent = json_value(data, "dynatemp_exponent", defaults.sampling.dynatemp_exponent);
    params.sampling.penalty_last_n = json_value(data, "repeat_last_n", defaults.sampling.penalty_last_n);
    params.sampling.penalty_repeat = json_value(data, "repeat_penalty", defaults.sampling.penalty_repeat);
    params.sampling.penalty_freq = json_value(data, "frequency_penalty", defaults.sampling.penalty_freq);
    params.sampling.penalty_present = json_value(data, "presence_penalty", defaults.sampling.penalty_present);
    params.sampling.dry_multiplier = json_value(data, "dry_multiplier", defaults.sampling.dry_multiplier);
    params.sampling.dry_base = json_value(data, "dry_base", defaults.sampling.dry_base);
    params.sampling.dry_allowed_length = json_value(data, "dry_allowed_length", defaults.sampling.dry_allowed_length);
    params.sampling.dry_penalty_last_n = json_value(data, "dry_penalty_last_n", defaults.sampling.dry_penalty_last_n);
    params.sampling.mirostat = json_value(data, "mirostat", defaults.sampling.mirostat);
    params.sampling.mirostat_tau = json_value(data, "mirostat_tau", defaults.sampling.mirostat_tau);
    params.sampling.mirostat_eta = json_value(data, "mirostat_eta", defaults.sampling.mirostat_eta);
    params.sampling.seed = json_value(data, "seed", defaults.sampling.seed);
    params.sampling.n_probs = json_value(data, "n_probs", defaults.sampling.n_probs);
    params.sampling.min_keep = json_value(data, "min_keep", defaults.sampling.min_keep);
    params.post_sampling_probs = json_value(data, "post_sampling_probs", defaults.post_sampling_probs);

    params.speculative.n_min = json_value(data, "speculative.n_min", defaults.speculative.n_min);
    params.speculative.n_max = json_value(data, "speculative.n_max", defaults.speculative.n_max);
    params.speculative.p_min = json_value(data, "speculative.p_min", defaults.speculative.p_min);

    params.speculative.n_min =
        std::min(params.speculative.n_max, params.speculative.n_min);
    params.speculative.n_min = std::max(params.speculative.n_min, 0);
    params.speculative.n_max = std::max(params.speculative.n_max, 0);

    // Use OpenAI API logprobs only if n_probs wasn't provided
    if (data.contains("logprobs") &&
        params.sampling.n_probs == defaults.sampling.n_probs) {
      params.sampling.n_probs =
          json_value(data, "logprobs", defaults.sampling.n_probs);
    }

    if (data.contains("lora")) {
      if (data.at("lora").is_array()) {
        params.lora =
            parse_lora_request(params_base.lora_adapters, data.at("lora"));
      } else {
        throw std::runtime_error(
            "Error: 'lora' must be an array of objects with 'id' and 'scale' "
            "fields");
      }
    } else {
      params.lora = params_base.lora_adapters;
    }

    // TODO: add more sanity checks for the input parameters

    if (params.sampling.penalty_last_n < -1) {
      throw std::runtime_error("Error: repeat_last_n must be >= -1");
    }

    if (params.sampling.dry_penalty_last_n < -1) {
      throw std::runtime_error("Error: dry_penalty_last_n must be >= -1");
    }

    if (params.sampling.penalty_last_n == -1) {
      // note: should be the slot's context and not the full context, but it's
      // ok
      params.sampling.penalty_last_n = llama_n_ctx(ctx);
    }

    if (params.sampling.dry_penalty_last_n == -1) {
      params.sampling.dry_penalty_last_n = llama_n_ctx(ctx);
    }

    if (params.sampling.dry_base < 1.0f) {
      params.sampling.dry_base = defaults.sampling.dry_base;
    }

    // sequence breakers for DRY
    {
      // Currently, this is not compatible with TextGen WebUI, Koboldcpp and
      // SillyTavern format Ref:
      // https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39

      if (data.contains("dry_sequence_breakers")) {
        params.sampling.dry_sequence_breakers = json_value(
            data, "dry_sequence_breakers", std::vector<std::string>());
        if (params.sampling.dry_sequence_breakers.empty()) {
          throw std::runtime_error(
              "Error: dry_sequence_breakers must be a non-empty array of "
              "strings");
        }
      }
    }

    // process "json_schema" and "grammar"
    if (data.contains("json_schema") && !data.contains("grammar")) {
      try {
        auto schema = json_value(data, "json_schema", json::object());
        SRV_DBG("JSON schema: %s\n", schema.dump(2).c_str());
        params.sampling.grammar = json_schema_to_grammar(schema);
        SRV_DBG("Converted grammar: %s\n", params.sampling.grammar.c_str());
      } catch (const std::exception &e) {
        throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
      }
    } else {
      params.sampling.grammar =
          json_value(data, "grammar", defaults.sampling.grammar);
      SRV_DBG("Grammar: %s\n", params.sampling.grammar.c_str());
      params.sampling.grammar_lazy =
          json_value(data, "grammar_lazy", defaults.sampling.grammar_lazy);
      SRV_DBG("Grammar lazy: %s\n",
              params.sampling.grammar_lazy ? "true" : "false");
    }
    {
        auto it = data.find("chat_format");
        if (it != data.end()) {
            params.oaicompat_chat_syntax.format = static_cast<common_chat_format>(it->get<int>());
            SRV_INF("Chat format: %s\n", common_chat_format_name(params.oaicompat_chat_syntax.format));
        } else {
            params.oaicompat_chat_syntax.format = defaults.oaicompat_chat_syntax.format;
        }
        params.oaicompat_chat_syntax.reasoning_format = params_base.reasoning_format;
        params.oaicompat_chat_syntax.reasoning_in_content =
            params.stream && (params_base.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
        params.oaicompat_chat_syntax.thinking_forced_open = json_value(data, "thinking_forced_open", false);
        params.oaicompat_chat_syntax.parse_tool_calls = json_value(data, "parse_tool_calls", false);
    }

    {
      const auto preserved_tokens = data.find("preserved_tokens");
      if (preserved_tokens != data.end()) {
        for (const auto &t : *preserved_tokens) {
          auto ids = common_tokenize(vocab, t.get<std::string>(),
                                     /* add_special= */ false,
                                     /* parse_special= */ true);
          if (ids.size() == 1) {
            SRV_DBG("Preserved token: %d\n", ids[0]);
            params.sampling.preserved_tokens.insert(ids[0]);
          } else {
            // This may happen when using a tool call style meant for a model
            // with special tokens to preserve on a model without said tokens.
            SRV_DBG("Not preserved because more than 1 token: %s\n",
                    t.get<std::string>().c_str());
          }
        }
      }
      const auto grammar_triggers = data.find("grammar_triggers");
      if (grammar_triggers != data.end()) {
        for (const auto &t : *grammar_triggers) {
            server_grammar_trigger ct(t);
            if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                const auto & word = ct.value.value;
                auto ids = common_tokenize(vocab, word, /* add_special= */ false, /* parse_special= */ true);
                if (ids.size() == 1) {
                    auto token = ids[0];
                    if (std::find(params.sampling.preserved_tokens.begin(), params.sampling.preserved_tokens.end(), (llama_token) token) == params.sampling.preserved_tokens.end()) {
                        throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                    }
                    SRV_DBG("Grammar trigger token: %d (`%s`)\n", token, word.c_str());
                    common_grammar_trigger trigger;
                    trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                    trigger.value = word;
                    trigger.token = token;
                    params.sampling.grammar_triggers.push_back(std::move(trigger));
                } else {
                    SRV_DBG("Grammar trigger word: `%s`\n", word.c_str());
                    params.sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
                }
            } else {
                if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN) {
                    SRV_DBG("Grammar trigger pattern: `%s`\n", ct.value.value.c_str());
                } else if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL) {
                    SRV_DBG("Grammar trigger pattern full: `%s`\n", ct.value.value.c_str());
                } else {
                    throw std::runtime_error("Unknown grammar trigger type");
                }
                params.sampling.grammar_triggers.emplace_back(std::move(ct.value));
            }
        }
      }
      if (params.sampling.grammar_lazy &&
          params.sampling.grammar_triggers.empty()) {
        throw std::runtime_error("Error: no triggers set for lazy grammar!");
      }
    }

    {
      params.sampling.logit_bias.clear();
      params.ignore_eos = json_value(data, "ignore_eos", false);

      const auto &logit_bias = data.find("logit_bias");
      if (logit_bias != data.end() && logit_bias->is_array()) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        SRV_DBG("++const int n_vocab is:%d\r\n",n_vocab);
        for (const auto &el : *logit_bias) {
          // TODO: we may want to throw errors here, in case "el" is incorrect
          if (el.is_array() && el.size() == 2) {
            float bias;
            if (el[1].is_number()) {
              bias = el[1].get<float>();
            } else if (el[1].is_boolean() && !el[1].get<bool>()) {
              bias = -INFINITY;
            } else {
              continue;
            }

            if (el[0].is_number_integer()) {
              llama_token tok = el[0].get<llama_token>();
              if (tok >= 0 && tok < n_vocab) {
                params.sampling.logit_bias.push_back({tok, bias});
              }
            } else if (el[0].is_string()) {
              auto toks =
                  common_tokenize(vocab, el[0].get<std::string>(), false);
              for (auto tok : toks) {
                params.sampling.logit_bias.push_back({tok, bias});
              }
            }
          }
        }
      }
    }

    {
      params.antiprompt.clear();

      const auto &stop = data.find("stop");
      if (stop != data.end() && stop->is_array()) {
        for (const auto &word : *stop) {
          if (!word.empty()) {
            params.antiprompt.push_back(word);
          }
        }
      }
    }

    {
      const auto samplers = data.find("samplers");
      if (samplers != data.end()) {
        if (samplers->is_array()) {
          params.sampling.samplers =
              common_sampler_types_from_names(*samplers, false);
        } else if (samplers->is_string()) {
          params.sampling.samplers =
              common_sampler_types_from_chars(samplers->get<std::string>());
        }
      } else {
        params.sampling.samplers = defaults.sampling.samplers;
      }
    }

    std::string model_name = params_base.model_alias.empty() ? DEFAULT_OAICOMPAT_MODEL : params_base.model_alias;
    params.oaicompat_model = json_value(data, "model", model_name);

    return params;
  }

  // utility function
  static std::unordered_set<int> get_list_id(
      const std::vector<server_task> &tasks) {
    std::unordered_set<int> ids(tasks.size());
    for (size_t i = 0; i < tasks.size(); i++) {
      ids.insert(tasks[i].id);
    }
    return ids;
  }
};

struct result_timings {
  int32_t prompt_n = -1;
  double prompt_ms;
  double prompt_per_token_ms;
  double prompt_per_second;

  int32_t predicted_n = -1;
  double predicted_ms;
  double predicted_per_token_ms;
  double predicted_per_second;

  // Optional speculative metrics - only included when > 0
  int32_t draft_n = 0;
  int32_t draft_n_accepted = 0;

  json to_json() const {
    json base = {
      {"prompt_n",               prompt_n},
      {"prompt_ms",              prompt_ms},
      {"prompt_per_token_ms",    prompt_per_token_ms},
      {"prompt_per_second",      prompt_per_second},

      {"predicted_n",            predicted_n},
      {"predicted_ms",           predicted_ms},
      {"predicted_per_token_ms", predicted_per_token_ms},
      {"predicted_per_second",   predicted_per_second},
  };

    if (draft_n > 0) {
      base["draft_n"] = draft_n;
      base["draft_n_accepted"] = draft_n_accepted;
    }

    return base;
  }
};

struct server_task_result {
  int id = -1;
  int id_slot = -1;
  virtual bool is_error() {
    // only used by server_task_result_error
    return false;
  }
  virtual bool is_stop() {
    // only used by server_task_result_cmpl_*
    return false;
  }
  virtual int get_index() { return -1; }
  virtual json to_json() = 0;
  virtual ~server_task_result() = default;
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

inline std::string stop_type_to_str(stop_type type) {
  switch (type) {
    case STOP_TYPE_EOS:
      return "eos";
    case STOP_TYPE_WORD:
      return "word";
    case STOP_TYPE_LIMIT:
      return "limit";
    default:
      return "none";
  }
}

struct completion_token_output {
  llama_token tok;
  float prob;
  std::string text_to_send;
  struct prob_info {
    llama_token tok;
    std::string txt;
    float prob;
  };
  std::vector<prob_info> probs;

  json to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto &p : probs) {
      std::string txt(p.txt);
      txt.resize(validate_utf8(txt));
      probs_for_token.push_back(json{
          {"id", p.tok},
          {"token", txt},
          {"bytes", str_to_bytes(p.txt)},
          {post_sampling_probs ? "prob" : "logprob",
           post_sampling_probs ? p.prob : logarithm(p.prob)},
      });
    }
    return probs_for_token;
  }

  static json probs_vector_to_json(
      const std::vector<completion_token_output> &probs,
      bool post_sampling_probs) {
    json out = json::array();
    for (const auto &p : probs) {
      std::string txt(p.text_to_send);
      txt.resize(validate_utf8(txt));
      out.push_back(json{
          {"id", p.tok},
          {"token", txt},
          {"bytes", str_to_bytes(p.text_to_send)},
          {post_sampling_probs ? "prob" : "logprob",
           post_sampling_probs ? p.prob : logarithm(p.prob)},
          {post_sampling_probs ? "top_probs" : "top_logprobs",
           p.to_json(post_sampling_probs)},
      });
    }
    return out;
  }

  static float logarithm(float x) {
    // nlohmann::json converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
  }

  static std::vector<unsigned char> str_to_bytes(const std::string &str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
      bytes.push_back(c);
    }
    return bytes;
  }
};

struct server_task_result_cmpl_final : server_task_result {
  int index = 0;

  std::string content;
  llama_tokens tokens;

  bool stream;
  result_timings timings;
  std::string prompt;

  bool truncated;
  int32_t n_decoded;
  int32_t n_prompt_tokens;
  int32_t n_tokens_cached;
  bool has_new_line;
  std::string stopping_word;
  stop_type stop = STOP_TYPE_NONE;

  bool post_sampling_probs;
  std::vector<completion_token_output> probs_output;
  std::vector<std::string> response_fields;

  slot_params generation_params;

  // OAI-compat fields
  bool verbose = false;
  oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;
  std::string oaicompat_model;
  std::string oaicompat_cmpl_id;
  common_chat_format oaicompat_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
  #if ENABLE_HOUMO_LLAMACPP
  common_chat_msg    oaicompat_msg;
  std::vector<common_chat_msg_diff> oaicompat_msg_diffs;
  #endif


  virtual int get_index() override { return index; }

  virtual bool is_stop() override {
    return true;  // in stream mode, final responses are considered stop
  }

  virtual json to_json() override {
    switch (oaicompat) {
      case OAICOMPAT_TYPE_NONE:
        return to_json_non_oaicompat();
      case OAICOMPAT_TYPE_COMPLETION:
        return to_json_oaicompat();
      case OAICOMPAT_TYPE_CHAT:
        return stream ? to_json_oaicompat_chat_stream()
                      : to_json_oaicompat_chat();
      default:
        GGML_ASSERT(false && "Invalid oaicompat_type");
    }
  }

  json to_json_non_oaicompat() {
    json res = json{
        {"index", index},
        {"content", stream ? "" : content},  // in stream mode, content is
                                             // already in last partial chunk
        {"tokens", stream ? llama_tokens{} : tokens},
        {"id_slot", id_slot},
        {"stop", true},
        {"model", oaicompat_model},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt", prompt},
        {"has_new_line", has_new_line},
        {"truncated", truncated},
        {"stop_type", stop_type_to_str(stop)},
        {"stopping_word", stopping_word},
        {"tokens_cached", n_tokens_cached},
        {"timings", timings.to_json()},
    };
    if (!stream && !probs_output.empty()) {
      res["completion_probabilities"] =
          completion_token_output::probs_vector_to_json(probs_output,
                                                        post_sampling_probs);
    }
    return response_fields.empty()
               ? res
               : json_get_nested_values(response_fields, res);
  }

  json to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr);  // OAI default to null
    if (!stream && probs_output.size() > 0) {
      logprobs = json{
          {"content", completion_token_output::probs_vector_to_json(
                          probs_output, post_sampling_probs)},
      };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
      finish_reason = "stop";
    }
    json res = json{
        {"choices",
         json::array({json{
             {"text", stream ? "" : content},  // in stream mode, content is
                                               // already in last partial chunk
             {"index", index},
             {"logprobs", logprobs},
             {"finish_reason", finish_reason},
         }})},
        {"created", t},
        {"model", oaicompat_model},
        {"system_fingerprint", build_info},
        {"object", "text_completion"},
        {"usage", json{{"completion_tokens", n_decoded},
                       {"prompt_tokens", n_prompt_tokens},
                       {"total_tokens", n_decoded + n_prompt_tokens}}},
        {"id", oaicompat_cmpl_id}};

    // extra fields for debugging purposes
    if (verbose) {
      res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
      res.push_back({"timings", timings.to_json()});
    }

    return res;
  }

  #if 0//ENABLE_HOUMO_LLAMACPP
  json to_json_oaicompat_chat() {
      std::string finish_reason = "length";
      common_chat_msg msg;
      if (!oaicompat_msg.empty()) {
          msg = oaicompat_msg;
      } else {
          msg.role = "assistant";
          msg.content = content;
      }
      if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
          finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
      }

      json choice {
          {"finish_reason", finish_reason},
          {"index", 0},
          {"message", msg.to_json_oaicompat<json>()},
      };

      if (!stream && probs_output.size() > 0) {
          choice["logprobs"] = json{
              {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
          };
      }

      std::time_t t = std::time(0);

      json res = json {
          {"choices",            json::array({choice})},
          {"created",            t},
          {"model",              oaicompat_model},
          {"system_fingerprint", build_info},
          {"object",             "chat.completion"},
          {"usage", json {
              {"completion_tokens", n_decoded},
              {"prompt_tokens",     n_prompt_tokens},
              {"total_tokens",      n_decoded + n_prompt_tokens}
          }},
          {"id", oaicompat_cmpl_id}
      };

      // extra fields for debugging purposes
      if (verbose) {
          res["__verbose"] = to_json_non_oaicompat();
      }
      if (timings.prompt_n >= 0) {
          res.push_back({"timings", timings.to_json()});
      }

      return res;
  }

  json to_json_oaicompat_chat_stream() {
        std::time_t t = std::time(0);
        std::string finish_reason = "length";
        if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
            finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
        }

        json deltas = json::array();
        for (const auto & diff : oaicompat_msg_diffs) {
            deltas.push_back({
                {"choices", json::array({
                    json {
                        {"finish_reason", nullptr},
                        {"index", 0},
                        {"delta", common_chat_msg_diff_to_json_oaicompat<json>(diff)},
                    },
                })},
                {"created", t},
                {"id", oaicompat_cmpl_id},
                {"model", oaicompat_model},
                {"system_fingerprint", build_info},
                {"object", "chat.completion.chunk"},
            });
        }

        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", finish_reason},
                    {"index", 0},
                    {"delta", json::object()},
                },
            })},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", build_info},
            {"object",             "chat.completion.chunk"},
            {"usage", json {
                {"completion_tokens", n_decoded},
                {"prompt_tokens",     n_prompt_tokens},
                {"total_tokens",      n_decoded + n_prompt_tokens},
            }},
        });

        if (timings.prompt_n >= 0) {
            deltas.back().push_back({"timings", timings.to_json()});
        }

        // extra fields for debugging purposes
        if (verbose && !deltas.empty()) {
            deltas.front()["__verbose"] = to_json_non_oaicompat();
        }

        return deltas;
    }

  #else
  json to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
      SRV_DBG("Parsing chat message: %s\n", content.c_str());
      #if ENABLE_HOUMO_LLAMACPP
      common_chat_syntax           oaicompat_chat_syntax;
      oaicompat_chat_syntax.format=oaicompat_chat_format;
      auto msg = common_chat_parse(
        content,
        /* is_partial= */ stop != STOP_TYPE_EOS,
        oaicompat_chat_syntax);
      #else
      msg = common_chat_parse(content, oaicompat_chat_format);
      #endif
      finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    } else {
      msg.content = content;
    }

    json message{
        {"role", "assistant"},
    };
    if (!msg.reasoning_content.empty()) {
      message["reasoning_content"] = msg.reasoning_content;
    }
    if (msg.content.empty() && !msg.tool_calls.empty()) {
      message["content"] = json();
    } else {
      message["content"] = msg.content;
    }
    if (!msg.tool_calls.empty()) {
      auto tool_calls = json::array();
      for (const auto &tc : msg.tool_calls) {
        tool_calls.push_back({
            {"type", "function"},
            {"function",
             {
                 {"name", tc.name},
                 {"arguments", tc.arguments},
             }},
            // Some templates generate and require an id (sometimes in a very
            // specific format, e.g. Mistral Nemo). We only generate a random id
            // for the ones that don't generate one by themselves (they also
            // won't get to see it as their template likely doesn't use it, so
            // it's all for the client)
            {"id", tc.id.empty() ? gen_tool_call_id() : tc.id},
        });
      }
      message["tool_calls"] = tool_calls;
    }

    json choice{
        {"finish_reason", finish_reason},
        {"index", 0},
        {"message", message},
    };

    if (!stream && probs_output.size() > 0) {
      choice["logprobs"] = json{
          {"content", completion_token_output::probs_vector_to_json(
                          probs_output, post_sampling_probs)},
      };
    }

    std::time_t t = std::time(0);

    json res =
        json{{"choices", json::array({choice})},
             {"created", t},
             {"model", oaicompat_model},
             {"system_fingerprint", build_info},
             {"object", "chat.completion"},
             {"usage", json{{"completion_tokens", n_decoded},
                            {"prompt_tokens", n_prompt_tokens},
                            {"total_tokens", n_decoded + n_prompt_tokens}}},
             {"id", oaicompat_cmpl_id}};

    // extra fields for debugging purposes
    if (verbose) {
      res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
      res.push_back({"timings", timings.to_json()});
    }

    return res;
  }
  #endif


  json to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
      finish_reason = "stop";
    }

    json choice = json{{"finish_reason", finish_reason},
                       {"index", 0},
                       {"delta", json::object()}};

    json ret = json{
        {"choices", json::array({choice})},
        {"created", t},
        {"id", oaicompat_cmpl_id},
        {"model", oaicompat_model},
        {"system_fingerprint", build_info},
        {"object", "chat.completion.chunk"},
        {"usage",
         json{
             {"completion_tokens", n_decoded},
             {"prompt_tokens", n_prompt_tokens},
             {"total_tokens", n_decoded + n_prompt_tokens},
         }},
    };

    if (timings.prompt_n >= 0) {
      ret.push_back({"timings", timings.to_json()});
    }

    // extra fields for debugging purposes
    if (verbose) {
      ret["__verbose"] = to_json_non_oaicompat();
    }

    return ret;
  }
};

struct server_task_result_cmpl_partial : server_task_result {
  int index = 0;

  std::string content;
  llama_tokens tokens;

  int32_t n_decoded;
  int32_t n_prompt_tokens;

  bool post_sampling_probs;
  completion_token_output prob_output;
  result_timings timings;

  // OAI-compat fields
  bool verbose = false;
  oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;
  std::string oaicompat_model;
  std::string oaicompat_cmpl_id;
  std::vector<common_chat_msg_diff> oaicompat_msg_diffs;

  virtual int get_index() override { return index; }

  virtual bool is_stop() override {
    return false;  // in stream mode, partial responses are not considered stop
  }

  virtual json to_json() override {
    switch (oaicompat) {
      case OAICOMPAT_TYPE_NONE:
        return to_json_non_oaicompat();
      case OAICOMPAT_TYPE_COMPLETION:
        return to_json_oaicompat();
      case OAICOMPAT_TYPE_CHAT:
        return to_json_oaicompat_chat();
      default:
        GGML_ASSERT(false && "Invalid oaicompat_type");
    }
  }

  json to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json{
        {"index", index},
        {"content", content},
        {"tokens", tokens},
        {"stop", false},
        {"id_slot", id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or
    // with timings_per_token enabled)
    if (timings.prompt_n > 0) {
      res.push_back({"timings", timings.to_json()});
    }
    if (!prob_output.probs.empty()) {
      res["completion_probabilities"] =
          completion_token_output::probs_vector_to_json({prob_output},
                                                        post_sampling_probs);
    }
    return res;
  }

  json to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr);  // OAI default to null
    if (prob_output.probs.size() > 0) {
      logprobs = json{
          {"content", completion_token_output::probs_vector_to_json(
                          {prob_output}, post_sampling_probs)},
      };
    }
    json res = json{{"choices", json::array({json{
                                    {"text", content},
                                    {"index", index},
                                    {"logprobs", logprobs},
                                    {"finish_reason", nullptr},
                                }})},
                    {"created", t},
                    {"model", oaicompat_model},
                    {"system_fingerprint", build_info},
                    {"object", "text_completion"},
                    {"id", oaicompat_cmpl_id}};

    // extra fields for debugging purposes
    if (verbose) {
      res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
      res.push_back({"timings", timings.to_json()});
    }

    return res;
  }

  json to_json_oaicompat_chat() {
    bool first = n_decoded == 0;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json& delta)
    {
      deltas.push_back({
          {"choices",
           json::array({
               json{
                   {"finish_reason", nullptr},
                   {"index", 0},
                   {"delta", delta},
               },
           })},
          {"created", t},
          {"id", oaicompat_cmpl_id},
          {"model", oaicompat_model},
          {"system_fingerprint", build_info},
          {"object", "chat.completion.chunk"},
      });
    };
    // We have to send an initial update to conform to openai behavior
    if (first) {
      add_delta({
          {"role", "assistant"},
          {"content", nullptr},
      });
    }

    for (const auto& diff : oaicompat_msg_diffs) {
      add_delta(common_chat_msg_diff_to_json_oaicompat<json>(diff));
    }

    if (!deltas.empty()) {
      GGML_ASSERT(deltas[deltas.size() - 1].at("choices").size() >= 1);

      if (prob_output.probs.size() > 0) {
        deltas[deltas.size() - 1].at("choices").at(0)["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
      }

      if (timings.prompt_n >= 0) {
        deltas[deltas.size() - 1].push_back({"timings", timings.to_json()});
      }
    }

    return deltas;
  }
};

struct server_task_result_embd : server_task_result {
  int index = 0;
  std::vector<std::vector<float>> embedding;

  int32_t n_tokens;

  // OAI-compat fields
  oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;

  virtual int get_index() override { return index; }

  virtual json to_json() override {
    return oaicompat == OAICOMPAT_TYPE_EMBEDDING ? to_json_oaicompat()
                                                 : to_json_non_oaicompat();
  }

  json to_json_non_oaicompat() {
    return json{
        {"index", index},
        {"embedding", embedding},
    };
  }

  json to_json_oaicompat() {
    return json{
        {"index", index},
        {"embedding", embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
  }
};

struct server_task_result_rerank : server_task_result {
  int index = 0;
  float score = -1e6;

  int32_t n_tokens;

  virtual int get_index() override { return index; }

  virtual json to_json() override {
    return json{
        {"index", index},
        {"score", score},
        {"tokens_evaluated", n_tokens},
    };
  }
};

// this function maybe used outside of server_task_result_error
static json format_error_response(const std::string &message,
                                  const enum error_type type) {
  std::string type_str;
  int code = 500;
  switch (type) {
    case ERROR_TYPE_INVALID_REQUEST:
      type_str = "invalid_request_error";
      code = 400;
      break;
    case ERROR_TYPE_AUTHENTICATION:
      type_str = "authentication_error";
      code = 401;
      break;
    case ERROR_TYPE_NOT_FOUND:
      type_str = "not_found_error";
      code = 404;
      break;
    case ERROR_TYPE_SERVER:
      type_str = "server_error";
      code = 500;
      break;
    case ERROR_TYPE_PERMISSION:
      type_str = "permission_error";
      code = 403;
      break;
    case ERROR_TYPE_NOT_SUPPORTED:
      type_str = "not_supported_error";
      code = 501;
      break;
    case ERROR_TYPE_UNAVAILABLE:
      type_str = "unavailable_error";
      code = 503;
      break;
  }
  return json{
      {"code", code},
      {"message", message},
      {"type", type_str},
  };
}

struct server_task_result_error : server_task_result {
  int index = 0;
  error_type err_type = ERROR_TYPE_SERVER;
  std::string err_msg;

  virtual bool is_error() override { return true; }

  virtual json to_json() override {
    return format_error_response(err_msg, err_type);
  }
};

struct server_task_result_metrics : server_task_result {
  int n_idle_slots;
  int n_processing_slots;
  int n_tasks_deferred;
  int64_t t_start;

  // TODO: somehow reuse server_metrics in the future, instead of duplicating
  // the fields
  uint64_t n_prompt_tokens_processed_total = 0;
  uint64_t t_prompt_processing_total = 0;
  uint64_t n_tokens_predicted_total = 0;
  uint64_t t_tokens_generation_total = 0;

  uint64_t n_prompt_tokens_processed = 0;
  uint64_t t_prompt_processing = 0;

  uint64_t n_tokens_predicted = 0;
  uint64_t t_tokens_generation = 0;

  uint64_t n_decode_total = 0;
  uint64_t n_busy_slots_total = 0;

  // while we can also use std::vector<server_slot> this requires copying the
  // slot object which can be quite messy therefore, we use json to temporarily
  // store the slot.to_json() result
  json slots_data = json::array();

  virtual json to_json() override {
    return json{
        {"idle", n_idle_slots},
        {"processing", n_processing_slots},
        {"deferred", n_tasks_deferred},
        {"t_start", t_start},

        {"n_prompt_tokens_processed_total", n_prompt_tokens_processed_total},
        {"t_tokens_generation_total", t_tokens_generation_total},
        {"n_tokens_predicted_total", n_tokens_predicted_total},
        {"t_prompt_processing_total", t_prompt_processing_total},

        {"n_prompt_tokens_processed", n_prompt_tokens_processed},
        {"t_prompt_processing", t_prompt_processing},
        {"n_tokens_predicted", n_tokens_predicted},
        {"t_tokens_generation", t_tokens_generation},

        {"n_decode_total", n_decode_total},
        {"n_busy_slots_total", n_busy_slots_total},

        {"slots", slots_data},
    };
  }
};

struct server_task_result_slot_save_load : server_task_result {
  std::string filename;
  bool is_save;  // true = save, false = load

  size_t n_tokens;
  size_t n_bytes;
  double t_ms;

  virtual json to_json() override {
    if (is_save) {
      return json{
          {"id_slot", id_slot},
          {"filename", filename},
          {"n_saved", n_tokens},
          {"n_written", n_bytes},
          {"timings", {{"save_ms", t_ms}}},
      };
    } else {
      return json{
          {"id_slot", id_slot},
          {"filename", filename},
          {"n_restored", n_tokens},
          {"n_read", n_bytes},
          {"timings", {{"restore_ms", t_ms}}},
      };
    }
  }
};

struct server_task_result_slot_erase : server_task_result {
  size_t n_erased;

  virtual json to_json() override {
    return json{
        {"id_slot", id_slot},
        {"n_erased", n_erased},
    };
  }
};

struct server_task_result_apply_lora : server_task_result {
  virtual json to_json() override { return json{{"success", true}}; }
};

struct server_slot {
  int id;
  int id_task = -1;

  // only used for completion/embedding/infill/rerank
  server_task_type task_type = SERVER_TASK_TYPE_COMPLETION;

  llama_batch batch_spec = {};

  llama_context *ctx = nullptr;
  llama_context *ctx_dft = nullptr;

  // multimodal
  mtmd_context * mctx = nullptr;

  common_speculative *spec = nullptr;

  std::vector<common_adapter_lora_info> lora;

  // the index relative to completion multi-task request
  size_t index = 0;

  struct slot_params params;

  slot_state state = SLOT_STATE_IDLE;

  // used to determine the slot that has been used the longest
  int64_t t_last_used = -1;

  // generation props
  int32_t n_ctx = 0;  // context size per slot
  int32_t n_past = 0;
  int32_t n_decoded = 0;
  int32_t n_remaining = -1;
  int32_t i_batch = -1;
  int32_t n_predict = -1;  // TODO: disambiguate from params.n_predict

  // n_prompt_tokens may not be equal to prompt_tokens.size(), because prompt
  // maybe truncated
  int32_t n_prompt_tokens = 0;
  int32_t n_prompt_tokens_processed = 0;

  // input prompt tokens
  server_tokens prompt_tokens;

  size_t last_nl_pos = 0;

  std::string generated_text;
  llama_tokens generated_tokens;
  common_chat_msg chat_msg;

  server_tokens cache_tokens;

  std::vector<completion_token_output> generated_token_probs;

  bool has_next_token = true;
  bool has_new_line = false;
  bool truncated = false;
  stop_type stop;

  std::string stopping_word;

  // sampling
  json json_schema;

  struct common_sampler *smpl = nullptr;

  llama_token sampled;

  common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
  std::vector<std::string> generated_tool_call_ids;

  // stats
  size_t n_sent_text = 0;  // number of sent text character

  int64_t t_start_process_prompt;
  int64_t t_start_generation;

  double t_prompt_processing;  // ms
  double t_token_generation;   // ms

  std::function<void(int)> callback_on_release;

  // Speculative decoding stats
  int32_t n_draft_total = 0;      // Total draft tokens generated
  int32_t n_draft_accepted = 0;   // Draft tokens actually accepted

  void reset() {
    SLT_DBG(*this, "%s", "\n");

    n_prompt_tokens = 0;
    last_nl_pos = 0;
    generated_text = "";
    has_new_line = false;
    truncated = false;
    stop = STOP_TYPE_NONE;
    stopping_word = "";
    n_past = 0;
    n_sent_text = 0;
    task_type = SERVER_TASK_TYPE_COMPLETION;
    chat_format        = COMMON_CHAT_FORMAT_CONTENT_ONLY;

    generated_tokens.clear();
    generated_token_probs.clear();
    chat_msg = {};
    json_schema = json();
    generated_tool_call_ids.clear();

    // clear speculative decoding stats
    n_draft_total = 0;
    n_draft_accepted = 0;
  }

  bool need_embd() const {
    return server_task_type_need_embd(task_type);
  }

  bool need_logits() const {
    return server_task_type_need_logits(task_type);
  }

  // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
  // also we cannot split if the pooling would require any past tokens
  bool can_split() const {
    return
        !need_embd() ||
        (llama_get_memory(ctx) && llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_LAST);
  }

  bool can_batch_with(server_slot & other_slot) const {
    return task_type == other_slot.task_type && are_lora_equal(lora, other_slot.lora);
  }

  bool has_budget(const common_params &global_params) {
    if (params.n_predict == -1 && global_params.n_predict == -1) {
      return true;  // limitless
    }

    n_remaining = -1;

    if (params.n_predict != -1) {
      n_remaining = params.n_predict - n_decoded;
    } else if (global_params.n_predict != -1) {
      n_remaining = global_params.n_predict - n_decoded;
    }

    return n_remaining > 0;  // no budget
  }

  bool is_processing() const { return state != SLOT_STATE_IDLE; }

  bool can_speculate() const {
    return ctx_dft && params.speculative.n_max > 0 && params.cache_prompt;
  }

  void add_token(const completion_token_output &token) {
    if (!is_processing()) {
      SLT_WRN(*this, "%s", "slot is not processing\n");
      return;
    }
    generated_token_probs.push_back(token);
  }

  void release() {
    if (is_processing()) {
      SLT_DBG(*this, "stop processing: n_past = %d, truncated = %d\n", n_past,
              truncated);

      t_last_used = ggml_time_us();
      t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;
      state = SLOT_STATE_IDLE;
      callback_on_release(id);
    }
  }

  result_timings get_timings() const {
    result_timings timings;
    timings.prompt_n = n_prompt_tokens_processed;
    timings.prompt_ms = t_prompt_processing;
    timings.prompt_per_token_ms =
        t_prompt_processing / n_prompt_tokens_processed;
    timings.prompt_per_second =
        1e3 / t_prompt_processing * n_prompt_tokens_processed;

    timings.predicted_n = n_decoded;
    timings.predicted_ms = t_token_generation;
    timings.predicted_per_token_ms = t_token_generation / n_decoded;
    timings.predicted_per_second = 1e3 / t_token_generation * n_decoded;

    // Add speculative metrics
    if (n_draft_total > 0) {
      timings.draft_n = n_draft_total;
      timings.draft_n_accepted = n_draft_accepted;
    }

    return timings;
  }

  const common_chat_msg & update_chat_msg(std::vector<common_chat_msg_diff> & diffs) {
    auto previous_msg = chat_msg;
    SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        /* is_partial= */ stop != STOP_TYPE_EOS,
        params.oaicompat_chat_syntax);
    if (!new_msg.empty()) {
      new_msg.ensure_tool_call_ids_set(generated_tool_call_ids, gen_tool_call_id);
      chat_msg = new_msg;
      diffs = common_chat_msg_diff::compute_diffs(previous_msg, new_msg.empty() ? previous_msg : new_msg);
    }
    return chat_msg;
  }

  size_t find_stopping_strings(const std::string &text,
                               const size_t last_token_size,
                               bool is_full_stop) {
    size_t stop_pos = std::string::npos;

    for (const std::string &word : params.antiprompt) {
      size_t pos;

      if (is_full_stop) {
        const size_t tmp = word.size() + last_token_size;
        const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

        pos = text.find(word, from_pos);
      } else {
        // otherwise, partial stop
        pos = string_find_partial_stop(text, word);
      }

      if (pos != std::string::npos &&
          (stop_pos == std::string::npos || pos < stop_pos)) {
        if (is_full_stop) {
          stop = STOP_TYPE_WORD;
          stopping_word = word;
          has_next_token = false;
        }
        stop_pos = pos;
      }
    }

    return stop_pos;
  }

  void print_timings() const {
    const double t_prompt = t_prompt_processing / n_prompt_tokens_processed;
    const double n_prompt_second =
        1e3 / t_prompt_processing * n_prompt_tokens_processed;

    const double t_gen = t_token_generation / n_decoded;
    const double n_gen_second = 1e3 / t_token_generation * n_decoded;

    SLT_DBG(*this,
            "\n"
            "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, "
            "%8.2f tokens per second)\n"
            "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, "
            "%8.2f tokens per second)\n"
            "      total time = %10.2f ms / %5d tokens\n",
            t_prompt_processing, n_prompt_tokens_processed, t_prompt,
            n_prompt_second, t_token_generation, n_decoded, t_gen, n_gen_second,
            t_prompt_processing + t_token_generation,
            n_prompt_tokens_processed + n_decoded);

    if (n_draft_total > 0) {
      const float draft_ratio = (float) n_draft_accepted / n_draft_total;
      SLT_INF(*this,
              "\n"
              "draft acceptance rate = %0.5f (%5d accepted / %5d generated)\n",
              draft_ratio, n_draft_accepted, n_draft_total
      );
    }
  }

  json to_json() const {
    return json{
        {"id", id},
        {"id_task", id_task},
        {"n_ctx", n_ctx},
        {"speculative", can_speculate()},
        {"is_processing", is_processing()},
        {"params", params.to_json()},
        {"prompt", prompt_tokens.detokenize(ctx, true)},
        {"next_token",
         {
             {"has_next_token", has_next_token},
             {"has_new_line", has_new_line},
             {"n_remain", n_remaining},
             {"n_decoded", n_decoded},
             {"stopping_word", stopping_word},
         }},
    };
  }
};

struct server_metrics {
  int64_t t_start = 0;

  uint64_t n_prompt_tokens_processed_total = 0;
  uint64_t t_prompt_processing_total = 0;
  uint64_t n_tokens_predicted_total = 0;
  uint64_t t_tokens_generation_total = 0;

  uint64_t n_prompt_tokens_processed = 0;
  uint64_t t_prompt_processing = 0;

  uint64_t n_tokens_predicted = 0;
  uint64_t t_tokens_generation = 0;

  uint64_t n_decode_total = 0;
  uint64_t n_busy_slots_total = 0;

  void init() { t_start = ggml_time_us(); }

  void on_prompt_eval(const server_slot &slot) {
    n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
    n_prompt_tokens_processed += slot.n_prompt_tokens_processed;
    t_prompt_processing += slot.t_prompt_processing;
    t_prompt_processing_total += slot.t_prompt_processing;
  }

  void on_prediction(const server_slot &slot) {
    n_tokens_predicted_total += slot.n_decoded;
    n_tokens_predicted += slot.n_decoded;
    t_tokens_generation += slot.t_token_generation;
    t_tokens_generation_total += slot.t_token_generation;
  }

  void on_decoded(const std::vector<server_slot> &slots) {
    n_decode_total++;
    for (const auto &slot : slots) {
      if (slot.is_processing()) {
        n_busy_slots_total++;
      }
    }
  }

  void reset_bucket() {
    n_prompt_tokens_processed = 0;
    t_prompt_processing = 0;
    n_tokens_predicted = 0;
    t_tokens_generation = 0;
  }
};

struct server_queue {
  int id = 0;
  bool running;

  // queues
  std::deque<server_task> queue_tasks;
  std::deque<server_task> queue_tasks_deferred;

  std::mutex mutex_tasks;
  std::condition_variable condition_tasks;

  // callback functions
  std::function<void(server_task&&)> callback_new_task;
  std::function<void(void)> callback_update_slots;

  // Add a new task to the end of the queue
  int post(server_task&& task, bool front = false) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    GGML_ASSERT(task.id != -1);
    // if this is cancel task make sure to clean up pending tasks
    if (task.type == SERVER_TASK_TYPE_CANCEL) {
      cleanup_pending_task(task.id_target);
    }
    const int task_id = task.id;
    QUE_DBG("new task, id = %d, front = %d\n", task_id, front);
    if (front) {
      queue_tasks.push_front(std::move(task));
    } else {
      queue_tasks.push_back(std::move(task));
    }
    condition_tasks.notify_one();
    return task_id;
  }

  // multi-task version of post()
  int post(std::vector<server_task> &&tasks, bool front = false) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    for (auto &task : tasks) {
      if (task.id == -1) {
        task.id = id++;
      }
      // if this is cancel task make sure to clean up pending tasks
      if (task.type == SERVER_TASK_TYPE_CANCEL) {
        cleanup_pending_task(task.id_target);
      }
      QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int)tasks.size(),
              front);
      if (front) {
        queue_tasks.push_front(std::move(task));
      } else {
        queue_tasks.push_back(std::move(task));
      }
    }
    condition_tasks.notify_one();
    return 0;
  }

  // Add a new task, but defer until one slot is available
  void defer(server_task&& task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    QUE_DBG("defer task, id = %d\n", task.id);
    queue_tasks_deferred.push_back(std::move(task));
    condition_tasks.notify_one();
  }

  // Get the next id for creating a new task
  int get_new_id() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    int new_id = id++;
    return new_id;
  }

  // Register function to process a new task
  void on_new_task(std::function<void(server_task&&)> callback) {
    SRV_DBG("%s", "++set on_new_task\r\n");
    callback_new_task = std::move(callback);
  }

  // Register the function to be called when all slots data is ready to be
  // processed
  void on_update_slots(std::function<void(void)> callback) {
    SRV_DBG("%s", "++set on_update_slots\r\n");
    callback_update_slots = std::move(callback);
  }

  // Call when the state of one slot is changed, it will move one task from
  // deferred to main queue
  void pop_deferred_task() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!queue_tasks_deferred.empty()) {
      queue_tasks.emplace_back(std::move(queue_tasks_deferred.front()));
      queue_tasks_deferred.pop_front();
    }
    condition_tasks.notify_one();
  }

  // end the start_loop routine
  void terminate() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    running = false;
    condition_tasks.notify_all();
  }

  /**
   * Main loop consists of these steps:
   * - Wait until a new task arrives
   * - Process the task (i.e. maybe copy data into slot)
   * - Check if multitask is finished
   * - Update all slots
   */
  void start_loop() {
    running = true;

    while (true) {
      QUE_DBG("%s", "processing new tasks\n");

      while (true) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (!running) {
          QUE_DBG("%s", "++here terminate 1\n");
          return;
        }
        if (queue_tasks.empty()) {
          lock.unlock();
          break;
        }
        server_task task = std::move(queue_tasks.front());
        queue_tasks.pop_front();
        lock.unlock();

        QUE_DBG("processing task, id = %d\n", task.id);
        callback_new_task(std::move(task));
      }

      // all tasks in the current loop is processed, slots data is now ready
      QUE_DBG("%s", "update slots\n");

      callback_update_slots();

      QUE_DBG("%s", "waiting for new tasks\n");
      {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (!running) {
          QUE_DBG("%s", "--here terminate 2\n");
          return;
        }
        if (queue_tasks.empty()) {
          condition_tasks.wait(
              lock, [&] { return (!queue_tasks.empty() || !running); });
        }
      }
    }
  }

 private:
  void cleanup_pending_task(int id_target) {
    // no need lock because this is called exclusively by post()
    auto rm_func = [id_target](const server_task &task) {
      return task.id_target == id_target;
    };
    queue_tasks.erase(
        std::remove_if(queue_tasks.begin(), queue_tasks.end(), rm_func),
        queue_tasks.end());
    queue_tasks_deferred.erase(
        std::remove_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(),
                       rm_func),
        queue_tasks_deferred.end());
  }
};

struct server_response {

    bool running = true;

  // for keeping track of all tasks waiting for the result
  std::unordered_set<int> waiting_task_ids;

  // the main result queue (using ptr for polymorphism)
  std::vector<server_task_result_ptr> queue_results;

  std::mutex mutex_results;
  std::condition_variable condition_results;

  // add the id_task to the list of tasks waiting for response
  void add_waiting_task_id(int id_task) {
    SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n",
            id_task, (int)waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.insert(id_task);
  }

  void add_waiting_tasks(const std::vector<server_task> &tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto &task : tasks) {
      SRV_DBG(
          "add task %d to waiting list. current waiting = %d (before add)\n",
          task.id, (int)waiting_task_ids.size());
      waiting_task_ids.insert(task.id);
    }
  }

  // when the request is finished, we can remove task associated with it
  void remove_waiting_task_id(int id_task) {
    SRV_DBG(
        "--remove task %d from waiting list. current waiting = %d (before "
        "remove)\n",
        id_task, (int)waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.erase(id_task);
    // make sure to clean up all pending results
    queue_results.erase(
        std::remove_if(queue_results.begin(), queue_results.end(),
                       [id_task](const server_task_result_ptr &res) {
                         return res->id == id_task;
                       }),
        queue_results.end());
  }

  void remove_waiting_task_ids(const std::unordered_set<int> &id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto &id_task : id_tasks) {
      SRV_DBG(
          "++remove task %d from waiting list. current waiting = %d (before "
          "remove)\n",
          id_task, (int)waiting_task_ids.size());
      waiting_task_ids.erase(id_task);
    }
  }

  // This function blocks the thread until there is a response for one of the
  // id_tasks
  server_task_result_ptr recv(const std::unordered_set<int> &id_tasks) {
      while (true) {
          std::unique_lock<std::mutex> lock(mutex_results);
          condition_results.wait(lock,
                                 [&]
                                 {
                                     if (!running) {
                                         SRV_DBG("%s : queue result stop\n", __func__);
                                         std::terminate(); // we cannot return here since the caller is HTTP code
                                     }
                                     return !queue_results.empty();
                                 });
          SRV_DBG("%s \r\n", "++call recv multitasks and after wait");
          for (size_t i = 0; i < queue_results.size(); i++) {
              if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                  server_task_result_ptr res = std::move(queue_results[i]);
                  queue_results.erase(queue_results.begin() + i);
                  return res;
              }
          }
    }

    // should never reach here
  }

  // This function blocks the thread until there is a response for one of the
  // id_tasks
  server_task_result *recv_aync(const std::unordered_set<int> &id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);
    // SRV_INF("recv_aync queue_results.size() %d\n", queue_results.size());
    for (int i = 0; i < (int)queue_results.size(); i++) {
      // SRV_INF("recv_aync i %d\n", i);
      if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
        server_task_result *res = queue_results[i].get();
        // queue_results.erase(queue_results.begin() + i);
        return res;
      }
    }
    return nullptr;
  }

  server_task_result_ptr recv_aync_confirm(
      const std::unordered_set<int> &id_tasks) {
    // SRV_INF("recv_aync_confirm %s\n", "1");
    std::unique_lock<std::mutex> lock(mutex_results);
    // SRV_INF("recv_aync_confirm queue_results.size() %d\n",
    // queue_results.size()); SRV_INF("queue_results.size() %d\n",
    // queue_results.size());
    for (int i = 0; i < (int)queue_results.size(); i++) {
      // SRV_INF("recv_aync_confirm i %d\n", i);
      // SRV_INF("recv_aync_confirm index %d\n", i);
      // SRV_INF("queue_results= %p\n", queue_results[i].get());
      // SRV_INF("find task id = %d\n", queue_results[i]->id);
      if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
        // SRV_INF("find task id = %d\n", queue_results[i]->id);
        // std::cout << "queue_results[i]->id: " << queue_results[i]->id <<
        // std::endl;
        server_task_result_ptr res = std::move(queue_results[i]);
        queue_results.erase(queue_results.begin() + i);
        return res;
      }
    }
    // std::cout << "nout found" << std::endl;
    // SRV_INF("not find %s\n", "task");
    return nullptr;
  }

  // same as recv(), but have timeout in seconds
  // if timeout is reached, nullptr is returned
  server_task_result_ptr recv_with_timeout(
      const std::unordered_set<int> &id_tasks, int timeout) {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_results);

      for (int i = 0; i < (int)queue_results.size(); i++) {
        if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
          server_task_result_ptr res = std::move(queue_results[i]);
          queue_results.erase(queue_results.begin() + i);
          return res;
        }
      }

      std::cv_status cr_res =
          condition_results.wait_for(lock, std::chrono::seconds(timeout));
      if (!running) {
        SRV_DBG("%s : queue result stop\n", __func__);
        std::terminate(); // we cannot return here since the caller is HTTP code
      }
      if (cr_res == std::cv_status::timeout) {
        return nullptr;
      }
    }

    // should never reach here
  }

  // single-task version of recv()
  server_task_result_ptr recv(int id_task) {
    SRV_DBG("%s \r\n","++call single-task version recv");
    std::unordered_set<int> id_tasks = {id_task};
    return recv(id_tasks);
  }

  // Send a new result to a waiting id_task
  void send(server_task_result_ptr &&result) {
    SRV_DBG("sending result for task id = %d\n", result->id);

    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto &id_task : waiting_task_ids) {
      if (result->id == id_task) {
        SRV_DBG("task id = %d pushed to result queue\n", result->id);

        queue_results.emplace_back(std::move(result));
        condition_results.notify_all();
        return;
      }
    }
  }

  // terminate the waiting loop
  void terminate() {
      running = false;
      condition_results.notify_all();
  }
};

struct server_context {
  int magic = 12345;
  std::mutex mutex_context;
  int generate_status = 0;  // 0 -> finished; 1 -> generating; 2 -> stopped; 3
                            // -> exceed; -1 -> failed;
  int model_status = 0;  // 0 -> init; 1 -> loaded; 2 -> unloaded; 3 -> infer; -1 -> error;
  bool stop = false;
  bool hm_running = false;
  bool houmo_sampling = true;
  int instance_no = 0; // Windows下多卡多服务实例场景
  int device_id = 0;   // Windows下依赖instance_no, Linux下依赖环境变量

  common_params *params=nullptr;
  std::thread ctx_t;
  std::thread srv_t;
  httplib::Server *svr;
  std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

  torch::nn::Embedding emb = nullptr;
  std::string model_path;

  std::map<std::string, tcim::Tensor> prefill_input_map;
  std::map<std::string, tcim::Tensor> prefill_output_map;

  std::map<std::string, tcim::Tensor> decode_input_map;
  std::map<std::string, tcim::Tensor> decode_output_map;

  std::map<std::string, tcim::Tensor> prefill_part1_tensormap;
  std::map<std::string, tcim::Tensor> prefill_part2_tensormap;
  std::map<std::string, tcim::Tensor> decode_part1_tensormap;
  std::map<std::string, tcim::Tensor> decode_part2_tensormap;

  int32_t context_length;

  llama_token token;
  torch::Tensor input_emd;
  int32_t next_ids[128]={0};//最大128 batch
  int32_t context_length_batch[128]={0};
  llama_pos decode_pos = 0;
  int32_t decode_n_tokens = 0;

  size_t n_finished = 0;

  std::vector<server_task> tasks;
  std::unordered_set<int> task_ids;

  common_params params_base;

  // note: keep these alive - they determine the lifetime of the model, context,
  // etc.
  common_init_result llama_init;
  common_init_result llama_init_dft;

  llama_model *model = nullptr;
  llama_context *ctx = nullptr;

  #if ENABLE_HOUMO_LLAMACPP
  // multimodal
  mtmd_context * mctx = nullptr;
  #endif

  const llama_vocab *vocab = nullptr;

  llama_model *model_dft = nullptr;

  llama_context_params cparams_dft;

  llama_batch batch = {};

  bool clean_kv_cache = true;
  bool add_bos_token = true;
  bool has_eos_token = false;

  int32_t n_ctx;  // total context for all clients / slots

  // slots / clients
  std::vector<server_slot> slots;
  json default_generation_settings_for_props;

  server_queue queue_tasks;
  server_response queue_results;

  server_metrics metrics;

  // Necessary similarity of prompt for slot selection
  float slot_prompt_similarity = 0.0f;

  common_chat_templates_ptr chat_templates;
  #if ENABLE_HOUMO_LLAMACPP
  oaicompat_parser_options  oai_parser_opt;
  #endif


  ~server_context() {
      SRV_DBG("delete server_context '%p'\n", this);

      mtmd_free(mctx);
      // Clear any sampling context
      for (server_slot& slot : slots) {
          common_sampler_free(slot.smpl);
          slot.smpl = nullptr;

          llama_free(slot.ctx_dft);
          slot.ctx_dft = nullptr;

          common_speculative_free(slot.spec);
          slot.spec = nullptr;

          llama_batch_free(slot.batch_spec);
      }

      llama_batch_free(batch);

      LOG_DBG("delete server_context finished '%p'\n", this);
  }

  bool is_special_token(const int token_id) const {
      const auto token_str = common_token_to_piece(ctx, token_id);

    // 检查是否以特殊符号开头
    if ((token_str.rfind("<|", 0) == 0) || (token_str.rfind("</", 0) == 0) || (token_str.rfind('<', 0) == 0)) {
        SRV_DBG("find special token:%s \r\n", token_str.c_str());
        return true;
    }

    return false;
  }

  int unload_model() {
    std::unique_lock<std::mutex> lock(mutex_context);
    model_status = 2;
    return 0;
  }


  bool load_model(const common_params & params, const std::string& model_path) {
    SRV_INF("loading model '%s'\n", params.model.path.c_str());

    params_base = params;

    llama_init = common_init_from_params(params_base);

    model = llama_init.model.get();
    ctx   = llama_init.context.get();

    if (model == nullptr) {
        SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
        return false;
    }

    vocab = llama_model_get_vocab(model);

    n_ctx = llama_n_ctx(ctx);

    add_bos_token = llama_vocab_get_add_bos(vocab);
    has_eos_token = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;

    if (!params_base.speculative.model.path.empty() || !params_base.speculative.model.hf_repo.empty()) {
        SRV_INF("loading draft model '%s'\n", params_base.speculative.model.path.c_str());

        auto params_dft = params_base;

        params_dft.devices      = params_base.speculative.devices;
        params_dft.model        = params_base.speculative.model;
        params_dft.n_ctx        = params_base.speculative.n_ctx == 0 ? params_base.n_ctx / params_base.n_parallel : params_base.speculative.n_ctx;
        params_dft.n_gpu_layers = params_base.speculative.n_gpu_layers;
        params_dft.n_parallel   = 1;
        params_dft.cache_type_k = params_base.speculative.cache_type_k;
        params_dft.cache_type_v = params_base.speculative.cache_type_v;

        llama_init_dft = common_init_from_params(params_dft);

        model_dft = llama_init_dft.model.get();

        if (model_dft == nullptr) {
            SRV_ERR("failed to load draft model, '%s'\n", params_base.speculative.model.path.c_str());
            return false;
        }

        if (!common_speculative_are_compatible(ctx, llama_init_dft.context.get())) {
            SRV_ERR("the draft model '%s' is not compatible with the target model '%s'\n", params_base.speculative.model.path.c_str(), params_base.model.path.c_str());

            return false;
        }

        const int n_ctx_dft = llama_n_ctx(llama_init_dft.context.get());

        cparams_dft = common_context_params_to_llama(params_dft);
        cparams_dft.n_batch = n_ctx_dft;

        // the context is not needed - we will create one for each slot
        llama_init_dft.context.reset();
    }

    chat_templates = common_chat_templates_init(model, params_base.chat_template);
    try {
        common_chat_format_example(chat_templates.get(), params.use_jinja);
    } catch (const std::exception & e) {
        SRV_WRN("%s: Chat template parsing error: %s\n", __func__, e.what());
        SRV_WRN("%s: The chat template that comes with this model is not yet supported, falling back to chatml. This may cause the model to output suboptimal responses\n", __func__);
        chat_templates = common_chat_templates_init(model, "chatml");
    }

    std::string & mmproj_path = params_base.mmproj.path;
    if (!mmproj_path.empty()) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu       = params_base.mmproj_use_gpu;
        mparams.print_timings = false;
        mparams.n_threads     = params_base.cpuparams.n_threads;
        mparams.verbosity     = params_base.verbosity > 0 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO;
        mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
        if (mctx == nullptr) {
            SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
            return false;
        }
        SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

        if (params_base.ctx_shift) {
            params_base.ctx_shift = false;
            SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
        }

        if (params_base.n_cache_reuse) {
            params_base.n_cache_reuse = 0;
            SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
        }

        if (!params_base.speculative.model.path.empty()) {
            SRV_ERR("%s\n", "err: speculative decode is not supported by multimodal");
            return false;
        }
    }

    if (!llama_memory_can_shift(llama_get_memory(ctx))) {
        if (params_base.ctx_shift) {
            params_base.ctx_shift = false;
            SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
        }

        if (params_base.n_cache_reuse) {
            params_base.n_cache_reuse = 0;
            SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
        }
    }

    return true;
  }

  void init_hm_llama(std::string model_path) {
    model_status = 0;
    params->model_alias=model_path;

    const int32_t n_ctx_slot = n_ctx / params_base.n_parallel;

    SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

    for (int i = 0; i < params_base.n_parallel; i++) {
      server_slot slot;

      slot.id = i;
      slot.ctx = ctx;
      slot.n_ctx = n_ctx_slot;
      slot.n_predict = params_base.n_predict;
      slot.mctx = mctx;
      slot.cache_tokens.has_mtmd = mctx != nullptr;

      if (model_dft) {
        slot.batch_spec =
            llama_batch_init(params_base.speculative.n_max + 1, 0, 1);

        slot.ctx_dft = llama_init_from_model(model_dft, cparams_dft);
        if (slot.ctx_dft == nullptr) {
          SRV_ERR("%s", "failed to create draft context\n");
          return;
        }

        slot.spec = common_speculative_init(slot.ctx_dft);
        if (slot.spec == nullptr) {
          SRV_ERR("%s", "failed to create speculator\n");
          return;
        }
      }

      SLT_INF(slot, "new slot n_ctx_slot = %d\n", slot.n_ctx);

      slot.params.sampling = params_base.sampling;

      slot.callback_on_release = [this](int) {
        queue_tasks.pop_deferred_task();
      };

      slot.reset();

      slots.push_back(std::move(slot));
    }

    default_generation_settings_for_props = slots[0].to_json();

    // the update_slots() logic will always submit a maximum of n_batch or
    // n_parallel tokens note that n_batch can be > n_ctx (e.g. for non-causal
    // attention models such as BERT where the KV cache is not used)
    {
      const int32_t n_batch = llama_n_batch(ctx);

      // only a single seq_id per token is needed
      batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
    }

    metrics.init();
    model_status = 1;

    //TODO szj check add this to init qwen3 8B or 14B
    oai_parser_opt = {
            /* use_jinja             */ params_base.use_jinja,
            /* prefill_assistant     */ params_base.prefill_assistant,
            /* reasoning_format      */ params_base.reasoning_format,
            /* chat_template_kwargs  */ params_base.default_template_kwargs,
            /* common_chat_templates */ chat_templates.get(),
            /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
            /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
            /* enable_thinking       */ params_base.reasoning_budget != 0,
        };
  }

  template <typename T>
  void save_raw_data(const T *data, size_t count, const std::string &filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
      throw std::runtime_error("无法打开文件: " + filename);
    }
    uint64_t file_size = count * sizeof(T);
    printf("++data addr is %p , write file_size is %llu\r\n", data, file_size);
    outfile.write(reinterpret_cast<const char *>(data), file_size);
    outfile.close();
  }

  void init_by_buffer(HANDLE h, uint8_t** buffers, size_t* bufferSize, size_t buffers_count, hardware_info hwinfo) {

      const int32_t n_ctx_slot = n_ctx / params_base.n_parallel;

      SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

      for (int i = 0; i < params_base.n_parallel; i++) {
          server_slot slot;

          slot.id = i;
          slot.ctx = ctx;
          slot.n_ctx = n_ctx_slot;
          slot.n_predict = params_base.n_predict;

          if (model_dft) {
              slot.batch_spec = llama_batch_init(params_base.speculative.n_max + 1, 0, 1);

              slot.ctx_dft = llama_init_from_model(model_dft, cparams_dft);
              if (slot.ctx_dft == nullptr) {
                  SRV_ERR("%s", "failed to create draft context\n");
                  return;
              }

              slot.spec = common_speculative_init(slot.ctx_dft);
              if (slot.spec == nullptr) {
                  SRV_ERR("%s", "failed to create speculator\n");
                  return;
              }
          }

          SLT_INF(slot, "new slot n_ctx_slot = %d\n", slot.n_ctx);

          slot.params.sampling = params_base.sampling;

          slot.callback_on_release = [this](int) { queue_tasks.pop_deferred_task(); };

          slot.reset();

          slots.push_back(std::move(slot));
      }

      default_generation_settings_for_props = slots[0].to_json();

      // the update_slots() logic will always submit a maximum of n_batch or
      // n_parallel tokens note that n_batch can be > n_ctx (e.g. for non-causal
      // attention models such as BERT where the KV cache is not used)
      {
          const int32_t n_batch = llama_n_batch(ctx);

          // only a single seq_id per token is needed
          batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
      }

      metrics.init();
      model_status = 1;
  }

  server_slot *get_slot_by_id(int id) {
    for (server_slot &slot : slots) {
      if (slot.id == id) {
        return &slot;
      }
    }

    return nullptr;
  }

  server_slot *get_available_slot(const server_task &task) {
    server_slot *ret = nullptr;

    // find the slot that has at least n% prompt similarity
    if (ret == nullptr && slot_prompt_similarity != 0.0f) {
      int lcs_len = 0;
      float similarity = 0;

      for (server_slot &slot : slots) {
        // skip the slot if it is not available
        if (slot.is_processing()) {
          continue;
        }

        // skip the slot if it does not contains cached tokens
        if (slot.cache_tokens.empty()) {
          continue;
        }

        // length of the Longest Common Subsequence between the current slot's
        // prompt and the input prompt
        int cur_lcs_len = slot.cache_tokens.get_common_prefix(task.prompt_tokens);

        // fraction of the common subsequence length compared to the current
        // slot's prompt length
        float cur_similarity = static_cast<float>(cur_lcs_len) /
                               static_cast<int>(slot.cache_tokens.size());

        // select the current slot if the criteria match
        if (cur_lcs_len > lcs_len && cur_similarity > slot_prompt_similarity) {
          lcs_len = cur_lcs_len;
          similarity = cur_similarity;
          ret = &slot;
        }
      }

      if (ret != nullptr) {
        SLT_DBG(
            *ret,
            "selected slot by lcs similarity, lcs_len = %d, similarity = %f\n",
            lcs_len, similarity);
      }
    }

    // find the slot that has been least recently used
    if (ret == nullptr) {
      int64_t t_last = -1;
      for (server_slot &slot : slots) {
        // skip the slot if it is not available
        if (slot.is_processing()) {
          continue;
        }

        // select the current slot if the criteria match
        if (!ret || slot.t_last_used <= t_last) {
          t_last = slot.t_last_used;
          ret = &slot;
        }
      }

      if (ret != nullptr) {
        SLT_DBG(*ret, "selected slot by lru, t_last = %" PRId64 "\n", t_last);
      }
    }

    return ret;
  }

  bool launch_slot_with_task(server_slot &slot, server_task &&task) {
    slot.reset();
    slot.id_task = task.id;
    slot.index = task.index;
    slot.task_type = task.type;
    slot.params = std::move(task.params);
    slot.prompt_tokens = std::move(task.prompt_tokens);

    if (!are_lora_equal(slot.params.lora, slot.lora)) {
      // if lora is changed, we cannot reuse cached tokens
      slot.cache_tokens.clear();
      slot.lora = slot.params.lora;
    }

    if (!slot.prompt_tokens.validate(ctx)) {
      send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
      return false;
    }
    SLT_DBG(slot, "launching slot : %s\n",
            safe_json_to_str(slot.to_json()).c_str());

    if (slot.n_predict > 0 && slot.params.n_predict > slot.n_predict) {
      // Might be better to reject the request with a 400 ?
      SLT_WRN(slot,
              "n_predict = %d exceeds server configuration, setting to %d\n",
              slot.params.n_predict, slot.n_predict);
      slot.params.n_predict = slot.n_predict;
    }

    if (slot.params.ignore_eos && has_eos_token) {
      slot.params.sampling.logit_bias.push_back(
          {llama_vocab_eos(vocab), -INFINITY});
    }

    {
      if (slot.smpl != nullptr) {
        SLT_WRN(slot, "++slot.smpl not nullptr %x\r\n", slot.smpl);
        common_sampler_free(slot.smpl);
      }

      slot.smpl = common_sampler_init(model, slot.params.sampling);
      if (slot.smpl == nullptr) {
        // for now, the only error that may happen here is invalid grammar
        send_error(task, "Failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
        return false;
      } else {
        SRV_DBG("%s", "++common_sampler_init success\n");
      }
    }

    if (slot.ctx_dft) {
      SRV_DBG("%s", "++llama_batch_free called\n");
      llama_batch_free(slot.batch_spec);

      slot.batch_spec =
          llama_batch_init(slot.params.speculative.n_max + 1, 0, 1);
    }

    slot.state = SLOT_STATE_STARTED;

    SLT_DBG(slot, "%s", "processing task\n");

    return true;
  }

  void kv_cache_clear() {
    SRV_DBG("%s", "clearing KV cache\n");

    // clear the entire KV cache
    llama_memory_clear(llama_get_memory(ctx), true);
    clean_kv_cache = false;
  }

  bool process_token(completion_token_output &result, server_slot &slot) {
    // remember which tokens were sampled - used for repetition penalties during
    // sampling
    const std::string token_str = result.text_to_send;
    slot.sampled = result.tok;

    slot.generated_text += token_str;
    if (slot.params.return_tokens) {
      SRV_DBG("++%s \r\n","call generated_tokens.push_back");
      slot.generated_tokens.push_back(result.tok);
    }
    slot.has_next_token = true;

    // check if there is incomplete UTF-8 character at the end
    bool incomplete =
        validate_utf8(slot.generated_text) < slot.generated_text.size();

    // search stop word and delete it
    if (!incomplete) {
      size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

      const std::string str_test = slot.generated_text.substr(pos);
      bool send_text = true;

      size_t stop_pos =
          slot.find_stopping_strings(str_test, token_str.size(), true);
      if (stop_pos != std::string::npos) {
        slot.generated_text.erase(slot.generated_text.begin() + pos + stop_pos,
                                  slot.generated_text.end());
        pos = std::min(slot.n_sent_text, slot.generated_text.size());
      } else if (slot.has_next_token) {
        stop_pos =
            slot.find_stopping_strings(str_test, token_str.size(), false);
        send_text = stop_pos == std::string::npos;
      }

      // check if there is any token to predict
      if (send_text) {
        // no send the stop word in the response
        result.text_to_send =
            slot.generated_text.substr(pos, std::string::npos);
        slot.n_sent_text += result.text_to_send.size();
        // add the token to slot queue and cache
      } else {
        result.text_to_send = "";
      }

      slot.add_token(result);
      if (slot.params.stream) {
        SRV_DBG("%s \r\n","call send_partial_response");
        send_partial_response(slot, result);
      }
    }

    if (incomplete) {
      slot.has_next_token = true;
    }

    // if context shifting is disabled, make sure that we don't run out of context
    if (!params_base.ctx_shift && slot.n_past + 1 >= slot.n_ctx) {
      slot.stop           = STOP_TYPE_LIMIT;
      slot.has_next_token = false;

      SLT_DBG(slot, "stopped due to running out of context, n_past = %d, n_ctx = %d\n", slot.n_past, slot.n_ctx);
    }

    // check the limits
    if (slot.n_decoded > 0 && slot.has_next_token &&
        !slot.has_budget(params_base)) {
      // 此处未进入
      slot.stop = STOP_TYPE_LIMIT;
      slot.has_next_token = false;

      SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n",
              slot.n_decoded, slot.params.n_predict);
    }

    if (slot.has_new_line) {
      SRV_DBG("++enter here %s\r\n", "has newline");
      // require that each new line has a whitespace prefix (i.e. indentation)
      // of at least slot.params.n_indent
      if (slot.params.n_indent > 0) {
        SRV_DBG("++enter here %s\r\n", "n_indent > 0");
        // check the current indentation
        // TODO: improve by not doing it more than once for each new line
        if (slot.last_nl_pos > 0) {
          size_t pos = slot.last_nl_pos;

          int n_indent = 0;
          while (pos < slot.generated_text.size() &&
                 (slot.generated_text[pos] == ' ' ||
                  slot.generated_text[pos] == '\t')) {
            n_indent++;
            pos++;
          }

          if (pos < slot.generated_text.size() &&
              n_indent < slot.params.n_indent) {
            slot.stop = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            // cut the last line
            slot.generated_text.erase(pos, std::string::npos);

            SLT_DBG(
                slot,
                "stopped by indentation limit, n_decoded = %d, n_indent = %d\n",
                slot.n_decoded, n_indent);
          }
        }

        // find the next new line
        {
          const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

          if (pos != std::string::npos) {
            slot.last_nl_pos = pos + 1;
          }
        }
      }
    }

    // check if there is a new line in the generated text
    if (result.text_to_send.find('\n') != std::string::npos) {
      slot.has_new_line = true;

      // if we have seen a new line, we stop after a certain time limit, but
      // only upon another new line
      if (slot.params.t_max_predict_ms > 0 &&
          (ggml_time_us() - slot.t_start_generation >
           1000.0f * slot.params.t_max_predict_ms)) {
        slot.stop = STOP_TYPE_LIMIT;
        slot.has_next_token = false;

        SLT_DBG(
            slot,
            "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n",
            slot.n_decoded, (int)slot.params.t_max_predict_ms);
      }
    }

    // if context shift is disabled, we stop when it reaches the context limit
    if (slot.n_past >= slot.n_ctx) {
      slot.truncated = true;
      slot.stop = STOP_TYPE_LIMIT;
      slot.has_next_token = false;

      SLT_DBG(slot,
              "stopped due to running out of context capacity, n_past = %d, "
              "n_prompt_tokens = %d, n_decoded = %d, n_ctx = %d\n",
              slot.n_decoded, slot.n_prompt_tokens, slot.n_past, slot.n_ctx);
    }

    if (llama_vocab_is_eog(vocab, result.tok)) {
      slot.stop = STOP_TYPE_EOS;
      slot.has_next_token = false;

      decode_pos = 0;

      SLT_DBG(slot, "%s", "stopped by EOS\n");
    }

    const auto n_ctx_train = llama_model_n_ctx_train(model);

    if (slot.params.n_predict < 1 && slot.n_predict < 1 &&
        slot.n_prompt_tokens + slot.n_decoded >= n_ctx_train) {
      slot.truncated = true;
      slot.stop = STOP_TYPE_LIMIT;
      slot.has_next_token = false;  // stop prediction

      SLT_WRN(slot,
              "n_predict (%d) is set for infinite generation. "
              "Limiting generated tokens to n_ctx_train (%d) to avoid EOS-less "
              "generation infinite loop\n",
              slot.params.n_predict, n_ctx_train);
    }

    SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n",
            slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

    return slot.has_next_token;  // continue
  }

  void populate_token_probs(const server_slot &slot,
                            completion_token_output &result, bool post_sampling,
                            bool special, int idx) {
    size_t n_probs = slot.params.sampling.n_probs;
    size_t n_vocab = llama_vocab_n_tokens(vocab);
    if (post_sampling) {
      const auto *cur_p = common_sampler_get_candidates(slot.smpl);
      const size_t max_probs = cur_p->size;

      // set probability for sampled token
      for (size_t i = 0; i < max_probs; i++) {
        if (cur_p->data[i].id == result.tok) {
          result.prob = cur_p->data[i].p;
          break;
        }
      }

      // set probability for top n_probs tokens
      result.probs.reserve(max_probs);
      for (size_t i = 0; i < std::min(max_probs, n_probs); i++) {
        result.probs.push_back(
            {cur_p->data[i].id,
             common_token_to_piece(ctx, cur_p->data[i].id, special),
             cur_p->data[i].p});
      }
    } else {
      // TODO: optimize this with min-p optimization
      std::vector<llama_token_data> cur = get_token_probabilities(ctx, idx);

      // set probability for sampled token
      for (size_t i = 0; i < n_vocab; i++) {
        // set probability for sampled token
        if (cur[i].id == result.tok) {
          result.prob = cur[i].p;
          break;
        }
      }

      // set probability for top n_probs tokens
      result.probs.reserve(n_probs);
      for (size_t i = 0; i < std::min(n_vocab, n_probs); i++) {
        result.probs.push_back({cur[i].id,
                                common_token_to_piece(ctx, cur[i].id, special),
                                cur[i].p});
      }
    }
  }

  void send_error(const server_task &task, const std::string &error,
                  const enum error_type type = ERROR_TYPE_SERVER) {
    send_error(task.id, error, type);
  }

  void send_error(const server_slot &slot, const std::string &error,
                  const enum error_type type = ERROR_TYPE_SERVER) {
    send_error(slot.id_task, error, type);
  }

  void send_error(const int id_task, const std::string &error,
                  const enum error_type type = ERROR_TYPE_SERVER) {
    SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

    auto res = std::make_unique<server_task_result_error>();
    res->id = id_task;
    res->err_type = type;
    res->err_msg = error;

    queue_results.send(std::move(res));
  }

  // if multimodal is enabled, send an error and return false
  bool ensure_no_mtmd(const int id_task) {
      if (mctx) {
          send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
          return false;
      }
      return true;
  }

  void send_partial_response(server_slot &slot,
                             const completion_token_output &tkn) {
    auto res = std::make_unique<server_task_result_cmpl_partial>();

    res->id = slot.id_task;
    res->index = slot.index;
    res->content = tkn.text_to_send;
    res->tokens = {tkn.tok};

    res->n_decoded = slot.n_decoded;
    res->n_prompt_tokens = slot.n_prompt_tokens;
    res->post_sampling_probs = slot.params.post_sampling_probs;

    res->verbose = slot.params.verbose;
    res->oaicompat = slot.params.oaicompat;
    res->oaicompat_model = slot.params.oaicompat_model;
    res->oaicompat_cmpl_id = slot.params.oaicompat_cmpl_id;

    slot.update_chat_msg(res->oaicompat_msg_diffs);

    // populate res.probs_output
    if (slot.params.sampling.n_probs > 0) {
      res->prob_output = tkn;  // copy the token probs
    }

    // populate timings if this is final response or timings_per_token is
    // enabled
    if (slot.stop != STOP_TYPE_NONE || slot.params.timings_per_token) {
      res->timings = slot.get_timings();
    }

    queue_results.send(std::move(res));
  }

  void send_final_response(server_slot &slot) {
    auto res = std::make_unique<server_task_result_cmpl_final>();
    res->id = slot.id_task;
    res->id_slot = slot.id;

    res->index = slot.index;
    res->content = slot.generated_text;
    res->tokens = std::move(slot.generated_tokens);
    res->timings = slot.get_timings();
    res->prompt = slot.prompt_tokens.detokenize(ctx, true);
    res->response_fields = std::move(slot.params.response_fields);

    res->truncated = slot.truncated;
    res->n_decoded = slot.n_decoded;
    res->n_prompt_tokens = slot.n_prompt_tokens;
    res->n_tokens_cached = slot.n_past;
    res->has_new_line = slot.has_new_line;
    res->stopping_word = slot.stopping_word;
    res->stop = slot.stop;
    res->post_sampling_probs = slot.params.post_sampling_probs;

    res->verbose = slot.params.verbose;
    res->stream = slot.params.stream;
    res->oaicompat = slot.params.oaicompat;
    res->oaicompat_model = slot.params.oaicompat_model;
    res->oaicompat_cmpl_id = slot.params.oaicompat_cmpl_id;
    res->oaicompat_msg = slot.update_chat_msg(res->oaicompat_msg_diffs);

    // populate res.probs_output
    if (slot.params.sampling.n_probs > 0) {
      if (!slot.params.stream && slot.stop == STOP_TYPE_WORD) {
        const llama_tokens stop_word_toks =
            common_tokenize(ctx, slot.stopping_word, false);

        size_t safe_offset =
            std::min(slot.generated_token_probs.size(), stop_word_toks.size());
        res->probs_output = std::vector<completion_token_output>(
            slot.generated_token_probs.begin(),
            slot.generated_token_probs.end() - safe_offset);
      } else {
        res->probs_output = std::vector<completion_token_output>(
            slot.generated_token_probs.begin(),
            slot.generated_token_probs.end());
      }
    }

    res->generation_params = slot.params;  // copy the parameters

    queue_results.send(std::move(res));
  }

  void send_embedding(const server_slot &slot, const llama_batch &batch) {
    auto res = std::make_unique<server_task_result_embd>();
    res->id = slot.id_task;
    res->index = slot.index;
    res->n_tokens = slot.n_prompt_tokens;
    res->oaicompat = slot.params.oaicompat;

    const int n_embd = llama_model_n_embd(model);

    std::vector<float> embd_res(n_embd, 0.0f);

    for (int i = 0; i < batch.n_tokens; ++i) {
      if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
        continue;
      }

      const float *embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
      if (embd == NULL) {
        embd = llama_get_embeddings_ith(ctx, i);
      }

      if (embd == NULL) {
        SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n",
                batch.token[i], batch.seq_id[i][0]);

        res->embedding.push_back(std::vector<float>(n_embd, 0.0f));
        continue;
      }

      // normalize only when there is pooling
      // TODO: configurable
      if (llama_pooling_type(slot.ctx) != LLAMA_POOLING_TYPE_NONE) {
        common_embd_normalize(embd, embd_res.data(), n_embd, 2);
        res->embedding.push_back(embd_res);
      } else {
        res->embedding.push_back({embd, embd + n_embd});
      }
    }

    SLT_DBG(slot, "%s", "sending embeddings\n");

    queue_results.send(std::move(res));
  }

  void send_rerank(const server_slot &slot, const llama_batch &batch) {
    auto res = std::make_unique<server_task_result_rerank>();
    res->id = slot.id_task;
    res->index = slot.index;
    res->n_tokens = slot.n_prompt_tokens;

    for (int i = 0; i < batch.n_tokens; ++i) {
      if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
        continue;
      }

      const float *embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
      if (embd == NULL) {
        embd = llama_get_embeddings_ith(ctx, i);
      }

      if (embd == NULL) {
        SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n",
                batch.token[i], batch.seq_id[i][0]);

        res->score = -1e6;
        continue;
      }

      res->score = embd[0];
    }

    SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

    queue_results.send(std::move(res));
  }

  //
  // Functions to create new task(s) and receive result(s)
  //

  void cancel_tasks(const std::unordered_set<int> &id_tasks) {
    std::vector<server_task> cancel_tasks;
    cancel_tasks.reserve(id_tasks.size());
    for (const auto &id_task : id_tasks) {
      SRV_WRN("cancel task, id_task = %d\n", id_task);

      server_task task(SERVER_TASK_TYPE_CANCEL);
      task.id_target = id_task;
      queue_results.remove_waiting_task_id(id_task);
      cancel_tasks.push_back(std::move(task));
    }
    // push to beginning of the queue, so it has highest priority
    queue_tasks.post(std::move(cancel_tasks), true);
    decode_pos = 0;
  }

  // receive the results from task(s)
  void receive_multi_results(
      const std::unordered_set<int> &id_tasks,
      const std::function<void(std::vector<server_task_result_ptr> &)>
          &result_handler,
      const std::function<void(json)> &error_handler,
      const std::function<bool()> &is_connection_closed) {
    std::vector<server_task_result_ptr> results(id_tasks.size());
    for (int i = 0; i < (int)id_tasks.size(); i++) {
      server_task_result_ptr result =
          queue_results.recv_with_timeout(id_tasks, HTTP_POLLING_SECONDS);

      if (is_connection_closed()) {
        cancel_tasks(id_tasks);
        return;
      }

      if (result == nullptr) {
        i--;  // retry
        continue;
      }

      if (result->is_error()) {
        error_handler(result->to_json());
        cancel_tasks(id_tasks);
        return;
      }

      GGML_ASSERT(
          dynamic_cast<server_task_result_cmpl_final *>(result.get()) !=
              nullptr ||
          dynamic_cast<server_task_result_embd *>(result.get()) != nullptr ||
          dynamic_cast<server_task_result_rerank *>(result.get()) != nullptr);
      const size_t idx = result->get_index();
      GGML_ASSERT(idx < results.size() && "index out of range");
      results[idx] = std::move(result);
    }
    result_handler(results);
  }

  // receive the results from task(s), in stream mode
  void receive_cmpl_results_stream(
      const std::unordered_set<int> &id_tasks,
      const std::function<bool(server_task_result_ptr &)> &result_handler,
      const std::function<void(json)> &error_handler,
      const std::function<bool()> &is_connection_closed) {
    size_t n_finished = 0;
    while (true) {
      server_task_result_ptr result =
          queue_results.recv_with_timeout(id_tasks, HTTP_POLLING_SECONDS);

      if (is_connection_closed && is_connection_closed()) {
        cancel_tasks(id_tasks);
        return;
      }

      if (result == nullptr) {
        continue;  // retry
      }

      if (result->is_error()) {
        error_handler(result->to_json());
        cancel_tasks(id_tasks);
        return;
      }

      GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial *>(
                      result.get()) != nullptr ||
                  dynamic_cast<server_task_result_cmpl_final *>(result.get()) !=
                      nullptr);
      if (!result_handler(result)) {
        cancel_tasks(id_tasks);
        break;
      }

      if (result->is_stop()) {
        if (++n_finished == id_tasks.size()) {
          break;
        }
      }
    }
  }
  // api_Generate 中调用了这个函数
  int completion_generate(json& data, std::vector<server_task_result_ptr>& results) {
      auto type = SERVER_TASK_TYPE_COMPLETION;
      GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

      std::unique_lock<std::mutex> lock(mutex_context);
      generate_status = 1;
      model_status = 3;
      auto completion_id = gen_chatcmplid();
      std::vector<server_task> tasks;
      oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;

      try {
          std::vector<server_tokens> inputs;
          const auto& prompt = data.at("prompt");
          auto tokenized_prompts = tokenize_input_prompts(vocab, prompt, true, true);
          for (auto& p : tokenized_prompts) {
              auto tmp = server_tokens(p, false);
              inputs.push_back(std::move(tmp));
          }
          tasks.reserve(inputs.size());
          for (size_t i = 0; i < tokenized_prompts.size(); i++) {
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

  int completion_generate_str(json &data, std::string &result_str) {
    bool stream = json_value(data, "stream", false);
    if (stream) {
      SRV_ERR("Error: %s\n", "This is not stream api!");
      return -1;
    }
    std::unique_lock<std::mutex> lock(mutex_context);
    generate_status = 1;
    model_status = 3;
    auto type = SERVER_TASK_TYPE_COMPLETION;
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION ||
                type == SERVER_TASK_TYPE_INFILL);
    auto completion_id = gen_chatcmplid();
    std::vector<server_task> tasks;
    oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;
    try {
        std::vector<server_tokens> inputs;
        const auto& prompt = data.at("prompt");
        auto tokenized_prompts = tokenize_input_prompts(vocab, prompt, true, true);
        for (auto& p : tokenized_prompts) {
            auto tmp = server_tokens(p, false);
            inputs.push_back(std::move(tmp));
        }
        tasks.reserve(inputs.size());
        SRV_DBG("++tokenized_prompts.size() is:%d\r\n", tokenized_prompts.size());
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
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

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_final *>(result.get()) !=
                nullptr ||
            dynamic_cast<server_task_result_embd *>(result.get()) != nullptr ||
            dynamic_cast<server_task_result_rerank *>(result.get()) != nullptr);
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

  int completion_stream_generate(json &data) {
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
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION ||
                type == SERVER_TASK_TYPE_INFILL);

    auto completion_id = gen_chatcmplid();

    oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;

    try {
        std::vector<server_tokens> inputs;
        const auto& prompt = data.at("prompt");
        auto tokenized_prompts = tokenize_input_prompts(vocab, prompt, true, true);
        for (auto& p : tokenized_prompts) {
            auto tmp = server_tokens(p, false);
            inputs.push_back(std::move(tmp));
        }
        tasks.reserve(inputs.size());
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
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

    for (auto it = task_ids.begin(); it != task_ids.end(); ++it) {
        std::cout << *it << " ";
    }
    task_ids = server_task::get_list_id(tasks);

    stop = false;
    SRV_INF("create stop: %p\n", &stop);
    SRV_INF("create stop: %d\n", stop);

    return 0;
  }

  int completion_token_count(json &data) {
    try {
      std::vector<llama_tokens> tokenized_prompts =
          tokenize_input_prompts(vocab, data.at("prompt"), true, true);
      if (tokenized_prompts.size() == 1) {
        return tokenized_prompts[0].size();
      } else {
        SRV_WRN("Warn: %s\n", "Multi prompts not supported yet!");
      }
    } catch (const std::exception &e) {
      log_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
      return -1;
    }

    return 0;
  }

  // szj 这个函数没调用，不用看
  int completion_stream_generate_next_token(server_task_result_ptr &result) {
    result = queue_results.recv(task_ids);

    if (result->is_error()) {
      cancel_tasks(task_ids);
      generate_status = -1;
      return -1;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial *>(result.get()) !=
                    nullptr ||
                dynamic_cast<server_task_result_cmpl_final *>(result.get()) !=
                    nullptr);

    if (result->is_stop()) {
      if (++n_finished == task_ids.size()) {
        SRV_DBG("all task finished = %d\n", n_finished);
        queue_results.remove_waiting_task_ids(task_ids);
        task_ids.clear();
        tasks.clear();
        n_finished = 0;
        generate_status = 0;
        model_status = 1;
        return 1;
      }
    }
    return 0;
  }

  int completion_stream_generate_next_token_str(std::string &result_str,
                                                bool confirm = true) {
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
      SRV_WRN("task_ids is empty and not confirm = %d , ignore it.\n",
              task_ids.size());
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
    server_task_result *result2 = nullptr;
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

    GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial *>(result.get()) !=
                    nullptr ||
                dynamic_cast<server_task_result_cmpl_final *>(result.get()) !=
                    nullptr);
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

  int completion_stream_generate_stop() {
    std::unique_lock<std::mutex> lock(mutex_context);

    if (task_ids.size() == 0) {
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

  //
  // Functions to process the task
  //
  void process_single_task(server_task&& task) {
    switch (task.type) {
      case SERVER_TASK_TYPE_COMPLETION:
      case SERVER_TASK_TYPE_INFILL:
      case SERVER_TASK_TYPE_EMBEDDING:
      case SERVER_TASK_TYPE_RERANK: {
        const int id_slot = task.id_selected_slot;

        server_slot *slot =
            id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

        if (slot == nullptr) {
          // if no slot is available, we defer this task for processing later
          SRV_DBG("no slot is available, defer task, id_task = %d\n", task.id);
          queue_tasks.defer(std::move(task));
          break;
        }
        if (slot->is_processing()) {
          // if requested slot is unavailable, we defer this task for processing
          // later
          SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n",
                  task.id);
          queue_tasks.defer(std::move(task));
          break;
        }

        if (!launch_slot_with_task(*slot, std::move(task))) {
          SRV_ERR("failed to launch slot with task, id_task = %d\n", task.id);
          break;
        }
      } break;
      case SERVER_TASK_TYPE_CANCEL: {
        // release slot linked with the task id
        for (auto &slot : slots) {
          if (slot.id_task == task.id_target) {
            slot.release();
            break;
          }
        }
      } break;
      case SERVER_TASK_TYPE_NEXT_RESPONSE: {
        // do nothing
      } break;
      case SERVER_TASK_TYPE_METRICS: {
        json slots_data = json::array();

        int n_idle_slots = 0;
        int n_processing_slots = 0;

        for (server_slot &slot : slots) {
          json slot_data = slot.to_json();

          if (slot.is_processing()) {
            n_processing_slots++;
          } else {
            n_idle_slots++;
          }

          slots_data.push_back(slot_data);
        }
        SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots,
                n_processing_slots);

        auto res = std::make_unique<server_task_result_metrics>();
        res->id = task.id;
        res->slots_data = std::move(slots_data);
        res->n_idle_slots = n_idle_slots;
        res->n_processing_slots = n_processing_slots;
        res->n_tasks_deferred = queue_tasks.queue_tasks_deferred.size();
        res->t_start = metrics.t_start;

        res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
        res->t_prompt_processing_total = metrics.t_prompt_processing_total;
        res->n_tokens_predicted_total = metrics.n_tokens_predicted_total;
        res->t_tokens_generation_total = metrics.t_tokens_generation_total;

        res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
        res->t_prompt_processing = metrics.t_prompt_processing;
        res->n_tokens_predicted = metrics.n_tokens_predicted;
        res->t_tokens_generation = metrics.t_tokens_generation;

        res->n_decode_total = metrics.n_decode_total;
        res->n_busy_slots_total = metrics.n_busy_slots_total;

        if (task.metrics_reset_bucket) {
          metrics.reset_bucket();
        }
        queue_results.send(std::move(res));
      } break;
      case SERVER_TASK_TYPE_SLOT_SAVE: {
        if (!ensure_no_mtmd(task.id)) {
              break;
        }
        int id_slot = task.slot_action.slot_id;
        server_slot *slot = get_slot_by_id(id_slot);
        if (slot == nullptr) {
          send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
          break;
        }
        if (slot->is_processing()) {
          // if requested slot is unavailable, we defer this task for processing
          // later
          SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n",
                  task.id);
          queue_tasks.defer(std::move(task));
          break;
        }

        const size_t token_count = slot->cache_tokens.size();
        const int64_t t_start = ggml_time_us();

        std::string filename = task.slot_action.filename;
        std::string filepath = task.slot_action.filepath;

        const llama_tokens & tokens = slot->cache_tokens.get_text_tokens();
        const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id, tokens.data(), token_count);

        const int64_t t_end = ggml_time_us();
        const double t_save_ms = (t_end - t_start) / 1000.0;

        auto res = std::make_unique<server_task_result_slot_save_load>();
        res->id = task.id;
        res->id_slot = id_slot;
        res->filename = filename;
        res->is_save = true;
        res->n_tokens = token_count;
        res->n_bytes = nwrite;
        res->t_ms = t_save_ms;
        queue_results.send(std::move(res));
      } break;
      case SERVER_TASK_TYPE_SLOT_RESTORE: {
        if (!ensure_no_mtmd(task.id)) break;
        int id_slot = task.slot_action.slot_id;
        server_slot *slot = get_slot_by_id(id_slot);
        if (slot == nullptr) {
          send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
          break;
        }
        if (slot->is_processing()) {
          // if requested slot is unavailable, we defer this task for processing
          // later
          SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n",
                  task.id);
          queue_tasks.defer(std::move(task));
          break;
        }

        const int64_t t_start = ggml_time_us();

        std::string filename = task.slot_action.filename;
        std::string filepath = task.slot_action.filepath;

        llama_tokens tokens;
        tokens.resize(slot->n_ctx);
        size_t token_count = 0;
        size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), slot->id, tokens.data(), slot->cache_tokens.size(), &token_count);
        if (nread == 0) {
          slot->cache_tokens.clear(); // KV may already been invalidated?
          send_error(task,
                     "Unable to restore slot, no available space in KV cache "
                     "or invalid slot save file",
                     ERROR_TYPE_INVALID_REQUEST);
          break;
        }
        tokens.resize(token_count);
        slot->cache_tokens.clear();
        slot->cache_tokens.insert(tokens);

        const int64_t t_end = ggml_time_us();
        const double t_restore_ms = (t_end - t_start) / 1000.0;

        auto res = std::make_unique<server_task_result_slot_save_load>();
        res->id = task.id;
        res->id_slot = id_slot;
        res->filename = filename;
        res->is_save = false;
        res->n_tokens = token_count;
        res->n_bytes = nread;
        res->t_ms = t_restore_ms;
        queue_results.send(std::move(res));
      } break;
      case SERVER_TASK_TYPE_SLOT_ERASE: {
        if (!ensure_no_mtmd(task.id)) break;
        int id_slot = task.slot_action.slot_id;
        server_slot *slot = get_slot_by_id(id_slot);
        if (slot == nullptr) {
          send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
          break;
        }
        if (slot->is_processing()) {
          // if requested slot is unavailable, we defer this task for processing
          // later
          SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n",
                  task.id);
          queue_tasks.defer(std::move(task));
          break;
        }

        // Erase token cache
        const size_t n_erased = slot->cache_tokens.size();
        llama_memory_seq_rm(llama_get_memory(ctx), slot->id, -1, -1);
        slot->cache_tokens.clear();

        auto res = std::make_unique<server_task_result_slot_erase>();
        res->id = task.id;
        res->id_slot = id_slot;
        res->n_erased = n_erased;
        queue_results.send(std::move(res));
      } break;
      case SERVER_TASK_TYPE_SET_LORA: {
        params_base.lora_adapters = std::move(task.set_lora);
        auto res = std::make_unique<server_task_result_apply_lora>();
        res->id = task.id;
        queue_results.send(std::move(res));
      } break;
    }
  }

  void update_slots() {
    // check if all slots are idle
    {
      bool all_idle = true;

      for (auto &slot : slots) {
        if (slot.is_processing()) {
          all_idle = false;
          break;
        }
      }

      if (all_idle) {
        model_status=1;
        SRV_DBG("all slots are idle  model_status=%d afte set to 1\n",model_status);
        if (clean_kv_cache) {
          kv_cache_clear();
        }

        return;
      }
    }

    {
      SRV_DBG("%s", "posting NEXT_RESPONSE\n");

      server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
      task.id = queue_tasks.get_new_id();
      queue_tasks.post(std::move(task));
    }

    // apply context-shift if needed
    // TODO: simplify and improve
    for (server_slot &slot : slots) {
      if (slot.is_processing() && slot.n_past + 1 >= slot.n_ctx) {
        // szj 此处未调用
        if (!params_base.ctx_shift) {
          // this check is redundant (for good)
          // we should never get here, because generation should already stopped
          // in process_token()
          slot.release();
          send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
          continue;
        }

        if (mctx) {
          // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
          // we don't support ctx_shift because an image chunk may contain multiple tokens
          GGML_ABORT("not supported by multimodal");
        }

        // Shift context
        const int n_keep = slot.params.n_keep + add_bos_token;
        const int n_left = slot.n_past - n_keep;
        const int n_discard =
            slot.params.n_discard ? slot.params.n_discard : (n_left / 2);

        SLT_WRN(
            slot,
            "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n",
            n_keep, n_left, n_discard);

        llama_memory_seq_rm(llama_get_memory(ctx), slot.id, n_keep, n_keep + n_discard);
        llama_memory_seq_add(llama_get_memory(ctx), slot.id, n_keep + n_discard, slot.n_past, -n_discard);

        // add generated tokens to cache
        {
          llama_tokens new_tokens = slot.cache_tokens.get_text_tokens(); // copy
          for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
            new_tokens[i - n_discard] = new_tokens[i];
          }

          new_tokens.resize(slot.cache_tokens.size() - n_discard);
          slot.cache_tokens.clear();
          slot.cache_tokens.insert(new_tokens);
        }

        slot.n_past -= n_discard;

        slot.truncated = true;
      }
    }

    // start populating the batch for this iteration
    common_batch_clear(batch);

    // track if given slot can be batched with slots already in the batch
    server_slot *slot_batched = nullptr;

    auto accept_special_token = [&](server_slot &slot, llama_token token) {
      return params_base.special ||
             slot.params.sampling.preserved_tokens.find(token) !=
                 slot.params.sampling.preserved_tokens.end();
    };

    // frist, add sampled tokens from any ongoing sequences
    for (auto &slot : slots) {
      if (slot.state != SLOT_STATE_GENERATING) {
        SRV_DBG("%s", "+++calling continue\r\n");
        continue;
      }

      // check if we can batch this slot with the previous one
      if (!slot_batched) {
        slot_batched = &slot;
      } else if (!slot_batched->can_batch_with(slot)) {
        SRV_DBG("##slot.id=%d  cannot batch with slot_batched->id:%d\r\n",slot.id,slot_batched->id);
        continue;
      }

      slot.i_batch = batch.n_tokens;
      SRV_DBG("before common_batch_add batch.n_tokens=%d\r\n",batch.n_tokens);
      common_batch_add(batch, slot.sampled, slot.n_past, {slot.id}, true);
      SRV_DBG("after common_batch_add batch.n_tokens=%d\r\n",batch.n_tokens);

      slot.n_past += 1;
      slot.cache_tokens.push_back(slot.sampled);

      SLT_DBG(slot,
              "slot decode token, n_ctx = %d, n_past = %d, n_cache_tokens = "
              "%d, truncated = %d\n",
              slot.n_ctx, slot.n_past, (int)slot.cache_tokens.size(),
              slot.truncated);
    }

    // process in chunks of params.n_batch
    int32_t n_batch = llama_n_batch(ctx);
    int32_t n_ubatch = llama_n_ubatch(ctx);
    SRV_DBG("n_batch=%d n_ubatch=%d \r\n",n_batch,n_ubatch);

    // next, batch any pending prompts without exceeding n_batch
    if (params_base.cont_batching || batch.n_tokens == 0) {
      size_t slot_index=0;
      size_t batched_slot_index=0;
      for (auto &slot : slots) {
        slot_index++;
        // check if we can batch this slot with the previous one
        if (slot.is_processing()) {
          if (!slot_batched) {
            slot_batched = &slot;
            SRV_DBG("set slot_batched to slot_index:%d \r\n",slot_index-1);
            batched_slot_index=slot_index-1;
          } else if (!slot_batched->can_batch_with(slot)) {
            SRV_DBG("slot_batched cannot batch  to slot_index:%d \r\n",slot_index-1);
            continue;
          } else {
            SRV_DBG("slot_batched index:%d can batch with slot_index:%d \r\n",batched_slot_index,slot_index-1);
          }
        }

        // this slot still has a prompt to be processed
        if (slot.state == SLOT_STATE_PROCESSING_PROMPT ||
            slot.state == SLOT_STATE_STARTED) {
          decode_pos = 0;
          auto &prompt_tokens = slot.prompt_tokens;

          // TODO: maybe move branch to outside of this loop in the future
          if (slot.state == SLOT_STATE_STARTED) {
            slot.t_start_process_prompt = ggml_time_us();
            slot.t_start_generation = 0;

            slot.n_past = 0;
            slot.n_prompt_tokens = prompt_tokens.size();
            slot.state = SLOT_STATE_PROCESSING_PROMPT;

            SLT_DBG(slot,
                    "new prompt, n_ctx_slot = %d, n_keep = %d, n_prompt_tokens "
                    "= %d\n",
                    slot.n_ctx, slot.params.n_keep, slot.n_prompt_tokens);

            // print prompt tokens (for debugging)
            /*if (1) {
              // first 16 tokens (avoid flooding logs)
              for (int i = 0; i < std::min<int>(16, prompt_tokens.size());
                   i++) {
                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i,
                        prompt_tokens[i],
                        common_token_to_piece(ctx, prompt_tokens[i]).c_str());
              }
            } else {
              // all
              for (int i = 0; i < (int)prompt_tokens.size(); i++) {
                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i,
                        prompt_tokens[i],
                        common_token_to_piece(ctx, prompt_tokens[i]).c_str());
              }
            }*/

            // empty prompt passed -> release the slot and send empty response
            if (prompt_tokens.empty()) {
              //szj 此处未调用
              SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

              slot.release();
              slot.print_timings();
              send_final_response(slot);
              continue;
            }

            // TODO: support memory-less logits computation
            if (slot.need_logits() && !llama_get_memory(ctx)) {
              slot.release();
              send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
              continue;
            }

            if (!slot.can_split()) {
              if (slot.n_prompt_tokens > n_ubatch) {
                slot.release();
                send_error(slot,
                           "input is too large to process. increase the "
                           "physical batch size",
                           ERROR_TYPE_SERVER);
                continue;
              }

              if (slot.n_prompt_tokens > slot.n_ctx) {
                slot.release();
                send_error(
                    slot, "input is larger than the max context size. skipping",
                    ERROR_TYPE_SERVER);
                continue;
              }
            } else {
              if (!params_base.ctx_shift) {
                // if context shift is disabled, we make sure prompt size is
                // smaller than KV size
                // TODO: there should be a separate parameter that control
                // prompt truncation
                //       context shift should be applied only during the
                //       generation phase
                if (slot.n_prompt_tokens >= slot.n_ctx) {
                  slot.release();
                  send_error(
                      slot,
                      "the request exceeds the available context size. try "
                      "increasing the context size or enable context shift",
                      ERROR_TYPE_INVALID_REQUEST);
                  continue;
                }
              }
              if (slot.params.n_keep < 0) {
                slot.params.n_keep = slot.n_prompt_tokens;
              }
              slot.params.n_keep = std::min(slot.n_ctx - 4, slot.params.n_keep);

              // if input prompt is too big, truncate it
              if (slot.n_prompt_tokens >= slot.n_ctx) {
                if (mctx) {
                  // we should never reach this
                  GGML_ABORT("not supported by multimodal");
                }

                //szj 此处未调用
                const int n_left = slot.n_ctx - slot.params.n_keep;

                const int n_block_size = n_left / 2;
                const int erased_blocks =
                    (slot.n_prompt_tokens - slot.params.n_keep - n_block_size) /
                    n_block_size;

                const llama_tokens & curr_tokens = slot.prompt_tokens.get_text_tokens();
                llama_tokens new_tokens(
                    curr_tokens.begin(),
                    curr_tokens.begin() + slot.params.n_keep);

                new_tokens.insert(new_tokens.end(), curr_tokens.begin() + slot.params.n_keep + erased_blocks * n_block_size,
                                  curr_tokens.end());

                prompt_tokens.clear();
                prompt_tokens.insert(new_tokens);

                slot.truncated = true;
                slot.n_prompt_tokens = prompt_tokens.size();

                SLT_WRN(slot,
                        "input truncated, n_ctx = %d, n_keep = %d, n_left = "
                        "%d, n_prompt_tokens = %d\n",
                        slot.n_ctx, slot.params.n_keep, n_left,
                        slot.n_prompt_tokens);

                GGML_ASSERT(slot.n_prompt_tokens < slot.n_ctx);
              }

              if (slot.params.cache_prompt) {
                // reuse any previously computed tokens that are common with the
                // new prompt
                slot.n_past = slot.cache_tokens.get_common_prefix(prompt_tokens);

                // reuse chunks from the cached prompt by shifting their KV
                // cache in the new position
                // 此处未调用
                if (params_base.n_cache_reuse > 0) {
                  size_t head_c = slot.n_past;  // cache
                  size_t head_p = slot.n_past;  // current prompt

                  if (mctx) {
                    // we should never reach this
                    GGML_ABORT("not supported by multimodal");
                  }

                  SLT_DBG(slot,
                          "trying to reuse chunks with size > %d, slot.n_past "
                          "= %d\n",
                          params_base.n_cache_reuse, slot.n_past);

                  while (head_c < slot.cache_tokens.size() &&
                         head_p < prompt_tokens.size()) {
                    size_t n_match = 0;
                    while (head_c + n_match < slot.cache_tokens.size() &&
                           head_p + n_match < prompt_tokens.size() &&
                           slot.cache_tokens[head_c + n_match] ==
                               prompt_tokens[head_p + n_match]) {
                      n_match++;
                    }

                    if (n_match >= (size_t)params_base.n_cache_reuse) {
                      SLT_DBG(slot,
                              "reusing chunk with size %zu, shifting KV cache "
                              "[%zu, %zu) -> [%zu, %zu)\n",
                              n_match, head_c, head_c + n_match, head_p,
                              head_p + n_match);
                      // for (size_t i = head_p; i < head_p + n_match; i++) {
                      //     SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i,
                      //     prompt_tokens[i], common_token_to_piece(ctx,
                      //     prompt_tokens[i]).c_str());
                      // }

                      const int64_t kv_shift =
                          (int64_t)head_p - (int64_t)head_c;

                      llama_memory_seq_rm (llama_get_memory(ctx), slot.id, head_p, head_c);
                      llama_memory_seq_add(llama_get_memory(ctx), slot.id, head_c, head_c + n_match, kv_shift);

                      for (size_t i = 0; i < n_match; i++) {
                        slot.cache_tokens.set_token(head_p + i, slot.cache_tokens[head_c + i]);
                        slot.n_past++;
                      }

                      head_c += n_match;
                      head_p += n_match;
                    } else {
                      head_c += 1;
                    }
                  }

                  SLT_DBG(slot, "after context reuse, new slot.n_past = %d\n",
                          slot.n_past);
                }
              } else {
                // if we don't cache the prompt, we have to remove the entire KV cache
                slot.n_past = 0;
              }

              if (slot.n_past > 0 && slot.n_past < (int) slot.cache_tokens.size()) {
                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                if (pos_min == -1) {
                  SLT_ERR(slot, "n_past = %d, cache_tokens.size() = %d, seq_id = %d, pos_min = %d\n", slot.n_past, (int) slot.cache_tokens.size(), slot.id, pos_min);
                  GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                }

                const auto n_swa = llama_model_n_swa(model);
                if (pos_min > std::max(0, slot.n_past - n_swa)) {
                  SLT_WRN(slot, "n_past = %d, cache_tokens.size() = %d, seq_id = %d, pos_min = %d, n_swa = %d\n", slot.n_past, (int) slot.cache_tokens.size(), slot.id, pos_min, n_swa);
                  SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA, see %s)\n",
                          "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                  slot.n_past = 0;
                }
              }
            }

            if (slot.n_past == slot.n_prompt_tokens && slot.n_past > 0) {
              //szj 此处未调用
              SLT_WRN(slot, "need to evaluate at least 1 token for each active slot, n_past = %d, n_prompt_tokens = %d\n", slot.n_past, slot.n_prompt_tokens);

              slot.n_past--;
            }

            slot.n_prompt_tokens_processed = 0;
          }

          if (!slot.can_split()) {
            // cannot fit the prompt in the current batch - will try next iter
            if (batch.n_tokens + slot.n_prompt_tokens > n_batch) {
              continue;
            }
          }

          // keep only the common part
          if (!llama_memory_seq_rm(llama_get_memory(ctx), slot.id, slot.n_past, -1)) {
            // could not partially delete (likely using a non-Transformer model)
            llama_memory_seq_rm(llama_get_memory(ctx), slot.id, -1, -1);

            // there is no common part left
            slot.n_past = 0;
          }

          SLT_DBG(slot, "kv cache rm [%d, end)\n", slot.n_past);

          // remove the non-common part from the cache
          slot.cache_tokens.keep_first(slot.n_past);

          // check if we should process the image
          if (slot.n_past < slot.n_prompt_tokens && slot.prompt_tokens[slot.n_past] == LLAMA_TOKEN_NULL) {
            // process the image
            int32_t new_n_past;
            int32_t res = slot.prompt_tokens.process_chunk(ctx, mctx, slot.n_past, slot.id, new_n_past);
            int32_t n_pos = new_n_past - slot.n_past;

            if (res != 0) {
              SLT_ERR(slot, "failed to process image, res = %d\n", res);
              slot.release();
              send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
              continue;
            }

            // add the image chunk to cache
            {
              const auto& chunk = slot.prompt_tokens.find_chunk(slot.n_past);
              slot.cache_tokens.push_back(chunk.get()); // copy
            }

            slot.n_past += n_pos;
            slot.n_prompt_tokens_processed += n_pos;
          }


          // add prompt tokens for processing in the current batch
          while (slot.n_past < slot.n_prompt_tokens && batch.n_tokens < n_batch) {
            SRV_DBG("++add token loopindex:%d\r\n",slot.n_past);
            // get next token to process
            llama_token cur_tok = slot.prompt_tokens[slot.n_past];
            if (cur_tok == LLAMA_TOKEN_NULL) {
              break; // end of text chunk
            }

            // embedding requires all tokens in the batch to be output
            const bool need_embd = server_task_type_need_embd(slot.task_type);

            common_batch_add(batch, cur_tok, slot.n_past, { slot.id }, need_embd);
            slot.cache_tokens.push_back(cur_tok);

            slot.n_prompt_tokens_processed++;
            slot.n_past++;
          }

          SLT_DBG(slot,
                  "prompt processing progress, n_past = %d, n_tokens = %d, "
                  "progress = %f\n",
                  slot.n_past, batch.n_tokens,
                  (float)slot.n_prompt_tokens_processed / slot.n_prompt_tokens);

          // entire prompt has been processed
          if (slot.n_past == slot.n_prompt_tokens) {
            slot.state = SLOT_STATE_DONE_PROMPT;

            GGML_ASSERT(batch.n_tokens > 0);
            GGML_ASSERT((size_t) slot.n_prompt_tokens == slot.prompt_tokens.size());

            common_sampler_reset(slot.smpl);

            // Process all prompt tokens through sampler system
            for (int i = 0; i < slot.n_prompt_tokens; ++i) {
              SLT_DBG(slot, "call common_sampler_accept1 loopcount=%d | slot.n_prompt_tokens=%d\r\n",i,slot.n_prompt_tokens);
              llama_token id = slot.prompt_tokens[i];
              if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(slot.smpl, id, false);
              }
            }

            // extract the logits only for the last token
            batch.logits[batch.n_tokens - 1] = true;

            slot.n_decoded = 0;
            slot.i_batch = batch.n_tokens - 1;

            SLT_DBG(slot, "prompt done, n_past = %d, n_tokens = %d slot.i_batch=%d\n",
                    slot.n_past, batch.n_tokens,slot.i_batch);
          }
        } //end slot.state==STARTED||PROMPT

        if (batch.n_tokens >= n_batch) {
          break;
        }
      } //for all slots
    } //end next-batch any pending prompts without exceeding n_batch

    if (batch.n_tokens == 0) {
      SRV_WRN("%s", "no tokens to decode\n");
      return;
    }

    SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

    if (slot_batched) {
      // apply lora, only need to do it once per batch
      common_set_adapter_lora(ctx, slot_batched->lora);

      llama_set_embeddings(ctx, slot_batched->need_embd());
    }

    int32_t i_next = 0;

    // process the created batch of tokens
    for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
      SRV_DBG("batch.n_tokens = %d\n", batch.n_tokens);
      SRV_DBG("decode_pos = %d\n", decode_pos);
      const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);
      SRV_DBG("n_tokens = %d\n", n_tokens);

      llama_batch batch_view = {
          n_tokens, batch.token + i, nullptr, batch.pos + i, batch.n_seq_id + i, batch.seq_id + i, batch.logits + i,
      };

      int ret = 0;

      SRV_DBG("%s\r\n","++call hm llama_decode");
      ret = llama_decode(ctx, batch_view);
      metrics.on_decoded(slots);

      if (ret != 0) {
        {
          std::string err;

          if (n_batch == 1 && ret == 1) {
            err = "Context size has been exceeded.";
          }

          if (ret == -1) {
            err = "Invalid input batch.";
          }

          if (ret < -1) {
            // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
            err = "Compute error.";
          }

          // TODO: handle ret == 2 (abort) when we start aborting

          if (!err.empty()) {
            SRV_ERR("%s, i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);
            for (auto & slot : slots) {
              slot.release();
              send_error(slot, err);
            }
            break;
          }
        }

        // retry with half the batch size to try to find a free slot in the KV
        // cache
        n_batch /= 2;

        SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

        continue;  // continue loop of n_batch
      }

      // move the head of the batch forward with the number of tokens we just processed
      i_next = i + n_tokens;

      // on successful decode, restore the original batch size
      n_batch = llama_n_batch(ctx);

      for (auto &slot : slots) {
        if (slot.i_batch < (int)i || slot.i_batch >= (int)(i + n_tokens)) {
          SRV_DBG("++enter continue here 55 i=%d slot.id=%d slot.i_batch:%d\r\n",i,slot.id,slot.i_batch);
          continue;  // continue loop of slots
        } else {
          SRV_DBG("--enter continue here 44 i=%d slot.id=%d slot.i_batch:%d\r\n",i,slot.id,slot.i_batch);
        }

        if (slot.state == SLOT_STATE_DONE_PROMPT) {
          if (slot.task_type == SERVER_TASK_TYPE_EMBEDDING) {
            // prompt evaluated for embedding
            // send_embedding(slot, batch_view);
            slot.release();
            slot.i_batch = -1;
            SRV_DBG("%s", "++enter continue here 66\r\n");
            continue;  // continue loop of slots
          }

          if (slot.task_type == SERVER_TASK_TYPE_RERANK) {
            // send_rerank(slot, batch_view);
            slot.release();
            slot.i_batch = -1;
            SRV_DBG("%s", "++enter continue here 77\r\n");
            continue;  // continue loop of slots
          }

          // prompt evaluated for next-token prediction
          slot.state = SLOT_STATE_GENERATING;
        } else if (slot.state != SLOT_STATE_GENERATING) {
          SRV_DBG("%s", "++enter continue here 88\r\n");
          continue;  // continue loop of slots
        }

        const int tok_idx = slot.i_batch - i;
        llama_token id = common_sampler_sample(slot.smpl, ctx, tok_idx);
        if(id!=tok_idx) {
           auto houmo_response = common_token_to_piece(ctx, id);
           auto argmax_response= common_token_to_piece(ctx, tok_idx);
           if(0)//(!containsChinese(houmo_response))
           {
            SRV_DBG("##HMLLama sample changeback: gen idx from %d backto %d , related string from %s backto %s\r\n",
                 id,tok_idx,houmo_response.c_str(),argmax_response.c_str());
            id=tok_idx;
           } else {
           SRV_DBG("##HMLLama sample change gen idx from %d to %d , related string from %s to %s\r\n",
                 tok_idx,id,argmax_response.c_str(),houmo_response.c_str());
           }
        }
        SRV_DBG("++call common_sampler_accept 99 slots size is:%d  slot.i_batch=%d i=%d || tok_idx=%d  token_id=%d  slot.id=%d\r\n",slots.size(),slot.i_batch,i,tok_idx, id,slot.id);
        slot.i_batch = -1;

        common_sampler_accept(slot.smpl, id, true);

        slot.n_decoded += 1;

        const int64_t t_current = ggml_time_us();

        if (slot.n_decoded == 1) {
          SRV_DBG("%s \r\n","call metrics.on_prompt_eval");//几次提问，几次打印
          slot.t_start_generation = t_current;
          slot.t_prompt_processing =
              (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
          metrics.on_prompt_eval(slot);
        }

        slot.t_token_generation = (t_current - slot.t_start_generation) / 1e3;

        completion_token_output result;
        result.tok = id;
        result.text_to_send = common_token_to_piece(
            ctx, result.tok, accept_special_token(slot, result.tok));
        result.prob = 1.0f;  // TODO: set it here instead of doing inside
                             // populate_token_probs

        if (slot.params.sampling.n_probs > 0) {
          populate_token_probs(slot, result, slot.params.post_sampling_probs,
                               params_base.special, tok_idx);
        }

        if (!process_token(result, slot)) {
          // release slot because of stop condition
          slot.release();
          slot.print_timings();
          send_final_response(slot);
          metrics.on_prediction(slot);
          SRV_DBG("%s\r\n","Release slot called and continue"); //几次提问，几次打印
          continue;
        }
      }

      // do speculative decoding-szj 此处未进入
      // 此处的for循环 第一个if处就一直continue
      for (auto &slot : slots) {
        if (!slot.is_processing() || !slot.can_speculate()) {
          // 此处continue
          SRV_DBG("%s", "++ continue here1\r\n");
          continue;
        }

        if (slot.state != SLOT_STATE_GENERATING) {
          SRV_DBG("%s", "++ continue here2\r\n");
          continue;
        }

        if (mctx) {
          // we should never reach this, as speculative is automatically disabled if mmproj is loaded
          GGML_ABORT("not supported by multimodal");
        }

        // determine the max draft that fits the current slot state
        int n_draft_max = slot.params.speculative.n_max;

        // note: n_past is not yet increased for the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context
        //       shifts
        n_draft_max = std::min(n_draft_max, slot.n_ctx - slot.n_past - 2);

        if (slot.n_remaining > 0) {
          n_draft_max = std::min(n_draft_max, slot.n_remaining - 1);
        }

        SLT_DBG(slot, "max possible draft: %d\n", n_draft_max);

        if (n_draft_max < slot.params.speculative.n_min) {
          SLT_DBG(slot,
                  "the max possible draft is too small: %d < %d - skipping "
                  "speculative decoding\n",
                  n_draft_max, slot.params.speculative.n_min);

          continue;
        }

        llama_token id = slot.sampled;

        struct common_speculative_params params_spec;
        params_spec.n_draft = n_draft_max;
        params_spec.n_reuse = llama_n_ctx(slot.ctx_dft) - slot.params.speculative.n_max;
        params_spec.p_min = slot.params.speculative.p_min;

        const llama_tokens & cached_text_tokens = slot.cache_tokens.get_text_tokens();
        llama_tokens draft = common_speculative_gen_draft(slot.spec, params_spec, cached_text_tokens, id);

        // ignore small drafts
        if (slot.params.speculative.n_min > (int)draft.size()) {
          SLT_DBG(slot, "ignoring small draft: %d < %d\n", (int)draft.size(),
                  slot.params.speculative.n_min);

          continue;
        }

        // keep track of total number of tokens generated in the draft
        slot.n_draft_total += draft.size();

        // construct the speculation batch
        common_batch_clear(slot.batch_spec);
        common_batch_add(slot.batch_spec, id, slot.n_past, {slot.id}, true);

        for (size_t i = 0; i < draft.size(); ++i) {
          common_batch_add(slot.batch_spec, draft[i], slot.n_past + 1 + i,
                           {slot.id}, true);
        }

        SLT_DBG(slot, "decoding speculative batch, size = %d\n",
                slot.batch_spec.n_tokens);

        llama_decode(ctx, slot.batch_spec);

        // the accepted tokens from the speculation
        SLT_INF(slot, "%s", "call common_sampler_sample_and_accept_n\r\n");
        const auto ids = common_sampler_sample_and_accept_n(slot.smpl, ctx, draft);

        slot.n_past += ids.size();
        slot.n_decoded += ids.size();

        // update how many tokens out of draft was accepted
        slot.n_draft_accepted += ids.size() - 1;

        slot.cache_tokens.push_back(id);
        slot.cache_tokens.insert({ids.begin(), ids.end() - 1});

        llama_memory_seq_rm(llama_get_memory(ctx), slot.id, slot.n_past, -1);

        for (size_t i = 0; i < ids.size(); ++i) {
          completion_token_output result;

          result.tok = ids[i];
          result.text_to_send = common_token_to_piece(
              ctx, result.tok, accept_special_token(slot, result.tok));
          result.prob = 1.0f;  // set later

          // TODO: set result.probs

          if (!process_token(result, slot)) {
            // release slot because of stop condition
            slot.release();
            slot.print_timings();
            send_final_response(slot);
            metrics.on_prediction(slot);
            SRV_DBG("%s\r\n","Release slot called and break");
            break;
          }
        }

        SLT_DBG(slot, "accepted %d/%d draft tokens, new n_past = %d\n",
                (int)ids.size() - 1, (int)draft.size(), slot.n_past);
      }
    } //End  process the created batch of tokens

    SRV_DBG("%s", "run slots completed\n");
  }

  json model_meta() const {
    return json{
        {"vocab_type", llama_vocab_type(vocab)},
        {"n_vocab", llama_vocab_n_tokens(vocab)},
        {"n_ctx_train", llama_model_n_ctx_train(model)},
        {"n_embd", llama_model_n_embd(model)},
        {"n_params", llama_model_n_params(model)},
        {"size", llama_model_size(model)},
    };
  }
};

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
    static server_context* ctx_server = nullptr;
    if (!ctx_server) {
        ctx_server = new server_context();

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
    }

    return ctx_server;
}


static std::shared_ptr<spdlog::logger> default_logger_ = nullptr;

int completion_load_model(HANDLE h, std::string model_path) {
    if (h == nullptr) {
        LOG_ERR("HANDLE is: %p\n", h);
        return -1;
    }
    auto* ctx_server = static_cast<server_context*>(h);

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

  std::time_t t = std::time(nullptr);
#ifdef _WIN32
  const char* userProfile = getenv("APPDATA");
  if (userProfile != nullptr) {
      std::cout << "userdata dir: " << userProfile << std::endl;
  } else {
      userProfile = "D:\\";
      std::cout << "get userdata dir failed,write do D:\\,please makesure D exists" << std::endl;
  }
  std::string log_folder(userProfile);
  std::string file_name = string_format("\\houmo-llm-%d.log", t);
  std::string log_file = log_folder + file_name;
#else
  std::string log_file = string_format("houmo-llm-%d.log", t);
#endif
  std::cout << "final logfile path:" << log_file << std::endl;

  const auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  const auto file_sink = std::make_shared<spdlog::sinks::daily_file_sink_mt>(log_file, 0, 0, false, 7);
  std::vector<spdlog::sink_ptr> sinks{file_sink, console_sink};
  default_logger_ = std::make_shared<spdlog::logger>("HLIECpp", sinks.begin(), sinks.end());
  default_logger_->set_level(spdlog::level::debug);
  spdlog::set_default_logger(default_logger_);

  int ctx_size = 4096;
  if (model_path.find("8k") != std::string::npos || model_path.find("8K") != std::string::npos) {
      ctx_size = 8192;
  }

  auto* p_params = new common_params();
  p_params->model.path = tok_model_path;
  p_params->n_parallel = 1;
  p_params->n_ctx = ctx_size;
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
  if (!ctx_server->load_model(*ctx_server->params, model_path)) {
      LOG_ERR("%s: exiting due to model loading error\n", __func__);
      return -1;
  }

  ctx_server->init_hm_llama(model_path);
  ctx_server->state.store(SERVER_STATE_READY);

  LOG_DBG("state 1: %d\n", ctx_server->state.load());
  LOG_DBG("%s: model loaded\n", __func__);

  ctx_server->queue_tasks.on_new_task(std::bind(&server_context::process_single_task, &*ctx_server, std::placeholders::_1));
  LOG_DBG("%s", "++after on new task\r\n");

  ctx_server->queue_tasks.on_update_slots(std::bind(&server_context::update_slots, &*ctx_server));

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

    auto* ctx_server = static_cast<server_context*>(h);
    if (ctx_server->magic != 12345) {
        LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
        return -1;
    }

    const auto tok_model_path = constructMemoryString(reinterpret_cast<void**>(buffers), bufferSize, buffers_count);
    std::cout << "++tok_model_path from mem:" << tok_model_path << std::endl;

    int ctx_size = 4096;
    const std::string model_path = ctx_server->model_path;
    if (model_path.find("8k") != std::string::npos || model_path.find("8K") != std::string::npos) {
        ctx_size = 8192;
    }

    auto* p_params = new common_params();
    p_params->model.path = tok_model_path;
    p_params->n_parallel = 1;
    p_params->n_ctx = ctx_size;
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

    if (!ctx_server->load_model(*ctx_server->params, model_path)) {
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return -1;
    }

    LOG_INF("%s", "++before call init_by_buffer\r\n");
    ctx_server->init_by_buffer(h, buffers, bufferSize, buffers_count, hwinfo);
    LOG_INF("%s", "++after call init_by_buffer\r\n");

    ctx_server->state.store(SERVER_STATE_READY);

    LOG_DBG("state 1: %d\n", ctx_server->state.load());

    LOG_DBG("%s: model loaded\n", __func__);

    ctx_server->queue_tasks.on_new_task(std::bind(&server_context::process_single_task, &*ctx_server, std::placeholders::_1));

    ctx_server->queue_tasks.on_update_slots(std::bind(&server_context::update_slots, &*ctx_server));

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

    auto* ctx_server = static_cast<server_context*>(h);
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
  auto *ctx_server = static_cast<server_context*>(h);
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

    auto* ctx_server = static_cast<server_context*>(h);
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

    auto* ctx_server = static_cast<server_context*>(h);
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
    auto* ctx_server = static_cast<server_context*>(h);

    const int ret = ctx_server->completion_stream_generate_stop();
    return ret;
}

int completion_destroy_instance(HANDLE h) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;

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

int completion_unload_model(HANDLE h) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;

  ctx_server->queue_tasks.terminate();
  if (ctx_server->ctx_t.joinable()) {
    ctx_server->ctx_t.join();
  }

  SRV_INF("join ctx_t finished: %s\n", "");

  return ctx_server->unload_model();
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
    // TODO only for Lenovo demo
    auto* ctx_server = static_cast<server_context*>(handle);
    std::unique_lock<std::mutex> lock(ctx_server->mutex_context);
    ctx_server->model_status = 2;
    model_real_load_flag = true;
    return true;


    int ret = completion_unload_model(handle);
    LOG_DBG("completion_unload_model: %s\n", ret);
    model_real_load_flag = false;
    if (ret == 0) {
        return true;
    }

    return false;
}

// model status see the enum status
// return enum status
int32 HTTP_SERVER_API api_status(HANDLE handle) {
  LOG_DBG("Call api_status atTime:%s\n",getCurrentTime().c_str());
  int32 status = 0;
  if (handle == nullptr) {
    LOG_ERR("HANDLE is: %p\n", handle);
    return -1;
  }
  auto *ctx_server = (server_context *)handle;
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return -1;
  }
  status = ctx_server->model_status;
  LOG_DBG("Return from api_status atTime:%s\n",getCurrentTime().c_str());
  return status;
}

// Generate
// if result == nullptr or *result_size is not enough, return E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HTTP_SERVER_API api_Generate(HANDLE handle, const char* prompt, const void* gen_params, size_t gen_params_size, char** result,
                                   size_t* result_size) {
    SRV_DBG("prompt: %s\n", prompt);
    auto p = (params*)gen_params;
    if (handle == nullptr) {
        LOG_ERR("HANDLE is: %p\n", handle);
        return -1;
    }

    std::string req_body = "{\"messages\":[{\"content\":\"%s”\",\"role\":\"user\"}],\"stream\":"
                           "false,\"model\": \"%s\", \"max_tokens\": %d}";
    req_body = string_format(req_body.c_str(), prompt, "fixed_hm_llama_qwen3_8b_szj", p->n_predict);
    json req = json::parse(req_body);
    std::string decode_result;
    int ret = completion_generate(handle, req, decode_result);
    const auto result_ptr = new char[decode_result.size() + 1]; // +1 for terminating NUL

    strncpy(result_ptr, decode_result.c_str(), decode_result.size() + 1);

    *result = result_ptr;
    *result_size = decode_result.size();
    SRV_DBG("decode_result: %s\n", decode_result.c_str());
    SRV_DBG("result: %s\n", *result);

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

    auto* ctx_server = static_cast<server_context*>(handle);
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
bool HTTP_SERVER_API api_warmup(HANDLE handle) { return true; }

// reset model not must
void HTTP_SERVER_API api_Reset(HANDLE handle) { }


//hi.h api for normal customers

#ifdef _MSC_VER
#define HMINFER_API __declspec(dllexport)
#else
#define HMINFER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// if fail, return nullptr
//HANDLE HMINFER_API create_infer_instance(params& params);
HANDLE HMINFER_API create_infer_instance(void* params, size_t params_bytes){
  return create_api_instance(params, params_bytes);
}

void HMINFER_API destroy_infer_instance(HANDLE handle){
  return destroy_api_instance(handle);
}

//load model with model name ,layer:-1 load all layer.
bool HMINFER_API infer_loadmodel_by_path(HANDLE handle, const char* model_path, const char** model_names, size_t model_name_count){
  hardware_info hwinfo;
  return api_loadmodel_by_path(handle, model_path, model_names,model_name_count,hwinfo);
}

//load model with buffer stream,layer:-1 load all layer.
bool HMINFER_API infer_loadmodel_by_buffer(HANDLE handle, uint8_t** buffers, size_t* bufferSize, size_t buffers_count){
  hardware_info hwinfo;
  return api_loadmodel_by_buffer(handle, buffers, bufferSize, buffers_count,hwinfo);
}

//load token model not must
bool HMINFER_API infer_loadtoken(HANDLE handle, const char* token_fullpath){
  return api_loadtoken(handle, token_fullpath);
}

//unload model
bool HMINFER_API infer_unloadmodel(HANDLE handle){
  return api_unloadmodel(handle);
}

// model status see the enum status
//return enum status
int32 HMINFER_API infer_status(HANDLE handle){
  return api_status(handle);
}

//Generate
//if result == nullptr or *result_size is not enough, return E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HMINFER_API infer_Generate(HANDLE handle, const char* prompt, const void* params, size_t params_bytes, char** result, size_t* result_size){
  return api_Generate(handle, prompt, params, params_bytes, result, result_size);
}

//get prompt token number
int32 HMINFER_API infer_token_num(HANDLE handle, const char* prompt){
  return api_token_num(handle, prompt);
}

//stream generate
bool HMINFER_API infer_Generate_stream(HANDLE handle, const char* prompt, const void* params, size_t params_size){
  return api_Generate_stream(handle, prompt, params, params_size);
}

//the generate token ,use for stream mode
//if result == nullptr or *result_size is not enough, return E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HMINFER_API infer_Get_Generate_token(HANDLE handle, char** result, size_t* result_size){
  return api_Get_Generate_token(handle, result, result_size);
}

//generate status see enum generater_status defined ahead
//return enum generater_status
int32 HMINFER_API infer_Generate_status(HANDLE handle){
  return api_Generate_status(handle);
}

//stop generate
bool HMINFER_API infer_stop(HANDLE handle){
  return api_stop(handle);
}

//warmup not must
bool HMINFER_API infer_warmup(HANDLE handle){
  return api_warmup(handle);
}

//reset model not must
void HMINFER_API infer_Reset(HANDLE handle){
  return api_Reset(handle);
}

#ifdef __cplusplus
}
#endif
#include "arg.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "utils.hpp"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
// mime type for sending response
#define MIMETYPE_JSON "application/json; charset=utf-8"

// auto generated files (update with ./deps.sh)
#include <signal.h>
#include <torch/torch.h>

#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "http_server_c_interface_3.h"
#include "INIReader.h"
#include "tcim/tcim_runtime.h"

#if (__GNUC__ < 8 && !defined(_MSC_VER))
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

using json = nlohmann::ordered_json;

#include <algorithm>
#include <cctype>
#include <locale>

#include "houmo-llm.h"
#include "houmo-llm.hpp"

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

static void int16toint16(int16_t *src, int16_t *dst, size_t size) {
  for (int i = 0; i < size; i++) {
    src[i] = dst[i];
  }
}

//
// Function: write_utf8_cstr_to_stdout(const char *) -> <writes to stdout>
//
// writes a string to standard output; taking into account that on Windows
// to display correctly you have to use special handling. Works even if the
// user has not set a unicode code page on a Windows cmd.exe.
//
// In case of invalid UTF-8, invalid_utf8 is set to true on Windows, and
// something a human-readable is written instead.
//
// On non-Windows systems, simply printfs() the string.
static void write_utf8_cstr_to_stdout(const char *str, bool &invalid_utf8) {
  invalid_utf8 = false;

#if defined(_WIN32)
  // Are we in a console?
  HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD dwMode = 0;

  // According to Microsoft docs:
  // "WriteConsole fails if it is used with a standard handle that is redirected
  // to a file." Also according to the docs, you can use GetConsoleMode to check
  // for that.
  if (hConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(hConsole, &dwMode)) {
    printf("%s", str);
    return;
  }

  // MultiByteToWideChar reports an error if str is empty, don't report
  // them as invalid_utf8.
  if (*str == 0) {
    return;
  }
  int length_needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str,
                                          strlen(str), NULL, 0);
  if (length_needed == 0) {
    DWORD err = GetLastError();
    if (err == ERROR_NO_UNICODE_TRANSLATION) {
      invalid_utf8 = true;
      int len = strlen(str);
      printf("<");
      for (int i = 0; i < len; ++i) {
        if (i > 0) {
          printf(" ");
        }
        printf("%02x", (uint8_t)str[i]);
      }
      printf(">");
      return;
    }
    GGML_ABORT("MultiByteToWideChar() failed in an unexpected way.");
  }

  LPWSTR wstr = (LPWSTR)calloc(length_needed + 1, sizeof(*wstr));
  GGML_ASSERT(wstr);

  MultiByteToWideChar(CP_UTF8, 0, str, strlen(str), wstr, length_needed);
  WriteConsoleW(hConsole, wstr, length_needed, NULL, NULL);

  free(wstr);
#else
  // TODO: reporting invalid_utf8 would be useful on non-Windows too.
  // printf will silently just write bad unicode.
  printf("%s", str);
#endif
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

//===========add sampling algorithm

struct llama_sampling_params {
  float temp = 1.0f;             // temperature
  int32_t top_k = 40;            // top-k采样
  float top_p = 0.95f;           // top-p采样
  float min_p = 0.05f;           // 最小概率阈值
  float typical_p = 1.0f;        // typical采样
  int32_t penalty_last_n = 64;   // 重复惩罚考虑的token数量
  float penalty_repeat = 1.1f;   // 重复惩罚系数
  float penalty_freq = 0.0f;     // 频率惩罚系数
  float penalty_present = 0.0f;  // 存在惩罚系数
                                 // ... 其他参数 ...
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
      grammar_triggers.push_back(trigger.to_json<json>());
    }

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


class LogitsProcessor {
public:
    virtual ~LogitsProcessor() = default;

    virtual void process(float* logits, int n_vocab, const std::vector<llama_token>& last_tokens) = 0;
    virtual void print_processor_param() = 0;

    virtual void update(const slot_params& params) = 0;
};

class TemperatureProcessor final : public LogitsProcessor {
    float temp;

public:
    explicit TemperatureProcessor(float temp) : temp(temp) {}

    void print_processor_param() override {
        SRV_INF("TemperatureProcessor temp:%f \r\n", temp);
    }

    void process(float* logits, int n_vocab, const std::vector<llama_token>&) override {
        //TODO:szj only for test must be deleted
        //temp=1.9;
        if (temp <= 0.0f || areFloatsEqual(0, temp) || areFloatsEqual(1, temp)) {
            SRV_DBG("%s || temp is:%f\r\n", "temperature M<=0 or equal 1, just return", temp);
            return;
            // 温度<=0时使用贪心搜索
            // float max_logit = *std::max_element(logits, logits + n_vocab);
            // std::fill(logits, logits + n_vocab, -INFINITY);
            // *std::max_element(logits, logits + n_vocab) = 0.0f;
            // int max_pos=std::distance(logits, max_it)
        }
        else {
            // 应用温度缩放
            for (int i = 0; i < n_vocab; ++i) {
                logits[i] /= temp;
            }
        }
        SRV_DBG("++%s \r\n", "call temperature");
    }

    void update(const slot_params& params) override {
        temp = params.sampling.temp;
    }
};

class TopKProcessor final : public LogitsProcessor {
    int k;

public:
    explicit TopKProcessor(int k) : k(k) {}

    void print_processor_param() override {
        SRV_INF("TopKProcessor topk:%d \r\n", k);
    }

    void process(float* logits, int n_vocab, const std::vector<llama_token>&) override {
        if (k <= 0) {
            SRV_ERR("set topk value invalid:%d \r\n", k);
            return;
        }

        std::vector<std::pair<float, int>> logits_id;
        logits_id.reserve(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            logits_id.emplace_back(logits[i], i);
        }

        // 找出top-k的logits
        std::partial_sort(logits_id.begin(), logits_id.begin() + std::min(k, n_vocab), logits_id.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });

        // 将非top-k的logits设为负无穷
        for (int i = k; i < n_vocab; ++i) {
            logits[logits_id[i].second] = -INFINITY;
        }
        SRV_DBG("++%s \r\n", "call topK");
    }

    void update(const slot_params& params) override {
        k = params.sampling.top_k;
    }
};

class TopPProcessor final : public LogitsProcessor {
    float p;

public:
    explicit TopPProcessor(float p) : p(p) {}

    void print_processor_param() override {
        SRV_INF("TopPProcessor topp:%f \r\n", p);
    }

    void process(float* logits, int n_vocab, const std::vector<llama_token>&) override {
        if (p >= 1.0f || areFloatsEqual(p, 1.0))
            return;
        if (p < 0) {
            SRV_ERR("Error:top_p value not right,please check input p:%f\r\n", p);
            return;
        }

        std::vector<std::pair<float, int>> logits_id;
        logits_id.reserve(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            logits_id.emplace_back(logits[i], i);
        }

        // 按概率降序排序
        std::sort(logits_id.begin(), logits_id.end(),
                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });

        // 计算softmax概率
        std::vector<float> probs(n_vocab);
        softmax(logits, probs.data(), n_vocab);

        // 累积概率直到超过p
        float cumsum = 0.0f;
        int last_idx = n_vocab - 1;
        for (int i = 0; i < n_vocab; ++i) {
            cumsum += probs[logits_id[i].second];
            if (cumsum >= p) {
                last_idx = i;
                break;
            }
        }

        // 将不在top-p中的logits设为负无穷
        for (int i = last_idx + 1; i < n_vocab; ++i) {
            logits[logits_id[i].second] = -INFINITY;
        }
        SRV_DBG("++%s \r\n", "call topp");
    }

    void update(const slot_params& params) override {
        p = params.sampling.top_p;
    }
};

class RepeatPenaltyProcessor final : public LogitsProcessor {
    int last_n;
    float penalty;

public:
    explicit RepeatPenaltyProcessor(int last_n, float penalty) : last_n(last_n), penalty(penalty) {}

    void print_processor_param() override {
        SRV_INF("RepeatPenaltyProcessor: last_n:%d penalty:%f\r\n", last_n, penalty);
    }

    void process(float* logits, int n_vocab, const std::vector<llama_token>& last_tokens) override {
        if (last_n <= 0 || penalty <= 1.0f || areFloatsEqual(0, penalty) || areFloatsEqual(1, penalty))
            return;

        // 统计最近last_n个token的出现次数
        std::unordered_map<llama_token, int> token_counts;
        for (auto token : last_tokens) {
            token_counts[token]++;
        }

        // 应用重复惩罚
        for (auto& pair : token_counts) {
            if (pair.first >= 0 && pair.first < n_vocab) {
                if (logits[pair.first] > 0) {
                    logits[pair.first] /= penalty;
                }
                else {
                    logits[pair.first] *= penalty;
                }
            }
        }
        SRV_DBG("++%s \r\n", "call repeat penalty");
    }

    void update(const slot_params& params) override {
        penalty = params.sampling.penalty_repeat;
        last_n = params.sampling.penalty_last_n;
    }
};

class FrequencyPenaltyProcessor final : public LogitsProcessor {
    float penalty;

public:
    explicit FrequencyPenaltyProcessor(float penalty) : penalty(penalty) {}

    void print_processor_param() override {
        SRV_INF("FrequencyPenaltyProcessor penalty:%f\r\n", penalty);
    }

    void process(float* logits, int n_vocab, const std::vector<llama_token>& last_tokens) override {
        if (penalty == 0.0f)
            return;

        // 统计所有token的出现频率
        std::unordered_map<llama_token, float> token_freqs;
        float sum = 0.0f;
        for (auto token : last_tokens) {
            token_freqs[token] += 1.0f;
            sum += 1.0f;
        }

        // 计算频率并应用惩罚
        for (auto& pair : token_freqs) {
            if (pair.first >= 0 && pair.first < n_vocab) {
                float freq = pair.second / sum;
                logits[pair.first] -= penalty * freq;
            }
        }
        SRV_DBG("++%s \r\n", "call freq penalty");
    }

    void update(const slot_params& params) override {
        penalty = params.sampling.penalty_repeat;
    }
};

class LogitsProcessorList {
public:
    void add(std::unique_ptr<LogitsProcessor> processor) {
        processors.push_back(std::move(processor));
    }

    void print_processors_param() const {
        for (const auto& processor : processors) {
            SRV_INF("%s", "++call print_processor_param\r\n");
            processor->print_processor_param();
        }
    }

    void process(float* logits, const int n_vocab, const std::vector<llama_token>& last_tokens, const slot_params& params) const {
        for (const auto& processor : processors) {
            SRV_DBG("--%s\r\n", "call process");
            processor->update(params);
            processor->process(logits, n_vocab, last_tokens);
        }
    }

private:
    std::vector<std::unique_ptr<LogitsProcessor>> processors;
};

struct server_task {
  int id = -1;     // to be filled by server_queue
  int index = -1;  // used when there are multiple prompts (batch request)

  server_task_type type;

  // used by SERVER_TASK_TYPE_CANCEL
  int id_target = -1;

  // used by SERVER_TASK_TYPE_INFERENCE
  slot_params params;
  llama_tokens prompt_tokens;
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

    // Sampling parameter defaults are loaded from the global server context
    // (but individual requests can still override them)
    slot_params defaults;
    defaults.sampling = params_base.sampling;
    defaults.speculative = params_base.speculative;

    // enabling this will output extra debug information in the HTTP responses
    // from the server
    params.verbose = params_base.verbosity > 9;
    params.timings_per_token = json_value(data, "timings_per_token", false);

    params.stream = json_value(data, "stream", false);
    params.cache_prompt = json_value(data, "cache_prompt", false);
    params.return_tokens = json_value(data, "return_tokens", true);
    params.n_predict = json_value(
        data, "n_predict", json_value(data, "max_tokens", defaults.n_predict));
    params.n_indent = json_value(data, "n_indent", defaults.n_indent);
    params.n_keep = json_value(data, "n_keep", defaults.n_keep);
    params.n_discard = json_value(data, "n_discard", defaults.n_discard);
    // params.t_max_prompt_ms  = json_value(data, "t_max_prompt_ms",
    // defaults.t_max_prompt_ms); // TODO: implement
    params.t_max_predict_ms =
        json_value(data, "t_max_predict_ms", defaults.t_max_predict_ms);
    params.response_fields =
        json_value(data, "response_fields", std::vector<std::string>());

    params.sampling.top_k = json_value(data, "top_k", defaults.sampling.top_k);
    params.sampling.top_p = json_value(data, "top_p", defaults.sampling.top_p);
    params.sampling.min_p = json_value(data, "min_p", defaults.sampling.min_p);
    params.sampling.xtc_probability =
        json_value(data, "xtc_probability", defaults.sampling.xtc_probability);
    params.sampling.xtc_threshold =
        json_value(data, "xtc_threshold", defaults.sampling.xtc_threshold);
    params.sampling.typ_p =
        json_value(data, "typical_p", defaults.sampling.typ_p);
    params.sampling.temp =
        json_value(data, "temperature", defaults.sampling.temp);
    params.sampling.dynatemp_range =
        json_value(data, "dynatemp_range", defaults.sampling.dynatemp_range);
    params.sampling.dynatemp_exponent = json_value(
        data, "dynatemp_exponent", defaults.sampling.dynatemp_exponent);
    params.sampling.penalty_last_n =
        json_value(data, "repeat_last_n", defaults.sampling.penalty_last_n);
    params.sampling.penalty_repeat =
        json_value(data, "repeat_penalty", defaults.sampling.penalty_repeat);
    params.sampling.penalty_freq =
        json_value(data, "frequency_penalty", defaults.sampling.penalty_freq);
    params.sampling.penalty_present =
        json_value(data, "presence_penalty", defaults.sampling.penalty_present);
    params.sampling.dry_multiplier =
        json_value(data, "dry_multiplier", defaults.sampling.dry_multiplier);
    params.sampling.dry_base =
        json_value(data, "dry_base", defaults.sampling.dry_base);
    params.sampling.dry_allowed_length = json_value(
        data, "dry_allowed_length", defaults.sampling.dry_allowed_length);
    params.sampling.dry_penalty_last_n = json_value(
        data, "dry_penalty_last_n", defaults.sampling.dry_penalty_last_n);
    params.sampling.mirostat =
        json_value(data, "mirostat", defaults.sampling.mirostat);
    params.sampling.mirostat_tau =
        json_value(data, "mirostat_tau", defaults.sampling.mirostat_tau);
    params.sampling.mirostat_eta =
        json_value(data, "mirostat_eta", defaults.sampling.mirostat_eta);
    params.sampling.seed = json_value(data, "seed", defaults.sampling.seed);
    params.sampling.n_probs =
        json_value(data, "n_probs", defaults.sampling.n_probs);
    params.sampling.min_keep =
        json_value(data, "min_keep", defaults.sampling.min_keep);
    params.post_sampling_probs =
        json_value(data, "post_sampling_probs", defaults.post_sampling_probs);

    params.speculative.n_min =
        json_value(data, "speculative.n_min", defaults.speculative.n_min);
    params.speculative.n_max =
        json_value(data, "speculative.n_max", defaults.speculative.n_max);
    params.speculative.p_min =
        json_value(data, "speculative.p_min", defaults.speculative.p_min);

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
        params.oaicompat_chat_format =
            static_cast<common_chat_format>(it->get<int>());
        SRV_DBG("Chat format: %s\n",
                common_chat_format_name(params.oaicompat_chat_format).c_str());
      } else {
        params.oaicompat_chat_format = defaults.oaicompat_chat_format;
      }
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
          auto ct = common_grammar_trigger::from_json(t);
          if (ct.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
            const auto &word = ct.value;
            auto ids = common_tokenize(vocab, word, /* add_special= */ false,
                                       /* parse_special= */ true);
            if (ids.size() == 1) {
              auto token = ids[0];
              if (std::find(params.sampling.preserved_tokens.begin(),
                            params.sampling.preserved_tokens.end(),
                            (llama_token)token) ==
                  params.sampling.preserved_tokens.end()) {
                throw std::runtime_error(
                    "Grammar trigger word should be marked as preserved "
                    "token: " +
                    word);
              }
              SRV_DBG("Grammar trigger token: %d (`%s`)\n", token,
                      word.c_str());
              common_grammar_trigger trigger;
              trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
              trigger.value = word;
              trigger.token = token;
              params.sampling.grammar_triggers.push_back(std::move(trigger));
            } else {
              SRV_DBG("Grammar trigger word: `%s`\n", word.c_str());
              params.sampling.grammar_triggers.push_back(
                  {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
            }
          } else {
            params.sampling.grammar_triggers.push_back(ct);
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

  json to_json() const {
    return {
        {"prompt_n", prompt_n},
        {"prompt_ms", prompt_ms},
        {"prompt_per_token_ms", prompt_per_token_ms},
        {"prompt_per_second", prompt_per_second},

        {"predicted_n", predicted_n},
        {"predicted_ms", predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second", predicted_per_second},
    };
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

  json to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
      SRV_DBG("Parsing chat message: %s\n", content.c_str());
      msg = common_chat_parse(content, oaicompat_chat_format);
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

    if (first) {
      if (content.empty()) {
        choices = json::array({json{{"finish_reason", nullptr},
                                    {"index", 0},
                                    {"delta", json{{"role", "assistant"}}}}});
      } else {
        // We have to send this as two updates to conform to openai behavior
        json initial_ret =
            json{{"choices",
                  json::array({json{{"finish_reason", nullptr},
                                    {"index", 0},
                                    {"delta", json{{"role", "assistant"}}}}})},
                 {"created", t},
                 {"id", oaicompat_cmpl_id},
                 {"model", oaicompat_model},
                 {"object", "chat.completion.chunk"}};

        json second_ret =
            json{{"choices",
                  json::array({json{{"finish_reason", nullptr},
                                    {"index", 0},
                                    {"delta", json{{"content", content}}}}})},
                 {"created", t},
                 {"id", oaicompat_cmpl_id},
                 {"model", oaicompat_model},
                 {"object", "chat.completion.chunk"}};

        return std::vector<json>({initial_ret, second_ret});
      }
    } else {
      choices = json::array({json{
          {"finish_reason", nullptr},
          {"index", 0},
          {"delta",
           json{
               {"content", content},
           }},
      }});
    }

    GGML_ASSERT(choices.size() >= 1);

    if (prob_output.probs.size() > 0) {
      choices[0]["logprobs"] = json{
          {"content", completion_token_output::probs_vector_to_json(
                          {prob_output}, post_sampling_probs)},
      };
    }

    json ret = json{{"choices", choices},
                    {"created", t},
                    {"id", oaicompat_cmpl_id},
                    {"model", oaicompat_model},
                    {"system_fingerprint", build_info},
                    {"object", "chat.completion.chunk"}};

    if (timings.prompt_n >= 0) {
      ret.push_back({"timings", timings.to_json()});
    }

    return std::vector<json>({ret});
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

  int32_t kv_cache_tokens_count;
  int32_t kv_cache_used_cells;

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

        {"kv_cache_tokens_count", kv_cache_tokens_count},
        {"kv_cache_used_cells", kv_cache_used_cells},

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
  llama_tokens prompt_tokens;

  size_t last_nl_pos = 0;

  std::string generated_text;
  llama_tokens generated_tokens;

  llama_tokens cache_tokens;

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

  // stats
  size_t n_sent_text = 0;  // number of sent text character

  int64_t t_start_process_prompt;
  int64_t t_start_generation;

  double t_prompt_processing;  // ms
  double t_token_generation;   // ms

  std::function<void(int)> callback_on_release;

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

    generated_tokens.clear();
    generated_token_probs.clear();
  }

  bool is_non_causal() const {
    return task_type == SERVER_TASK_TYPE_EMBEDDING ||
           task_type == SERVER_TASK_TYPE_RERANK;
  }

  bool can_batch_with(server_slot &other_slot) const {
    return is_non_causal() == other_slot.is_non_causal() &&
           are_lora_equal(lora, other_slot.lora);
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

    return timings;
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
        pos = find_partial_stop_string(word, text);
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
  }

  json to_json() const {
    return json{
        {"id", id},
        {"id_task", id_task},
        {"n_ctx", n_ctx},
        {"speculative", can_speculate()},
        {"is_processing", is_processing()},
        {"non_causal", is_non_causal()},
        {"params", params.to_json()},
        {"prompt", common_detokenize(ctx, prompt_tokens)},
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
  std::function<void(server_task)> callback_new_task;
  std::function<void(void)> callback_update_slots;

  // Add a new task to the end of the queue
  int post(server_task task, bool front = false) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    GGML_ASSERT(task.id != -1);
    // if this is cancel task make sure to clean up pending tasks
    if (task.type == SERVER_TASK_TYPE_CANCEL) {
      cleanup_pending_task(task.id_target);
    }
    QUE_DBG("new task, id = %d, front = %d\n", task.id, front);
    if (front) {
      queue_tasks.push_front(std::move(task));
    } else {
      queue_tasks.push_back(std::move(task));
    }
    condition_tasks.notify_one();
    return task.id;
  }

  // multi-task version of post()
  int post(std::vector<server_task> &tasks, bool front = false) {
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
  void defer(server_task task) {
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
  void on_new_task(std::function<void(server_task)> callback) {
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
          QUE_DBG("%s", "terminate\n");
          return;
        }
        if (queue_tasks.empty()) {
          lock.unlock();
          break;
        }
        server_task task = queue_tasks.front();
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
          QUE_DBG("%s", "terminate\n");
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
      condition_results.wait(lock, [&] { return !queue_results.empty(); });
      SRV_DBG("%s \r\n","++call recv multitasks and after wait");
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
};

struct houmo_context {
  std::string model_type = "deepseek";
  std::string model_name = "Unknown";

  int prefill_length = 256;
  int decode_length = 4096;
  int nblocks = 28;

  std::vector<std::string> dummy_tensor_names;

  tcim::Module::WeightManager wm;
  tcim::Stream stream;
  tcim::Module::Option option_prefill;
  tcim::Module::Option option_decode;
  tcim::Module prefill_model;
  tcim::Module decode_model;
};

struct houmo_context_14B {
  std::string model_type = "deepseek";
  std::string model_name = "Unknown";

  int prefill_length = 256;
  int decode_length = 8192;
  int nblocks = 48;

  std::vector<std::string> dummy_tensor_names;

  tcim::Module::WeightManager* wm0_;
  tcim::Module::WeightManager* wm1_;
  tcim::Stream stream;
  tcim::Stream stream2;
  tcim::Module prefill_model_part1_;
  tcim::Module prefill_model_part2_;
  tcim::Module decode_model_part1_;
  tcim::Module decode_model_part2_;
};

struct server_context {
  int magic = 12345;
  std::mutex mutex_context;
  int generate_status = 0;  // 0 -> finished; 1 -> generating; 2 -> stopped; 3
                            // -> exceed; -1 -> failed;
  int model_status =
      0;  // 0 -> init; 1 -> loaded; 2 -> unloaded; 3 -> infer; -1 -> error;
  bool stop = false;
  bool hm_running = false;

  common_params *params;
  std::thread ctx_t;
  std::thread srv_t;
  httplib::Server *svr;
  std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

  torch::nn::Embedding emb = nullptr;
  std::string model_path;
  houmo_context *hmctx;
  houmo_context_14B *hmctx_14B;
  model_scale_type hm_model_scale_type=MODEL_7B;  //根据加载的模型，修改成对应的宏
  LogitsProcessorList  hm_sampling_processors; //后采样实现

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

  // server_context() {}
  //
  // server_context(const server_context& p) = delete;
  //
  // server_context& operator=(const server_context& p) = delete;

  // server_context(const server_context& p) = delete;
  //
  // server_context(const server_context* p) = delete;
  //
  // server_context& operator=(const server_context& p) = delete;
  // server_context& operator=(const server_context* p) = delete;

  ~server_context() {
    SRV_DBG("delete server_context '%p'\n", this);
    // Clear any sampling context
    for (server_slot &slot : slots) {
      common_sampler_free(slot.smpl);
      slot.smpl = nullptr;

      llama_free(slot.ctx_dft);
      slot.ctx_dft = nullptr;

      common_speculative_free(slot.spec);
      slot.spec = nullptr;

      llama_batch_free(slot.batch_spec);
    }

    llama_batch_free(batch);
    // srv_t.detach();
    // ctx_t.detach();
    if (hmctx != nullptr) {
      delete hmctx;
      hmctx=nullptr;
    }

    if (hmctx_14B != nullptr) {
      delete hmctx_14B;
      hmctx_14B=nullptr;
    }

    QUE_DBG("delete hmctx finished '%p'\n", this);
    QUE_DBG("delete hmctx_14B finished '%p'\n", this);
    LOG_DBG("delete server_context finished '%p'\n", this);
  }

  int unload_model() {
    std::unique_lock<std::mutex> lock(mutex_context);

    LOG_INF("unload_model '%p'\n", hmctx);

    if (hmctx != nullptr) {
      delete hmctx;
      hmctx = nullptr;
    }

    if (hmctx_14B != nullptr) {
      delete hmctx_14B;
      hmctx_14B = nullptr;
    }

    LOG_INF("delete hmctx  finished '%p'\n", hmctx);
    LOG_INF("delete hmctx_14B  finished '%p'\n", hmctx_14B);
    model_status = 2;
    return 0;
  }

  int destroy_server_context() {
    SRV_INF("destroy_server_context '%p'\n", this);
    // Clear any sampling context
    for (server_slot &slot : slots) {
      common_sampler_free(slot.smpl);
      slot.smpl = nullptr;

      llama_free(slot.ctx_dft);
      slot.ctx_dft = nullptr;

      common_speculative_free(slot.spec);
      slot.spec = nullptr;

      llama_batch_free(slot.batch_spec);
    }

    llama_batch_free(batch);
    SRV_INF("destroy server_context over'%p'\n", this);
    SRV_INF("hmctx '%p'\n", hmctx);
    // hmctx->destroy();

    // int input_num = hmctx->prefill_part1_model.GetInputNum();
    // std::cout << "Count of Input: " << input_num << std::endl;
    // for (int idx = 0; idx < input_num; idx++) {
    //     auto input_name = hmctx->prefill_part1_model.GetInputName(idx);
    //     auto input_info =
    //     hmctx->prefill_part1_model.GetInputInfo(input_name).AsContiguous();
    //     std::cout << "Input[" << input_name << "] " << input_info <<
    //     std::endl;
    // }
    if (hmctx != nullptr) {
      delete hmctx;
      hmctx=nullptr;
    }
    if (hmctx_14B != nullptr) {
      delete hmctx_14B;
      hmctx_14B = nullptr;
    }
    // SRV_INF("hmctx '%p'\n", hmctx);
    // input_num = hmctx->prefill_part1_model.GetInputNum();
    // std::cout << "Count of Input: " << input_num << std::endl;
    // for (int idx = 0; idx < input_num; idx++) {
    //     auto input_name = hmctx->prefill_part1_model.GetInputName(idx);
    //     auto input_info =
    //     hmctx->prefill_part1_model.GetInputInfo(input_name).AsContiguous();
    //     std::cout << "Input[" << input_name << "] " << input_info <<
    //     std::endl;
    // }

    SRV_INF("destroy_server_context finished '%p'\n", this);
  }

  bool load_model(const common_params &params, std::string model_path) {
    SRV_INF("loading model '%s'\n", params.model.c_str());

    params_base = params;

    llama_init = common_init_from_params(params_base);

    model = llama_init.model.get();
    ctx = llama_init.context.get();

    if (model == nullptr) {
      SRV_ERR("failed to load model, '%s'\n", params_base.model.c_str());
      return false;
    }

    vocab = llama_model_get_vocab(model);

    n_ctx = llama_n_ctx(ctx);

    add_bos_token = llama_vocab_get_add_bos(vocab);
    has_eos_token = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;

    // 草稿模型-此处未调用-szj
    if (!params_base.speculative.model.empty() ||
        !params_base.speculative.hf_repo.empty()) {
      SRV_INF("loading draft model '%s'\n",
              params_base.speculative.model.c_str());

      auto params_dft = params_base;

      params_dft.devices = params_base.speculative.devices;
      params_dft.hf_file = params_base.speculative.hf_file;
      params_dft.hf_repo = params_base.speculative.hf_repo;
      params_dft.model = params_base.speculative.model;
      params_dft.model_url = params_base.speculative.model_url;
      params_dft.n_ctx = params_base.speculative.n_ctx == 0
                             ? params_base.n_ctx / params_base.n_parallel
                             : params_base.speculative.n_ctx;
      params_dft.n_gpu_layers = params_base.speculative.n_gpu_layers;
      params_dft.n_parallel = 1;

      // force F16 KV cache for the draft model for extra performance
      params_dft.cache_type_k = GGML_TYPE_F16;
      params_dft.cache_type_v = GGML_TYPE_F16;

      llama_init_dft = common_init_from_params(params_dft);

      model_dft = llama_init_dft.model.get();

      if (model_dft == nullptr) {
        SRV_ERR("failed to load draft model, '%s'\n",
                params_base.speculative.model.c_str());
        return false;
      }

      if (!common_speculative_are_compatible(ctx,
                                             llama_init_dft.context.get())) {
        SRV_ERR(
            "the draft model '%s' is not compatible with the target model "
            "'%s'\n",
            params_base.speculative.model.c_str(), params_base.model.c_str());

        return false;
      }

      const int n_ctx_dft = llama_n_ctx(llama_init_dft.context.get());

      cparams_dft = common_context_params_to_llama(params_dft);
      cparams_dft.n_batch = n_ctx_dft;

      // the context is not needed - we will create one for each slot
      llama_init_dft.context.reset();
    }

    chat_templates =
        common_chat_templates_init(model, params_base.chat_template);
    try {
      common_chat_format_example(chat_templates.get(), params.use_jinja);
    } catch (const std::exception &e) {
      SRV_WRN("%s: Chat template parsing error: %s\n", __func__, e.what());
      SRV_WRN(
          "%s: The chat template that comes with this model is not yet "
          "supported, falling back to chatml. This may cause the model to "
          "output suboptimal responses\n",
          __func__);
      chat_templates = common_chat_templates_init(model, "chatml");
    }

    return true;
  }

  void init_houmo_sampling(LogitsProcessorList &processors, ::params* p) {
    // 创建处理器列表
    processors.add(std::make_unique<TopKProcessor>(p->top_k));
    processors.add(std::make_unique<TopPProcessor>(p->top_p));
    processors.add(std::make_unique<TemperatureProcessor>(p->temperature));

    // 只保留RepeatPenaltyProcessor，效果很差-两次都是答非所问
    processors.add(std::make_unique<RepeatPenaltyProcessor>(p->repeat_last_n, p->repeat_penalty));

    // 只保留FrequencyPenaltyProcessor，效果也很差-部分答非所问
    processors.add(std::make_unique<FrequencyPenaltyProcessor>(std::fabs(p->repeat_penalty-1.0)));

  }

  void fillinfomap_from_modelinfo(houmo_context *hmctx) {
    // 2. get input info
        int input_num = hmctx->prefill_model.GetInputNum();
        printf("input_num is：%d\r\n",input_num);
        // std::cout << "Count of Input: " << input_num << std::endl;
        for (int idx = 0; idx < input_num; idx++) {
        auto input_name = hmctx->prefill_model.GetInputName(idx);
        auto input_info =
            hmctx->prefill_model.GetInputInfo(input_name).AsContiguous();
        // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        prefill_input_map.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get output info
        int output_num = hmctx->prefill_model.GetOutputNum();
        printf("output_num is：%d\r\n",output_num);
        // std::cout << "Count of Output: " << output_num << std::endl;
        for (int idx = 0; idx < output_num; idx++) {
        auto output_name = hmctx->prefill_model.GetOutputName(idx);
        auto output_info =
            hmctx->prefill_model.GetOutputInfo(output_name).AsContiguous();
        // std::cout << "Output[" << output_name << "] " << output_info <<
        // std::endl;
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        prefill_input_map.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 2. get input info
        int input_num_decode = hmctx->decode_model.GetInputNum();
        printf("input_num_decode is：%d\r\n",input_num_decode);
        // std::cout << "Count of Input: " << input_num_decode << std::endl;
        for (int idx = 0; idx < input_num_decode; idx++) {
        auto input_name = hmctx->decode_model.GetInputName(idx);
        auto input_info =
            hmctx->decode_model.GetInputInfo(input_name).AsContiguous();
        // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        decode_input_map.insert(
            std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get output info
        int output_num_decode = hmctx->decode_model.GetOutputNum();
        printf("output_num_decode is：%d\r\n",output_num_decode);
        // std::cout << "Count of Output: " << output_num_decode << std::endl;
        for (int idx = 0; idx < output_num_decode; idx++) {
        auto output_name = hmctx->decode_model.GetOutputName(idx);
        auto output_info =
            hmctx->decode_model.GetOutputInfo(output_name).AsContiguous();
        // std::cout << "Output[" << output_name << "] " << output_info <<
        // std::endl;
        auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
        decode_input_map.insert(
            std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

  }

  void fillinfomap_from_modelinfo(houmo_context_14B *hmctx_14B) {
        // 1. get prefillpart1 input info
        int input_num = hmctx_14B->prefill_model_part1_.GetInputNum();
        // std::cout << "Count of Input: " << input_num << std::endl;
        for (int idx = 0; idx < input_num; idx++) {
          auto input_name = hmctx_14B->prefill_model_part1_.GetInputName(idx);
          auto input_info =
              hmctx_14B->prefill_model_part1_.GetInputInfo(input_name).AsContiguous();
          // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
          auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
          prefill_part1_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 1. get prefillpart1 info
        int output_num = hmctx_14B->prefill_model_part1_.GetOutputNum();
        // std::cout << "Count of Output: " << output_num << std::endl;
        for (int idx = 0; idx < output_num; idx++) {
          auto output_name = hmctx_14B->prefill_model_part1_.GetOutputName(idx);
          auto output_info =
              hmctx_14B->prefill_model_part1_.GetOutputInfo(output_name).AsContiguous();
          // std::cout << "Output[" << output_name << "] " << output_info <<
          // std::endl;
          auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
          prefill_part1_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 2. get prefillpart2 input info
        input_num = hmctx_14B->prefill_model_part2_.GetInputNum();
        // std::cout << "Count of Input: " << input_num << std::endl;
        for (int idx = 0; idx < input_num; idx++) {
          auto input_name = hmctx_14B->prefill_model_part2_.GetInputName(idx);
          auto input_info =
              hmctx_14B->prefill_model_part2_.GetInputInfo(input_name).AsContiguous();
          // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
          auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
          prefill_part2_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 2. get prefillpart2 info
        output_num = hmctx_14B->prefill_model_part2_.GetOutputNum();
        // std::cout << "Count of Output: " << output_num << std::endl;
        for (int idx = 0; idx < output_num; idx++) {
          auto output_name = hmctx_14B->prefill_model_part2_.GetOutputName(idx);
          auto output_info =
              hmctx_14B->prefill_model_part2_.GetOutputInfo(output_name).AsContiguous();
          // std::cout << "Output[" << output_name << "] " << output_info <<
          // std::endl;
          auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
          prefill_part2_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 3. get decodepart1 input info
        input_num = hmctx_14B->decode_model_part1_.GetInputNum();
        // std::cout << "Count of Input: " << input_num << std::endl;
        for (int idx = 0; idx < input_num; idx++) {
          auto input_name = hmctx_14B->decode_model_part1_.GetInputName(idx);
          auto input_info =
              hmctx_14B->decode_model_part1_.GetInputInfo(input_name).AsContiguous();
          // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
          auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
          decode_part1_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 3. get decodepart1 info
        output_num = hmctx_14B->decode_model_part1_.GetOutputNum();
        // std::cout << "Count of Output: " << output_num << std::endl;
        for (int idx = 0; idx < output_num; idx++) {
          auto output_name = hmctx_14B->decode_model_part1_.GetOutputName(idx);
          auto output_info =
              hmctx_14B->decode_model_part1_.GetOutputInfo(output_name).AsContiguous();
          // std::cout << "Output[" << output_name << "] " << output_info <<
          // std::endl;
          auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
          decode_part1_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 4. get decodepart2 input info
        input_num = hmctx_14B->decode_model_part2_.GetInputNum();
        // std::cout << "Count of Input: " << input_num << std::endl;
        for (int idx = 0; idx < input_num; idx++) {
          auto input_name = hmctx_14B->decode_model_part2_.GetInputName(idx);
          auto input_info =
              hmctx_14B->decode_model_part2_.GetInputInfo(input_name).AsContiguous();
          // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
          auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
          decode_part2_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get decodepart2 info
        output_num = hmctx_14B->decode_model_part2_.GetOutputNum();
        // std::cout << "Count of Output: " << output_num << std::endl;
        for (int idx = 0; idx < output_num; idx++) {
          auto output_name = hmctx_14B->decode_model_part2_.GetOutputName(idx);
          auto output_info =
              hmctx_14B->decode_model_part2_.GetOutputInfo(output_name).AsContiguous();
          // std::cout << "Output[" << output_name << "] " << output_info <<
          // std::endl;
          auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
          decode_part2_tensormap.insert(
              std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }
  }
  void init_qwen3_8B(std::string model_path) {
    SRV_DBG("%s\r\n","++call init_qwen3_8B");
    model_status = 0;
    hm_model_scale_type=MODEL_8B;
    if (model_path == "") {
      model_path = "./";
    }
    //model_path = model_path + "/";

    hmctx = new houmo_context();
    hmctx->prefill_length = 256;
    hmctx->decode_length = 8192;
    hmctx->nblocks = 36;
    if (model_path.find("qwen2.5") != std::string::npos) {
      hmctx->model_type = "qwen2.5";
    } else if (model_path.find("qwen2") != std::string::npos) {
      hmctx->model_type = "qwen2";
    } else if (model_path.find("qwen3") != std::string::npos) {
      hmctx->model_type = "qwen3";
    }

    if (hmctx->model_type == "qwen2") {
      hmctx->model_name = "Qwen2-7B-Instruct";
    } else if (hmctx->model_type == "qwen2.5") {
      hmctx->model_name = "Qwen2.5-7B-Instruct";
    } else if (hmctx->model_type == "qwen3") {
      hmctx->model_name = "Qwen3-8B";
    } else if (hmctx->model_type == "deepseek") {
      hmctx->model_name = "DeepSeek-R1-Distill-Qwen-14B";
    } else {
      LOG_ERR(
          "model type %s not support, only support qwen2, qwen2.5 and "
          "deepseek.\n",
          hmctx->model_type);
    }

    // /v1/models 接口依赖获取对应的model_id
    // params->model_alias = hmctx_14B->model_name;
    // hmctx->model_name=hmctx_14B->model_name;

        //printf("nblocks=%d,args.nblocks=%d \r\n",nblocks,args.nblocks);
        std::string HMM_MODEL_PATH = model_path;
        HMM_MODEL_PATH.resize(HMM_MODEL_PATH.length() - 1);//去掉多余的 "/"
        tcim::Module::WeightManager weight_manager=tcim::Module::WeightManager::CreateWeightManager(0);
        tcim::Module::Option option1(weight_manager);
        tcim::Module::Option option2(weight_manager);
        std::vector<std::string> dummy_tensor_names;
        for (int i = 0; i < hmctx->nblocks; ++i) {
            dummy_tensor_names.push_back("model_layers_" + std::to_string(i) + "_self_attn_kcache_input");
            dummy_tensor_names.push_back("model_layers_" + std::to_string(i) + "_self_attn_vcache_input");
        }
        option2.SetDummyTensors(dummy_tensor_names);
        hmctx->prefill_model = tcim::Module::LoadFromFile((fs::path(HMM_MODEL_PATH) / "prefill.hmm").string(),option1);
        printf("++1 prefill model loaded\r\n");
        hmctx->decode_model = tcim::Module::LoadFromFile((fs::path(HMM_MODEL_PATH) / "decoder.hmm").string(),option2);
        printf("++2 decode model loaded\r\n");
        fillinfomap_from_modelinfo(hmctx);
        printf("++2.1 after fillmap\r\n");

        // 设置KVCache输入
        for (int i = 0; i < hmctx->nblocks; ++i) {
            std::string kcache_input_name = "model_layers_" + std::to_string(i) + "_self_attn_kcache_input";
            std::string vcache_input_name = "model_layers_" + std::to_string(i) + "_self_attn_vcache_input";
            auto kcache = hmctx->prefill_model.GetInput(kcache_input_name);
            hmctx->decode_model.SetInput(kcache_input_name, kcache);
            auto vcache = hmctx->prefill_model.GetInput(vcache_input_name);
            hmctx->decode_model.SetInput(vcache_input_name, vcache);
        }

        // 设置decode输入
        std::vector<int16_t> current_length = {1};
        printf("++3\r\n");
        //TODO:szj 确认 shape 是1维的1 还是二维的1
        TensorConverter::FromVector(current_length, decode_input_map.at("current_length"));
        hmctx->decode_model.SetInput("current_length", decode_input_map.at("current_length"));

        // 加载embedding权重
        // 这里应该实现加载embedding权重的逻辑
        printf("++4 EMBEDDING_PATH.string() is:%s\r\n",(HMM_MODEL_PATH+"/quant_embedding_traced.pt").c_str());
        torch::Tensor embedding_weight_;
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

        const int32_t n_ctx_slot = n_ctx / params_base.n_parallel;

        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        for (int i = 0; i < params_base.n_parallel; i++) {
          server_slot slot;

          slot.id = i;
          slot.ctx = ctx;
          slot.n_ctx = n_ctx_slot;
          slot.n_predict = params_base.n_predict;

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

          slots.push_back(slot);
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
        hm_sampling_processors.print_processors_param();
  }

  void init_14B(std::string model_path) {
    #define SPLIT_COUNT (26)
    model_status = 0;
    hm_model_scale_type=MODEL_14B;
    if (model_path == "") {
      model_path = "./";
    }
    model_path = model_path + "/";

    hmctx_14B = new houmo_context_14B();
    hmctx = new houmo_context();
    hmctx_14B->prefill_length = 256;
    hmctx_14B->decode_length = 8192;
    hmctx_14B->nblocks = 48;
    if (model_path.find("qwen2.5") != std::string::npos) {
      hmctx_14B->model_type = "qwen2.5";
    } else if (model_path.find("qwen2") != std::string::npos) {
      hmctx_14B->model_type = "qwen2";
    }

    if (hmctx_14B->model_type == "qwen2") {
      hmctx_14B->model_name = "Qwen2-7B-Instruct";
    } else if (hmctx_14B->model_type == "qwen2.5") {
      hmctx_14B->model_name = "Qwen2.5-7B-Instruct";
    } else if (hmctx_14B->model_type == "deepseek") {
      hmctx_14B->model_name = "DeepSeek-R1-Distill-Qwen-14B";
    } else {
      LOG_ERR(
          "model type %s not support, only support qwen2, qwen2.5 and "
          "deepseek.\n",
          hmctx_14B->model_type);
    }

    // /v1/models 接口依赖获取对应的model_id
    params->model_alias = hmctx_14B->model_name;
    hmctx->model_name=hmctx_14B->model_name;

    try {
        std::string input_path = model_path + "hmquant/quant_embedding_traced.pt";
        torch::Tensor input;
        torch::load(input, input_path);
        input = input.view({-1, 5120});
        // std::cout << "input options: " << input.options() << "\n";
        // std::cout << "input dim: " << input.dim() << "\n";
        // std::cout << "input size: " << input.size(0) << "\n";
        // std::cout << "embedding content: " << std::endl;
        // for (int i = 0; i < 20; i++) {
        //     std::cout << input[0][i] << std::endl;
        // }
        // std::cout << std::endl;
        emb = torch::nn::Embedding::from_pretrained(input);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model: " << e.what_without_backtrace() << std::endl;
        return;
    }

    LOG_DBG("%s\n", "emb loaded");
    LOG_DBG("tcim version: %s\n", tcim::GetVersion().c_str());

    // 初始化权重管理器
    tcim::Module::WeightManager weight_manager_0=tcim::Module::WeightManager::CreateWeightManager(0);
    tcim::Module::WeightManager weight_manager_1=tcim::Module::WeightManager::CreateWeightManager(1);

    // 创建模型选项
    tcim::Module::Option option1_0(weight_manager_0);
    tcim::Module::Option option1_1(weight_manager_1);
    tcim::Module::Option option2_0(weight_manager_0);
    tcim::Module::Option option2_1(weight_manager_1);

    // 设置dummy tensors
    std::vector<std::string> dummy_tensor_names;
    for (int i = 0; i < hmctx_14B->nblocks; ++i) {
        dummy_tensor_names.push_back("model_layers_" + std::to_string(i) + "_self_attn_kcache_input");
        dummy_tensor_names.push_back("model_layers_" + std::to_string(i) + "_self_attn_vcache_input");
        dummy_tensor_names.push_back("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum");
    }
    option2_0.SetDummyTensors(dummy_tensor_names);
    option2_1.SetDummyTensors(dummy_tensor_names);

    hmctx_14B->stream = tcim::Stream(true);
    hmctx_14B->stream2 = tcim::Stream(true);

    std::string prefill_model = "qwen2_prefill.hmm";
    std::string decode_model = "qwen2_decode.hmm";
    if (hmctx_14B->model_type == "qwen2.5") {
      decode_model = "qwen2.5_decode.hmm";
      prefill_model = "qwen2.5_prefill.hmm";
    } else if (hmctx_14B->model_type == "deepseek") {
      decode_model = "deepseek_decode.hmm";
      prefill_model = "deepseek_prefill.hmm";
    }
    // std::cout << "prefill_model: " << prefill_model << "\n";
    // std::cout << "decode_model: " << decode_model << "\n";

    // 加载模型
    hmctx_14B->prefill_model_part1_ = tcim::Module::LoadFromFile((model_path +"/deepseek_prefill_part1.hmm"), option1_0);
    std::cout << " loaded prefill part1" << std::endl;

    hmctx_14B->prefill_model_part2_ = tcim::Module::LoadFromFile((model_path +"/deepseek_prefill_part2.hmm"), option1_1);
    std::cout << " loaded prefill part2" << std::endl;

    hmctx_14B->decode_model_part1_ = tcim::Module::LoadFromFile((model_path  +"/deepseek_decode_part1.hmm"), option2_0);
    std::cout << " loaded decoder part1" << std::endl;

    hmctx_14B->decode_model_part2_ = tcim::Module::LoadFromFile((model_path  +"/deepseek_decode_part2.hmm"), option2_1);
    std::cout << " loaded decoder part2" << std::endl;

    hmctx_14B->prefill_model_part1_.SetStream(hmctx_14B->stream);
    hmctx_14B->prefill_model_part2_.SetStream(hmctx_14B->stream2);
    hmctx_14B->decode_model_part1_.SetStream(hmctx_14B->stream);
    hmctx_14B->decode_model_part2_.SetStream(hmctx_14B->stream2);

    fillinfomap_from_modelinfo(hmctx_14B);
    printf("++1\r\n");
    // 设置KVCache输入
    for (int i = 0; i < SPLIT_COUNT; ++i) {
        //printf("++i =%d\r\n",i);
        auto kcache = hmctx_14B->prefill_model_part1_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_input");
        //printf("##1\r\n");
        hmctx_14B->decode_model_part1_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_input", kcache);
        //printf("##2\r\n");
        auto vcache = hmctx_14B->prefill_model_part1_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_vcache_input");
        //printf("##3\r\n");
        hmctx_14B->decode_model_part1_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_vcache_input", vcache);
        //printf("##4\r\n");
        auto kcache_history_sum = hmctx_14B->prefill_model_part1_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum");
        //printf("##5\r\n");
        hmctx_14B->decode_model_part1_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum", kcache_history_sum);
        //printf("##6\r\n");
    }
    printf("++2\r\n");
    for (int i = SPLIT_COUNT; i < hmctx_14B->nblocks; ++i) {
        auto kcache = hmctx_14B->prefill_model_part2_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_input");
        hmctx_14B->decode_model_part2_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_input", kcache);

        auto vcache = hmctx_14B->prefill_model_part2_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_vcache_input");
        hmctx_14B->decode_model_part2_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_vcache_input", vcache);

        auto kcache_history_sum = hmctx_14B->prefill_model_part2_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum");
        hmctx_14B->decode_model_part2_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum", kcache_history_sum);
    }
    printf("++3\r\n");
    // 设置decode输入
    std::vector<int16_t> current_length_input_1 = {1};
    //TODO:szj 确认 shape 是1维的1 还是二维的1
    TensorConverter::FromVector(current_length_input_1, decode_part1_tensormap.at("current_length"));
    hmctx_14B->decode_model_part1_.SetInput("current_length", decode_part1_tensormap.at("current_length"));
    TensorConverter::FromVector(current_length_input_1, decode_part2_tensormap.at("current_length"));
    hmctx_14B->decode_model_part2_.SetInput("current_length",  decode_part2_tensormap.at("current_length"));



    const int32_t n_ctx_slot = n_ctx / params_base.n_parallel;

    SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

    for (int i = 0; i < params_base.n_parallel; i++) {
      server_slot slot;

      slot.id = i;
      slot.ctx = ctx;
      slot.n_ctx = n_ctx_slot;
      slot.n_predict = params_base.n_predict;

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

      slots.push_back(slot);
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
    hm_sampling_processors.print_processors_param();
  }

  void init(std::string model_path) {
    model_status = 0;
    hm_model_scale_type=MODEL_7B;
    if (model_path == "") {
      model_path = "./";
    }
    model_path = model_path + "/";

    hmctx = new houmo_context();
    if (model_path.find("qwen2.5") != std::string::npos) {
      hmctx->model_type = "qwen2.5";
    } else if (model_path.find("qwen2") != std::string::npos) {
      hmctx->model_type = "qwen2";
    }

    if (hmctx->model_type == "qwen2") {
      hmctx->model_name = "Qwen2-7B-Instruct";
    } else if (hmctx->model_type == "qwen2.5") {
      hmctx->model_name = "Qwen2.5-7B-Instruct";
    } else if (hmctx->model_type == "deepseek") {
      hmctx->model_name = "DeepSeek-R1-Distill-Qwen-7B";
    } else {
      LOG_ERR(
          "model type %s not support, only support qwen2, qwen2.5 and "
          "deepseek.\n",
          hmctx->model_type);
    }

    // /v1/models 接口依赖获取对应的model_id
    params->model_alias = hmctx->model_name;

    std::string prefill_model = "qwen2_prefill.hmm";
    std::string decode_model = "qwen2_decode.hmm";
    if (hmctx->model_type == "qwen2.5") {
      decode_model = "qwen2.5_decode.hmm";
      prefill_model = "qwen2.5_prefill.hmm";
    } else if (hmctx->model_type == "deepseek") {
      decode_model = "deepseek_decode.hmm";
      prefill_model = "deepseek_prefill.hmm";
    }
    // std::cout << "prefill_model: " << prefill_model << "\n";
    // std::cout << "decode_model: " << decode_model << "\n";


    try {
        torch::Tensor input;
        std::string input_path = model_path + "hmquant/quant_embedding_traced.pt";
        torch::load(input, input_path);
        input = input.view({-1, 3584});
        // std::cout << "input options: " << input.options() << "\n";
        // std::cout << "input dim: " << input.dim() << "\n";
        // std::cout << "input size: " << input.size(0) << "\n";
        // std::cout << "embedding content: " << std::endl;
        // for (int i = 0; i < 20; i++) {
        //     std::cout << input[0][i] << std::endl;
        // }
        // std::cout << std::endl;
        emb = torch::nn::Embedding::from_pretrained(input);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model: " << e.what_without_backtrace() << std::endl;
        return;
    }

    LOG_DBG("%s\n", "emb loaded");

    // std::cout << "options: " << input.options() << "\n";
    // std::cout << "dim: " << input.dim() << "\n";
    // std::cout << "size: " << input.size(0) << "\n";

    LOG_DBG("tcim version: %s\n", tcim::GetVersion().c_str());

    // for (int i = 0; i < hmctx->nblocks; i++) {
    //     hmctx->dummy_tensor_names.push_back(hm_string_format("model_layers_%d_self_attn_kcache_input",
    //     i));
    // }
    // for (int i = 0; i < hmctx->nblocks; i++) {
    //     hmctx->dummy_tensor_names.push_back(hm_string_format("model_layers_%d_self_attn_vcache_input",
    //     i));
    // }

    hmctx->stream = tcim::Stream(true);
    hmctx->wm = tcim::Module::WeightManager::CreateWeightManager(0);
    hmctx->option_prefill = tcim::Module::Option(hmctx->wm);
    hmctx->option_decode = tcim::Module::Option(hmctx->wm);
    // hmctx->option_decode.set_dummy_tensors(hmctx->dummy_tensor_names);
    LOG_DBG("%s\n", "hmctx created.");

    // 1. load model
    // std::cout << "LoadFromFile prefill_part1" << std::endl;
    std::string prefill_model_path = model_path + prefill_model;
    LOG_DBG("model %s to load.\n", prefill_model_path.c_str());
    if (!fs::exists(prefill_model_path)) {
      LOG_ERR(
          "model %s not exist. you should run build.py in prefill_part1 "
          "example first.\n",
          prefill_model_path.c_str());
      exit(-1);
    }
    if (!fs::is_regular_file(prefill_model_path.c_str())) {
      LOG_ERR("File %s Not Found!\n", prefill_model_path.c_str());
    }
    hmctx->prefill_model =
        tcim::Module::LoadFromFile(prefill_model_path, hmctx->option_prefill);
    if (!hmctx->prefill_model) {
      std::cerr << " load model " << prefill_model_path << " fail."
                << std::endl;
      exit(-1);
    }
    LOG_DBG("model %s loaded.\n", prefill_model_path.c_str());
    hmctx->prefill_model.SetStream(hmctx->stream);

    // 2. get input info
    int input_num = hmctx->prefill_model.GetInputNum();
    // std::cout << "Count of Input: " << input_num << std::endl;
    for (int idx = 0; idx < input_num; idx++) {
      auto input_name = hmctx->prefill_model.GetInputName(idx);
      auto input_info =
          hmctx->prefill_model.GetInputInfo(input_name).AsContiguous();
      // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
      auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
      prefill_input_map.insert(
          std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }

    // 4. get output info
    int output_num = hmctx->prefill_model.GetOutputNum();
    // std::cout << "Count of Output: " << output_num << std::endl;
    for (int idx = 0; idx < output_num; idx++) {
      auto output_name = hmctx->prefill_model.GetOutputName(idx);
      auto output_info =
          hmctx->prefill_model.GetOutputInfo(output_name).AsContiguous();
      // std::cout << "Output[" << output_name << "] " << output_info <<
      // std::endl;
      auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
      prefill_input_map.insert(
          std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
    }

    // 5. set input
    // for (const auto& input : prefill_input_map) {
    //    hmctx->prefill_model.SetInput(input.first, input.second);
    //}
    //
    //// 6. run and sync
    ////hmctx->prefill_model.Run();
    ////hmctx->prefill_model.Sync();
    //
    //// 7. get output
    // for (auto& output : prefill_output_map) {
    //     hmctx->prefill_model.GetOutput(output.first, output.second);
    // }
    //
    //// 8. postprocess, with no softmax
    // for (auto& output : prefill_output_map) {
    //     //std::cout << "output name: " << output.first << "output data: " <<
    //     output.second << std::endl;
    //     //std::cout << "output name: " << output.first << std::endl;
    // }

    // 1. load model
    // std::cout << "LoadFromFile decode_part1" << std::endl;
    std::string decode_model_path = model_path + decode_model;
    if (!fs::exists(decode_model_path)) {
      std::cerr << decode_model_path
                << " not exist. you should run build.py in decode_part1 "
                   "example first."
                << std::endl;
      exit(-1);
    }
    hmctx->decode_model =
        tcim::Module::LoadFromFile(decode_model_path, hmctx->option_decode);
    if (!hmctx->decode_model) {
      std::cerr << " load model " << decode_model_path << " fail." << std::endl;
      exit(-1);
    }
    LOG_DBG("model %s loaded.\n", decode_model_path.c_str());
    hmctx->decode_model.SetStream(hmctx->stream);

    // 2. get input info
    int input_num_decode = hmctx->decode_model.GetInputNum();
    // std::cout << "Count of Input: " << input_num_decode << std::endl;
    for (int idx = 0; idx < input_num_decode; idx++) {
      auto input_name = hmctx->decode_model.GetInputName(idx);
      auto input_info =
          hmctx->decode_model.GetInputInfo(input_name).AsContiguous();
      // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
      auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
      decode_input_map.insert(
          std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }

    // 4. get output info
    int output_num_decode = hmctx->decode_model.GetOutputNum();
    // std::cout << "Count of Output: " << output_num_decode << std::endl;
    for (int idx = 0; idx < output_num_decode; idx++) {
      auto output_name = hmctx->decode_model.GetOutputName(idx);
      auto output_info =
          hmctx->decode_model.GetOutputInfo(output_name).AsContiguous();
      // std::cout << "Output[" << output_name << "] " << output_info <<
      // std::endl;
      auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
      decode_output_map.insert(
          std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
    }

    // 5. set input
    // for (const auto& input : decode_part1_input_map) {
    //    hmctx->decode_model.SetInput(input.first, input.second);
    //}

    // 6. run and sync
    // hmctx->decode_model.Run();
    // hmctx->decode_model.Sync();

    // 7. get output
    // for (auto& output : decode_output_map) {
    //    hmctx->decode_model.GetOutput(output.first, output.second);
    //}

    // 8. postprocess, with no softmax
    // for (auto& output : decode_output_map) {
    //    //std::cout << "output name: " << output.first << "output data: " <<
    //    output.second << std::endl; std::cout << "output name: " <<
    //    output.first << std::endl;
    //}

    // set kvcache input
    for (int idx = 0; idx < hmctx->nblocks; idx++) {
      std::string k_cache_name =
          hm_string_format("model_layers_%d_self_attn_kcache_input", idx);
      std::string v_cache_name =
          hm_string_format("model_layers_%d_self_attn_vcache_input", idx);
      auto kcache = hmctx->prefill_model.GetInput(k_cache_name);
      auto vcache = hmctx->prefill_model.GetInput(v_cache_name);
      hmctx->decode_model.SetInput(k_cache_name, kcache);
      hmctx->decode_model.SetInput(v_cache_name, vcache);
      if (hmctx->model_type == "deepseek") {
        std::string kcache_history_sum_name = hm_string_format(
            "model_layers_%d_self_attn_kcache_history_sum", idx);
        auto kcache_history_sum =
            hmctx->prefill_model.GetInput(kcache_history_sum_name);
        hmctx->decode_model.SetInput(kcache_history_sum_name,
                                     kcache_history_sum);
      }
    }

    // set decode input
    std::vector<int16_t> current_length_input = {1};
    auto current_length_input_tensor_opts =
        torch::TensorOptions().dtype(torch::kInt16);
    auto current_length_input_tensor =
        torch::from_blob(current_length_input.data(),
                         {int64_t(current_length_input.size())},
                         current_length_input_tensor_opts)
            .clone();

    // std::cout << "current_length_input_tensor: " <<
    // current_length_input_tensor << std::endl;

    auto decode_current_length = decode_input_map.at("current_length");
    // int16toint16((int16_t *)decode_current_length.Data(), (int16_t
    // *)current_length_input.data(), current_length_input.size());
    // int16toint16((int16_t *)decode_current_length.Data(), (int16_t
    // *)current_length_input_tensor.data<int16_t>(),
    // current_length_input_tensor.size(0)); int16toint16((int16_t
    // *)decode_current_length.Data(), (int16_t
    // *)current_length_input_tensor.data<int16_t>(),
    // current_length_input_tensor.numel());
    int16toint16((int16_t *)decode_current_length.Data(),
                 (int16_t *)current_length_input_tensor.data_ptr<int16_t>(),
                 current_length_input_tensor.numel());

    hmctx->decode_model.SetInput("current_length", decode_current_length);

    const int32_t n_ctx_slot = n_ctx / params_base.n_parallel;

    SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

    for (int i = 0; i < params_base.n_parallel; i++) {
      server_slot slot;

      slot.id = i;
      slot.ctx = ctx;
      slot.n_ctx = n_ctx_slot;
      slot.n_predict = params_base.n_predict;

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

      slots.push_back(slot);
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
    hm_sampling_processors.print_processors_param();
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

  void init_by_buffer(HANDLE h, uint8_t **buffers, size_t *bufferSize,
                      size_t buffers_count, hardware_info hwinfo) {
    std::cout << "++enter 1: buffers_count is:" << buffers_count << std::endl;
    if (0 == buffers_count) {
      std::cout << "++enter 2 buffers_count is invalid:" << std::endl;
      // 在exit前手动刷新所有日志缓冲区
      fflush(stdout);
      fflush(stderr);
      exit(-1);
    }
    if (nullptr == buffers) {
      std::cout << "++enter 3 buffers is invalid:" << std::endl;
      // 在exit前手动刷新所有日志缓冲区
      fflush(stdout);
      fflush(stderr);
      exit(-1);
    }
    for (int i = 0; i < buffers_count; i++) {
      if (nullptr == buffers[i]) {
        std::cout << "++enter 4 buffers index: " << i
                  << "_address is invalid:" << std::endl;
        // 在exit前手动刷新所有日志缓冲区
        fflush(stdout);
        fflush(stderr);
        exit(-1);
      }
    }
    model_status = 0;
    std::cout << "++enter 1.1" << std::endl;
    // TODO 默认写死的deepseek
    hmctx = new houmo_context();

    if (hmctx->model_type == "qwen2") {
      hmctx->model_name = "Qwen2-7B-Instruct";
    } else if (hmctx->model_type == "qwen2.5") {
      hmctx->model_name = "Qwen2.5-7B-Instruct";
    } else if (hmctx->model_type == "deepseek") {
      hmctx->model_name = "DeepSeek-R1-Distill-Qwen-7B";
    } else {
      LOG_ERR(
          "model type %s not support, only support qwen2, qwen2.5 and "
          "deepseek.\n",
          hmctx->model_type);
    }

    // hmctx->model_type = "deepseek";
    // hmctx->model_type = "qwen2";
    // hmctx->model_type = "qwen2.5";
    std::string prefill_model = "qwen2_prefill.hmm";
    std::string decode_model = "qwen2_decode.hmm";
    if (hmctx->model_type == "qwen2.5") {
      decode_model = "qwen2.5_decode.hmm";
      prefill_model = "qwen2.5_prefill.hmm";
    } else if (hmctx->model_type == "deepseek") {
      decode_model = "deepseek_decode.hmm";
      prefill_model = "deepseek_prefill.hmm";
    }
    // std::cout << "prefill_model: " << prefill_model << "\n";
    // std::cout << "decode_model: " << decode_model << "\n";

    try {
        torch::Tensor input;
        std::string input_path = model_path + "hmquant/quant_embedding_traced.pt";
        torch::load(input, input_path);
        input = input.view({-1, 3584});
        // std::cout << "input options: " << input.options() << "\n";
        // std::cout << "input dim: " << input.dim() << "\n";
        // std::cout << "input size: " << input.size(0) << "\n";
        // std::cout << "embedding content: " << std::endl;
        // for (int i = 0; i < 20; i++) {
        //     std::cout << input[0][i] << std::endl;
        // }
        // std::cout << std::endl;
        emb = torch::nn::Embedding::from_pretrained(input);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model: " << e.what_without_backtrace() << std::endl;
        return;
    }

    LOG_DBG("%s\n", "emb loaded");

    // std::cout << "options: " << input.options() << "\n";
    // std::cout << "dim: " << input.dim() << "\n";
    // std::cout << "size: " << input.size(0) << "\n";

    LOG_DBG("tcim version: %s\n", tcim::GetVersion().c_str());

    // for (int i = 0; i < hmctx->nblocks; i++) {
    //     hmctx->dummy_tensor_names.push_back(hm_string_format("model_layers_%d_self_attn_kcache_input",
    //     i));
    // }
    // for (int i = 0; i < hmctx->nblocks; i++) {
    //     hmctx->dummy_tensor_names.push_back(hm_string_format("model_layers_%d_self_attn_vcache_input",
    //     i));
    // }

    hmctx->stream = tcim::Stream(true);
    hmctx->wm = tcim::Module::WeightManager::CreateWeightManager(0);
    hmctx->option_prefill = tcim::Module::Option(hmctx->wm);
    hmctx->option_decode = tcim::Module::Option(hmctx->wm);
    // hmctx->option_decode.set_dummy_tensors(hmctx->dummy_tensor_names);
    LOG_INF("%s\n", "hmctx created.");

    // 1. load prefill model from buffer
    uint64_t prefill_model_size =*(uint64_t* )bufferSize;
    LOG_INF("++prefill model buffer addr:%p  bufferSize[0] is:%llu\r\n",
            buffers[0], prefill_model_size);

    //如果需要保存Ram到文件-用下面的代码
    // LOG_INF("++before save raw data\r\n");
    // save_raw_data(buffers[0], *bufferSize, "prefill_model_frombuffer.hmm");
    // LOG_INF("++after save raw data\r\n");

    // hmctx->prefill_model = tcim::Module::LoadFromMem(buffers[0], *bufferSize,
    //                                                  hmctx->option_prefill);
    hmctx->prefill_model.LoadModel(buffers[0], prefill_model_size,
                                                     hmctx->option_prefill);
    // hmctx->prefill_model = tcim::Module::LoadFromFile(prefill_model_path,
    // hmctx->option_prefill);
    if (!hmctx->prefill_model) {
      std::cerr << " load  prefill_model from buffer failed.*bufferSize:"
                << *bufferSize
                << " || prefillmodel buffer[0] address:" << buffers[0]
                << std::endl;
      exit(-1);
    }
    LOG_INF("++Prefill model load from buffer loaded success\n");
    hmctx->prefill_model.SetStream(hmctx->stream);

    // 2. get input info
    int input_num = hmctx->prefill_model.GetInputNum();
    // std::cout << "Count of Input: " << input_num << std::endl;
    for (int idx = 0; idx < input_num; idx++) {
      auto input_name = hmctx->prefill_model.GetInputName(idx);
      auto input_info =
          hmctx->prefill_model.GetInputInfo(input_name).AsContiguous();
      // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
      auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
      prefill_input_map.insert(
          std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }

    // 4. get output info
    int output_num = hmctx->prefill_model.GetOutputNum();
    // std::cout << "Count of Output: " << output_num << std::endl;
    for (int idx = 0; idx < output_num; idx++) {
      auto output_name = hmctx->prefill_model.GetOutputName(idx);
      auto output_info =
          hmctx->prefill_model.GetOutputInfo(output_name).AsContiguous();
      // std::cout << "Output[" << output_name << "] " << output_info <<
      // std::endl;
      auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
      prefill_input_map.insert(
          std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
    }

    // 5. set input
    // for (const auto& input : prefill_input_map) {
    //    hmctx->prefill_model.SetInput(input.first, input.second);
    //}
    //
    //// 6. run and sync
    ////hmctx->prefill_model.Run();
    ////hmctx->prefill_model.Sync();
    //
    //// 7. get output
    // for (auto& output : prefill_output_map) {
    //     hmctx->prefill_model.GetOutput(output.first, output.second);
    // }
    //
    //// 8. postprocess, with no softmax
    // for (auto& output : prefill_output_map) {
    //     //std::cout << "output name: " << output.first << "output data: " <<
    //     output.second << std::endl;
    //     //std::cout << "output name: " << output.first << std::endl;
    // }

    // 1. load decode model
    uint64_t decode_model_size =*((uint64_t* )bufferSize+1);
    LOG_DBG("++decode model buffer addr:%x  bufferSize[1] is:%llu\r\n",
            buffers[1], decode_model_size);
    // hmctx->decode_model = tcim::Module::LoadFromMem(buffers[1], *bufferSize,
    //                                                 hmctx->option_decode);

    //TODO: buffersize 应该和模型实际尺寸一致，不然会有问题-不能按照最大值传递-4451267656
    hmctx->decode_model.LoadModel(buffers[1], decode_model_size,
                                                    hmctx->option_decode);
    // hmctx->decode_model = tcim::Module::LoadFromFile(decode_model_path,
    // hmctx->option_decode);
    if (!hmctx->decode_model) {
      std::cerr << " load  decode_model from buffer failed.*bufferSize:"
                << *bufferSize
                << " || decodemodel buffer[1] address:" << buffers[1]
                << std::endl;
      exit(-1);
    }
    LOG_DBG("++Decode model load from buffer loaded success\n");
    hmctx->decode_model.SetStream(hmctx->stream);

    // 2. get input info
    int input_num_decode = hmctx->decode_model.GetInputNum();
    // std::cout << "Count of Input: " << input_num_decode << std::endl;
    for (int idx = 0; idx < input_num_decode; idx++) {
      auto input_name = hmctx->decode_model.GetInputName(idx);
      auto input_info =
          hmctx->decode_model.GetInputInfo(input_name).AsContiguous();
      // std::cout << "Input[" << input_name << "] " << input_info << std::endl;
      auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
      decode_input_map.insert(
          std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }

    // 4. get output info
    int output_num_decode = hmctx->decode_model.GetOutputNum();
    // std::cout << "Count of Output: " << output_num_decode << std::endl;
    for (int idx = 0; idx < output_num_decode; idx++) {
      auto output_name = hmctx->decode_model.GetOutputName(idx);
      auto output_info =
          hmctx->decode_model.GetOutputInfo(output_name).AsContiguous();
      // std::cout << "Output[" << output_name << "] " << output_info <<
      // std::endl;
      auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
      decode_output_map.insert(
          std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
    }

    // 5. set input
    // for (const auto& input : decode_part1_input_map) {
    //    hmctx->decode_model.SetInput(input.first, input.second);
    //}

    // 6. run and sync
    // hmctx->decode_model.Run();
    // hmctx->decode_model.Sync();

    // 7. get output
    // for (auto& output : decode_output_map) {
    //    hmctx->decode_model.GetOutput(output.first, output.second);
    //}

    // 8. postprocess, with no softmax
    // for (auto& output : decode_output_map) {
    //    //std::cout << "output name: " << output.first << "output data: " <<
    //    output.second << std::endl; std::cout << "output name: " <<
    //    output.first << std::endl;
    //}

    // set kvcache input
    for (int idx = 0; idx < hmctx->nblocks; idx++) {
      std::string k_cache_name =
          hm_string_format("model_layers_%d_self_attn_kcache_input", idx);
      std::string v_cache_name =
          hm_string_format("model_layers_%d_self_attn_vcache_input", idx);
      auto kcache = hmctx->prefill_model.GetInput(k_cache_name);
      auto vcache = hmctx->prefill_model.GetInput(v_cache_name);
      hmctx->decode_model.SetInput(k_cache_name, kcache);
      hmctx->decode_model.SetInput(v_cache_name, vcache);
      if (hmctx->model_type == "deepseek") {
        std::string kcache_history_sum_name = hm_string_format(
            "model_layers_%d_self_attn_kcache_history_sum", idx);
        auto kcache_history_sum =
            hmctx->prefill_model.GetInput(kcache_history_sum_name);
        hmctx->decode_model.SetInput(kcache_history_sum_name,
                                     kcache_history_sum);
      }
    }

    // set decode input
    std::vector<int16_t> current_length_input = {1};
    auto current_length_input_tensor_opts =
        torch::TensorOptions().dtype(torch::kInt16);
    auto current_length_input_tensor =
        torch::from_blob(current_length_input.data(),
                         {int64_t(current_length_input.size())},
                         current_length_input_tensor_opts)
            .clone();

    // std::cout << "current_length_input_tensor: " <<
    // current_length_input_tensor << std::endl;

    auto decode_current_length = decode_input_map.at("current_length");
    // int16toint16((int16_t *)decode_current_length.Data(), (int16_t
    // *)current_length_input.data(), current_length_input.size());
    // int16toint16((int16_t *)decode_current_length.Data(), (int16_t
    // *)current_length_input_tensor.data<int16_t>(),
    // current_length_input_tensor.size(0)); int16toint16((int16_t
    // *)decode_current_length.Data(), (int16_t
    // *)current_length_input_tensor.data<int16_t>(),
    // current_length_input_tensor.numel());
    int16toint16((int16_t *)decode_current_length.Data(),
                 (int16_t *)current_length_input_tensor.data_ptr<int16_t>(),
                 current_length_input_tensor.numel());

    hmctx->decode_model.SetInput("current_length", decode_current_length);

    const int32_t n_ctx_slot = n_ctx / params_base.n_parallel;

    SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

    for (int i = 0; i < params_base.n_parallel; i++) {
      server_slot slot;

      slot.id = i;
      slot.ctx = ctx;
      slot.n_ctx = n_ctx_slot;
      slot.n_predict = params_base.n_predict;

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

      slots.push_back(slot);
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
        int cur_lcs_len = common_lcs(slot.cache_tokens, task.prompt_tokens);

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
      int64_t t_last = ggml_time_us();
      for (server_slot &slot : slots) {
        // skip the slot if it is not available
        if (slot.is_processing()) {
          continue;
        }

        // select the current slot if the criteria match
        if (slot.t_last_used < t_last) {
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

  bool can_be_detokenized(const struct llama_context *ctx,
                          const std::vector<llama_token> &tokens) {
    const llama_model *model = llama_get_model(ctx);
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (const auto &token : tokens) {
      if (token < 0 || token >= n_vocab) {
        return false;
      }
    }
    return true;
  }

  bool launch_slot_with_task(server_slot &slot, const server_task &task) {
    slot.reset();
    slot.id_task = task.id;
    slot.index = task.index;
    slot.task_type = task.type;
    slot.params = std::move(task.params);
    slot.prompt_tokens = std::move(task.prompt_tokens);

    if (!are_lora_equal(task.params.lora, slot.lora)) {
      // if lora is changed, we cannot reuse cached tokens
      slot.cache_tokens.clear();
      slot.lora = task.params.lora;
    }

    bool can_detokenize = can_be_detokenized(ctx, slot.prompt_tokens);
    if (!can_detokenize) {
      send_error(task, "Prompt contains invalid tokens",
                 ERROR_TYPE_INVALID_REQUEST);
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
    llama_kv_self_clear(ctx);
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
    res->content = std::move(slot.generated_text);
    res->tokens = std::move(slot.generated_tokens);
    res->timings = slot.get_timings();
    res->prompt = common_detokenize(ctx, slot.prompt_tokens, true);
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
    res->oaicompat_chat_format = slot.params.oaicompat_chat_format;
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
      cancel_tasks.push_back(task);
    }
    // push to beginning of the queue, so it has highest priority
    queue_tasks.post(cancel_tasks, true);
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
  int completion_generate(json &data,
                          std::vector<server_task_result_ptr> &results) {
    auto type = SERVER_TASK_TYPE_COMPLETION;
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION ||
                type == SERVER_TASK_TYPE_INFILL);

    std::unique_lock<std::mutex> lock(mutex_context);
    generate_status = 1;
    model_status = 3;
    auto completion_id = gen_chatcmplid();
    std::vector<server_task> tasks;
    oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;
    bool oaicompat_chat = false;

    try {
      std::vector<llama_tokens> tokenized_prompts =
          tokenize_input_prompts(vocab, data.at("prompt"), true, true);
      tasks.reserve(tokenized_prompts.size());
      for (size_t i = 0; i < tokenized_prompts.size(); i++) {
        server_task task = server_task(type);

        task.id = queue_tasks.get_new_id();
        task.index = i;

        task.prompt_tokens = std::move(tokenized_prompts[i]);
        task.params =
            server_task::params_from_json_cmpl(ctx, params_base, data);
        task.id_selected_slot = json_value(data, "id_slot", -1);
        SRV_INF("++task.id_selected_slot is:%d \r\n", task.id_selected_slot);

        // OAI-compat
        task.params.oaicompat = oaicompat;
        task.params.oaicompat_cmpl_id = completion_id;
        // oaicompat_model is already populated by params_from_json_cmpl

        tasks.push_back(task);
      }
    } catch (const std::exception &e) {
      log_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
      return -1;
    }

        queue_results.add_waiting_tasks(tasks);
        queue_tasks.post(tasks);

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

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_final *>(result.get()) !=
                nullptr ||
            dynamic_cast<server_task_result_embd *>(result.get()) != nullptr ||
            dynamic_cast<server_task_result_rerank *>(result.get()) != nullptr);
        const size_t idx = result->get_index();
        GGML_ASSERT(idx < results.size() && "index out of range");
        results[idx] = std::move(result);
      }

      queue_results.remove_waiting_task_ids(task_ids);
      return 0;
    } else {
      SRV_ERR("Error: %s\n", "This is not stream api!");
      queue_results.remove_waiting_task_ids(task_ids);
      return -1;
    }
    return 0;
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
    bool oaicompat_chat = false;
    try {
      std::vector<llama_tokens> tokenized_prompts =
          tokenize_input_prompts(vocab, data.at("prompt"), true, true);
      tasks.reserve(tokenized_prompts.size());
      SRV_DBG("++tokenized_prompts.size() is:%d\r\n", tokenized_prompts.size());
      for (size_t i = 0; i < tokenized_prompts.size(); i++) {
        server_task task = server_task(type);

        task.id = queue_tasks.get_new_id();
        task.index = i;

        task.prompt_tokens = std::move(tokenized_prompts[i]);
        task.params =
            server_task::params_from_json_cmpl(ctx, params_base, data);
        task.id_selected_slot = json_value(data, "id_slot", -1);

        // OAI-compat
        task.params.oaicompat = oaicompat;
        task.params.oaicompat_cmpl_id = completion_id;
        // oaicompat_model is already populated by params_from_json_cmpl

        tasks.push_back(task);
      }
    } catch (const std::exception &e) {
      log_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
      return -1;
    }
    queue_results.add_waiting_tasks(tasks);
    queue_tasks.post(tasks);
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
    SRV_INF("this: %p\n", this);
    bool stream = json_value(data, "stream", false);
    if (!stream) {
      SRV_ERR("Error: %s\n", "This is stream api!");
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
    bool oaicompat_chat = false;

    try {
      std::vector<llama_tokens> tokenized_prompts =
          tokenize_input_prompts(vocab, data.at("prompt"), true, true);
      SRV_INF("++tokenized_prompts.size() is:%d\r\n",tokenized_prompts.size());
      tasks.reserve(tokenized_prompts.size());
      for (size_t i = 0; i < tokenized_prompts.size(); i++) {
        server_task task = server_task(type);

        task.id = queue_tasks.get_new_id();
        task.index = i;

        task.prompt_tokens = std::move(tokenized_prompts[i]);
        task.params =
            server_task::params_from_json_cmpl(ctx, params_base, data);
        task.id_selected_slot = json_value(data, "id_slot", -1);

        // OAI-compat
        task.params.oaicompat = oaicompat;
        task.params.oaicompat_cmpl_id = completion_id;
        // oaicompat_model is already populated by params_from_json_cmpl
        SRV_DBG("push_back one task i=%d",i);
        tasks.push_back(task);
      }
    } catch (const std::exception &e) {
      log_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
      return -1;
    }

    queue_results.add_waiting_tasks(tasks);
    queue_tasks.post(tasks);

    for (auto it = task_ids.begin(); it != task_ids.end(); ++it) {
      std::cout << *it << " ";
    }
    task_ids = server_task::get_list_id(tasks);
    // for (auto it=task_ids.begin(); it!=task_ids.end(); ++it) {
    //     std::cout << *it << " ";
    // }
    // std::cout << std::endl;
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
      SRV_WRN("%s\n", "stop and not confirm, ignore it.");
      generate_status = 0;
      // result_str = "Stopped";
      return -1;
    }

    if (task_ids.size() == 0 && !confirm) {
      SRV_WRN("task_ids is empty and not confirm = %d , ignore it.\n",
              task_ids.size());
      generate_status = 0;
      model_status = 1;
      // result_str = "Done";
      return -1;
    }
    if (stop) {
      SRV_WRN("%s\n", "stop, ignore it.");
      generate_status = 2;
      return -1;
    }

    if (task_ids.size() == 0) {
      SRV_WRN("task_ids is empty = %d , ignore it.\n", task_ids.size());
      generate_status = 0;
      model_status = 1;
      return -1;
    }

    // auto result = queue_results.recv(task_ids);
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
    // SRV_INF("!result: %d\n", !result);
    // SRV_INF("!result2: %d\n", !result2);
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

      // if (result2->is_stop()) {
      //     SRV_DBG("result2->is_stop n_finished = %d\n", n_finished);
      //     SRV_DBG("result2->is_stop task_ids size  = %d\n", task_ids.size());
      //     if (++n_finished == task_ids.size()) {
      //         SRV_DBG("all task finished = %d\n", n_finished);
      //         queue_results.remove_waiting_task_ids(task_ids);
      //         task_ids.clear();
      //         tasks.clear();
      //         n_finished = 0;
      //         //for (auto it=task_ids.begin(); it!=task_ids.end(); ++it) {
      //         //    std::cout << *it << " ";
      //         //}
      //         //std::cout << std::endl;
      //         generate_status = 0;
      //         model_status = 1;
      //         return 0;
      //     }
      // }
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
        // for (auto it=task_ids.begin(); it!=task_ids.end(); ++it) {
        //     std::cout << *it << " ";
        // }
        // std::cout << std::endl;
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
    return 0;
  }

  // receive the results from task(s)
  int generate(const std::unordered_set<int> &id_tasks,
               std::vector<server_task_result_ptr> &results) {
    for (size_t i = 0; i < id_tasks.size(); i++) {
      server_task_result_ptr result = queue_results.recv(id_tasks);

      if (result->is_error()) {
        cancel_tasks(id_tasks);
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

    queue_results.remove_waiting_task_ids(id_tasks);
    return 0;
  }

  int generate_stream(const std::unordered_set<int> &id_tasks) {
    n_finished = 0;
    return 0;
  }

  int generate_stream_next_token(const std::unordered_set<int> &id_tasks,
                                 server_task_result_ptr &result) {
    result = queue_results.recv(id_tasks);

    if (result->is_error()) {
      cancel_tasks(id_tasks);
      return -1;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial *>(result.get()) !=
                    nullptr ||
                dynamic_cast<server_task_result_cmpl_final *>(result.get()) !=
                    nullptr);

    if (result->is_stop()) {
      if (++n_finished == id_tasks.size()) {
        SRV_DBG("all task finished = %d\n", n_finished);
        queue_results.remove_waiting_task_ids(id_tasks);
        task_ids.clear();
        tasks.clear();
        n_finished = 0;
        return 1;
      }
    }
    return 0;
  }

  //
  // Functions to process the task
  //

  void process_single_task(server_task task) {
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
          queue_tasks.defer(task);
          break;
        }
        if (slot->is_processing()) {
          // if requested slot is unavailable, we defer this task for processing
          // later
          SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n",
                  task.id);
          queue_tasks.defer(task);
          break;
        }

        if (!launch_slot_with_task(*slot, task)) {
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

        res->kv_cache_tokens_count = llama_kv_self_n_tokens(ctx);
        res->kv_cache_used_cells = llama_kv_self_used_cells(ctx);

        res->n_prompt_tokens_processed_total =
            metrics.n_prompt_tokens_processed_total;
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
          queue_tasks.defer(task);
          break;
        }

        const size_t token_count = slot->cache_tokens.size();
        const int64_t t_start = ggml_time_us();

        std::string filename = task.slot_action.filename;
        std::string filepath = task.slot_action.filepath;

        const size_t nwrite =
            llama_state_seq_save_file(ctx, filepath.c_str(), slot->id,
                                      slot->cache_tokens.data(), token_count);

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
          queue_tasks.defer(task);
          break;
        }

        const int64_t t_start = ggml_time_us();

        std::string filename = task.slot_action.filename;
        std::string filepath = task.slot_action.filepath;

        slot->cache_tokens.resize(slot->n_ctx);
        size_t token_count = 0;
        size_t nread = llama_state_seq_load_file(
            ctx, filepath.c_str(), slot->id, slot->cache_tokens.data(),
            slot->cache_tokens.size(), &token_count);
        if (nread == 0) {
          slot->cache_tokens.resize(0);
          send_error(task,
                     "Unable to restore slot, no available space in KV cache "
                     "or invalid slot save file",
                     ERROR_TYPE_INVALID_REQUEST);
          break;
        }
        slot->cache_tokens.resize(token_count);

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
          queue_tasks.defer(task);
          break;
        }

        // Erase token cache
        const size_t n_erased = slot->cache_tokens.size();
        llama_kv_self_seq_rm(ctx, slot->id, -1, -1);
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

  int houmo_decode_14B(llama_batch batch) {
    if (decode_pos == 0) {
      SRV_DBG("%s", "prefill stage\n");
      // std::vector<llama_token> tokens(batch.token, batch.token +
      // batch.n_tokens);
      std::vector<llama_token> tokens;
      copy(batch.token, batch.token + batch.n_tokens, back_inserter(tokens));
      // for(const int& i : tokens) {
      //     std::cout << "i = " << i << std::endl;
      // }
      size_t input_echo_len = tokens.size();
      // auto input_echo_len = test_tokens.size();

      // std::cout << "input_echo_len: " << input_echo_len << std::endl;
      SRV_DBG("++input_echo_len is:%d\r\n", input_echo_len);
      // std::cout << "prefill_length: " << prefill_length << std::endl;

      // int16_t prefill_loop_round = ceil(static_cast<float>(input_echo_len) /
      //                                   static_cast<float>(prefill_length));
      int16_t valid_length = 0;
      int16_t current_length = 0;
      int16_t gather_index = 0;
      if (hmctx_14B->model_type == "deepseek") {
        //TODO:
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
            hmctx_14B->prefill_model_part1_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum", input_data);
        }
        for (int i = SPLIT_COUNT; i < hmctx_14B->nblocks; ++i) {
            //auto kcache_history_sum = prefill_model_part2_.GetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum");
            //std::vector<float> zeros(kcache_history_sum.info().size(), 0.0f);

            auto input_data = prefill_part2_tensormap.at("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum");
            auto kcache_history_sum_init = torch::zeros(
              {1, (int16_t)(input_data.MemSize() / 2)}, torch::kInt16);
            int16toint16((int16_t *)input_data.Data(),
                       (int16_t *)kcache_history_sum_init.data_ptr<int16_t>(),
                       kcache_history_sum_init.numel());

            hmctx_14B->prefill_model_part2_.SetInput("model_layers_" + std::to_string(i) + "_self_attn_kcache_history_sum", input_data);
        }
        printf("@@1+with kvcache clear\r\n");
        // 预填充阶段
        int prefill_loop_round = std::ceil(static_cast<double>(input_echo_len) / hmctx_14B->prefill_length);
        printf("++prefill_loop_round is:%d\r\n",prefill_loop_round);
        for (int round = 0; round < prefill_loop_round; ++round) {
            int16_t valid_length = round * hmctx_14B->prefill_length;
            int16_t current_length;
            std::vector<int32_t> input_ids;

            if (round == prefill_loop_round - 1) {
                current_length = input_echo_len - round * hmctx_14B->prefill_length;
                input_ids.assign(tokens.begin() + round * hmctx_14B->prefill_length, tokens.end());
            } else {
                current_length = hmctx_14B->prefill_length;
                input_ids.assign(tokens.begin() + round * hmctx_14B->prefill_length,
                                tokens.begin() + (round + 1) * hmctx_14B->prefill_length);
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
            torch::Tensor pad_embeds = torch::zeros({hmctx_14B->prefill_length - effective_length, inputs_embeds.size(-1)}, inputs_embeds.options());
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

            //torch::Tensor input_data = torch::cat({inputs_embeds, pad_embeds}, 0).reshape({4, hmctx_14B->prefill_length / 4, 5120});
            torch::Tensor input_data = torch::cat({inputs_embeds, pad_embeds});
            input_data=input_data.view({4, hmctx_14B->prefill_length / 4, 5120}).to(torch::kInt16);
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
            hmctx_14B->prefill_model_part1_.SetInput("input_1",prefill_part1_tensormap.at("input_1"));

            TensorConverter::FromVector(valid_length_data, prefill_part1_tensormap.at("valid_length"));
            hmctx_14B->prefill_model_part1_.SetInput("valid_length",  prefill_part1_tensormap.at("valid_length"));

            TensorConverter::FromVector(current_length_data, prefill_part1_tensormap.at("current_length"));
            hmctx_14B->prefill_model_part1_.SetInput("current_length", prefill_part1_tensormap.at("current_length"));
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

              hmctx_14B->prefill_model_part1_.SetInput("input_1",prefill_part1_input_data);
              hmctx_14B->prefill_model_part1_.SetInput("valid_length",  prefill_valid_length);
              hmctx_14B->prefill_model_part1_.SetInput("current_length", prefill_current_length);
            #endif
            printf("@@4\r\n");
            if (1) {
                printf("+++call prefill part1 run+sync\r\n");
                hmctx_14B->prefill_model_part1_.Run();
                hmctx_14B->prefill_model_part1_.Sync();
                printf("@@4.1\r\n");
                tcim::Tensor prefill_part1_output = hmctx_14B->prefill_model_part1_.GetOutput("model_layers_25_resadd2");
                //打印prefill1输出-每隔512个打印输出
                //print_tcimtensor(prefill_part1_output);
                printf("@@4.2\r\n");
                hmctx_14B->prefill_model_part2_.SetInput("model_layers_25_resadd2", prefill_part1_output);
                printf("@@4.3\r\n");

                TensorConverter::FromVector(valid_length_data, prefill_part2_tensormap.at("valid_length"));
                hmctx_14B->prefill_model_part2_.SetInput("valid_length", prefill_part2_tensormap.at("valid_length"));
                printf("@@4.4\r\n");
                TensorConverter::FromVector(current_length_data, prefill_part2_tensormap.at("current_length"));
                hmctx_14B->prefill_model_part2_.SetInput("current_length", prefill_part2_tensormap.at("current_length"));
                printf("@@4.5\r\n");
                printf("+++call prefill part2 run+sync\r\n");
                hmctx_14B->prefill_model_part2_.Run();
                hmctx_14B->prefill_model_part2_.Sync();
                printf("@@4.6\r\n");
            }
        }
        printf("@@5\r\n");
        // 获取预填充结果
        tcim::Tensor output = hmctx_14B->prefill_model_part2_.GetOutput("Output_lm_head_add_list_1");
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
        next_id = *(int32_t *)selected_output.data_ptr<int32_t>();
        printf("++selected next_id is:%d\r\n",next_id);
        #endif

      // std::cout << "selected_output: " << selected_output << std::endl;
      // 下面打印的selected_output是乱码，index
      // int值，调下面的common_token_to_piece转成字符串
      SRV_DBG("++selected_output(tensor addr not selected index) is:%u\r\n", selected_output);
      auto prefill_token = *(int32_t *)selected_output.data_ptr<int32_t>();
      SRV_DBG("++selected tensor index is:%d\r\n",*(int32_t *)selected_output.data_ptr<int32_t>());
      auto response = common_token_to_piece(ctx, prefill_token);
      std::cout << "rrefill response: ";
      auto invalid_utf8 = false;
      write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
      std::cout << std::endl;
      // 下面打印的是正常的response 字符串
      SRV_DBG("++prefill response is:%s\r\n", response);

      context_length = input_echo_len;

      std::vector<llama_token> prefill_input_ids = {prefill_token};
      torch::Tensor prefill_inputs_embeds;
      auto prefill_opts = torch::TensorOptions().dtype(torch::kInt);
      auto prefill_inp =
          torch::from_blob(prefill_input_ids.data(),
                           {int64_t(prefill_input_ids.size())}, prefill_opts)
              .clone();
      // std::cout << "prefill_inp: " << prefill_inp << "\n";
      SRV_DBG("++prefill_inp is:%s\r\n", prefill_inp);
      try {
        prefill_inputs_embeds = emb->forward(prefill_inp);
      } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n" << e.what_without_backtrace();
        return -1;
      }
      // std::cout << "prefill_inputs_embeds options: " <<
      // prefill_inputs_embeds.options() << std::endl; std::cout <<
      // "prefill_inputs_embeds dim: " << prefill_inputs_embeds.dim() <<
      // std::endl; std::cout << "prefill_inputs_embeds size: " <<
      // prefill_inputs_embeds.size(0) << std::endl; for (int i = 0; i < 3; i++)
      // {
      //     std::cout << prefill_inputs_embeds[0][i] << " ";
      // }
      // std::cout << std::endl;
      prefill_inputs_embeds =
          prefill_inputs_embeds.view({1, 1, -1}).to(torch::kInt16);
      // std::cout << "prefill_inputs_embeds options: " <<
      // prefill_inputs_embeds.options() << std::endl; std::cout <<
      // "prefill_inputs_embeds dim: " << prefill_inputs_embeds.dim() <<
      // std::endl; std::cout << "prefill_inputs_embeds size: " <<
      // prefill_inputs_embeds.size(0) << std::endl;

      input_emd = prefill_inputs_embeds;
      // write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
      // std::cout.flush();
      token = prefill_token;
      decode_pos = batch.n_tokens;
      SRV_DBG("prefill stage:decode_pos is :%d \r\n",decode_pos);
    } else {
      SRV_DBG("%s", "decode stage\n");
      printf("@@7\r\n");
      // 设置并运行解码模型
      TensorConverter::FromTorch(input_emd.to(torch::kInt16),decode_part1_tensormap.at("input_1"));
      hmctx_14B->decode_model_part1_.SetInput("input_1", decode_part1_tensormap.at("input_1"));
      std::vector<int16_t> valid_length_data = {static_cast<int16_t>(context_length)};
      TensorConverter::FromVector(valid_length_data, decode_part1_tensormap.at("valid_length"));
      hmctx_14B->decode_model_part1_.SetInput("valid_length", decode_part1_tensormap.at("valid_length"));

      printf("@@8\r\n");

      hmctx_14B->decode_model_part1_.Run();
      hmctx_14B->decode_model_part1_.Sync();
      tcim::Tensor decode_part1_output = hmctx_14B->decode_model_part1_.GetOutput("model_layers_25_resadd2");

      hmctx_14B->decode_model_part2_.SetInput("model_layers_25_resadd2", decode_part1_output);
      TensorConverter::FromVector(valid_length_data, decode_part2_tensormap.at("valid_length"));
      hmctx_14B->decode_model_part2_.SetInput("valid_length", decode_part2_tensormap.at("valid_length"));

      hmctx_14B->decode_model_part2_.Run();
      hmctx_14B->decode_model_part2_.Sync();
      tcim::Tensor decode_data = hmctx_14B->decode_model_part2_.GetOutput("Output_lm_head_add_list_1");
      //output_data.assign(decode_output.data(), decode_output.data() + decode_output.size());
      //output_data=TensorConverter::ToVector(decode_output);

      //7B 获取输出
      // infer_end = clock();
      //tcim::Tensor decode_data;
      //tcim::Tensor decode_output_tensor_float;
      //TODO:szj
      // if (hmctx->model_type == "deepseek") {
      //   decode_data =
      //       hmctx->decode_model.GetOutput("Output_lm_head_add_list_0");
      // } else {
      //   decode_data = hmctx->decode_model.GetOutput("output");
      // }

      // std::cout << "decode_data Size: " << decode_data.Size() << std::endl;
      // std::this_thread::sleep_for(std::chrono::seconds(5));
      #if 0 //old int to float ,not right
      std::vector<int16_t> decode_output;
      for (int idx = 0; idx < decode_data.MemSize() / 2; idx++) {
        auto data = *((int16_t *)decode_data.Data() + idx);
        // if (data != 0) {
        //     std::cout << "data at " << idx << ": " << data << std::endl;
        // }

        decode_output.push_back(data);
      }
      SRV_DBG("vector<int16_t> decode_output.size is:%d\r\n ",decode_output.size());
      auto decode_output_opts = torch::TensorOptions().dtype(torch::kInt16);
      decode_output_tensor =
          torch::from_blob(decode_output.data(),
                           {int64_t(decode_output.size())}, decode_output_opts)
              .clone();
      #else
        auto output_info = decode_data.Info().AsContiguous().AsType(tcim::FLOAT32);
        std::cout << "Output info:" << output_info << std::endl;
        auto decode_output_tensor_float = tcim::Tensor::CreateHostTensor(output_info);
        decode_data.CastTo(decode_output_tensor_float);
        std::vector<float> decode_output;
        float logits_sum=0;
        for (int idx = 0; idx < decode_output_tensor_float.MemSize()/4 ; idx++) {
          float data = *((float *)decode_output_tensor_float.Data() + idx);
          logits_sum+=data;
          #if PRINT_DEBUGINFO
          if (std::fabs(data) > 0.01) {
              std::cout << "float data at " << idx << ": " << data << std::endl;
          }
          #endif
          decode_output.push_back(data);
        }
        printf("++logits_sum is:%f\r\n",logits_sum);
        SRV_DBG("float decode_output.size() is:%d \r\n",decode_output.size());
        auto decode_output_opts = torch::TensorOptions().dtype(torch::kFloat);
        auto decode_output_tensor =
          torch::from_blob(decode_output.data(),
                           {int64_t(decode_output.size())}, decode_output_opts)
              .clone();
      #endif

      // std::cout << "decode_output_tensor options: " <<
      // decode_output_tensor.options() << std::endl; std::cout <<
      // "decode_output_tensor dim: " << decode_output_tensor.dim() <<
      // std::endl; std::cout << "decode_output_tensor size: " <<
      // decode_output_tensor.size(0) << std::endl; for (int i = 0; i < 3; i++)
      // {
      //     std::cout << decode_output_tensor[i] << " ";
      // }
      // std::cout << std::endl;
      // for (int i = 152060; i < 152064; i++) {
      //     std::cout << decode_output_tensor[i] << " ";
      // }
      // std::cout << std::endl;
      int32_t generate_token=0;
      //default默认的处理流程-没有复杂的后采样
      if(0) {
        auto selected_output =
            torch::argmax(decode_output_tensor).to(torch::kInt32);
        // std::cout << "selected_output: " << selected_output << std::endl;
        generate_token = *(int32_t *)selected_output.data_ptr<int32_t>();
      } else {
        #if 1 //TODO remember delete ,only for compare token between argmax and houmo sampling
        auto selected_output_argmax =
            torch::argmax(decode_output_tensor).to(torch::kInt32);
        // std::cout << "selected_output: " << selected_output << std::endl;
        auto argmax_generate_token = *(int32_t *)selected_output_argmax.data_ptr<int32_t>();
        SRV_DBG("++before houmosample argmax_generate_token=%d \r\n",argmax_generate_token);
        #endif

        //float test_nouse=0.1234;
        //float *logits =&test_nouse;                         // 获取模型输出的logits
        int n_vocab = decode_output.size();                           // 词汇表大小
        //std::vector<llama_token> last_tokens = {};  // 最近生成的token

        //torch::Tensor decode_floatTensor = decode_output_tensor.to(torch::kFloat);
        //float *p_logits=(float *)decode_floatTensor.data_ptr<float>();

        auto *p_logits= decode_output_tensor.data_ptr<float>();
        for (auto &slot : slots) {
          if (!slot.generated_tokens.empty()) {
            SRV_DBG("%s\r\n","++call hm_sampling_processors process");
            SRV_DBG("HI.CPP >>> [%d][%d] slot.params top_k: %d, top_p: %f, temp: %f, penalty_repeat: %f, penalty_last_n: %d \r\n",
                slot.id, slot.id_task, slot.params.sampling.top_k, slot.params.sampling.top_p, slot.params.sampling.temp,
                slot.params.sampling.penalty_repeat, slot.params.sampling.penalty_last_n);
            hm_sampling_processors.process(p_logits, n_vocab, slot.generated_tokens, slot.params);
          }
          else{
            SRV_ERR("slot.generated_tokens.size() is:%d ,dont need sample\r\n",slot.generated_tokens.size());
          }
        }
        // 计算softmax概率
        std::vector<float> probs(n_vocab);
        softmax(p_logits, probs.data(), n_vocab);
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto tensor = torch::from_blob(probs.data(), {int64_t(probs.size())}, opts).clone();

        auto selected_output = torch::multinomial(tensor, 1).to(torch::kInt32);
        generate_token = *(int32_t *)selected_output.data_ptr<int32_t>();
        #if 1 //TODO remember delete ,only for compare token between argmax and houmo sampling
        SRV_DBG("++after houmosample generate_token=%d \r\n",generate_token);
        if(generate_token!=argmax_generate_token) {
           auto houmo_response = common_token_to_piece(ctx, generate_token);
           auto argmax_response= common_token_to_piece(ctx, argmax_generate_token);
           SRV_DBG("Houmo sample change gen idx from %d to %d , related string from %s to %s\r\n",
                 argmax_generate_token,generate_token,argmax_response,houmo_response);
        }
        #endif
      }
      // auto response = common_token_to_piece(ctx, generate_token);
      // std::cout << "response: ";
      // auto invalid_utf8 = false;
      // write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
      // std::cout.flush();
      // std::cout << std::endl;

      std::vector<llama_token> decode_input_ids = {generate_token};
      torch::Tensor decode_inputs_embeds;
      auto decode_opts = torch::TensorOptions().dtype(torch::kInt);
      auto decode_inp =
          torch::from_blob(decode_input_ids.data(),
                           {int64_t(decode_input_ids.size())}, decode_opts)
              .clone();
      // std::cout << "decode_inp: " << decode_inp << "\n";
      try {
        input_emd = emb->forward(decode_inp);
      } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n" << e.what_without_backtrace();
        return -1;
      }
      // std::cout << "input_emd options: " << input_emd.options() << std::endl;
      // std::cout << "input_emd dim: " << input_emd.dim() << std::endl;
      // std::cout << "input_emd size: " << input_emd.size(0) << std::endl;
      // for (int i = 0; i < 3; i++) {
      //     std::cout << input_emd[0][i] << " ";
      // }
      // std::cout << std::endl;
      input_emd = input_emd.view({1, 1, -1}).to(torch::kInt16);
      // std::cout << "input_emd options: " << input_emd.options() << std::endl;
      // std::cout << "input_emd dim: " << input_emd.dim() << std::endl;
      // std::cout << "input_emd size: " << input_emd.size(0) << std::endl;
      context_length = context_length + 1;
      if (llama_token_is_eog(vocab, generate_token)) {
        LOG_DBG("llama_token_is_eog: %d\n", generate_token);
      }
      token = generate_token;
      decode_pos = decode_pos + batch.n_tokens;
      SRV_DBG("decode stage:decode_pos is :%d \r\n",decode_pos);

      auto response = common_token_to_piece(ctx, generate_token);
      std::cout << "decode response: ";
      bool invalid_utf8 = false;
      write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
      std::cout << std::endl;
      // 下面打印的是正常的response 字符串
      SRV_DBG("--decode response is:%s\r\n", response);
    }

    return 0;
  }

  int houmo_decode_8B(llama_batch batch) {
    SRV_DBG("%s\r\n","++call houmo_decode_8B");
    if (decode_pos == 0) {
      SRV_DBG("%s", "prefill stage\n");
      // std::vector<llama_token> tokens(batch.token, batch.token +
      // batch.n_tokens);
      std::vector<llama_token> tokens;
      copy(batch.token, batch.token + batch.n_tokens, back_inserter(tokens));
      // for(const int& i : tokens) {
      //     std::cout << "i = " << i << std::endl;
      // }
      size_t input_echo_len = tokens.size();
      // auto input_echo_len = test_tokens.size();

      int16_t prefill_length = 256;
      // std::cout << "input_echo_len: " << input_echo_len << std::endl;
      SRV_DBG("++input_echo_len is:%d\r\n", input_echo_len);
      // std::cout << "prefill_length: " << prefill_length << std::endl;

      int16_t prefill_loop_round = ceil(static_cast<float>(input_echo_len) /
                                        static_cast<float>(prefill_length));
      int16_t valid_length = 0;
      int16_t current_length = 0;
      int16_t gather_index = 0;

      // std::cout << "prefill_loop_round: " << prefill_loop_round << std::endl;
      std::vector<llama_token> input_ids;
      for (int round = 0; round < prefill_loop_round; round++) {
        valid_length = round * prefill_length;
        if (round == prefill_loop_round - 1) {
          current_length = input_echo_len - round * prefill_length;
          gather_index = current_length - 1;
          std::vector<llama_token>::const_iterator first =
              tokens.begin() + round * prefill_length;
          std::vector<llama_token>::const_iterator last =
              tokens.begin() + input_echo_len;
          std::vector<llama_token> round_tokens(first, last);
          input_ids = round_tokens;
        } else {
          current_length = prefill_length;
          std::vector<llama_token>::const_iterator first =
              tokens.begin() + round * prefill_length;
          std::vector<llama_token>::const_iterator last =
              tokens.begin() + (round + 1) * prefill_length;
          std::vector<llama_token> round_tokens(first, last);
          input_ids = round_tokens;
        }
        // std::cout << "input_ids: " << input_ids << std::endl;
        torch::Tensor inputs_embeds;
        auto opts = torch::TensorOptions().dtype(torch::kInt);
        auto inp = torch::from_blob(input_ids.data(),
                                    {int64_t(input_ids.size())}, opts)
                       .clone();
        // std::cout << "inp: " << inp << "\n";
        try {
          inputs_embeds = emb->forward(inp);
        } catch (const c10::Error &e) {
          std::cerr << "error loading the model\n"
                    << e.what_without_backtrace();
          return -1;
        }

        auto effective_length = input_ids.size();
        auto pad_embeds = torch::zeros(
            {(int64_t)(prefill_length - effective_length), (int64_t)4096},
            torch::kInt16);
        auto padded_embeds = torch::cat({inputs_embeds, pad_embeds});
        padded_embeds =
            padded_embeds.view({4, prefill_length / 4, 4096}).to(torch::kInt16);

        auto input_data = prefill_input_map.at("input_1");
        int16toint16((int16_t *)input_data.Data(),
                     (int16_t *)padded_embeds.data_ptr<int16_t>(),
                     padded_embeds.numel());

        std::vector<int16_t> valid_length_input = {valid_length};
        auto valid_length_input_tensor_opts =
            torch::TensorOptions().dtype(torch::kInt16);
        auto valid_length_input_tensor =
            torch::from_blob(valid_length_input.data(),
                             {int64_t(valid_length_input.size())},
                             valid_length_input_tensor_opts)
                .clone();

        // std::cout << "valid_length_input_tensor: " <<
        // valid_length_input_tensor << std::endl;
        auto prefill_valid_length = prefill_input_map.at("valid_length");
        int16toint16((int16_t *)prefill_valid_length.Data(),
                     (int16_t *)valid_length_input_tensor.data_ptr<int16_t>(),
                     valid_length_input_tensor.numel());

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
        auto prefill_current_length = prefill_input_map.at("current_length");
        int16toint16((int16_t *)prefill_current_length.Data(),
                     (int16_t *)current_length_input_tensor.data_ptr<int16_t>(),
                     current_length_input_tensor.numel());

        hmctx->prefill_model.SetInput("input_1", input_data);
        hmctx->prefill_model.SetInput("valid_length", prefill_valid_length);
        hmctx->prefill_model.SetInput("current_length", prefill_current_length);
        hmctx->prefill_model.Run();
        hmctx->prefill_model.Sync();
        // std::cout << "prefill run finished" << std::endl;
      }

      tcim::Tensor input_data;
      //if (hmctx->model_type == "deepseek") {
      if (1) {
        input_data =
            hmctx->prefill_model.GetOutput("Output_lm_head_add_list_0");
      } else {
        input_data = hmctx->prefill_model.GetOutput("output");
      }

      SRV_DBG("input_data.info.memSize:%d  input_data.DataType() is:%d\r\n",input_data.Info().MemSize(),input_data.Info().DataType());
      // std::this_thread::sleep_for(std::chrono::seconds(5));
      std::vector<int16_t> prefill_output;
      for (int idx = 0; idx < input_data.MemSize() / 2; idx++) {
        auto data = *((int16_t *)input_data.Data() + idx);
        // if (data != 0) {
        //     std::cout << "data at " << idx << ": " << data << std::endl;
        // }

        prefill_output.push_back(data);
      }
      SRV_DBG("vector<int16_t> prefill_output.size is:%d\r\n ",prefill_output.size());

      auto prefill_output_opts = torch::TensorOptions().dtype(torch::kInt16);
      auto prefill_output_tensor =
          torch::from_blob(prefill_output.data(),
                           {int64_t(prefill_output.size())},
                           prefill_output_opts)
              .clone();
      // std::cout << "prefill_output options: " <<
      // prefill_output_tensor.options() << std::endl; std::cout <<
      // "prefill_output dim: " << prefill_output_tensor.dim() << std::endl;
      // std::cout << "prefill_output size: " << prefill_output_tensor.size(0)
      // << std::endl; for (int i = 0; i < 3; i++) {
      //     std::cout << prefill_output_tensor[i] << " ";
      // }
      // std::cout << std::endl;
      // for (int i = 152060; i < 152064; i++) {
      //     std::cout << prefill_output_tensor[i] << " ";
      // }
      // std::cout << std::endl;
      auto selected_output =
          torch::argmax(prefill_output_tensor).to(torch::kInt32);
      // std::cout << "selected_output: " << selected_output << std::endl;
      // 下面打印的selected_output是乱码，index
      // int值，调下面的common_token_to_piece转成字符串
      SRV_DBG("++selected_output(tensor addr not selected index) is:%u\r\n", selected_output);
      auto prefill_token = *(int32_t *)selected_output.data_ptr<int32_t>();
      SRV_DBG("++selected tensor index is:%d\r\n",*(int32_t *)selected_output.data_ptr<int32_t>());
      auto response = common_token_to_piece(ctx, prefill_token);
      std::cout << "rrefill response: ";
      auto invalid_utf8 = false;
      write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
      std::cout << std::endl;
      // 下面打印的是正常的response 字符串
      SRV_DBG("++prefill response is:%s\r\n", response);

      context_length = input_echo_len;

      std::vector<llama_token> prefill_input_ids = {prefill_token};
      torch::Tensor prefill_inputs_embeds;
      auto prefill_opts = torch::TensorOptions().dtype(torch::kInt);
      auto prefill_inp =
          torch::from_blob(prefill_input_ids.data(),
                           {int64_t(prefill_input_ids.size())}, prefill_opts)
              .clone();
      // std::cout << "prefill_inp: " << prefill_inp << "\n";
      SRV_DBG("++prefill_inp is:%s\r\n", prefill_inp);
      try {
        prefill_inputs_embeds = emb->forward(prefill_inp);
      } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n" << e.what_without_backtrace();
        return -1;
      }
      // std::cout << "prefill_inputs_embeds options: " <<
      // prefill_inputs_embeds.options() << std::endl; std::cout <<
      // "prefill_inputs_embeds dim: " << prefill_inputs_embeds.dim() <<
      // std::endl; std::cout << "prefill_inputs_embeds size: " <<
      // prefill_inputs_embeds.size(0) << std::endl; for (int i = 0; i < 3; i++)
      // {
      //     std::cout << prefill_inputs_embeds[0][i] << " ";
      // }
      // std::cout << std::endl;
      prefill_inputs_embeds =
          prefill_inputs_embeds.view({1, 1, -1}).to(torch::kInt16);
      // std::cout << "prefill_inputs_embeds options: " <<
      // prefill_inputs_embeds.options() << std::endl; std::cout <<
      // "prefill_inputs_embeds dim: " << prefill_inputs_embeds.dim() <<
      // std::endl; std::cout << "prefill_inputs_embeds size: " <<
      // prefill_inputs_embeds.size(0) << std::endl;

      input_emd = prefill_inputs_embeds;
      // write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
      // std::cout.flush();
      token = prefill_token;
      decode_pos = batch.n_tokens;
      SRV_DBG("prefill stage:decode_pos is :%d \r\n",decode_pos);
    } else {
      SRV_DBG("%s", "decode stage\n");
      auto invalid_utf8 = false;
      auto input_data = decode_input_map.at("input_1");
      int16toint16((int16_t *)input_data.Data(),
                   (int16_t *)input_emd.data_ptr<int16_t>(), input_emd.numel());

      std::vector<int16_t> valid_length_input = {(int16_t)(context_length - 1)};
      auto valid_length_input_tensor_opts =
          torch::TensorOptions().dtype(torch::kInt16);
      auto valid_length_input_tensor =
          torch::from_blob(valid_length_input.data(),
                           {int64_t(valid_length_input.size())},
                           valid_length_input_tensor_opts)
              .clone();

      // std::cout << "valid_length_input_tensor: " << valid_length_input_tensor
      // << std::endl;
      auto decode_valid_length = decode_input_map.at("valid_length");
      int16toint16((int16_t *)decode_valid_length.Data(),
                   (int16_t *)valid_length_input_tensor.data_ptr<int16_t>(),
                   valid_length_input_tensor.numel());

      hmctx->decode_model.SetInput("input_1", input_data);
      hmctx->decode_model.SetInput("valid_length", decode_valid_length);

      // clock_t infer_start,infer_end;
      // infer_start = clock();
      hmctx->decode_model.Run();
      hmctx->decode_model.Sync();
      // infer_end = clock();
      tcim::Tensor decode_data;
      tcim::Tensor decode_output_tensor_float;
      //if (hmctx->model_type == "deepseek") {
      if (1) {
        decode_data =
            hmctx->decode_model.GetOutput("Output_lm_head_add_list_0");
      } else {
        decode_data = hmctx->decode_model.GetOutput("output");
      }

      // std::cout << "decode_data Size: " << decode_data.Size() << std::endl;
      // std::this_thread::sleep_for(std::chrono::seconds(5));
      #if 0 //old int to float ,not right
      std::vector<int16_t> decode_output;
      for (int idx = 0; idx < decode_data.MemSize() / 2; idx++) {
        auto data = *((int16_t *)decode_data.Data() + idx);
        // if (data != 0) {
        //     std::cout << "data at " << idx << ": " << data << std::endl;
        // }

        decode_output.push_back(data);
      }
      SRV_DBG("vector<int16_t> decode_output.size is:%d\r\n ",decode_output.size());
      auto decode_output_opts = torch::TensorOptions().dtype(torch::kInt16);
      decode_output_tensor =
          torch::from_blob(decode_output.data(),
                           {int64_t(decode_output.size())}, decode_output_opts)
              .clone();
      #else
        auto output_info = decode_data.Info().AsContiguous().AsType(tcim::FLOAT32);
        std::cout << "Output info:" << output_info << std::endl;
        decode_output_tensor_float = tcim::Tensor::CreateHostTensor(output_info);
        decode_data.CastTo(decode_output_tensor_float);
        std::vector<float> decode_output;
        for (int idx = 0; idx < decode_output_tensor_float.MemSize()/4 ; idx++) {
          auto data = *((float *)decode_output_tensor_float.Data() + idx);
          // if (data != 0) {
          //     std::cout << "data at " << idx << ": " << data << std::endl;
          // }
          decode_output.push_back(data);
        }
        SRV_DBG("float decode_output.size() is:%d \r\n",decode_output.size());
        auto decode_output_opts = torch::TensorOptions().dtype(torch::kFloat);
        auto decode_output_tensor =
          torch::from_blob(decode_output.data(),
                           {int64_t(decode_output.size())}, decode_output_opts)
              .clone();
      #endif

      // std::cout << "decode_output_tensor options: " <<
      // decode_output_tensor.options() << std::endl; std::cout <<
      // "decode_output_tensor dim: " << decode_output_tensor.dim() <<
      // std::endl; std::cout << "decode_output_tensor size: " <<
      // decode_output_tensor.size(0) << std::endl; for (int i = 0; i < 3; i++)
      // {
      //     std::cout << decode_output_tensor[i] << " ";
      // }
      // std::cout << std::endl;
      // for (int i = 152060; i < 152064; i++) {
      //     std::cout << decode_output_tensor[i] << " ";
      // }
      // std::cout << std::endl;
      int32_t generate_token=0;
      //default默认的处理流程-没有复杂的后采样
      if(0) {
        auto selected_output =
            torch::argmax(decode_output_tensor).to(torch::kInt32);
        // std::cout << "selected_output: " << selected_output << std::endl;
        generate_token = *(int32_t *)selected_output.data_ptr<int32_t>();
      } else {
        #if 1 //TODO remember delete ,only for compare token between argmax and houmo sampling
        auto selected_output_argmax =
            torch::argmax(decode_output_tensor).to(torch::kInt32);
        // std::cout << "selected_output: " << selected_output << std::endl;
        auto argmax_generate_token = *(int32_t *)selected_output_argmax.data_ptr<int32_t>();
        SRV_DBG("++before houmosample argmax_generate_token=%d \r\n",argmax_generate_token);
        #endif

        //float test_nouse=0.1234;
        //float *logits =&test_nouse;                         // 获取模型输出的logits
        int n_vocab = decode_output.size();                           // 词汇表大小
        //std::vector<llama_token> last_tokens = {};  // 最近生成的token

        //torch::Tensor decode_floatTensor = decode_output_tensor.to(torch::kFloat);
        //float *p_logits=(float *)decode_floatTensor.data_ptr<float>();

        auto *p_logits= decode_output_tensor.data_ptr<float>();
        for (auto &slot : slots) {
          if (!slot.generated_tokens.empty()) {
            SRV_DBG("%s\r\n","++call hm_sampling_processors process");
            SRV_DBG("HI.CPP >>> [%d][%d] slot.params top_k: %d, top_p: %f, temp: %f, penalty_repeat: %f, penalty_last_n: %d \r\n",
                slot.id, slot.id_task, slot.params.sampling.top_k, slot.params.sampling.top_p, slot.params.sampling.temp,
                slot.params.sampling.penalty_repeat, slot.params.sampling.penalty_last_n);
            hm_sampling_processors.process(p_logits, n_vocab, slot.generated_tokens, slot.params);
          }
          else{
            SRV_ERR("slot.generated_tokens.size() is:%d ,dont need sample\r\n",slot.generated_tokens.size());
          }
        }
        // 计算softmax概率
        std::vector<float> probs(n_vocab);
        softmax(p_logits, probs.data(), n_vocab);
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto tensor = torch::from_blob(probs.data(), {int64_t(probs.size())}, opts).clone();

        auto selected_output = torch::multinomial(tensor, 1).to(torch::kInt32);
        generate_token = *(int32_t *)selected_output.data_ptr<int32_t>();
        #if 1 //TODO remember delete ,only for compare token between argmax and houmo sampling
        SRV_DBG("++after houmosample generate_token=%d \r\n",generate_token);
        if(generate_token!=argmax_generate_token) {
           auto houmo_response = common_token_to_piece(ctx, generate_token);
           auto argmax_response= common_token_to_piece(ctx, argmax_generate_token);
           SRV_DBG("Houmo sample change gen idx from %d to %d , related string from %s to %s\r\n",
                 argmax_generate_token,generate_token,argmax_response,houmo_response);
        }
        #endif
      }
      // auto response = common_token_to_piece(ctx, generate_token);
      // std::cout << "response: ";
      // auto invalid_utf8 = false;
      // write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
      // std::cout.flush();
      // std::cout << std::endl;

      std::vector<llama_token> decode_input_ids = {generate_token};
      torch::Tensor decode_inputs_embeds;
      auto decode_opts = torch::TensorOptions().dtype(torch::kInt);
      auto decode_inp =
          torch::from_blob(decode_input_ids.data(),
                           {int64_t(decode_input_ids.size())}, decode_opts)
              .clone();
      // std::cout << "decode_inp: " << decode_inp << "\n";
      try {
        input_emd = emb->forward(decode_inp);
      } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n" << e.what_without_backtrace();
        return -1;
      }
      // std::cout << "input_emd options: " << input_emd.options() << std::endl;
      // std::cout << "input_emd dim: " << input_emd.dim() << std::endl;
      // std::cout << "input_emd size: " << input_emd.size(0) << std::endl;
      // for (int i = 0; i < 3; i++) {
      //     std::cout << input_emd[0][i] << " ";
      // }
      // std::cout << std::endl;
      input_emd = input_emd.view({1, 1, -1}).to(torch::kInt16);
      // std::cout << "input_emd options: " << input_emd.options() << std::endl;
      // std::cout << "input_emd dim: " << input_emd.dim() << std::endl;
      // std::cout << "input_emd size: " << input_emd.size(0) << std::endl;
      context_length = context_length + 1;
      if (llama_token_is_eog(vocab, generate_token)) {
        LOG_DBG("llama_token_is_eog: %d\n", generate_token);
      }
      token = generate_token;
      decode_pos = decode_pos + batch.n_tokens;
      SRV_DBG("decode stage:decode_pos is :%d \r\n",decode_pos);

      auto response = common_token_to_piece(ctx, generate_token);
      std::cout << "decode response: ";
      invalid_utf8 = false;
      write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
      std::cout << std::endl;
      // 下面打印的是正常的response 字符串
      SRV_DBG("--decode response is:%s\r\n", response);
    }

    return 0;
  }

  int houmo_decode(llama_batch batch) {
    if (decode_pos == 0) {
      SRV_DBG("%s", "prefill stage\n");
      // std::vector<llama_token> tokens(batch.token, batch.token +
      // batch.n_tokens);
      std::vector<llama_token> tokens;
      copy(batch.token, batch.token + batch.n_tokens, back_inserter(tokens));
      // for(const int& i : tokens) {
      //     std::cout << "i = " << i << std::endl;
      // }
      size_t input_echo_len = tokens.size();
      // auto input_echo_len = test_tokens.size();

      int16_t prefill_length = 256;
      // std::cout << "input_echo_len: " << input_echo_len << std::endl;
      SRV_DBG("++input_echo_len is:%d\r\n", input_echo_len);
      // std::cout << "prefill_length: " << prefill_length << std::endl;

      int16_t prefill_loop_round = ceil(static_cast<float>(input_echo_len) /
                                        static_cast<float>(prefill_length));
      int16_t valid_length = 0;
      int16_t current_length = 0;
      int16_t gather_index = 0;
      if (hmctx->model_type == "deepseek") {
        for (int idx = 0; idx < hmctx->nblocks; idx++) {
          std::string kcache_history_sum_name = hm_string_format(
              "model_layers_%d_self_attn_kcache_history_sum", idx);
          auto kcache_history_sum =
              hmctx->prefill_model.GetInput(kcache_history_sum_name);
          auto input_data = prefill_input_map.at(kcache_history_sum_name);
          auto kcache_history_sum_init = torch::zeros(
              {1, (int16_t)(input_data.MemSize() / 2)}, torch::kInt16);

          int16toint16((int16_t *)input_data.Data(),
                       (int16_t *)kcache_history_sum_init.data_ptr<int16_t>(),
                       kcache_history_sum_init.numel());
          hmctx->decode_model.SetInput(kcache_history_sum_name, input_data);
        }
      }
      // std::cout << "prefill_loop_round: " << prefill_loop_round << std::endl;
      std::vector<llama_token> input_ids;
      for (int round = 0; round < prefill_loop_round; round++) {
        valid_length = round * prefill_length;
        if (round == prefill_loop_round - 1) {
          current_length = input_echo_len - round * prefill_length;
          gather_index = current_length - 1;
          std::vector<llama_token>::const_iterator first =
              tokens.begin() + round * prefill_length;
          std::vector<llama_token>::const_iterator last =
              tokens.begin() + input_echo_len;
          std::vector<llama_token> round_tokens(first, last);
          input_ids = round_tokens;
        } else {
          current_length = prefill_length;
          std::vector<llama_token>::const_iterator first =
              tokens.begin() + round * prefill_length;
          std::vector<llama_token>::const_iterator last =
              tokens.begin() + (round + 1) * prefill_length;
          std::vector<llama_token> round_tokens(first, last);
          input_ids = round_tokens;
        }
        // std::cout << "input_ids: " << input_ids << std::endl;
        torch::Tensor inputs_embeds;
        auto opts = torch::TensorOptions().dtype(torch::kInt);
        auto inp = torch::from_blob(input_ids.data(),
                                    {int64_t(input_ids.size())}, opts)
                       .clone();
        // std::cout << "inp: " << inp << "\n";
        try {
          inputs_embeds = emb->forward(inp);
        } catch (const c10::Error &e) {
          std::cerr << "error loading the model\n"
                    << e.what_without_backtrace();
          return -1;
        }
        // std::cout << "inputs_embeds options: " << inputs_embeds.options() <<
        // std::endl; std::cout << "inputs_embeds dim: " << inputs_embeds.dim()
        // << std::endl; std::cout << "inputs_embeds size: " <<
        // inputs_embeds.size(0) << std::endl; std::cout << "output: " << output
        // << "\n"; for (int i = 0; i < 5; i++) {
        //     std::cout << inputs_embeds[0][i] << " ";
        // }
        // std::cout << std::endl;

        auto effective_length = input_ids.size();
        auto pad_embeds = torch::zeros(
            {(int64_t)(prefill_length - effective_length), (int64_t)3584},
            torch::kInt16);
        auto padded_embeds = torch::cat({inputs_embeds, pad_embeds});
        padded_embeds =
            padded_embeds.view({4, prefill_length / 4, 3584}).to(torch::kInt16);
        // std::cout << "padded_embeds: " << padded_embeds << std::endl;
        // std::cout << "padded_embeds numel: " << padded_embeds.numel() <<
        // std::endl;

        // for (int i = 0; i < 1; i++) {
        //     std::cout << padded_embeds[0][i] << " ";
        // }
        // std::cout << std::endl;
        // for (int i = 0; i < 1; i++) {
        //     std::cout << padded_embeds[1][i] << " ";
        // }
        // std::cout << std::endl;
        // for (int i = 0; i < 1; i++) {
        //     std::cout << padded_embeds[2][i] << " ";
        // }
        // std::cout << std::endl;
        // for (int i = 0; i < 1; i++) {
        //     std::cout << padded_embeds[3][i] << " ";
        // }
        // std::cout << std::endl;
        // std::this_thread::sleep_for(std::chrono::seconds(5));

        auto input_data = prefill_input_map.at("input_1");
        int16toint16((int16_t *)input_data.Data(),
                     (int16_t *)padded_embeds.data_ptr<int16_t>(),
                     padded_embeds.numel());

        std::vector<int16_t> valid_length_input = {valid_length};
        auto valid_length_input_tensor_opts =
            torch::TensorOptions().dtype(torch::kInt16);
        auto valid_length_input_tensor =
            torch::from_blob(valid_length_input.data(),
                             {int64_t(valid_length_input.size())},
                             valid_length_input_tensor_opts)
                .clone();

        // std::cout << "valid_length_input_tensor: " <<
        // valid_length_input_tensor << std::endl;
        auto prefill_valid_length = prefill_input_map.at("valid_length");
        int16toint16((int16_t *)prefill_valid_length.Data(),
                     (int16_t *)valid_length_input_tensor.data_ptr<int16_t>(),
                     valid_length_input_tensor.numel());

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
        auto prefill_current_length = prefill_input_map.at("current_length");
        int16toint16((int16_t *)prefill_current_length.Data(),
                     (int16_t *)current_length_input_tensor.data_ptr<int16_t>(),
                     current_length_input_tensor.numel());

        hmctx->prefill_model.SetInput("input_1", input_data);
        hmctx->prefill_model.SetInput("valid_length", prefill_valid_length);
        hmctx->prefill_model.SetInput("current_length", prefill_current_length);
        hmctx->prefill_model.Run();
        hmctx->prefill_model.Sync();
        // std::cout << "prefill run finished" << std::endl;
      }

      tcim::Tensor input_data;
      if (hmctx->model_type == "deepseek") {
        input_data =
            hmctx->prefill_model.GetOutput("Output_lm_head_add_list_0");
      } else {
        input_data = hmctx->prefill_model.GetOutput("output");
      }

      SRV_DBG("input_data.info.memSize:%d  input_data.DataType() is:%d\r\n",input_data.Info().MemSize(),input_data.Info().DataType());
      // std::this_thread::sleep_for(std::chrono::seconds(5));
      std::vector<int16_t> prefill_output;
      for (int idx = 0; idx < input_data.MemSize() / 2; idx++) {
        auto data = *((int16_t *)input_data.Data() + idx);
        // if (data != 0) {
        //     std::cout << "data at " << idx << ": " << data << std::endl;
        // }

        prefill_output.push_back(data);
      }
      SRV_DBG("vector<int16_t> prefill_output.size is:%d\r\n ",prefill_output.size());

      auto prefill_output_opts = torch::TensorOptions().dtype(torch::kInt16);
      auto prefill_output_tensor =
          torch::from_blob(prefill_output.data(),
                           {int64_t(prefill_output.size())},
                           prefill_output_opts)
              .clone();
      // std::cout << "prefill_output options: " <<
      // prefill_output_tensor.options() << std::endl; std::cout <<
      // "prefill_output dim: " << prefill_output_tensor.dim() << std::endl;
      // std::cout << "prefill_output size: " << prefill_output_tensor.size(0)
      // << std::endl; for (int i = 0; i < 3; i++) {
      //     std::cout << prefill_output_tensor[i] << " ";
      // }
      // std::cout << std::endl;
      // for (int i = 152060; i < 152064; i++) {
      //     std::cout << prefill_output_tensor[i] << " ";
      // }
      // std::cout << std::endl;
      auto selected_output =
          torch::argmax(prefill_output_tensor).to(torch::kInt32);
      // std::cout << "selected_output: " << selected_output << std::endl;
      // 下面打印的selected_output是乱码，index
      // int值，调下面的common_token_to_piece转成字符串
      SRV_DBG("++selected_output(tensor addr not selected index) is:%u\r\n", selected_output);
      auto prefill_token = *(int32_t *)selected_output.data_ptr<int32_t>();
      SRV_DBG("++selected tensor index is:%d\r\n",*(int32_t *)selected_output.data_ptr<int32_t>());
      auto response = common_token_to_piece(ctx, prefill_token);
      std::cout << "rrefill response: ";
      auto invalid_utf8 = false;
      write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
      std::cout << std::endl;
      // 下面打印的是正常的response 字符串
      SRV_DBG("++prefill response is:%s\r\n", response);

      context_length = input_echo_len;

      std::vector<llama_token> prefill_input_ids = {prefill_token};
      torch::Tensor prefill_inputs_embeds;
      auto prefill_opts = torch::TensorOptions().dtype(torch::kInt);
      auto prefill_inp =
          torch::from_blob(prefill_input_ids.data(),
                           {int64_t(prefill_input_ids.size())}, prefill_opts)
              .clone();
      // std::cout << "prefill_inp: " << prefill_inp << "\n";
      SRV_DBG("++prefill_inp is:%s\r\n", prefill_inp);
      try {
        prefill_inputs_embeds = emb->forward(prefill_inp);
      } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n" << e.what_without_backtrace();
        return -1;
      }
      // std::cout << "prefill_inputs_embeds options: " <<
      // prefill_inputs_embeds.options() << std::endl; std::cout <<
      // "prefill_inputs_embeds dim: " << prefill_inputs_embeds.dim() <<
      // std::endl; std::cout << "prefill_inputs_embeds size: " <<
      // prefill_inputs_embeds.size(0) << std::endl; for (int i = 0; i < 3; i++)
      // {
      //     std::cout << prefill_inputs_embeds[0][i] << " ";
      // }
      // std::cout << std::endl;
      prefill_inputs_embeds =
          prefill_inputs_embeds.view({1, 1, -1}).to(torch::kInt16);
      // std::cout << "prefill_inputs_embeds options: " <<
      // prefill_inputs_embeds.options() << std::endl; std::cout <<
      // "prefill_inputs_embeds dim: " << prefill_inputs_embeds.dim() <<
      // std::endl; std::cout << "prefill_inputs_embeds size: " <<
      // prefill_inputs_embeds.size(0) << std::endl;

      input_emd = prefill_inputs_embeds;
      // write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
      // std::cout.flush();
      token = prefill_token;
      decode_pos = batch.n_tokens;
      SRV_DBG("prefill stage:decode_pos is :%d \r\n",decode_pos);
    } else {
      SRV_DBG("%s", "decode stage\n");
      auto invalid_utf8 = false;
      auto input_data = decode_input_map.at("input_1");
      int16toint16((int16_t *)input_data.Data(),
                   (int16_t *)input_emd.data_ptr<int16_t>(), input_emd.numel());

      std::vector<int16_t> valid_length_input = {(int16_t)(context_length - 1)};
      auto valid_length_input_tensor_opts =
          torch::TensorOptions().dtype(torch::kInt16);
      auto valid_length_input_tensor =
          torch::from_blob(valid_length_input.data(),
                           {int64_t(valid_length_input.size())},
                           valid_length_input_tensor_opts)
              .clone();

      // std::cout << "valid_length_input_tensor: " << valid_length_input_tensor
      // << std::endl;
      auto decode_valid_length = decode_input_map.at("valid_length");
      int16toint16((int16_t *)decode_valid_length.Data(),
                   (int16_t *)valid_length_input_tensor.data_ptr<int16_t>(),
                   valid_length_input_tensor.numel());

      hmctx->decode_model.SetInput("input_1", input_data);
      hmctx->decode_model.SetInput("valid_length", decode_valid_length);

      // clock_t infer_start,infer_end;
      // infer_start = clock();
      hmctx->decode_model.Run();
      hmctx->decode_model.Sync();
      // infer_end = clock();
      tcim::Tensor decode_data;
      tcim::Tensor decode_output_tensor_float;
      if (hmctx->model_type == "deepseek") {
        decode_data =
            hmctx->decode_model.GetOutput("Output_lm_head_add_list_0");
      } else {
        decode_data = hmctx->decode_model.GetOutput("output");
      }

      // std::cout << "decode_data Size: " << decode_data.Size() << std::endl;
      // std::this_thread::sleep_for(std::chrono::seconds(5));
      #if 0 //old int to float ,not right
      std::vector<int16_t> decode_output;
      for (int idx = 0; idx < decode_data.MemSize() / 2; idx++) {
        auto data = *((int16_t *)decode_data.Data() + idx);
        // if (data != 0) {
        //     std::cout << "data at " << idx << ": " << data << std::endl;
        // }

        decode_output.push_back(data);
      }
      SRV_DBG("vector<int16_t> decode_output.size is:%d\r\n ",decode_output.size());
      auto decode_output_opts = torch::TensorOptions().dtype(torch::kInt16);
      decode_output_tensor =
          torch::from_blob(decode_output.data(),
                           {int64_t(decode_output.size())}, decode_output_opts)
              .clone();
      #else
        auto output_info = decode_data.Info().AsContiguous().AsType(tcim::FLOAT32);
        std::cout << "Output info:" << output_info << std::endl;
        decode_output_tensor_float = tcim::Tensor::CreateHostTensor(output_info);
        decode_data.CastTo(decode_output_tensor_float);
        std::vector<float> decode_output;
        for (int idx = 0; idx < decode_output_tensor_float.MemSize()/4 ; idx++) {
          auto data = *((float *)decode_output_tensor_float.Data() + idx);
          // if (data != 0) {
          //     std::cout << "data at " << idx << ": " << data << std::endl;
          // }
          decode_output.push_back(data);
        }
        SRV_DBG("float decode_output.size() is:%d \r\n",decode_output.size());
        auto decode_output_opts = torch::TensorOptions().dtype(torch::kFloat);
        auto decode_output_tensor =
          torch::from_blob(decode_output.data(),
                           {int64_t(decode_output.size())}, decode_output_opts)
              .clone();
      #endif

      // std::cout << "decode_output_tensor options: " <<
      // decode_output_tensor.options() << std::endl; std::cout <<
      // "decode_output_tensor dim: " << decode_output_tensor.dim() <<
      // std::endl; std::cout << "decode_output_tensor size: " <<
      // decode_output_tensor.size(0) << std::endl; for (int i = 0; i < 3; i++)
      // {
      //     std::cout << decode_output_tensor[i] << " ";
      // }
      // std::cout << std::endl;
      // for (int i = 152060; i < 152064; i++) {
      //     std::cout << decode_output_tensor[i] << " ";
      // }
      // std::cout << std::endl;
      int32_t generate_token=0;
      //default默认的处理流程-没有复杂的后采样
      if(0) {
        auto selected_output =
            torch::argmax(decode_output_tensor).to(torch::kInt32);
        // std::cout << "selected_output: " << selected_output << std::endl;
        generate_token = *(int32_t *)selected_output.data_ptr<int32_t>();
      } else {
        #if 1 //TODO remember delete ,only for compare token between argmax and houmo sampling
        auto selected_output_argmax =
            torch::argmax(decode_output_tensor).to(torch::kInt32);
        // std::cout << "selected_output: " << selected_output << std::endl;
        auto argmax_generate_token = *(int32_t *)selected_output_argmax.data_ptr<int32_t>();
        SRV_DBG("++before houmosample argmax_generate_token=%d \r\n",argmax_generate_token);
        #endif

        //float test_nouse=0.1234;
        //float *logits =&test_nouse;                         // 获取模型输出的logits
        int n_vocab = decode_output.size();                           // 词汇表大小
        //std::vector<llama_token> last_tokens = {};  // 最近生成的token

        //torch::Tensor decode_floatTensor = decode_output_tensor.to(torch::kFloat);
        //float *p_logits=(float *)decode_floatTensor.data_ptr<float>();

        auto *p_logits= decode_output_tensor.data_ptr<float>();
        for (auto &slot : slots) {
          if (!slot.generated_tokens.empty()) {
            SRV_DBG("%s\r\n","++call hm_sampling_processors process");
            SRV_DBG("HI.CPP >>> [%d][%d] slot.params top_k: %d, top_p: %f, temp: %f, penalty_repeat: %f, penalty_last_n: %d \r\n",
                slot.id, slot.id_task, slot.params.sampling.top_k, slot.params.sampling.top_p, slot.params.sampling.temp,
                slot.params.sampling.penalty_repeat, slot.params.sampling.penalty_last_n);
            hm_sampling_processors.process(p_logits, n_vocab, slot.generated_tokens, slot.params);
          }
          else{
            SRV_ERR("slot.generated_tokens.size() is:%d ,dont need sample\r\n",slot.generated_tokens.size());
          }
        }
        // 计算softmax概率
        std::vector<float> probs(n_vocab);
        softmax(p_logits, probs.data(), n_vocab);
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto tensor = torch::from_blob(probs.data(), {int64_t(probs.size())}, opts).clone();

        auto selected_output = torch::multinomial(tensor, 1).to(torch::kInt32);
        generate_token = *(int32_t *)selected_output.data_ptr<int32_t>();
        #if 1 //TODO remember delete ,only for compare token between argmax and houmo sampling
        SRV_DBG("++after houmosample generate_token=%d \r\n",generate_token);
        if(generate_token!=argmax_generate_token) {
           auto houmo_response = common_token_to_piece(ctx, generate_token);
           auto argmax_response= common_token_to_piece(ctx, argmax_generate_token);
           SRV_DBG("Houmo sample change gen idx from %d to %d , related string from %s to %s\r\n",
                 argmax_generate_token,generate_token,argmax_response,houmo_response);
        }
        #endif
      }
      // auto response = common_token_to_piece(ctx, generate_token);
      // std::cout << "response: ";
      // auto invalid_utf8 = false;
      // write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
      // std::cout.flush();
      // std::cout << std::endl;

      std::vector<llama_token> decode_input_ids = {generate_token};
      torch::Tensor decode_inputs_embeds;
      auto decode_opts = torch::TensorOptions().dtype(torch::kInt);
      auto decode_inp =
          torch::from_blob(decode_input_ids.data(),
                           {int64_t(decode_input_ids.size())}, decode_opts)
              .clone();
      // std::cout << "decode_inp: " << decode_inp << "\n";
      try {
        input_emd = emb->forward(decode_inp);
      } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n" << e.what_without_backtrace();
        return -1;
      }
      // std::cout << "input_emd options: " << input_emd.options() << std::endl;
      // std::cout << "input_emd dim: " << input_emd.dim() << std::endl;
      // std::cout << "input_emd size: " << input_emd.size(0) << std::endl;
      // for (int i = 0; i < 3; i++) {
      //     std::cout << input_emd[0][i] << " ";
      // }
      // std::cout << std::endl;
      input_emd = input_emd.view({1, 1, -1}).to(torch::kInt16);
      // std::cout << "input_emd options: " << input_emd.options() << std::endl;
      // std::cout << "input_emd dim: " << input_emd.dim() << std::endl;
      // std::cout << "input_emd size: " << input_emd.size(0) << std::endl;
      context_length = context_length + 1;
      if (llama_token_is_eog(vocab, generate_token)) {
        LOG_DBG("llama_token_is_eog: %d\n", generate_token);
      }
      token = generate_token;
      decode_pos = decode_pos + batch.n_tokens;
      SRV_DBG("decode stage:decode_pos is :%d \r\n",decode_pos);

      auto response = common_token_to_piece(ctx, generate_token);
      std::cout << "decode response: ";
      invalid_utf8 = false;
      write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
      std::cout << std::endl;
      // 下面打印的是正常的response 字符串
      SRV_DBG("--decode response is:%s\r\n", response);
    }

    return 0;
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
        SRV_DBG("%s", "all slots are idle\n");
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
      queue_tasks.post(task);
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

        // Shift context
        const int n_keep = slot.params.n_keep + add_bos_token;
        const int n_left = slot.n_past - n_keep;
        const int n_discard =
            slot.params.n_discard ? slot.params.n_discard : (n_left / 2);

        SLT_WRN(
            slot,
            "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n",
            n_keep, n_left, n_discard);

        llama_kv_self_seq_rm(ctx, slot.id, n_keep, n_keep + n_discard);
        llama_kv_self_seq_add(ctx, slot.id, n_keep + n_discard, slot.n_past,
                              -n_discard);

        if (slot.params.cache_prompt) {
          for (size_t i = n_keep + n_discard; i < slot.cache_tokens.size();
               i++) {
            slot.cache_tokens[i - n_discard] = slot.cache_tokens[i];
          }

          slot.cache_tokens.resize(slot.cache_tokens.size() - n_discard);
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
        continue;
      }

      slot.i_batch = batch.n_tokens;
      SRV_DBG("before common_batch_add batch.n_tokens=%d\r\n",batch.n_tokens);
      common_batch_add(batch, slot.sampled, slot.n_past, {slot.id}, true);
      SRV_DBG("after common_batch_add batch.n_tokens=%d\r\n",batch.n_tokens);

      slot.n_past += 1;

      if (slot.params.cache_prompt) {
        SLT_DBG(slot,"%s\r\n","push back cache_tokens");
        slot.cache_tokens.push_back(slot.sampled);
      }

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
      for (auto &slot : slots) {
        // check if we can batch this slot with the previous one
        if (slot.is_processing()) {
          if (!slot_batched) {
            slot_batched = &slot;
          } else if (!slot_batched->can_batch_with(slot)) {
            continue;
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
            if (1) {
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
            }

            // empty prompt passed -> release the slot and send empty response
            if (prompt_tokens.empty()) {
              //szj 此处未调用
              SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

              slot.release();
              slot.print_timings();
              send_final_response(slot);
              continue;
            }

            if (slot.is_non_causal()) {
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
                //szj 此处未调用
                const int n_left = slot.n_ctx - slot.params.n_keep;

                const int n_block_size = n_left / 2;
                const int erased_blocks =
                    (slot.n_prompt_tokens - slot.params.n_keep - n_block_size) /
                    n_block_size;

                llama_tokens new_tokens(
                    prompt_tokens.begin(),
                    prompt_tokens.begin() + slot.params.n_keep);

                new_tokens.insert(new_tokens.end(),
                                  prompt_tokens.begin() + slot.params.n_keep +
                                      erased_blocks * n_block_size,
                                  prompt_tokens.end());

                prompt_tokens = std::move(new_tokens);

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
                slot.n_past = common_lcp(slot.cache_tokens, prompt_tokens);

                // reuse chunks from the cached prompt by shifting their KV
                // cache in the new position
                if (params_base.n_cache_reuse > 0) {
                  size_t head_c = slot.n_past;  // cache
                  size_t head_p = slot.n_past;  // current prompt

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

                      llama_kv_self_seq_rm(ctx, slot.id, head_p, head_c);
                      llama_kv_self_seq_add(ctx, slot.id, head_c,
                                            head_c + n_match, kv_shift);

                      for (size_t i = 0; i < n_match; i++) {
                        slot.cache_tokens[head_p + i] =
                            slot.cache_tokens[head_c + i];
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
              }
            }

            if (slot.n_past == slot.n_prompt_tokens && slot.n_past > 0) {
              //szj 此处未调用
              // we have to evaluate at least 1 token to generate logits.
              SLT_WRN(slot,
                      "need to evaluate at least 1 token to generate logits, "
                      "n_past = %d, n_prompt_tokens = %d\n",
                      slot.n_past, slot.n_prompt_tokens);

              slot.n_past--;
            }

            slot.n_prompt_tokens_processed = 0;
          }

          // non-causal tasks require to fit the entire prompt in the physical
          // batch
          if (slot.is_non_causal()) {
            // cannot fit the prompt in the current batch - will try next iter
            if (batch.n_tokens + slot.n_prompt_tokens > n_batch) {
              continue;
            }
          }

          // keep only the common part
          if (!llama_kv_self_seq_rm(ctx, slot.id, slot.n_past, -1)) {
            // could not partially delete (likely using a non-Transformer model)
            llama_kv_self_seq_rm(ctx, slot.id, -1, -1);

            // there is no common part left
            slot.n_past = 0;
          }

          SLT_DBG(slot, "kv cache rm [%d, end)\n", slot.n_past);

          // remove the non-common part from the cache
          slot.cache_tokens.resize(slot.n_past);

          // add prompt tokens for processing in the current batch
          while (slot.n_past < slot.n_prompt_tokens &&
                 batch.n_tokens < n_batch) {
            SRV_DBG("++add token loopindex:%d\r\n",slot.n_past);
            // without pooling, we want to output the embeddings for all the
            // tokens in the batch
            const bool need_embd =
                slot.task_type == SERVER_TASK_TYPE_EMBEDDING &&
                llama_pooling_type(slot.ctx) == LLAMA_POOLING_TYPE_NONE;

            common_batch_add(batch, prompt_tokens[slot.n_past], slot.n_past,
                             {slot.id}, need_embd);

            if (slot.params.cache_prompt) {
              slot.cache_tokens.push_back(prompt_tokens[slot.n_past]);
            }

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

            common_sampler_reset(slot.smpl);

            // Process all prompt tokens through sampler system
            for (int i = 0; i < slot.n_prompt_tokens; ++i) {
              SLT_DBG(slot, "call common_sampler_accept1 loopcount=%d | slot.n_prompt_tokens=%d\r\n",i,slot.n_prompt_tokens);
              common_sampler_accept(slot.smpl, prompt_tokens[i], false);
            }

            // extract the logits only for the last token
            batch.logits[batch.n_tokens - 1] = true;

            slot.n_decoded = 0;
            slot.i_batch = batch.n_tokens - 1;

            SLT_DBG(slot, "prompt done, n_past = %d, n_tokens = %d\n",
                    slot.n_past, batch.n_tokens);
          }
        }

        if (batch.n_tokens >= n_batch) {
          break;
        }
      }
    }

    if (batch.n_tokens == 0) {
      SRV_WRN("%s", "no tokens to decode\n");
      return;
    }

    SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

    if (slot_batched) {
      SRV_DBG("%s\r\n","enter slot_batched");
      // make sure we're in the right embedding mode
      llama_set_embeddings(ctx, slot_batched->is_non_causal());
      // apply lora, only need to do it once per batch
      common_set_adapter_lora(ctx, slot_batched->lora);
    }

    // process the created batch of tokens
    for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
      SRV_DBG("batch.n_tokens = %d\n", batch.n_tokens);
      SRV_DBG("decode_pos = %d\n", decode_pos);
      const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);
      SRV_DBG("n_tokens = %d\n", n_tokens);

      llama_batch batch_view = {
          n_tokens,           batch.token + i,  nullptr,          batch.pos + i,
          batch.n_seq_id + i, batch.seq_id + i, batch.logits + i,
      };

      int ret = 0;
      switch(hm_model_scale_type) {
        case MODEL_14B:
             ret=houmo_decode_14B(batch);
             break;
        case MODEL_7B:
             ret=houmo_decode(batch);
             break;
        case MODEL_8B:
             ret=houmo_decode_8B(batch);
             break;
        default:
             SRV_ERR("%s\r\n","hm_model_scale_type not supported,please check!");
      }
      metrics.on_decoded(slots);

      if (ret != 0) {
        if (n_batch == 1 || ret < 0) {
          // if you get here, it means the KV cache is full - try increasing it
          // via the context size
          SRV_ERR(
              "failed to decode the batch: KV cache is full - try increasing "
              "it via the context size, i = %d, n_batch = %d, ret = %d\n",
              i, n_batch, ret);
          for (auto &slot : slots) {
            slot.release();
            send_error(slot,
                       "Input prompt is too big compared to KV size. Please "
                       "try increasing KV size.");
          }
          break;  // break loop of n_batch
        }

        // retry with half the batch size to try to find a free slot in the KV
        // cache
        n_batch /= 2;
        i -= n_batch;

        SRV_WRN(
            "failed to find free space in the KV cache, retrying with smaller "
            "batch size - try increasing it via the context size or enable "
            "defragmentation, i = %d, n_batch = %d, ret = %d\n",
            i, n_batch, ret);

        continue;  // continue loop of n_batch
      }

      for (auto &slot : slots) {
        if (slot.i_batch < (int)i || slot.i_batch >= (int)(i + n_tokens)) {
          SRV_DBG("%s", "++enter continue here 55\r\n");
          continue;  // continue loop of slots
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

        llama_token id = token;

        slot.i_batch = -1;
        SRV_DBG("++call common_sampler_accept 99 slots size is:%d\r\n",slots.size());
        common_sampler_accept(slot.smpl, id, true);

        slot.n_decoded += 1;

        const int64_t t_current = ggml_time_us();

        if (slot.n_decoded == 1) {
          SRV_DBG("%s \r\n","call metrics.on_prompt_eval");
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
          SRV_DBG("%s\r\n","Release slot called and continue");
          continue;
        }
      }

      // do speculative decoding-szj 此处未进入
      // 此处的for循环不会执行
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
        params_spec.n_reuse =
            llama_n_ctx(slot.ctx_dft) - slot.params.speculative.n_max;
        params_spec.p_min = slot.params.speculative.p_min;

        llama_tokens draft = common_speculative_gen_draft(
            slot.spec, params_spec, slot.cache_tokens, id);

        // ignore small drafts
        if (slot.params.speculative.n_min > (int)draft.size()) {
          SLT_DBG(slot, "ignoring small draft: %d < %d\n", (int)draft.size(),
                  slot.params.speculative.n_min);

          continue;
        }

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
        const auto ids =
            common_sampler_sample_and_accept_n(slot.smpl, ctx, draft);

        slot.n_past += ids.size();
        slot.n_decoded += ids.size();

        slot.cache_tokens.push_back(id);
        slot.cache_tokens.insert(slot.cache_tokens.end(), ids.begin(),
                                 ids.end() - 1);

        llama_kv_self_seq_rm(ctx, slot.id, slot.n_past, -1);

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
    }

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

static void log_server_request(const httplib::Request &req,
                               const httplib::Response &res) {
  // skip GH copilot requests when using default port
  if (req.path == "/v1/health" || req.path == "/v1/completions") {
    return;
  }

  LOG_INF("request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(),
          req.remote_addr.c_str(), res.status);

  LOG_DBG("request:  %s\n", req.body.c_str());
  LOG_DBG("response: %s\n", res.body.c_str());
}


#ifdef _WIN32

int init_params(std::string model_path, const INIReader& config_reader, common_params* &p_params) {
    model_path = model_path + "/";

    std::string tok_model_path = model_path + "DeepSeek-R1-Distill-Qwen-1.5B-IQ2_M.gguf";
    if (model_path.find("qwen2") != std::string::npos) {
        tok_model_path = model_path + "qwen2-1_5b-instruct-q4_0.gguf";
    }

    const auto server_port = config_reader.Get("server", "port", "8808");

    const std::string log_location = config_reader.Get("log", "location", "UNKNOWN");
    if (log_location == "UNKNOWN") {
        return -1;
    }

    const int log_level = config_reader.GetInteger("log", "level", 0);

    try {
        if (std::filesystem::create_directories(log_location)) {
            std::cout << "Log Dir created successfully.\n";
        } else {
            std::cout << "Log Dir already exist.\n";
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Log Dir created failed, exception: " << e.what() << "\n";
        return -1;
    }


    const std::time_t t = std::time(nullptr);
    const std::string log_file = log_location + string_format("houmo-llm-%d.log", t);
    char* argv[] = {"houmo-llm-demo",
                    "-m",
                    &tok_model_path[0],
                    "--port",
                    const_cast<char*>(server_port.c_str()),
                    "--host",
                    "0.0.0.0",
                    "--device",
                    "none",
                    "-np",
                    "1",
                    "--ctx-size",
                    "4096",
                    "--no-context-shift",
                    "--no-warmup",
                    "--log-verbosity",
                    const_cast<char*>(std::to_string(log_level).c_str()),
                    "--log-file",
                    const_cast<char*>(log_file.c_str()),
                    nullptr};

    char** p = argv;
    int argc = 0;
    while (*p++ != nullptr)
    {
        argc++;
    }

    // own arguments required by this example
    //common_params *p_params = new common_params();

    // own arguments required by this example
    p_params = new common_params();

    if (!common_params_parse(argc, argv, *p_params, LLAMA_EXAMPLE_SERVER)) {
        return -1;
    }

    // struct that contains llama context and inference
    static server_context* ctx_server = nullptr;
    ctx_server = new server_context();
    SRV_INF("ctx_server: %p\n", ctx_server);

    // 服务化版本自定义初始后采样参数配置
    p_params->sampling.top_k = 40;
    p_params->sampling.top_p = 0.9;
    p_params->sampling.temp = 1.1f;
    p_params->sampling.penalty_repeat = 1.05;
    p_params->sampling.penalty_last_n = 60;
    ctx_server->params = p_params;

    return 0;
}

static bool server_running = false;
static bool server_terminating = false;

/**
 * Win Service Main Logic
 *
 * @return error_code
 */
int run_server() {
    std::string configPath = "../config/sys.cfg";
    char exePath[MAX_PATH];
    GetModuleFileName(nullptr, exePath, MAX_PATH); // 获取exe绝对路径
    const auto programPath = std::string(exePath);
    configPath = std::string(exePath).substr(0, std::string(exePath).find_last_of('\\') + 1) + "..\\config\\sys.cfg";

    const INIReader config_reader(configPath);
    if (config_reader.ParseError() < 0) {
        LOG_ERR("WINSERVICE cannot parse config/sys.cfg");
        return 1;
    }

    const int thread_count = config_reader.GetInteger("server", "thread_count", 5);
    const std::string cur_model = config_reader.Get("model", "cur_model", "UNKNOWN");
    if (cur_model == "UNKNOWN") {
        return 2;
    }

    common_params* p_params = nullptr;
    const std::string cur_model_path = cur_model + "/";
    if (const int error_code = init_params(cur_model_path, config_reader, p_params); error_code != 0) {
        return error_code;
    }


    LOG_INF("WINSERVICE info: n_threads = %d, cur_model = %s \n", thread_count, cur_model.c_str());

    HANDLE instance = create_instance(cur_model_path, p_params);
    if (!instance) {
        LOG_ERR("instance is null\n");
    }

    while (server_running) {
        Sleep(1000);
    }

    const int ret = destroy_instance(instance);

    printf("ret = %d\n", ret);
    server_terminating = false;
    return ret;
}

void stop_server() {
    LOG_INF("WINSERVICE begin to terminate. \n");
    server_running = false;
    server_terminating = true;
    while (server_terminating) {
        Sleep(1000);
    }
    LOG_INF("WINSERVICE terminated. \n");
}

#endif


HANDLE create_instance(std::string model_path, common_params* p_params = nullptr) {
    if (!p_params) {
        model_path = model_path + "/";

        std::string tok_model_path = model_path + "DeepSeek-R1-Distill-Qwen-1.5B-IQ2_M.gguf";
        if (model_path.find("qwen2") != std::string::npos) {
            tok_model_path = model_path + "qwen2-1_5b-instruct-q4_0.gguf";
        }
        std::time_t t = std::time(nullptr);
        std::string log_file = string_format("houmo-llm-%d.log", t);
        char* argv[] = {"houmo-llm-demo",
                        "-m",
                        &tok_model_path[0],
                        "--port",
                        "8808",
                        "--host",
                        "0.0.0.0",
                        "--device",
                        "none",
                        "-np",
                        "1",
                        "--ctx-size",
                        "4096",
                        "--no-context-shift",
                        "--no-warmup",
                        "--log-verbose",
                        "--log-file",
                        (char*)log_file.c_str(),
                        NULL};

        char** p = argv;
        int argc = 0;
        while (*p++ != NULL)
        {
            argc++;
        }

        //for (int i = 0; i < argc; i++) {
        //    printf("argv[%d] = %s\n", i, argv[i]);
        //}

        // own arguments required by this example
        p_params = new common_params();
        p_params->sampling.top_k = 40;
        p_params->sampling.top_p = 0.9;
        p_params->sampling.temp = 1.1f;
        p_params->sampling.penalty_repeat = 1.05;
        p_params->sampling.penalty_last_n = 60;

        if (!common_params_parse(argc, argv, *p_params, LLAMA_EXAMPLE_SERVER)) {
            return nullptr;
        }

        common_init();
    }

    // struct that contains llama context and inference
    static server_context * ctx_server = nullptr;
    ctx_server = new server_context();
    SRV_INF("ctx_server: %p\n", ctx_server);
    params params;
    params.top_k = p_params->sampling.top_k;
    params.top_p = p_params->sampling.top_p;
    params.temperature = p_params->sampling.temp;
    params.repeat_penalty = p_params->sampling.penalty_repeat;
    params.repeat_last_n = p_params->sampling.penalty_last_n;
    ctx_server->init_houmo_sampling(ctx_server->hm_sampling_processors, &params);
    ctx_server->params = p_params;


    llama_backend_init();
    llama_numa_init(ctx_server->params->numa);


    LOG_INF(
      "system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n",
      ctx_server->params->cpuparams.n_threads,
      ctx_server->params->cpuparams_batch.n_threads,
      std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(*ctx_server->params).c_str());
    LOG_INF("\n");

  httplib::Server *svr = new httplib::Server();

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  if (ctx_server->params->ssl_file_key != "" &&
      ctx_server->params->ssl_file_cert != "") {
    LOG_INF("Running with SSL: key = %s, cert = %s\n",
            ctx_server->params->ssl_file_key.c_str(),
            ctx_server->params->ssl_file_cert.c_str());
    svr = new httplib::SSLServer(ctx_server->params->ssl_file_cert.c_str(),
                                 ctx_server->params->ssl_file_key.c_str());
  } else {
    LOG_INF("Running without SSL\n");
    svr = new httplib::Server();
  }
#else
  if (ctx_server->params->ssl_file_key != "" &&
      ctx_server->params->ssl_file_cert != "") {
    LOG_ERR("Server is built without SSL support\n");
    return nullptr;
  }
  svr = new httplib::Server();
#endif

  ctx_server->svr = svr;

  ctx_server->svr->set_default_headers({{"Server", "HoumoHI"}});
  ctx_server->svr->set_logger(log_server_request);

  auto res_error = [](httplib::Response &res, const json &error_data) {
    json final_response{{"error", error_data}};
    res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
    res.status = json_value(error_data, "code", 500);
  };

  auto res_ok = [](httplib::Response &res, const json &data) {
    res.set_content(safe_json_to_str(data), MIMETYPE_JSON);
    res.status = 200;
  };

  ctx_server->svr->set_exception_handler(
      [&res_error](const httplib::Request &, httplib::Response &res,
                   const std::exception_ptr &ep) {
        std::string message;
        try {
          std::rethrow_exception(ep);
        } catch (const std::exception &e) {
          message = e.what();
        } catch (...) {
          message = "Unknown Exception";
        }

        json formatted_error =
            format_error_response(message, ERROR_TYPE_SERVER);
        LOG_WRN("got exception: %s\n", formatted_error.dump().c_str());
        res_error(res, formatted_error);
      });

  ctx_server->svr->set_error_handler([&res_error](const httplib::Request &,
                                                  httplib::Response &res) {
    if (res.status == 404) {
      res_error(res,
                format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
    }
    // for other error codes, we skip processing here because it's already done
    // by res_error()
  });

  // set timeouts and change hostname and port
  ctx_server->svr->set_read_timeout(ctx_server->params->timeout_read);
  ctx_server->svr->set_write_timeout(ctx_server->params->timeout_write);

  std::unordered_map<std::string, std::string> log_data;

  log_data["hostname"] = ctx_server->params->hostname;
  log_data["port"] = std::to_string(ctx_server->params->port);

  if (ctx_server->params->api_keys.size() == 1) {
    auto key = ctx_server->params->api_keys[0];
    log_data["api_key"] =
        "api_key: ****" + key.substr(std::max((int)(key.length() - 4), 0));
  } else if (ctx_server->params->api_keys.size() > 1) {
    log_data["api_key"] =
        "api_key: " + std::to_string(ctx_server->params->api_keys.size()) +
        " keys loaded";
  }

  // Necessary similarity of prompt for slot selection
  ctx_server->slot_prompt_similarity = ctx_server->params->slot_prompt_similarity;

  //
  // Middlewares
  //

  auto middleware_validate_api_key = [&res_error](const httplib::Request &req,
                                                  httplib::Response &res) {
    static const std::unordered_set<std::string> public_endpoints = {
        "/health",
        "/models",
        "/v1/models",
    };

    // If API key is not set, skip validation
    if (ctx_server->params->api_keys.empty()) {
      return true;
    }

    // If path is public or is static file, skip validation
    if (public_endpoints.find(req.path) != public_endpoints.end() ||
        req.path == "/") {
      return true;
    }

    // Check for API key in the header
    auto auth_header = req.get_header_value("Authorization");

    std::string prefix = "Bearer ";
    if (auth_header.substr(0, prefix.size()) == prefix) {
      std::string received_api_key = auth_header.substr(prefix.size());
      if (std::find(ctx_server->params->api_keys.begin(),
                    ctx_server->params->api_keys.end(),
                    received_api_key) != ctx_server->params->api_keys.end()) {
        return true;  // API key is valid
      }
    }

    // API key is invalid or not provided
    res_error(res, format_error_response("Invalid API Key",
                                         ERROR_TYPE_AUTHENTICATION));

    LOG_WRN("Unauthorized: Invalid API Key\n");

    return false;
  };

  auto middleware_server_state = [&res_error](const httplib::Request& req, httplib::Response& res) {
      LOG_INF("state 2: %d\n", ctx_server->state.load());
      if (ctx_server->state.load() == SERVER_STATE_LOADING_MODEL) {
          res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
          return false;
      }
      return true;
  };

  // register server middlewares
  ctx_server->svr->set_pre_routing_handler(
      [&middleware_validate_api_key, &middleware_server_state](
          const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin",
                       req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't
        // include Authorization header
        if (req.method == "OPTIONS") {
          res.set_header("Access-Control-Allow-Credentials", "true");
          res.set_header("Access-Control-Allow-Methods", "GET, POST");
          res.set_header("Access-Control-Allow-Headers", "*");
          res.set_content("", "text/html");  // blank response, no data
          return httplib::Server::HandlerResponse::Handled;  // skip further
                                                             // processing
        }
        if (!middleware_server_state(req, res)) {
          return httplib::Server::HandlerResponse::Handled;
        }
        if (!middleware_validate_api_key(req, res)) {
          return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
      });

  //
  // Route handlers (or controllers)
  //

  const auto handle_health = [&](const httplib::Request &, httplib::Response &res) {
      // error and loading states are handled by middleware
      json health = {{"status", "ok"}};
      res_ok(res, health);
  };

  const auto handle_slots = [&](const httplib::Request &req,
                                httplib::Response &res) {
    if (!ctx_server->params->endpoint_slots) {
      res_error(res, format_error_response("This server does not support slots "
                                           "endpoint. Start it with `--slots`",
                                           ERROR_TYPE_NOT_SUPPORTED));
      return;
    }

    // request slots data using task queue
    server_task task(SERVER_TASK_TYPE_METRICS);
    task.id = ctx_server->queue_tasks.get_new_id();
    ctx_server->queue_results.add_waiting_task_id(task.id);
    ctx_server->queue_tasks.post(task, true);  // high-priority task

    // get the result
    server_task_result_ptr result = ctx_server->queue_results.recv(task.id);
    ctx_server->queue_results.remove_waiting_task_id(task.id);

    if (result->is_error()) {
      res_error(res, result->to_json());
      return;
    }

    // TODO: get rid of this dynamic_cast
    auto res_metrics = dynamic_cast<server_task_result_metrics *>(result.get());
    GGML_ASSERT(res_metrics != nullptr);

    // optionally return "fail_on_no_slot" error
    if (req.has_param("fail_on_no_slot")) {
      if (res_metrics->n_idle_slots == 0) {
        res_error(res, format_error_response("no slot available",
                                             ERROR_TYPE_UNAVAILABLE));
        return;
      }
    }

    res_ok(res, res_metrics->slots_data);
  };

  const auto handle_metrics = [&](const httplib::Request &,
                                  httplib::Response &res) {
    if (!ctx_server->params->endpoint_metrics) {
      res_error(res,
                format_error_response("This server does not support metrics "
                                      "endpoint. Start it with `--metrics`",
                                      ERROR_TYPE_NOT_SUPPORTED));
      return;
    }

    // request slots data using task queue
    server_task task(SERVER_TASK_TYPE_METRICS);
    task.id = ctx_server->queue_tasks.get_new_id();
    task.metrics_reset_bucket = true;

    ctx_server->queue_results.add_waiting_task_id(task.id);
    ctx_server->queue_tasks.post(task, true);  // high-priority task

    // get the result
    server_task_result_ptr result = ctx_server->queue_results.recv(task.id);
    ctx_server->queue_results.remove_waiting_task_id(task.id);

    if (result->is_error()) {
      res_error(res, result->to_json());
      return;
    }

    // TODO: get rid of this dynamic_cast
    auto res_metrics = dynamic_cast<server_task_result_metrics *>(result.get());
    GGML_ASSERT(res_metrics != nullptr);

    // metrics definition:
    // https://prometheus.io/docs/practices/naming/#metric-names
    json all_metrics_def = json{
        {"counter",
         {{{"name", "prompt_tokens_total"},
           {"help", "Number of prompt tokens processed."},
           {"value", (uint64_t)res_metrics->n_prompt_tokens_processed_total}},
          {{"name", "prompt_seconds_total"},
           {"help", "Prompt process time"},
           {"value", (uint64_t)res_metrics->t_prompt_processing_total / 1.e3}},
          {{"name", "tokens_predicted_total"},
           {"help", "Number of generation tokens processed."},
           {"value", (uint64_t)res_metrics->n_tokens_predicted_total}},
          {{"name", "tokens_predicted_seconds_total"},
           {"help", "Predict process time"},
           {"value", (uint64_t)res_metrics->t_tokens_generation_total / 1.e3}},
          {{"name", "n_decode_total"},
           {"help", "Total number of llama_decode() calls"},
           {"value", res_metrics->n_decode_total}},
          {{"name", "n_busy_slots_per_decode"},
           {"help", "Average number of busy slots per llama_decode() call"},
           {"value", (float)res_metrics->n_busy_slots_total /
                         (float)res_metrics->n_decode_total}}}},
        {"gauge",
         {{{"name", "prompt_tokens_seconds"},
           {"help", "Average prompt throughput in tokens/s."},
           {"value", res_metrics->n_prompt_tokens_processed
                         ? 1.e3 / res_metrics->t_prompt_processing *
                               res_metrics->n_prompt_tokens_processed
                         : 0.}},
          {{"name", "predicted_tokens_seconds"},
           {"help", "Average generation throughput in tokens/s."},
           {"value", res_metrics->n_tokens_predicted
                         ? 1.e3 / res_metrics->t_tokens_generation *
                               res_metrics->n_tokens_predicted
                         : 0.}},
          {{"name", "kv_cache_usage_ratio"},
           {"help", "KV-cache usage. 1 means 100 percent usage."},
           {"value",
            1. * res_metrics->kv_cache_used_cells / ctx_server->params->n_ctx}},
          {{"name", "kv_cache_tokens"},
           {"help", "KV-cache tokens."},
           {"value", (uint64_t)res_metrics->kv_cache_tokens_count}},
          {{"name", "requests_processing"},
           {"help", "Number of request processing."},
           {"value", (uint64_t)res_metrics->n_processing_slots}},
          {{"name", "requests_deferred"},
           {"help", "Number of request deferred."},
           {"value", (uint64_t)res_metrics->n_tasks_deferred}}}}};

    std::stringstream prometheus;

    for (const auto &el : all_metrics_def.items()) {
      const auto &type = el.key();
      const auto &metrics_def = el.value();

      for (const auto &metric_def : metrics_def) {
        const std::string name = metric_def.at("name");
        const std::string help = metric_def.at("help");

        auto value = json_value(metric_def, "value", 0.);
        prometheus << "# HELP llamacpp:" << name << " " << help << "\n"
                   << "# TYPE llamacpp:" << name << " " << type << "\n"
                   << "llamacpp:" << name << " " << value << "\n";
      }
    }

    res.set_header("Process-Start-Time-Unix",
                   std::to_string(res_metrics->t_start));

    res.set_content(prometheus.str(), "text/plain; version=0.0.4");
    res.status = 200;  // HTTP OK
  };

  const auto handle_slots_save = [&res_error, &res_ok](
                                     const httplib::Request &req,
                                     httplib::Response &res, int id_slot) {
    json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
      res_error(res, format_error_response("Invalid filename",
                                           ERROR_TYPE_INVALID_REQUEST));
      return;
    }
    std::string filepath = ctx_server->params->slot_save_path + filename;

    server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
    task.id = ctx_server->queue_tasks.get_new_id();
    task.slot_action.slot_id = id_slot;
    task.slot_action.filename = filename;
    task.slot_action.filepath = filepath;

    ctx_server->queue_results.add_waiting_task_id(task.id);
    ctx_server->queue_tasks.post(task);

    server_task_result_ptr result = ctx_server->queue_results.recv(task.id);
    ctx_server->queue_results.remove_waiting_task_id(task.id);

    if (result->is_error()) {
      res_error(res, result->to_json());
      return;
    }

    res_ok(res, result->to_json());
  };

  const auto handle_slots_restore = [&res_error, &res_ok](
                                        const httplib::Request &req,
                                        httplib::Response &res, int id_slot) {
    json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
      res_error(res, format_error_response("Invalid filename",
                                           ERROR_TYPE_INVALID_REQUEST));
      return;
    }
    std::string filepath = ctx_server->params->slot_save_path + filename;

    server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
    task.id = ctx_server->queue_tasks.get_new_id();
    task.slot_action.slot_id = id_slot;
    task.slot_action.filename = filename;
    task.slot_action.filepath = filepath;

    ctx_server->queue_results.add_waiting_task_id(task.id);
    ctx_server->queue_tasks.post(task);

    server_task_result_ptr result = ctx_server->queue_results.recv(task.id);
    ctx_server->queue_results.remove_waiting_task_id(task.id);

    if (result->is_error()) {
      res_error(res, result->to_json());
      return;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load *>(
                    result.get()) != nullptr);
    res_ok(res, result->to_json());
  };

  const auto handle_slots_erase = [&res_error, &res_ok](
                                      const httplib::Request & /* req */,
                                      httplib::Response &res, int id_slot) {
    server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
    task.id = ctx_server->queue_tasks.get_new_id();
    task.slot_action.slot_id = id_slot;

    ctx_server->queue_results.add_waiting_task_id(task.id);
    ctx_server->queue_tasks.post(task);

    server_task_result_ptr result = ctx_server->queue_results.recv(task.id);
    ctx_server->queue_results.remove_waiting_task_id(task.id);

    if (result->is_error()) {
      res_error(res, result->to_json());
      return;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase *>(result.get()) !=
                nullptr);
    res_ok(res, result->to_json());
  };

  const auto handle_slots_action = [&res_error, &handle_slots_save,
                                    &handle_slots_restore, &handle_slots_erase](
                                       const httplib::Request &req,
                                       httplib::Response &res) {
    if (ctx_server->params->slot_save_path.empty()) {
      res_error(
          res, format_error_response("This server does not support slots "
                                     "action. Start it with `--slot-save-path`",
                                     ERROR_TYPE_NOT_SUPPORTED));
      return;
    }

    std::string id_slot_str = req.path_params.at("id_slot");
    int id_slot;

    try {
      id_slot = std::stoi(id_slot_str);
    } catch (const std::exception &) {
      res_error(res, format_error_response("Invalid slot ID",
                                           ERROR_TYPE_INVALID_REQUEST));
      return;
    }

    std::string action = req.get_param_value("action");

    if (action == "save") {
      handle_slots_save(req, res, id_slot);
    } else if (action == "restore") {
      handle_slots_restore(req, res, id_slot);
    } else if (action == "erase") {
      handle_slots_erase(req, res, id_slot);
    } else {
      res_error(res, format_error_response("Invalid action",
                                           ERROR_TYPE_INVALID_REQUEST));
    }
  };

  const auto handle_props = [&res_ok](const httplib::Request &,
                                      httplib::Response &res) {
    // this endpoint is publicly available, please only return what is safe to
    // be exposed
    json data = {
        {"default_generation_settings",
         ctx_server->default_generation_settings_for_props},
        {"total_slots", ctx_server->params_base.n_parallel},
        {"model_path", ctx_server->params_base.model},
        {"chat_template",
         common_chat_templates_source(ctx_server->chat_templates.get())},
        {"bos_token", common_token_to_piece(ctx_server->ctx,
                                            llama_vocab_bos(ctx_server->vocab),
                                            /* special= */ true)},
        {"eos_token", common_token_to_piece(ctx_server->ctx,
                                            llama_vocab_eos(ctx_server->vocab),
                                            /* special= */ true)},
        {"build_info", build_info},
    };
    if (ctx_server->params_base.use_jinja) {
      if (auto tool_use_src = common_chat_templates_source(
              ctx_server->chat_templates.get(), "tool_use")) {
        data["chat_template_tool_use"] = tool_use_src;
      }
    }

    res_ok(res, data);
  };

  const auto handle_props_change = [&res_error, &res_ok](
                                       const httplib::Request &req,
                                       httplib::Response &res) {
    if (!ctx_server->params_base.endpoint_props) {
      res_error(res, format_error_response(
                         "This server does not support changing global "
                         "properties. Start it with `--props`",
                         ERROR_TYPE_NOT_SUPPORTED));
      return;
    }

    json data = oaicompat_completion_params_parse(
        json::parse(req.body), ctx_server->params_base.use_jinja,
        ctx_server->params_base.reasoning_format,
        ctx_server->chat_templates.get());

    // update any props here

    res_ok(res, {{"success", true}});
  };

  // handle completion-like requests (completion, chat, infill)
  // we can optionally provide a custom format for partial results and final
  // results
  const auto handle_completions_generic =
      [&res_error, &res_ok](server_task_type type, json &data,
                            std::function<bool()> is_connection_closed,
                            httplib::Response &res,
                            oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE,
                            bool oaicompat_chat = false) {
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION ||
                    type == SERVER_TASK_TYPE_INFILL);

        if (ctx_server->params_base.embedding) {
          res_error(res, format_error_response(
                             "This server does not support completions. Start "
                             "it without `--embeddings`",
                             ERROR_TYPE_NOT_SUPPORTED));
          return;
        }

        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;

        try {
          std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(
              ctx_server->vocab, data.at("prompt"), true, true);
          tasks.reserve(tokenized_prompts.size());
          for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(type);

            task.id = ctx_server->queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens = std::move(tokenized_prompts[i]);
            task.params = server_task::params_from_json_cmpl(
                ctx_server->ctx, ctx_server->params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.oaicompat = oaicompat;
            task.params.oaicompat_cmpl_id = completion_id;
            // oaicompat_model is already populated by params_from_json_cmpl

            tasks.push_back(task);
          }
        } catch (const std::exception &e) {
          res_error(
              res, format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
          return;
        }

        ctx_server->queue_results.add_waiting_tasks(tasks);
        ctx_server->queue_tasks.post(tasks);

        bool stream = json_value(data, "stream", false);
        const auto task_ids = server_task::get_list_id(tasks);

        if (!stream) {
          ctx_server->receive_multi_results(
              task_ids,
              [&](std::vector<server_task_result_ptr> &results) {
                if (results.size() == 1) {
                  // single result
                  res_ok(res, results[0]->to_json());
                } else {
                  // multiple results (multitask)
                  json arr = json::array();
                  for (auto &res : results) {
                    arr.push_back(res->to_json());
                  }
                  res_ok(res, arr);
                }
              },
              [&](const json &error_data) { res_error(res, error_data); },
              is_connection_closed);

          ctx_server->queue_results.remove_waiting_task_ids(task_ids);
        } else {
          const auto chunked_content_provider =
              [task_ids, oaicompat](size_t, httplib::DataSink &sink) {
                ctx_server->receive_cmpl_results_stream(
                    task_ids,
                    [&](server_task_result_ptr &result) -> bool {
                      json res_json = result->to_json();
                      if (res_json.is_array()) {
                        for (const auto &res : res_json) {
                          if (!server_sent_event(sink, "data", res)) {
                            // sending failed (HTTP connection closed), cancel
                            // the generation
                            return false;
                          }
                        }
                        return true;
                      } else {
                        return server_sent_event(sink, "data", res_json);
                      }
                    },
                    [&](const json &error_data) {
                      server_sent_event(sink, "error", error_data);
                    },
                    [&sink]() {
                      // note: do not use req.is_connection_closed here because
                      // req is already destroyed
                      return !sink.is_writable();
                    });
                if (oaicompat != OAICOMPAT_TYPE_NONE) {
                  static const std::string ev_done = "data: [DONE]\n\n";
                  sink.write(ev_done.data(), ev_done.size());
                }
                sink.done();
                return false;
              };

          auto on_complete = [task_ids](bool) {
            ctx_server->queue_results.remove_waiting_task_ids(task_ids);
          };

          res.set_chunked_content_provider(
              "text/event-stream", chunked_content_provider, on_complete);
        }
      };

  auto generate_error = [](const json &error_data) {
    const std::string str =
        error_data.dump(-1, ' ', false, json::error_handler_t::replace) +
        "\n\n";  // required by RFC 8895 - A message is terminated by a blank
                 // line (two line terminators in a row).

    SRV_ERR("%s\n", str.c_str());
  };

  auto generate_ok = [](const json &data) {
    const std::string str =
        data.dump(-1, ' ', false, json::error_handler_t::replace) +
        "\n\n";  // required by RFC 8895 - A message is terminated by a blank
                 // line (two line terminators in a row).
    SRV_INF("%s\n", str.c_str());
  };

  // handle completion-like requests (completion, chat, infill)
  // we can optionally provide a custom format for partial results and final
  // results
  const auto handle_completions_generate =
      [&generate_error, &generate_ok](
          server_task_type type, json &data,
          std::function<bool()> is_connection_closed,
          oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE,
          bool oaicompat_chat = false) {
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION ||
                    type == SERVER_TASK_TYPE_INFILL);

        if (ctx_server->params_base.embedding) {
          generate_error(
              format_error_response("This server does not support completions. "
                                    "Start it without `--embeddings`",
                                    ERROR_TYPE_NOT_SUPPORTED));
          return;
        }

        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;

        try {
          std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(
              ctx_server->vocab, data.at("prompt"), true, true);
          tasks.reserve(tokenized_prompts.size());
          for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(type);

            task.id = ctx_server->queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens = std::move(tokenized_prompts[i]);
            task.params = server_task::params_from_json_cmpl(
                ctx_server->ctx, ctx_server->params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.oaicompat = oaicompat;
            task.params.oaicompat_cmpl_id = completion_id;
            // oaicompat_model is already populated by params_from_json_cmpl

            tasks.push_back(task);
          }
        } catch (const std::exception &e) {
          generate_error(
              format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
          return;
        }

        ctx_server->queue_results.add_waiting_tasks(tasks);
        ctx_server->queue_tasks.post(tasks);

        bool stream = json_value(data, "stream", false);
        const auto task_ids = server_task::get_list_id(tasks);

        if (!stream) {
          SRV_DBG("mode: %s\n", "no stream");
          ctx_server->receive_multi_results(
              task_ids,
              [&](std::vector<server_task_result_ptr> &results) {
                if (results.size() == 1) {
                  // single result
                  generate_ok(results[0]->to_json());
                } else {
                  // multiple results (multitask)
                  json arr = json::array();
                  for (auto &res : results) {
                    arr.push_back(res->to_json());
                  }
                  generate_ok(arr);
                }
              },
              [&](const json &error_data) { generate_error(error_data); },
              is_connection_closed);

          ctx_server->queue_results.remove_waiting_task_ids(task_ids);
        } else {
          SRV_DBG("mode: %s\n", "stream");

          ctx_server->receive_cmpl_results_stream(
              task_ids,
              [&](server_task_result_ptr &result) -> bool {
                json res_json = result->to_json();
                if (res_json.is_array()) {
                  for (const auto &res : res_json) {
                    generate_ok(res);
                  }
                  return true;
                } else {
                  generate_ok(res_json);
                }
              },
              [&](const json &error_data) { generate_error(error_data); },
              nullptr);

          ctx_server->queue_results.remove_waiting_task_ids(task_ids);
        }
      };

  const auto handle_completions_generate_v2 =
      [&generate_error, &generate_ok](
          server_task_type type, json &data,
          std::function<bool()> is_connection_closed,
          oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE,
          bool oaicompat_chat = false) {
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION ||
                    type == SERVER_TASK_TYPE_INFILL);

        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;

        try {
          std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(
              ctx_server->vocab, data.at("prompt"), true, true);
          tasks.reserve(tokenized_prompts.size());
          for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(type);

            task.id = ctx_server->queue_tasks.get_new_id();
            task.index = i;

            task.prompt_tokens = std::move(tokenized_prompts[i]);
            task.params = server_task::params_from_json_cmpl(
                ctx_server->ctx, ctx_server->params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.oaicompat = oaicompat;
            task.params.oaicompat_cmpl_id = completion_id;
            // oaicompat_model is already populated by params_from_json_cmpl

            tasks.push_back(task);
          }
        } catch (const std::exception &e) {
          generate_error(
              format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
          return;
        }

        ctx_server->queue_results.add_waiting_tasks(tasks);
        ctx_server->queue_tasks.post(tasks);

        bool stream = json_value(data, "stream", false);
        const auto task_ids = server_task::get_list_id(tasks);

        if (!stream) {
          SRV_DBG("mode: %s\n", "no stream");
          std::vector<server_task_result_ptr> results(task_ids.size());
          ctx_server->generate(task_ids, results);
          if (results.size() == 1) {
            // single result
            generate_ok(results[0]->to_json());
          } else {
            // multiple results (multitask)
            json arr = json::array();
            for (auto &res : results) {
              arr.push_back(res->to_json());
            }
            generate_ok(arr);
          }
        } else {
          SRV_DBG("mode: %s\n", "stream");
          ctx_server->generate_stream(task_ids);
          while (true) {
            server_task_result_ptr result;
            int ret = ctx_server->generate_stream_next_token(task_ids, result);
            if (ret == 1) {
              SRV_INF("stop found: %s\n",
                      dump_result(result->to_json()).c_str());
              break;
            } else {
              SRV_INF("content: %s\n", dump_result(result->to_json()).c_str());
            }
          }
        }
      };

  const auto handle_completions = [&handle_completions_generic](
                                      const httplib::Request &req,
                                      httplib::Response &res) {
    json data = oaicompat_completion_params_parse(
        json::parse(req.body), ctx_server->params_base.use_jinja,
        ctx_server->params_base.reasoning_format,
        ctx_server->chat_templates.get());
    return handle_completions_generic(SERVER_TASK_TYPE_COMPLETION, data,
                                      req.is_connection_closed, res,
                                      /* oaicompat */ OAICOMPAT_TYPE_NONE,
                                      /* oaicompat_chat */ false);
  };

  const auto handle_infill = [&res_error, &handle_completions_generic](
                                 const httplib::Request &req,
                                 httplib::Response &res) {
    // check model compatibility
    std::string err;
    if (llama_token_fim_pre(ctx_server->vocab) == LLAMA_TOKEN_NULL) {
      err += "prefix token is missing. ";
    }
    if (llama_token_fim_suf(ctx_server->vocab) == LLAMA_TOKEN_NULL) {
      err += "suffix token is missing. ";
    }
    if (llama_token_fim_mid(ctx_server->vocab) == LLAMA_TOKEN_NULL) {
      err += "middle token is missing. ";
    }
    if (!err.empty()) {
      res_error(res,
                format_error_response(
                    string_format("Infill is not supported by this model: %s",
                                  err.c_str()),
                    ERROR_TYPE_NOT_SUPPORTED));
      return;
    }

    json data = oaicompat_completion_params_parse(
        json::parse(req.body), ctx_server->params_base.use_jinja,
        ctx_server->params_base.reasoning_format,
        ctx_server->chat_templates.get());

    // validate input
    if (data.contains("prompt") && !data.at("prompt").is_string()) {
      // prompt is optional
      res_error(res, format_error_response("\"prompt\" must be a string",
                                           ERROR_TYPE_INVALID_REQUEST));
    }

    if (!data.contains("input_prefix")) {
      res_error(res, format_error_response("\"input_prefix\" is required",
                                           ERROR_TYPE_INVALID_REQUEST));
    }

    if (!data.contains("input_suffix")) {
      res_error(res, format_error_response("\"input_suffix\" is required",
                                           ERROR_TYPE_INVALID_REQUEST));
    }
    if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
      // input_extra is optional
      res_error(
          res, format_error_response("\"input_extra\" must be an array of "
                                     "{\"filename\": string, \"text\": string}",
                                     ERROR_TYPE_INVALID_REQUEST));
      return;
    }

    json input_extra = json_value(data, "input_extra", json::array());
    for (const auto &chunk : input_extra) {
      // { "text": string, "filename": string }
      if (!chunk.contains("text") || !chunk.at("text").is_string()) {
        res_error(res,
                  format_error_response("extra_context chunk must contain a "
                                        "\"text\" field with a string value",
                                        ERROR_TYPE_INVALID_REQUEST));
        return;
      }
      // filename is optional
      if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
        res_error(
            res,
            format_error_response(
                "extra_context chunk's \"filename\" field must be a string",
                ERROR_TYPE_INVALID_REQUEST));
        return;
      }
    }
    data["input_extra"] =
        input_extra;  // default to empty array if it's not exist

    std::string prompt = json_value(data, "prompt", std::string());
    std::vector<llama_tokens> tokenized_prompts =
        tokenize_input_prompts(ctx_server->vocab, prompt, true, true);
    SRV_DBG("creating infill tasks, n_prompts = %d\n",
            (int)tokenized_prompts.size());
    data["prompt"] = format_infill(
        ctx_server->vocab, data.at("input_prefix"), data.at("input_suffix"),
        data.at("input_extra"), ctx_server->params_base.n_batch,
        ctx_server->params_base.n_predict,
        ctx_server->slots[0].n_ctx,  // TODO: there should be a better way
        ctx_server->params_base.spm_infill, tokenized_prompts[0]);

    return handle_completions_generic(SERVER_TASK_TYPE_INFILL, data,
                                      req.is_connection_closed, res);
  };

  const auto handle_chat_completions =
      [&res_error, &handle_completions_generic](const httplib::Request &req,
                                                httplib::Response &res) {
        if (ctx_server->params_base.embedding) {
          res_error(res, format_error_response(
                             "This server does not support completions. Start "
                             "it without `--embeddings`",
                             ERROR_TYPE_NOT_SUPPORTED));
          return;
        }

        json data = oaicompat_completion_params_parse(
            json::parse(req.body), ctx_server->params_base.use_jinja,
            ctx_server->params_base.reasoning_format,
            ctx_server->chat_templates.get());
        return handle_completions_generic(SERVER_TASK_TYPE_COMPLETION, data,
                                          req.is_connection_closed, res,
                                          /* oaicompat */ OAICOMPAT_TYPE_CHAT,
                                          /* oaicompat_chat */ true);
      };

  const auto handle_models = [&res_ok](const httplib::Request &, httplib::Response &res) {
    const json models = {{"object", "list"},
                   {"data",
                    {
                        {{"id", ctx_server->params->model_alias},
                         {"object", "model"},
                         {"created", std::time(0)},
                         {"owned_by", "owner"},
                         {"meta", ctx_server->model_meta()}},
                    }}};

    res_ok(res, models);
  };

  const auto handle_tokenize = [&res_ok](const httplib::Request &req,
                                         httplib::Response &res) {
    const json body = json::parse(req.body);

    json tokens_response = json::array();
    if (body.count("content") != 0) {
      const bool add_special = json_value(body, "add_special", false);
      const bool with_pieces = json_value(body, "with_pieces", false);

      llama_tokens tokens = tokenize_mixed(
          ctx_server->vocab, body.at("content"), add_special, true);

      if (with_pieces) {
        for (const auto &token : tokens) {
          std::string piece = common_token_to_piece(ctx_server->ctx, token);
          json piece_json;

          // Check if the piece is valid UTF-8
          if (is_valid_utf8(piece)) {
            piece_json = piece;
          } else {
            // If not valid UTF-8, store as array of byte values
            piece_json = json::array();
            for (unsigned char c : piece) {
              piece_json.push_back(static_cast<int>(c));
            }
          }

          tokens_response.push_back({{"id", token}, {"piece", piece_json}});
        }
      } else {
        tokens_response = tokens;
      }
    }

    const json data = format_tokenizer_response(tokens_response);
    res_ok(res, data);
  };

  const auto handle_detokenize = [&res_ok](const httplib::Request &req,
                                           httplib::Response &res) {
    const json body = json::parse(req.body);

    std::string content;
    if (body.count("tokens") != 0) {
      const llama_tokens tokens = body.at("tokens");
      content = tokens_to_str(ctx_server->ctx, tokens.cbegin(), tokens.cend());
    }

    const json data = format_detokenized_response(content);
    res_ok(res, data);
  };

  const auto handle_embeddings = [&res_error, &res_ok](
                                     const httplib::Request &req,
                                     httplib::Response &res) {
    const json body = json::parse(req.body);
    bool oaicompat = false;

    // an input prompt can be a string or a list of tokens (integer)
    json prompt;
    if (body.count("input") != 0) {
      oaicompat = true;
      prompt = body.at("input");
    } else if (body.count("content") != 0) {
      // with "content", we only support single prompt
      prompt = std::vector<std::string>{body.at("content")};
    } else {
      res_error(res, format_error_response(
                         "\"input\" or \"content\" must be provided",
                         ERROR_TYPE_INVALID_REQUEST));
      return;
    }

    // create and queue the task
    json responses = json::array();
    bool error = false;
    {
      std::vector<server_task> tasks;
      std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(
          ctx_server->vocab, prompt, /* add_special */ false, true);
      for (size_t i = 0; i < tokenized_prompts.size(); i++) {
        server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);
        task.id = ctx_server->queue_tasks.get_new_id();
        task.index = i;
        task.prompt_tokens = std::move(tokenized_prompts[i]);
        tasks.push_back(task);
      }

      ctx_server->queue_results.add_waiting_tasks(tasks);
      ctx_server->queue_tasks.post(tasks);

      // get the result
      std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

      ctx_server->receive_multi_results(
          task_ids,
          [&](std::vector<server_task_result_ptr> &results) {
            for (auto &res : results) {
              GGML_ASSERT(dynamic_cast<server_task_result_embd *>(res.get()) !=
                          nullptr);
              responses.push_back(res->to_json());
            }
          },
          [&](const json &error_data) {
            res_error(res, error_data);
            error = true;
          },
          req.is_connection_closed);

      ctx_server->queue_results.remove_waiting_task_ids(task_ids);
    }

    if (error) {
      return;
    }

    // write JSON response
    json root = oaicompat
                    ? format_embeddings_response_oaicompat(body, responses)
                : responses.size() == 1 ? responses[0]
                                        : json(responses);
    res_ok(res, root);
  };

  const auto handle_rerank = [&res_error, &res_ok](const httplib::Request &req,
                                                   httplib::Response &res) {
    if (!ctx_server->params_base.reranking ||
        ctx_server->params_base.embedding) {
      res_error(res, format_error_response(
                         "This server does not support reranking. Start it "
                         "with `--reranking` and without `--embedding`",
                         ERROR_TYPE_NOT_SUPPORTED));
      return;
    }

    const json body = json::parse(req.body);

    // TODO: implement
    // int top_n = 1;
    // if (body.count("top_n") != 1) {
    //    top_n = body.at("top_n");
    //} else {
    //    res_error(res, format_error_response("\"top_n\" must be provided",
    //    ERROR_TYPE_INVALID_REQUEST)); return;
    //}
    bool is_tei_format = body.contains("texts");

    json query;
    if (body.count("query") == 1) {
      query = body.at("query");
      if (!query.is_string()) {
        res_error(res, format_error_response("\"query\" must be a string",
                                             ERROR_TYPE_INVALID_REQUEST));
        return;
      }
    } else {
      res_error(res, format_error_response("\"query\" must be provided",
                                           ERROR_TYPE_INVALID_REQUEST));
      return;
    }

    std::vector<std::string> documents =
        json_value(body, "documents", std::vector<std::string>());
    if (documents.empty()) {
      res_error(res, format_error_response(
                         "\"documents\" must be a non-empty string array",
                         ERROR_TYPE_INVALID_REQUEST));
      return;
    }

    llama_tokens tokenized_query = tokenize_input_prompts(
        ctx_server->vocab, query, /* add_special */ false, true)[0];

    // create and queue the task
    json responses = json::array();
    bool error = false;
    {
      std::vector<server_task> tasks;
      std::vector<llama_tokens> tokenized_docs = tokenize_input_prompts(
          ctx_server->vocab, documents, /* add_special */ false, true);
      tasks.reserve(tokenized_docs.size());
      for (size_t i = 0; i < tokenized_docs.size(); i++) {
        server_task task = server_task(SERVER_TASK_TYPE_RERANK);
        task.id = ctx_server->queue_tasks.get_new_id();
        task.index = i;
        task.prompt_tokens = format_rerank(ctx_server->vocab, tokenized_query,
                                           tokenized_docs[i]);
        tasks.push_back(task);
      }

      ctx_server->queue_results.add_waiting_tasks(tasks);
      ctx_server->queue_tasks.post(tasks);

      // get the result
      std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

      ctx_server->receive_multi_results(
          task_ids,
          [&](std::vector<server_task_result_ptr> &results) {
            for (auto &res : results) {
              GGML_ASSERT(dynamic_cast<server_task_result_rerank *>(
                              res.get()) != nullptr);
              responses.push_back(res->to_json());
            }
          },
          [&](const json &error_data) {
            res_error(res, error_data);
            error = true;
          },
          req.is_connection_closed);
    }

    if (error) {
      return;
    }

    // write JSON response
    json root =
        format_response_rerank(body, responses, is_tei_format, documents);
    res_ok(res, root);
  };

  const auto handle_lora_adapters_list = [&](const httplib::Request &,
                                             httplib::Response &res) {
    json result = json::array();
    const auto &loras = ctx_server->params_base.lora_adapters;
    for (size_t i = 0; i < loras.size(); ++i) {
      auto &lora = loras[i];
      result.push_back({
          {"id", i},
          {"path", lora.path},
          {"scale", lora.scale},
      });
    }
    res_ok(res, result);
    res.status = 200;  // HTTP OK
  };

  const auto handle_lora_adapters_apply = [&](const httplib::Request &req,
                                              httplib::Response &res) {
    const json body = json::parse(req.body);
    if (!body.is_array()) {
      res_error(res, format_error_response("Request body must be an array",
                                           ERROR_TYPE_INVALID_REQUEST));
      return;
    }
    server_task task(SERVER_TASK_TYPE_SET_LORA);
    task.id = ctx_server->queue_tasks.get_new_id();
    task.set_lora =
        parse_lora_request(ctx_server->params_base.lora_adapters, body);
    ctx_server->queue_results.add_waiting_task_id(task.id);
    ctx_server->queue_tasks.post(task);

    server_task_result_ptr result = ctx_server->queue_results.recv(task.id);
    ctx_server->queue_results.remove_waiting_task_id(task.id);

    if (result->is_error()) {
      res_error(res, result->to_json());
      return;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_apply_lora *>(result.get()) !=
                nullptr);
    res_ok(res, result->to_json());
  };


  // register API routes
  ctx_server->svr->Get("/health",  handle_health);  // public endpoint (no API key check)
  ctx_server->svr->Get("/metrics", handle_metrics);
  ctx_server->svr->Get("/props", handle_props);
  ctx_server->svr->Post("/props", handle_props_change);
  ctx_server->svr->Get("/models", handle_models);  // public endpoint (no API key check)
  ctx_server->svr->Get("/v1/models", handle_models);  // public endpoint (no API key check)
  ctx_server->svr->Post("/completion", handle_completions);  // legacy
  ctx_server->svr->Post("/completions", handle_completions);
  ctx_server->svr->Post("/v1/completions", handle_completions);
  ctx_server->svr->Post("/chat/completions", handle_chat_completions);
  ctx_server->svr->Post("/v1/chat/completions", handle_chat_completions);
  ctx_server->svr->Post("/infill", handle_infill);
  ctx_server->svr->Post("/embedding", handle_embeddings);  // legacy
  ctx_server->svr->Post("/embeddings", handle_embeddings);
  ctx_server->svr->Post("/v1/embeddings", handle_embeddings);
  ctx_server->svr->Post("/rerank", handle_rerank);
  ctx_server->svr->Post("/reranking", handle_rerank);
  ctx_server->svr->Post("/v1/rerank", handle_rerank);
  ctx_server->svr->Post("/v1/reranking", handle_rerank);
  ctx_server->svr->Post("/tokenize", handle_tokenize);
  ctx_server->svr->Post("/detokenize", handle_detokenize);
  // LoRA adapters hotswap
  ctx_server->svr->Get("/lora-adapters", handle_lora_adapters_list);
  ctx_server->svr->Post("/lora-adapters", handle_lora_adapters_apply);
  // Save & load slots
  ctx_server->svr->Get("/slots", handle_slots);
  ctx_server->svr->Post("/slots/:id_slot", handle_slots_action);

  //
  // Start the server
  //
  if (ctx_server->params->n_threads_http < 1) {
    // +2 threads for monitoring endpoints
    ctx_server->params->n_threads_http =
        std::max(ctx_server->params->n_parallel + 2,
                 (int32_t)std::thread::hardware_concurrency() - 1);
  }
  log_data["n_threads_http"] =
      std::to_string(ctx_server->params->n_threads_http);
  ctx_server->svr->new_task_queue = [] {
    return new httplib::ThreadPool(ctx_server->params->n_threads_http);
  };

  // clean up function, to be called before exit
  auto clean_up = []() {
    ctx_server->svr->stop();
    llama_backend_free();
  };

  // bind HTTP listen port
  bool was_bound = false;
  if (ctx_server->params->port == 0) {
    int bound_port =
        ctx_server->svr->bind_to_any_port(ctx_server->params->hostname);
    if ((was_bound = (bound_port >= 0))) {
      ctx_server->params->port = bound_port;
    }
  } else {
    was_bound = ctx_server->svr->bind_to_port(ctx_server->params->hostname,
                                              ctx_server->params->port);
  }

  if (!was_bound) {
    // LOG_ERROR("couldn't bind HTTP server socket", {
    //     {"hostname", params.hostname},
    //     {"port", params.port},
    // });
    LOG_ERR("%s: couldn't bind HTTP server socket, hostname: %s, port: %d\n",
            __func__, ctx_server->params->hostname.c_str(),
            ctx_server->params->port);
    clean_up();
    return nullptr;
  }

  // run the HTTP server in a thread
  ctx_server->srv_t = std::thread([&] { ctx_server->svr->listen_after_bind(); });
  ctx_server->svr->wait_until_ready();

  LOG_INF(
      "%s: HTTP server is listening, hostname: %s, port: %d, http threads: "
      "%d\n",
      __func__, ctx_server->params->hostname.c_str(), ctx_server->params->port,
      ctx_server->params->n_threads_http);

  // load the model
  LOG_INF("%s: loading model\n", __func__);

  if (!ctx_server->load_model(*ctx_server->params, "")) {
    clean_up();
    ctx_server->srv_t.join();
    LOG_ERR("%s: exiting due to model loading error\n", __func__);
    return nullptr;
  }

  if (model_path.find("14B") != std::string::npos || model_path.find("14b") != std::string::npos) {
      ctx_server->init_14B(model_path);
  } else {
      ctx_server->init(model_path);
  }

  ctx_server->state.store(SERVER_STATE_READY);

  LOG_INF("state 1: %d\n", ctx_server->state.load());

  LOG_INF("%s: model loaded\n", __func__);

  ctx_server->queue_tasks.on_new_task(
      std::bind(&server_context::process_single_task, &*ctx_server,
                std::placeholders::_1));

  ctx_server->queue_tasks.on_update_slots(
      std::bind(&server_context::update_slots, &*ctx_server));

  LOG_INF("%s: server is listening on http://%s:%d - starting the main loop\n",
          __func__, ctx_server->params->hostname.c_str(),
          ctx_server->params->port);

  // ctx_server->queue_tasks.start_loop();
  ctx_server->ctx_t =
      std::thread([&]() { ctx_server->queue_tasks.start_loop(); });

  std::string req_body = string_format(
      "{\"messages\":[{\"content\":\"你好！\",\"role\":\"user\"}],\"stream\":"
      "false,\"model\": \"%s\", \"max_tokens\": 20}",
      ctx_server->hmctx->model_name.c_str());
  // std::string req_body =
  // "{\"messages\":[{\"content\":\"现在假设你是一名会议记录人员，需要你对一段会议记录的每一句话依次进行文本规整化，并直接返回规整化后的文本，回答的文本中不要增加任何额外内容。要求：1、规整化后的文本要忠于原始会议记录，保留会议记录中全部的有用信息；2、去除原文中明显的无意义表达；3、如果原文中存在明显的错别字，也需要进行改正。4、不要删除任何有用的信息。5、请用中文进行回答。6、不允许在回答中添加编造成分。7、如果原始会议记录中存在明显重复的表达，需用清晰通顺的方式改写。8、需注意原文中的数值不得擅自修改。原始会议记录为：“哦，喂啊。啊，你讲啊啊啊。”\",\"role\":\"user\"}],\"stream\":true,\"model\":\"Qwen2.5-32B-Instruct\"}";
  json data = oaicompat_completion_params_parse(
      json::parse(req_body), ctx_server->params_base.use_jinja,
      ctx_server->params_base.reasoning_format,
      ctx_server->chat_templates.get());
  int ret = 0;
  const std::string str =
      data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
  LOG_INF("data: %s\n", str.c_str());
  // handle_completions_generate(
  //     SERVER_TASK_TYPE_COMPLETION,
  //     data,
  //     /* oaicompat */ true,
  //     /* oaicompat_chat */ true);
  //
  // handle_completions_generate_v2(
  //    SERVER_TASK_TYPE_COMPLETION,
  //    data,
  //    /* oaicompat */ true,
  //    /* oaicompat_chat */ true);
  //
  // std::vector<server_task_result_ptr> results;
  // int ret = ctx_server->completion_generate(data, results);
  // if (ret == 0) {
  //    if (results.size() == 1) {
  //        log_ok(results[0]->to_json());
  //    } else {
  //        // multiple results (multitask)
  //        json arr = json::array();
  //        for (auto & res : results) {
  //            arr.push_back(res->to_json());
  //        }
  //        log_ok(arr);
  //    }
  //} else {
  //    LOG_ERR("Error: %d\n", ret);
  //}
  //
  //
  // ret = ctx_server->completion_stream_generate(data);
  // if (ret == 0) {
  //    while (true) {
  //        server_task_result_ptr result;
  //        int ret = ctx_server->completion_stream_generate_next_token(result);
  //        if (ret == 1) {
  //            SRV_INF("stop found: %s\n",
  //            dump_result(result->to_json()).c_str()); break;
  //        } else {
  //            SRV_INF("content: %s\n",
  //            dump_result(result->to_json()).c_str());
  //        }
  //    }
  //} else {
  //    LOG_ERR("Error: %d\n", ret);
  //}

  LOG_INF("Start str generate111\n");

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
      ret = ctx_server->completion_stream_generate_next_token_str(
          result_json_str);
      SRV_INF("next_token_str ret: %d\n", ret);
      if (ret == 1) {
        SRV_INF("stop found str: %s\n", result_json_str.c_str());
        break;
      } else if (ret == -1) {
        LOG_ERR("Error: %d\n", ret);
      } else {
        SRV_INF("completion_stream_generate_next_token_str str: %s\n",
                result_json_str.c_str());
      }
    }
  } else {
    LOG_ERR("Error: %d\n", ret);
  }

  // LOG_INF("%s: thead join\n", __func__);
  // t.join();
  // ctx_t.join();
  // clean_up();
  // t.detach();
  // ctx_t.detach();

  SRV_INF("ctx_server: %p\n", ctx_server);
  SRV_INF("ctx_server->model: %p\n", ctx_server->model);
  SRV_INF("ctx_server->ctx: %p\n", ctx_server->ctx);

  SRV_INF("ctx_server->params: %p\n", ctx_server->params);
  SRV_INF("svr: %p\n", svr);

  SRV_INF("svr running: %d\n", ctx_server->svr->is_running());

#ifdef _WIN32
    server_running = true;
#endif

    return ctx_server;
}

HANDLE completion_create_instance(void* param) {
  static server_context *ctx_server = nullptr;
  // if (ctx_server) {
  //     SRV_INF("ctx_server exist, use it: %p\n", ctx_server);
  //     return ctx_server;
  // }
  ctx_server = new server_context();
  if (ctx_server) {
    SRV_DBG("ctx_server: %p\n", ctx_server);
    SRV_DBG("ctx_server->model: %p\n", ctx_server->model);
    SRV_DBG("ctx_server->ctx: %p\n", ctx_server->ctx);
    SRV_DBG("ctx_server->params: %p\n", ctx_server->params);
    ctx_server->model_path = getenv_or_default("HM_MODEL_PATH", "./models/hmm_deepseek_256_4096_4cores_20250317/");
    const auto p = static_cast<::params*>(param);
    ctx_server->init_houmo_sampling(ctx_server->hm_sampling_processors, p);
  }

  return ctx_server;
}

int completion_load_model(HANDLE h, std::string model_path,
                          std::vector<std::string> model_name_list) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return -1;
  }
  if (model_path == "") {
    model_path = "./";
  }

  model_path = model_path + "/";
  LOG_DBG("model_name_list size = %d\n", model_name_list.size());
  std::string tok_model_path = model_path + "qwen2-1_5b-instruct-q4_0.gguf";
  if (tok_model_path.find("deepseek") != std::string::npos) {
    tok_model_path = model_path + "DeepSeek-R1-Distill-Qwen-1.5B-IQ2_M.gguf";
  } else if(tok_model_path.find("qwen3") != std::string::npos) {
    tok_model_path = model_path + "Qwen3-0.6B-Q8_0.gguf";
  }
  //tok_model_path = model_path + "DeepSeek-R1-Distill-Qwen-1.5B-IQ2_M.gguf";
  //tok_model_path = model_path + "qwen2-1_5b-instruct-q4_0.gguf"; //this works for qwen3-8B szj
  LOG_DBG("load tok_model_path:%s\n",tok_model_path.c_str());

  std::time_t t = std::time(nullptr);
  std::string log_file = string_format("houmo-llm-%d.log", t);

  char *argv[] = {"houmo-llm-demo",
                  "-m",
                  (char *)tok_model_path.c_str(),
                  "--port",
                  "8808",
                  "--host",
                  "0.0.0.0",
                  "--device",
                  "none",
                  "-np",
                  "1",
                  "--ctx-size",
                  "4096",
                  "--no-context-shift",
                  "--no-warmup",
                  "--log-file",
                  (char *)log_file.c_str(),
                  NULL};

  char **p = argv;
  int argc = 0;
  while (*p++ != NULL) {
    argc++;
  }

  // for (int i = 0; i < argc; i++) {
  //     printf("argv[%d] = %s\n", i, argv[i]);
  // }

  // own arguments required by this example
  common_params *p_params = new common_params();

  if (!common_params_parse(argc, argv, *p_params, LLAMA_EXAMPLE_SERVER)) {
    return -1;
  }

  common_init();

  ctx_server->params = p_params;

  llama_backend_init();
  llama_numa_init(ctx_server->params->numa);

  LOG_DBG(
      "system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n",
      ctx_server->params->cpuparams.n_threads,
      ctx_server->params->cpuparams_batch.n_threads,
      std::thread::hardware_concurrency());
  LOG_DBG("\n");
  LOG_DBG("%s\n", common_params_get_system_info(*ctx_server->params).c_str());
  LOG_DBG("\n");

  std::unordered_map<std::string, std::string> log_data;

  log_data["hostname"] = ctx_server->params->hostname;
  log_data["port"] = std::to_string(ctx_server->params->port);

  if (ctx_server->params->api_keys.size() == 1) {
    auto key = ctx_server->params->api_keys[0];
    log_data["api_key"] =
        "api_key: ****" + key.substr(std::max((int)(key.length() - 4), 0));
  } else if (ctx_server->params->api_keys.size() > 1) {
    log_data["api_key"] =
        "api_key: " + std::to_string(ctx_server->params->api_keys.size()) +
        " keys loaded";
  }

  // Necessary similarity of prompt for slot selection
  ctx_server->slot_prompt_similarity =
      ctx_server->params->slot_prompt_similarity;

  // load the model
  LOG_DBG("%s: loading model\n", __func__);

  if (!ctx_server->load_model(*ctx_server->params, model_path)) {
    LOG_ERR("%s: exiting due to model loading error\n", __func__);
    return -1;
  }
  if ((model_path.find("14B") != std::string::npos) || (model_path.find("14b") != std::string::npos)) {
    ctx_server->init_14B(model_path);
  } else if((model_path.find("qwen3") != std::string::npos)&&(model_path.find("8b") != std::string::npos)) {
    ctx_server->init_qwen3_8B(model_path);
  } else {
    ctx_server->init(model_path);
  }
  LOG_DBG("%s","++before store\r\n");
  ctx_server->state.store(SERVER_STATE_READY);
  LOG_DBG("%s","++after store\r\n");

  LOG_DBG("state 1: %d\n", ctx_server->state.load());
  LOG_DBG("%s","++after load\r\n");

  LOG_DBG("%s: model loaded\n", __func__);

  ctx_server->queue_tasks.on_new_task(
      std::bind(&server_context::process_single_task, &*ctx_server,
                std::placeholders::_1));
  LOG_DBG("%s","++after on new task\r\n");

  ctx_server->queue_tasks.on_update_slots(
      std::bind(&server_context::update_slots, &*ctx_server));

  LOG_DBG("%s","++after on update slots\r\n");

  // ctx_server->queue_tasks.start_loop();
  ctx_server->ctx_t =
      std::thread([&]() { ctx_server->queue_tasks.start_loop(); });

  std::string req_body = string_format(
      "{\"messages\":[{\"content\":\"你好！\",\"role\":\"user\"}],\"stream\":"
      "false,\"model\": \"%s\", \"max_tokens\": 20}",
      ctx_server->hmctx->model_name.c_str());
  // std::string req_body =
  // "{\"messages\":[{\"content\":\"现在假设你是一名会议记录人员，需要你对一段会议记录的每一句话依次进行文本规整化，并直接返回规整化后的文本，回答的文本中不要增加任何额外内容。要求：1、规整化后的文本要忠于原始会议记录，保留会议记录中全部的有用信息；2、去除原文中明显的无意义表达；3、如果原文中存在明显的错别字，也需要进行改正。4、不要删除任何有用的信息。5、请用中文进行回答。6、不允许在回答中添加编造成分。7、如果原始会议记录中存在明显重复的表达，需用清晰通顺的方式改写。8、需注意原文中的数值不得擅自修改。原始会议记录为：“哦，喂啊。啊，你讲啊啊啊。”\",\"role\":\"user\"}],\"stream\":true,\"model\":\"Qwen2.5-32B-Instruct\"}";
  json data = oaicompat_completion_params_parse(
      json::parse(req_body), ctx_server->params_base.use_jinja,
      ctx_server->params_base.reasoning_format,
      ctx_server->chat_templates.get());
  LOG_DBG("%s","++after oaicompat param parse\r\n");
  int ret = 0;
  const std::string str =
      data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
  LOG_DBG("data: %s\n", str.c_str());
  // handle_completions_generate(
  //     SERVER_TASK_TYPE_COMPLETION,
  //     data,
  //     /* oaicompat */ true,
  //     /* oaicompat_chat */ true);
  //
  // handle_completions_generate_v2(
  //    SERVER_TASK_TYPE_COMPLETION,
  //    data,
  //    /* oaicompat */ true,
  //    /* oaicompat_chat */ true);
  //
  // std::vector<server_task_result_ptr> results;
  // int ret = ctx_server->completion_generate(data, results);
  // if (ret == 0) {
  //    if (results.size() == 1) {
  //        log_ok(results[0]->to_json());
  //    } else {
  //        // multiple results (multitask)
  //        json arr = json::array();
  //        for (auto & res : results) {
  //            arr.push_back(res->to_json());
  //        }
  //        log_ok(arr);
  //    }
  //} else {
  //    LOG_ERR("Error: %d\n", ret);
  //}
  //
  //
  // ret = ctx_server->completion_stream_generate(data);
  // if (ret == 0) {
  //    while (true) {
  //        server_task_result_ptr result;
  //        int ret = ctx_server->completion_stream_generate_next_token(result);
  //        if (ret == 1) {
  //            SRV_INF("stop found: %s\n",
  //            dump_result(result->to_json()).c_str()); break;
  //        } else {
  //            SRV_INF("content: %s\n",
  //            dump_result(result->to_json()).c_str());
  //        }
  //    }
  //} else {
  //    LOG_ERR("Error: %d\n", ret);
  //}

  LOG_DBG("Start str generate222\n");
  LOG_DBG("%s","++before completion_generate_str\r\n");
  std::string result_json_str;
  ret = ctx_server->completion_generate_str(data, result_json_str);
  if (ret == 0) {
    SRV_INF("completion_generate_str content: %s\n", result_json_str.c_str());
  } else {
    LOG_ERR("Error: %d\n", ret);
  }
  LOG_DBG("%s","++after completion_generate_str before completion_stream_generate\r\n");
  ret = ctx_server->completion_stream_generate(data);
  if (ret == 0) {
    while (true) {
      std::string result_json_str;
      ret = ctx_server->completion_stream_generate_next_token_str(
          result_json_str);
      SRV_INF("next_token_str ret: %d\n", ret);
      if (ctx_server->generate_status == 0) {
        SRV_INF("stop found str: %s\n", result_json_str.c_str());
        break;
      } else if (ret == 2) {
        SRV_INF("Data not ready: %d\n", ret);
      } else if (ret == -1) {
        LOG_ERR("Error: %d\n", ret);
      } else {
        SRV_INF("completion_stream_generate_next_token_str str: %s\n",
                result_json_str.c_str());
      }
    }
  } else {
    LOG_ERR("Error: %d\n", ret);
  }
  LOG_DBG("%s","###after completion_stream_generate\r\n");
  // LOG_INF("%s: thead join\n", __func__);
  // t.join();
  // ctx_t.join();
  // clean_up();
  // t.detach();
  // ctx_t.detach();

  LOG_DBG("ctx_server: %p\n", ctx_server);
  LOG_DBG("ctx_server->model: %p\n", ctx_server->model);
  LOG_DBG("ctx_server->ctx: %p\n", ctx_server->ctx);

  LOG_DBG("ctx_server->params: %p\n", ctx_server->params);

  return 0;
}

int completion_load_model_by_buffer(HANDLE h, uint8_t **buffers,
                                    size_t *bufferSize, size_t buffers_count,
                                    hardware_info hwinfo) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is null: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return -1;
  }

  std::string model_path = ctx_server->model_path;
  std::string tok_model_path = model_path + "qwen2-1_5b-instruct-q4_0.gguf";
  if (tok_model_path.find("deepseek") != std::string::npos) {
    tok_model_path = model_path + "DeepSeek-R1-Distill-Qwen-1.5B-IQ2_M.gguf";
  }

  std::time_t t = std::time(nullptr);
  std::string log_file = string_format("houmo-llm-%d.log", t);

  char *argv[] = {"houmo-llm-demo",
                  "-m",
                  (char *)tok_model_path.c_str(),
                  "--port",
                  "8808",
                  "--host",
                  "0.0.0.0",
                  "--device",
                  "none",
                  "-np",
                  "1",
                  "--ctx-size",
                  "4096",
                  "--no-context-shift",
                  "--no-warmup",
                  "--log-file",
                  (char *)log_file.c_str(),
                  NULL};

  char **p = argv;
  int argc = 0;
  while (*p++ != NULL) {
    argc++;
  }

  // for (int i = 0; i < argc; i++) {
  //     printf("argv[%d] = %s\n", i, argv[i]);
  // }

  // own arguments required by this example
  common_params *p_params = new common_params();

  if (!common_params_parse(argc, argv, *p_params, LLAMA_EXAMPLE_SERVER)) {
    return -1;
  }

  common_init();

  ctx_server->params = p_params;

  llama_backend_init();
  llama_numa_init(ctx_server->params->numa);

  LOG_DBG(
      "system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n",
      ctx_server->params->cpuparams.n_threads,
      ctx_server->params->cpuparams_batch.n_threads,
      std::thread::hardware_concurrency());
  LOG_DBG("\n");
  LOG_DBG("%s\n", common_params_get_system_info(*ctx_server->params).c_str());
  LOG_DBG("\n");

  std::unordered_map<std::string, std::string> log_data;

  log_data["hostname"] = ctx_server->params->hostname;
  log_data["port"] = std::to_string(ctx_server->params->port);

  if (ctx_server->params->api_keys.size() == 1) {
    auto key = ctx_server->params->api_keys[0];
    log_data["api_key"] =
        "api_key: ****" + key.substr(std::max((int)(key.length() - 4), 0));
  } else if (ctx_server->params->api_keys.size() > 1) {
    log_data["api_key"] =
        "api_key: " + std::to_string(ctx_server->params->api_keys.size()) +
        " keys loaded";
  }

  // Necessary similarity of prompt for slot selection
  ctx_server->slot_prompt_similarity =
      ctx_server->params->slot_prompt_similarity;

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

  ctx_server->queue_tasks.on_new_task(
      std::bind(&server_context::process_single_task, &*ctx_server,
                std::placeholders::_1));

  ctx_server->queue_tasks.on_update_slots(
      std::bind(&server_context::update_slots, &*ctx_server));

  // ctx_server->queue_tasks.start_loop();
  ctx_server->ctx_t =
      std::thread([&]() { ctx_server->queue_tasks.start_loop(); });

  std::string req_body = string_format(
      "{\"messages\":[{\"content\":\"你好！\",\"role\":\"user\"}],\"stream\":"
      "false,\"model\": \"%s\", \"max_tokens\": 20}",
      ctx_server->hmctx->model_name.c_str());
  // std::string req_body =
  // "{\"messages\":[{\"content\":\"现在假设你是一名会议记录人员，需要你对一段会议记录的每一句话依次进行文本规整化，并直接返回规整化后的文本，回答的文本中不要增加任何额外内容。要求：1、规整化后的文本要忠于原始会议记录，保留会议记录中全部的有用信息；2、去除原文中明显的无意义表达；3、如果原文中存在明显的错别字，也需要进行改正。4、不要删除任何有用的信息。5、请用中文进行回答。6、不允许在回答中添加编造成分。7、如果原始会议记录中存在明显重复的表达，需用清晰通顺的方式改写。8、需注意原文中的数值不得擅自修改。原始会议记录为：“哦，喂啊。啊，你讲啊啊啊。”\",\"role\":\"user\"}],\"stream\":true,\"model\":\"Qwen2.5-32B-Instruct\"}";
  json data = oaicompat_completion_params_parse(
      json::parse(req_body), ctx_server->params_base.use_jinja,
      ctx_server->params_base.reasoning_format,
      ctx_server->chat_templates.get());
  int ret = 0;
  const std::string str =
      data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
  LOG_DBG("data: %s\n", str.c_str());
  // handle_completions_generate(
  //     SERVER_TASK_TYPE_COMPLETION,
  //     data,
  //     /* oaicompat */ true,
  //     /* oaicompat_chat */ true);
  //
  // handle_completions_generate_v2(
  //    SERVER_TASK_TYPE_COMPLETION,
  //    data,
  //    /* oaicompat */ true,
  //    /* oaicompat_chat */ true);
  //
  // std::vector<server_task_result_ptr> results;
  // int ret = ctx_server->completion_generate(data, results);
  // if (ret == 0) {
  //    if (results.size() == 1) {
  //        log_ok(results[0]->to_json());
  //    } else {
  //        // multiple results (multitask)
  //        json arr = json::array();
  //        for (auto & res : results) {
  //            arr.push_back(res->to_json());
  //        }
  //        log_ok(arr);
  //    }
  //} else {
  //    LOG_ERR("Error: %d\n", ret);
  //}
  //
  //
  // ret = ctx_server->completion_stream_generate(data);
  // if (ret == 0) {
  //    while (true) {
  //        server_task_result_ptr result;
  //        int ret = ctx_server->completion_stream_generate_next_token(result);
  //        if (ret == 1) {
  //            SRV_INF("stop found: %s\n",
  //            dump_result(result->to_json()).c_str()); break;
  //        } else {
  //            SRV_INF("content: %s\n",
  //            dump_result(result->to_json()).c_str());
  //        }
  //    }
  //} else {
  //    LOG_ERR("Error: %d\n", ret);
  //}

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
      ret = ctx_server->completion_stream_generate_next_token_str(
          result_json_str);
      SRV_INF("next_token_str ret: %d\n", ret);
      if (ctx_server->generate_status == 0) {
        SRV_INF("stop found str: %s\n", result_json_str.c_str());
        break;
      } else if (ret == 2) {
        SRV_INF("Data not ready: %d\n", ret);
      } else if (ret == -1) {
        LOG_ERR("Error: %d\n", ret);
      } else {
        SRV_INF("completion_stream_generate_next_token_str str: %s\n",
                result_json_str.c_str());
      }
    }
  } else {
    LOG_ERR("Error: %d\n", ret);
  }

  // LOG_INF("%s: thead join\n", __func__);
  // t.join();
  // ctx_t.join();
  // clean_up();
  // t.detach();
  // ctx_t.detach();

  LOG_DBG("ctx_server: %p\n", ctx_server);
  LOG_DBG("ctx_server->model: %p\n", ctx_server->model);
  LOG_DBG("ctx_server->ctx: %p\n", ctx_server->ctx);

  LOG_DBG("ctx_server->params: %p\n", ctx_server->params);

  return 0;
}

// 似乎也没被调用
int generate(HANDLE h) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return -1;
  }

  LOG_DBG("ctx_server: %p\n", ctx_server);

  LOG_DBG("ctx_server->model: %p\n", ctx_server->model);
  LOG_DBG("ctx_server->ctx: %p\n", ctx_server->ctx);

  LOG_DBG("ctx_server->params: %p\n", ctx_server->params);

  std::string req_body = string_format(
      "{\"messages\":[{\"content\":\"你好！\",\"role\":\"user\"}],\"stream\":"
      "false,\"model\": \"%s\", \"max_tokens\": 20}",
      ctx_server->hmctx->model_name.c_str());
  // std::string req_body =
  // "{\"messages\":[{\"content\":\"现在假设你是一名会议记录人员，需要你对一段会议记录的每一句话依次进行文本规整化，并直接返回规整化后的文本，回答的文本中不要增加任何额外内容。要求：1、规整化后的文本要忠于原始会议记录，保留会议记录中全部的有用信息；2、去除原文中明显的无意义表达；3、如果原文中存在明显的错别字，也需要进行改正。4、不要删除任何有用的信息。5、请用中文进行回答。6、不允许在回答中添加编造成分。7、如果原始会议记录中存在明显重复的表达，需用清晰通顺的方式改写。8、需注意原文中的数值不得擅自修改。原始会议记录为：“哦，喂啊。啊，你讲啊啊啊。”\",\"role\":\"user\"}],\"stream\":true,\"model\":\"Qwen2.5-32B-Instruct\"}";
  json data = oaicompat_completion_params_parse(
      json::parse(req_body), ctx_server->params_base.use_jinja,
      ctx_server->params_base.reasoning_format,
      ctx_server->chat_templates.get());
  int ret = 0;
  const std::string str =
      data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
  LOG_DBG("data: %s\n", str.c_str());

  LOG_DBG("Start str generate 444\n");

  std::string result_json_str;
  ret = ctx_server->completion_generate_str(data, result_json_str);
  if (ret == 0) {
    LOG_DBG("completion_generate_str content: %s\n", result_json_str.c_str());
  } else {
    LOG_ERR("Error: %d\n", ret);
  }

  LOG_DBG("ctx_server: %p\n", ctx_server);
  LOG_DBG("ctx_server->svr 0: %d\n", ctx_server->svr->is_running());

  return 0;
}

int completion_generate(HANDLE h, json &req, std::string &result) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return -1;
  }

  LOG_DBG("ctx_server: %p\n", ctx_server);

  json data = oaicompat_completion_params_parse(
      req, ctx_server->params_base.use_jinja,
      ctx_server->params_base.reasoning_format,
      ctx_server->chat_templates.get());
  int ret = 0;
  const std::string str =
      data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
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

// 未被调用
int generate_stream(HANDLE h) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return -1;
  }

  std::string req_body = string_format(
      "{\"messages\":[{\"content\":\"你好！\",\"role\":\"user\"}],\"stream\":"
      "true,\"model\": \"%s\", \"max_tokens\": 20}",
      ctx_server->hmctx->model_name.c_str());
  json data = oaicompat_completion_params_parse(
      json::parse(req_body), ctx_server->params_base.use_jinja,
      ctx_server->params_base.reasoning_format,
      ctx_server->chat_templates.get());
  int ret = 0;
  const std::string str =
      data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
  LOG_DBG("data: %s\n", str.c_str());

  LOG_DBG("Start str generate 5555\n");

  ret = ctx_server->completion_stream_generate(data);
  if (ret == 0) {
  } else {
    LOG_ERR("Error: %d\n", ret);
  }

  LOG_DBG("ctx_server: %p\n", ctx_server);
  LOG_DBG("ctx_server->svr 0: %d\n", ctx_server->svr->is_running());

  return 0;
}

int completion_generate_stream(HANDLE h, json &req) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;
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

  LOG_DBG("Start str generate 666\n");

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
  server_context *ctx_server = (server_context *)h;
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
  // req_body = "{\"prompt\": \"你好！\"}";
  json data = json::parse(req_body);
  int ret = 0;
  const std::string str =
      data.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
  LOG_INF("data: %s\n", str.c_str());

  ret = ctx_server->completion_token_count(data);
  SRV_INF("generate_token_count ret: %d\n", ret);

  return ret;
}

// szj 此函数未调用
int generate_stream_next_token_all(HANDLE h) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return -1;
  }

  LOG_DBG("Start str generate 7777\n");

  while (true) {
    std::string result_json_str;
    int ret =
        ctx_server->completion_stream_generate_next_token_str(result_json_str);
    if (ret == 1) {
      SRV_INF("stop found str: %s\n", result_json_str.c_str());
      break;
    } else if (ret == -1) {
      LOG_ERR("Error: %d\n", ret);
      break;
    } else {
      SRV_INF("completion_stream_generate_next_token_str str: %s\n",
              result_json_str.c_str());
    }
  }

  LOG_DBG("ctx_server: %p\n", ctx_server);
  LOG_DBG("ctx_server->svr 0: %d\n", ctx_server->svr->is_running());

  return 0;
}

// szj 此函数未调用
int generate_stream_next_token_string(HANDLE h, std::string &result_json_str) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return -1;
  }

  int ret =
      ctx_server->completion_stream_generate_next_token_str(result_json_str);

  return ret;
}

// api_Get_Generate_token 调用下面的函数
int generate_stream_next_token(HANDLE h, char **result, size_t *result_size) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  if (result == nullptr) {
    LOG_ERR("result is nullptr\n");
  }

  // SRV_INF("*result: %p\n", *result);
  server_context *ctx_server = (server_context *)h;
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return -1;
  }
  std::string result_json_str = "";

  int ret = 0;
  if (result != nullptr && *result) {
    SRV_INF("%s",
            "++confirm True：completion_stream_generate_next_token_str\r\n");
    ret =
        ctx_server->completion_stream_generate_next_token_str(result_json_str);
  } else {
    SRV_INF("%s",
            "--confirm False：completion_stream_generate_next_token_str\r\n");
    ret = ctx_server->completion_stream_generate_next_token_str(result_json_str,
                                                                false);
  }
  SRV_INF("+++ret: %d\n", ret);

  // SRV_INF("result str: %s\n", result_json_str.c_str());
  if (ret == -1) {
    return ret;
  }
  // if (ret == 2) {
  //     return ret;
  // }

  // char * data = new char [result_json_str.length()+1];
  // size_t size = result_json_str.length();
  // strcpy (data, result_json_str.c_str());
  ////SRV_INF("result: %p\n", result);
  //*result = data;
  // result_size = &size;
  size_t len = strlen(result_json_str.c_str()) + 1;
  SRV_INF("len: %d\n", len);

  // SRV_INF("result_json_str len: %d\n", result_json_str.length());
  if (len > 0) {
    if (result == nullptr || !*result || *result_size < len) {
      SRV_INF("%s ","enter 1*result: \n");
      *result_size = len;
      return ret;
    } else {
      SRV_INF("2 *result: %p\n", *result);
      SRV_INF("result_size: %d\n", *result_size);
      strcpy(*result, result_json_str.c_str());

      SRV_INF("*result length: %d\n", strlen(*result));
      SRV_INF("*result size: %d\n", (sizeof(*result) / sizeof(**result)));

      //*result[*result_size - 1] = '\0';
      // strncpy(*result, result_json_str.c_str(), len);
      // SRV_INF("result_json_str: %s\n", result_json_str.c_str());
      SRV_INF("result str: %s\n", *result);
      //*result[len - 1] = '\0';
      // SRV_INF("2 result: %s\n", *result);
    }
    // SRV_INF("result_size: %d\n", *result_size);
    // SRV_INF("data: %s\n", data);
    // SRV_INF("2 *result: %s\n", *result);
  }

  return ret;
}

int generate_stream_stop(HANDLE h) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;

  int ret = ctx_server->completion_stream_generate_stop();
  return ret;
}

int destroy_instance(HANDLE h) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;
  SRV_INF("ctx_server->svr 0: %d\n", ctx_server->svr->is_running());

  ctx_server->queue_tasks.terminate();
  SRV_INF("ctx_server->svr 1: %d\n", ctx_server->svr->is_running());
  ctx_server->svr->stop();
  SRV_INF("ctx_server->svr 2: %d\n", ctx_server->svr->is_running());
  // ctx_server->destroy_server_context();
  ctx_server->ctx_t.join();
  SRV_DBG("join ctx_t finished: %s\n", "");
  ctx_server->srv_t.join();
  SRV_INF("join srv_t finished: %s\n", "");
  SRV_INF("ctx_server->svr 3: %d\n", ctx_server->svr->is_running());
  delete ctx_server;
  ctx_server = nullptr;

  SRV_INF("ctx_server: %p\n", ctx_server);
  // ctx_server = nullptr;
  SRV_INF("ctx_server: %p\n", ctx_server);
  return 0;
}

int completion_destroy_instance(HANDLE h) {
  if (h == nullptr) {
    LOG_ERR("HANDLE is: %p\n", h);
    return -1;
  }
  server_context *ctx_server = (server_context *)h;

  ctx_server->queue_tasks.terminate();
  // ctx_server->destroy_server_context();
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
  // ctx_server->destroy_server_context();
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
  // return;
  completion_destroy_instance(handle);
  return;
}

// load model with model name ,layer:-1 load all layer.
// bool HTTP_SERVER_API api_loadmodel_by_path(HANDLE handle, const std::string&
// model_path, std::vector<std::string>& model_name, hardware_info hwinfo);
bool HTTP_SERVER_API api_loadmodel_by_path(HANDLE handle,
                                           const char *model_path,
                                           const char **model_names,
                                           size_t model_name_count,
                                           hardware_info hwinfo) {
  SRV_DBG("model_path: %s\n", model_path);
  SRV_DBG("model_name: %s\n", model_names[model_name_count - 1]);

  common_log_pause(common_log_main());
  std::vector<std::string> model_name_list;
  for (int i = 0; i < model_name_count; i++) {
    model_name_list.push_back(model_names[i]);
  }
  int ret = completion_load_model(handle, model_path, model_name_list);
  common_log_resume(common_log_main());

  if (ret == 0) {
    return true;
  }
  return false;
}

// load model with buffer stream,layer:-1 load all layer.
// bool HTTP_SERVER_API api_loadmodel_by_buffer(HANDLE handle,
// std::vector<uint8_t*>& buffers, std::vector<size_t>& buffersSize,
// hardware_info hwinfo);
bool HTTP_SERVER_API api_loadmodel_by_buffer(HANDLE handle, uint8_t **buffers,
                                             size_t *bufferSize,
                                             size_t buffers_count,
                                             hardware_info hwinfo) {
  SRV_DBG("buffers_count: %d\n", buffers_count);

  common_log_pause(common_log_main());
  int ret = completion_load_model_by_buffer(handle, buffers, bufferSize,
                                            buffers_count, hwinfo);
  common_log_resume(common_log_main());

  if (ret == 0) {
    return true;
  }
  SRV_ERR("%s", "++Loadmodel from buffer error, Please Check!\r\n");
  return false;
}

// load token model not must
// bool HTTP_SERVER_API api_loadtoken(HANDLE handle, const std::string&
// token_fullpath);
bool HTTP_SERVER_API api_loadtoken(HANDLE handle, const char *token_fullpath) {
  SRV_DBG("token_fullpath: %s\n", token_fullpath);
  return true;
}

// unload model
bool HTTP_SERVER_API api_unloadmodel(HANDLE handle) {
  // return true;
  int ret = completion_unload_model(handle);
  LOG_DBG("completion_unload_model: %s\n", ret);
  if (ret == 0) {
    return true;
  }
  return false;
}

// model status see the enum status
// return enum status
int32 HTTP_SERVER_API api_status(HANDLE handle) {
  int32 status = 0;
  if (handle == nullptr) {
    LOG_ERR("HANDLE is: %p\n", handle);
    return -1;
  }
  server_context *ctx_server = (server_context *)handle;
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return -1;
  }
  status = ctx_server->model_status;
  return status;
}

// Generate
// std::string HTTP_SERVER_API api_Generate(HANDLE handle, const std::string&
// prompt, const params& params); if result == nullptr or *result_size is not
// enough, return E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HTTP_SERVER_API api_Generate(HANDLE handle, const char *prompt,
                                   const void *gen_params,
                                   size_t gen_params_size, char **result,
                                   size_t *result_size) {
  SRV_DBG("prompt: %s\n", prompt);
  auto p = (params *)gen_params;
  if (handle == nullptr) {
    LOG_ERR("HANDLE is: %p\n", handle);
    return -1;
  }

  server_context *ctx_server = (server_context *)handle;

  std::string req_body =
      "{\"messages\":[{\"content\":\"%s”\",\"role\":\"user\"}],\"stream\":"
      "false,\"model\": \"%s\", \"max_tokens\": %d}";
  req_body = string_format(req_body.c_str(), prompt,
                           ctx_server->hmctx->model_name.c_str(), p->n_predict);
  json req = json::parse(req_body);
  std::string decode_result;
  int ret = completion_generate(handle, req, decode_result);
  char *result_ptr =
      new char[decode_result.size() + 1];  // +1 for terminating NUL

  strncpy(result_ptr, decode_result.c_str(), decode_result.size() + 1);

  *result = result_ptr;
  *result_size = decode_result.size();
  SRV_DBG("decode_result: %s\n", decode_result.c_str());
  SRV_DBG("result: %s\n", *result);

  return 0;
}
/*
{
"content": "我是联想基于通义千问打造的天禧大模型。",
"id_slot": 0,
"stop": true,
"model": "model0_v2.0.2.0-confused.gguf",
"tokens_predicted": 15,
"tokens_evaluated": 23,
"generation_settings": {
        "n_ctx": 4096,
        "n_predict": 1024,
        "model": "model0_v2.0.2.0-confused.gguf",
        "seed": 4294967295,
        "temperature": 0.699999988079071,
        "dynatemp_range": 0,
        "dynatemp_exponent": 1,
        "top_k": 20,
        "top_p": 0.800000011920929,
        "min_p": 0.05000000074505806,
        "tfs_z": 1,
        "typical_p": 1,
        "repeat_last_n": 64,
        "repeat_penalty": 1.0499999523162842,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "penalty_prompt_tokens": [],
        "use_penalty_prompt_tokens": false,
        "mirostat": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.10000000149011612,
        "penalize_nl": false,
        "stop": [],
        "n_keep": 0,
        "ignore_eos": false,
        "stream": false,
        "logit_bias": [],
        "n_probs": 0,
        "min_keep": 0,
        "grammar": "",
        "samplers": [
                "top_k",
                "tfs_z",
                "typical_p",
                "top_p",
                "min_p",
                "temperature"
        ]
},
"prompt": "",
"truncated": false,
"stopped_eos": true,
"stopped_word": false,
"stopped_limit": false,
"stopping_word": "",
"tokens_cached": 37,
"timings": {
        "prompt_n": 23,
        "prompt_ms": 178.442,
        "prompt_per_token_ms": 7.758347826086957,
        "prompt_per_second": 128.89342195223097,
        "predicted_n": 15,
        "predicted_ms": 1012.613,
        "predicted_per_token_ms": 67.50753333333334,
        "predicted_per_second": 14.813161592829639
}
}
*/

// get prompt token number
// int32 HTTP_SERVER_API api_token_num(HANDLE handle, const std::string&
// prompt);
int32 HTTP_SERVER_API api_token_num(HANDLE handle, const char *prompt) {
  SRV_DBG("prompt: %s\n", prompt);
  return generate_token_count(handle, prompt);
}

// stream generate
// bool HTTP_SERVER_API api_Generate_stream(HANDLE handle, const std::string&
// prompt, const params& params);
bool HTTP_SERVER_API api_Generate_stream(HANDLE handle, const char *prompt,
                                         const void *gen_params,
                                         size_t gen_params_size) {
  SRV_DBG("stream prompt: %s\n", prompt);
  std::string pp(prompt);
  std::string input = pp;
  input = std::regex_replace(input, std::regex(R"(\n)"), "\\n");
  input = std::regex_replace(input, std::regex(R"(\r)"), "\\r");
  input = std::regex_replace(input, std::regex(R"(\")"), "\\\"");
  SRV_INF("input: %s\n", input.c_str());
  rtrim(input);
  auto p = (params *)gen_params;
  int ret = 0;
  if (handle == nullptr) {
    LOG_ERR("HANDLE is: %p\n", handle);
    return -1;
  }

  server_context *ctx_server = (server_context *)handle;
  try {
    std::string req_body =
        "{\"messages\":[{\"content\":\"%s\",\"role\":\"user\"}],\"stream\":"
        "true,\"model\": \"%s\", \"max_tokens\": %d}";
    req_body =
        string_format(req_body.c_str(), input.c_str(),
                      ctx_server->hmctx->model_name.c_str(), p->n_predict);
    // SRV_INF("req_body: %s\n", req_body.c_str());
    // std::cout.flush();
    // bool flag = true;
    // write_utf8_cstr_to_stdout(req_body.c_str(), flag);
    // std::cout.flush();
    // std::cout << std::endl;
    // std::cout.flush();
    json req = json::parse(req_body);
    ret = completion_generate_stream(handle, req);
  } catch (const std::exception &e) {
    // std::cout.flush();
    std::cout << e.what() << std::endl;
    throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
  }
  if (ret != 0) {
    return false;
  }
  return true;
}

// the generate token ,use for stream mode
// if result == nullptr or *result_size is not enough, return
// E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HTTP_SERVER_API api_Get_Generate_token(HANDLE handle, char **result,
                                             size_t *result_size) {
  LOG_DBG("+++++++++++++++++++call api_Get_Generate_token\r\n");
  int ret = 0;
  try {
    ret = generate_stream_next_token(handle, result, result_size);
  } catch (const std::exception &e) {
    // std::cout.flush();
    std::cout << e.what() << std::endl;
    ret = -1;
  }
  return ret;
}
//{"content":" ","stop":false,"id_slot":0,"multimodal":false}
//{"content":"","id_slot":0,"stop":true,"model":"model0_v2.0.2.0-confused.gguf","tokens_predicted":79,"tokens_evaluated":23,"generation_settings":{"n_ctx":4096,"n_predict":1024,"model":"model0_v2.0.2.0-confused.gguf","seed":4294967295,"temperature":0.699999988079071,"dynatemp_range":0.0,"dynatemp_exponent":1.0,"top_k":20,"top_p":0.800000011920929,"min_p":0.05000000074505806,"tfs_z":1.0,"typical_p":1.0,"repeat_last_n":64,"repeat_penalty":1.0499999523162842,"presence_penalty":0.0,"frequency_penalty":0.0,"penalty_prompt_tokens":[],"use_penalty_prompt_tokens":false,"mirostat":0,"mirostat_tau":5.0,"mirostat_eta":0.10000000149011612,"penalize_nl":false,"stop":[],"n_keep":0,"ignore_eos":false,"stream":true,"logit_bias":[],"n_probs":0,"min_keep":0,"grammar":"","samplers":["top_k","tfs_z","typical_p","top_p","min_p","temperature"]},"prompt":"","truncated":false,"stopped_eos":true,"stopped_word":false,"stopped_limit":false,"stopping_word":"","tokens_cached":101,"timings":{"prompt_n":23,"prompt_ms":183.544,"prompt_per_token_ms":7.980173913043479,"prompt_per_second":125.31055223815542,"predicted_n":79,"predicted_ms":5507.942,"predicted_per_token_ms":69.72078481012659,"predicted_per_second":14.342925179676909}}

// generate status see generater_status
// return enum generater_status
int32 HTTP_SERVER_API api_Generate_status(HANDLE handle) {
  int32 status = 0;
  if (handle == nullptr) {
    LOG_ERR("HANDLE is: %p\n", handle);
    return -1;
  }

  server_context *ctx_server = (server_context *)handle;
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return -1;
  }
  status = ctx_server->generate_status;
  return status;
}

// stop generate
bool HTTP_SERVER_API api_stop(HANDLE handle) {
  int ret = generate_stream_stop(handle);
  if (ret == 0) {
    return true;
  }
  return false;
}

// warmup not must
bool HTTP_SERVER_API api_warmup(HANDLE handle) { return true; }

// reset model not must
void HTTP_SERVER_API api_Reset(HANDLE handle) { return; }

void HTTP_SERVER_API api_get_info(HANDLE handle) {
  if (handle == nullptr) {
    LOG_ERR("HANDLE is: %p\n", handle);
    return;
  }
  server_context *ctx_server = (server_context *)handle;
  if (ctx_server->magic != 12345) {
    LOG_ERR("HANDLE is wrong: %p\n", ctx_server);
    return;
  }
  printf("info status: %d\n", ctx_server->generate_status);
  printf("info stop: %d\n", ctx_server->stop);
  return;
}


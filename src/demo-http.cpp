#include "utils.hpp"

#include "arg.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
// mime type for sending response
#define MIMETYPE_JSON "application/json; charset=utf-8"

// auto generated files (update with ./deps.sh)
#include "index.html.hpp"
#include "loading.html.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cinttypes>
#include <deque>
#include <memory>
#include <mutex>
#include <signal.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <torch/torch.h>
#include "tcim/tcim_runtime.h"
#if (__GNUC__ < 8 && !defined(_MSC_VER))
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

using json = nlohmann::ordered_json;

template<typename ... Args>
std::string hm_string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

static void int16toint16(int16_t *src, int16_t *dst, size_t size) {
    for (int i=0;i<size;i++){
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
// In case of invalid UTF-8, invalid_utf8 is set to true on Windows, and something
// a human-readable is written instead.
//
// On non-Windows systems, simply printfs() the string.
static void write_utf8_cstr_to_stdout(const char * str, bool & invalid_utf8) {
        invalid_utf8 = false;

#if defined(_WIN32)
        // Are we in a console?
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD dwMode = 0;

        // According to Microsoft docs:
        // "WriteConsole fails if it is used with a standard handle that is redirected to a file."
        // Also according to the docs, you can use GetConsoleMode to check for that.
        if (hConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(hConsole, &dwMode)) {
            printf("%s", str);
            return;
        }

        // MultiByteToWideChar reports an error if str is empty, don't report
        // them as invalid_utf8.
        if (*str == 0) {
            return;
        }
        int length_needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, str, strlen(str), NULL, 0);
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
                    printf("%02x", (uint8_t) str[i]);
                }
                printf(">");
                return;
            }
            GGML_ABORT("MultiByteToWideChar() failed in an unexpected way.");
        }

        LPWSTR wstr = (LPWSTR) calloc(length_needed+1, sizeof(*wstr));
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

std::string dump_result(const json & data) {
    const std::string str =
    data.dump(-1, ' ', false, json::error_handler_t::replace) +
        "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
    return str;
}

void log_error(const json & error_data) {
    const std::string str =
    error_data.dump(-1, ' ', false, json::error_handler_t::replace) +
        "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).

    SRV_ERR("%s\n", str.c_str());
}

void log_ok(const json & data) {
    const std::string str =
    data.dump(-1, ' ', false, json::error_handler_t::replace) +
        "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
    SRV_INF("%s\n", str.c_str());
}

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

// state diagram: https://github.com/ggerganov/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_STARTED, // TODO: this state is only used for setting up the initial prompt processing; maybe merge it with launch_slot_with_task in the future
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
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

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
};

struct slot_params {
    bool stream       = true;
    bool cache_prompt = false; // remember the prompt to avoid reprocessing all prompt

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // mininum line indentation for the generated text in number of whitespace characters

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::vector<std::string> antiprompt;
    bool timings_per_token = false;
    bool ignore_eos = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // OAI-compat fields
    bool        verbose        = false;
    bool        oaicompat      = false;
    bool        oaicompat_chat = true;
    std::string oaicompat_model;
    std::string oaicompat_cmpl_id;

    json to_json() const {
        std::vector<std::string> samplers;
        samplers.reserve(sampling.samplers.size());
        for (const auto & sampler : sampling.samplers) {
            samplers.emplace_back(common_sampler_type_to_str(sampler));
        }

        return json {
            {"n_predict",                 n_predict},     // Server configured n_predict
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
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
            {"penalize_nl",               sampling.penalize_nl},
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
            {"samplers",                  samplers},
            {"speculative.n_max",         speculative.n_max},
            {"speculative.n_min",         speculative.n_min},
            {"speculative.p_min",         speculative.p_min},
            {"timings_per_token",         timings_per_token},
        };
    }
};

struct server_task {
    int id    = -1; // to be filled by server_queue
    int index = -1; // used when there are multiple prompts (batch request)

    server_task_type type;

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;

    // used by SERVER_TASK_TYPE_INFERENCE
    slot_params  params;
    llama_tokens prompt_tokens;
    int id_selected_slot = -1;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int slot_id;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    server_task(server_task_type type) : type(type) {}

    static slot_params params_from_json_cmpl(
            const llama_model * model,
            const common_params & params_base,
            const json & data) {
        slot_params params;

        // Sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
        slot_params defaults;
        defaults.sampling    = params_base.sampling;
        defaults.speculative = params_base.speculative;

        // enabling this will output extra debug information in the HTTP responses from the server
        params.verbose           = params_base.verbosity > 9;
        params.timings_per_token = json_value(data, "timings_per_token", false);

        params.stream           = json_value(data, "stream",             false);
        params.cache_prompt     = json_value(data, "cache_prompt",       false);
        params.n_predict        = json_value(data, "n_predict",          json_value(data, "max_tokens", defaults.n_predict));
        params.n_indent         = json_value(data, "n_indent",           defaults.n_indent);
        params.n_keep           = json_value(data, "n_keep",             defaults.n_keep);
        params.n_discard        = json_value(data, "n_discard",          defaults.n_discard);
      //params.t_max_prompt_ms  = json_value(data, "t_max_prompt_ms",    defaults.t_max_prompt_ms); // TODO: implement
        params.t_max_predict_ms = json_value(data, "t_max_predict_ms",   defaults.t_max_predict_ms);

        params.sampling.top_k              = json_value(data, "top_k",              defaults.sampling.top_k);
        params.sampling.top_p              = json_value(data, "top_p",              defaults.sampling.top_p);
        params.sampling.min_p              = json_value(data, "min_p",              defaults.sampling.min_p);
        params.sampling.xtc_probability    = json_value(data, "xtc_probability",    defaults.sampling.xtc_probability);
        params.sampling.xtc_threshold      = json_value(data, "xtc_threshold",      defaults.sampling.xtc_threshold);
        params.sampling.typ_p              = json_value(data, "typical_p",          defaults.sampling.typ_p);
        params.sampling.temp               = json_value(data, "temperature",        defaults.sampling.temp);
        params.sampling.dynatemp_range     = json_value(data, "dynatemp_range",     defaults.sampling.dynatemp_range);
        params.sampling.dynatemp_exponent  = json_value(data, "dynatemp_exponent",  defaults.sampling.dynatemp_exponent);
        params.sampling.penalty_last_n     = json_value(data, "repeat_last_n",      defaults.sampling.penalty_last_n);
        params.sampling.penalty_repeat     = json_value(data, "repeat_penalty",     defaults.sampling.penalty_repeat);
        params.sampling.penalty_freq       = json_value(data, "frequency_penalty",  defaults.sampling.penalty_freq);
        params.sampling.penalty_present    = json_value(data, "presence_penalty",   defaults.sampling.penalty_present);
        params.sampling.dry_multiplier     = json_value(data, "dry_multiplier",     defaults.sampling.dry_multiplier);
        params.sampling.dry_base           = json_value(data, "dry_base",           defaults.sampling.dry_base);
        params.sampling.dry_allowed_length = json_value(data, "dry_allowed_length", defaults.sampling.dry_allowed_length);
        params.sampling.dry_penalty_last_n = json_value(data, "dry_penalty_last_n", defaults.sampling.dry_penalty_last_n);
        params.sampling.mirostat           = json_value(data, "mirostat",           defaults.sampling.mirostat);
        params.sampling.mirostat_tau       = json_value(data, "mirostat_tau",       defaults.sampling.mirostat_tau);
        params.sampling.mirostat_eta       = json_value(data, "mirostat_eta",       defaults.sampling.mirostat_eta);
        params.sampling.penalize_nl        = json_value(data, "penalize_nl",        defaults.sampling.penalize_nl);
        params.sampling.seed               = json_value(data, "seed",               defaults.sampling.seed);
        params.sampling.n_probs            = json_value(data, "n_probs",            defaults.sampling.n_probs);
        params.sampling.min_keep           = json_value(data, "min_keep",           defaults.sampling.min_keep);

        params.speculative.n_min = json_value(data, "speculative.n_min", defaults.speculative.n_min);
        params.speculative.n_max = json_value(data, "speculative.n_max", defaults.speculative.n_max);
        params.speculative.p_min = json_value(data, "speculative.p_min", defaults.speculative.p_min);

        params.speculative.n_min = std::min(params.speculative.n_max, params.speculative.n_min);
        params.speculative.n_min = std::max(params.speculative.n_min, 2);
        params.speculative.n_max = std::max(params.speculative.n_max, 0);

        if (params.sampling.dry_base < 1.0f) {
           params.sampling.dry_base = defaults.sampling.dry_base;
        }

        // sequence breakers for DRY
        {
            // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
            // Ref: https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39

            if (data.contains("dry_sequence_breakers")) {
                params.sampling.dry_sequence_breakers = json_value(data, "dry_sequence_breakers", std::vector<std::string>());
                if (params.sampling.dry_sequence_breakers.empty()) {
                    throw std::runtime_error("Error: dry_sequence_breakers must be a non-empty array of strings");
                }
            }
        }

        // process "json_schema" and "grammar"
        if (data.contains("json_schema") && !data.at("json_schema").is_null() && data.contains("grammar") && !data.at("grammar").is_null()) {
            throw std::runtime_error("Either \"json_schema\" or \"grammar\" can be specified, but not both");
        }
        if (data.contains("json_schema") && !data.contains("grammar")) {
            try {
                auto schema                  = json_value(data, "json_schema", json::object());
                params.sampling.grammar = json_schema_to_grammar(schema);
            } catch (const std::exception & e) {
                throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
            }
        } else {
            params.sampling.grammar = json_value(data, "grammar", defaults.sampling.grammar);
        }

        {
            params.sampling.logit_bias.clear();
            params.ignore_eos = json_value(data, "ignore_eos", false);

            const auto & logit_bias = data.find("logit_bias");
            if (logit_bias != data.end() && logit_bias->is_array()) {
                const int n_vocab = llama_n_vocab(model);
                for (const auto & el : *logit_bias) {
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
                            auto toks = common_tokenize(model, el[0].get<std::string>(), false);
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

            const auto & stop = data.find("stop");
            if (stop != data.end() && stop->is_array()) {
                for (const auto & word : *stop) {
                    if (!word.empty()) {
                        params.antiprompt.push_back(word);
                    }
                }
            }
        }

        {
            const auto & samplers = data.find("samplers");
            if (samplers != data.end()) {
                if (samplers->is_array()) {
                    std::vector<std::string> sampler_names;
                    for (const auto & name : *samplers) {
                        if (name.is_string()) {
                            sampler_names.emplace_back(name);
                        }
                    }
                    params.sampling.samplers = common_sampler_types_from_names(sampler_names, false);
                } else if (samplers->is_string()){
                    std::string sampler_string;
                    for (const auto & name : *samplers) {
                        sampler_string += name;
                    }
                    params.sampling.samplers = common_sampler_types_from_chars(sampler_string);
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
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
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
            {"prompt_n",               prompt_n},
            {"prompt_ms",              prompt_ms},
            {"prompt_per_token_ms",    prompt_per_token_ms},
            {"prompt_per_second",      prompt_per_second},

            {"predicted_n",            predicted_n},
            {"predicted_ms",           predicted_ms},
            {"predicted_per_token_ms", predicted_per_token_ms},
            {"predicted_per_second",   predicted_per_second},
        };
    }
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;
    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return false;
    }
    virtual int get_index() {
        return -1;
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

struct completion_token_output {
    llama_token tok;
    std::string text_to_send;
    struct token_prob {
        llama_token tok;
        std::string tok_str;
        float prob;
    };
    std::vector<token_prob> probs;

    json to_json() const {
        json probs_for_token = json::array();
        for (const auto & p : probs) {
            probs_for_token.push_back(json {
                {"tok_str", p.tok_str},
                {"prob",    p.prob},
            });
        }
        return probs_for_token;
    }

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs) {
        json out = json::array();
        for (const auto & prob : probs) {
            const std::string tok_str = prob.text_to_send;
            out.push_back(json {
                {"content", tok_str},
                {"probs",   prob.to_json()},
            });
        }
        return out;
    }
};

struct server_task_result_cmpl_final : server_task_result {
    int index = 0;
    std::string content;
    bool stream;
    result_timings timings;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_tokens_cached;
    int32_t has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    std::vector<completion_token_output> probs_output;

    slot_params generation_params;

    // OAI-compat fields
    bool        verbose        = false;
    bool        oaicompat      = false;
    bool        oaicompat_chat = true; // TODO: support oaicompat for non-chat
    std::string oaicompat_model;
    std::string oaicompat_cmpl_id;

    virtual int get_index() override {
        return index;
    }

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override {
        return oaicompat
            ? (stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat())
            : to_json_non_oaicompat();
    }

    json to_json_non_oaicompat() {
        json res = json {
            {"index",               index},
            {"content",             stream ? "" : content}, // in stream mode, content is already in last partial chunk
            {"id_slot",             id_slot},
            {"stop",                true},
            {"model",               oaicompat_model},
            {"tokens_predicted",    n_decoded},
            {"tokens_evaluated",    n_prompt_tokens},
            {"generation_settings", generation_params.to_json()},
            {"prompt",              prompt},
            {"has_new_line",        has_new_line},
            {"truncated",           truncated},
            {"stop_type",           stop_type_to_str(stop)},
            {"stopping_word",       stopping_word},
            {"tokens_cached",       n_tokens_cached},
            {"timings",             timings.to_json()},
        };
        if (!probs_output.empty()) {
            res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output);
        }
        return res;
    }

    json to_json_oaicompat_chat() {
        std::string finish_reason = "length";
        if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
            finish_reason = "stop";
        }

        json choices = json::array({json{
            {"finish_reason", finish_reason},
            {"index", 0},
            {"message", json{
                {"content", content},
                {"role", "assistant"}
            }
        }}});

        std::time_t t = std::time(0);

        json res = json {
            {"choices", choices},
            {"created", t},
            {"model", oaicompat_model},
            {"object", "chat.completion"},
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
            finish_reason = "stop";
        }

        json choices = json::array({json{{"finish_reason", finish_reason},
                                        {"index", 0},
                                        {"delta", json::object()}}});

        json ret = json {
            {"choices", choices},
            {"created", t},
            {"id",      oaicompat_cmpl_id},
            {"model",   oaicompat_model},
            {"object",  "chat.completion.chunk"},
            {"usage", json {
                {"completion_tokens", n_decoded},
                {"prompt_tokens",     n_prompt_tokens},
                {"total_tokens",      n_decoded + n_prompt_tokens},
            }},
        };

        if (timings.prompt_n >= 0) {
            ret.push_back({"timings", timings.to_json()});
        }

        return ret;
    }
};

struct server_task_result_cmpl_partial : server_task_result {
    int index = 0;
    std::string content;

    int32_t n_decoded;
    int32_t n_prompt_tokens;

    std::vector<completion_token_output> probs_output;
    result_timings timings;

    // OAI-compat fields
    bool        verbose        = false;
    bool        oaicompat      = false;
    bool        oaicompat_chat = true; // TODO: support oaicompat for non-chat
    std::string oaicompat_model;
    std::string oaicompat_cmpl_id;

    virtual int get_index() override {
        return index;
    }

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual json to_json() override {
        return oaicompat ? to_json_oaicompat() : to_json_non_oaicompat();
    }

    json to_json_non_oaicompat() {
        // non-OAI-compat JSON
        json res = json {
            {"index",            index},
            {"content",          content},
            {"stop",             false},
            {"id_slot",          id_slot},
            {"tokens_predicted", n_decoded},
            {"tokens_evaluated", n_prompt_tokens},
        };
        // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
        if (timings.prompt_n > 0) {
            res.push_back({"timings", timings.to_json()});
        }
        if (!probs_output.empty()) {
            res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output);
        }
        return res;
    }

    json to_json_oaicompat() {
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
                json initial_ret = json{{"choices", json::array({json{
                                        {"finish_reason", nullptr},
                                        {"index", 0},
                                        {"delta", json{
                                            {"role", "assistant"}
                                        }}}})},
                            {"created", t},
                            {"id", oaicompat_cmpl_id},
                            {"model", oaicompat_model},
                            {"object", "chat.completion.chunk"}};

                json second_ret = json{
                            {"choices", json::array({json{{"finish_reason", nullptr},
                                                            {"index", 0},
                                                            {"delta", json{
                                                            {"content", content}}}
                                                            }})},
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

        json ret = json {
            {"choices", choices},
            {"created", t},
            {"id",      oaicompat_cmpl_id},
            {"model",   oaicompat_model},
            {"object",  "chat.completion.chunk"}
        };

        if (timings.prompt_n >= 0) {
            ret.push_back({"timings", timings.to_json()});
        }

        return std::vector<json>({ret});
    }
};

struct server_task_result_embd : server_task_result {
    int index = 0;
    std::vector<float> embedding;

    virtual int get_index() override {
        return index;
    }

    virtual json to_json() override {
        return json {
            {"index",     index},
            {"embedding", embedding},
        };
    }
};

struct server_task_result_rerank : server_task_result {
    int index = 0;
    float score = -1e6;

    virtual int get_index() override {
        return index;
    }

    virtual json to_json() override {
        return json {
            {"index", index},
            {"score", score},
        };
    }
};

// this function maybe used outside of server_task_result_error
static json format_error_response(const std::string & message, const enum error_type type) {
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
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}

struct server_task_result_error : server_task_result {
    int index = 0;
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    virtual bool is_error() override {
        return true;
    }

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

    // TODO: somehow reuse server_metrics in the future, instead of duplicating the fields
    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override {
        return json {
            { "idle",                            n_idle_slots },
            { "processing",                      n_processing_slots },
            { "deferred",                        n_tasks_deferred },
            { "t_start",                         t_start },

            { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
            { "t_tokens_generation_total",       t_tokens_generation_total },
            { "n_tokens_predicted_total",        n_tokens_predicted_total },
            { "t_prompt_processing_total",       t_prompt_processing_total },

            { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
            { "t_prompt_processing",             t_prompt_processing },
            { "n_tokens_predicted",              n_tokens_predicted },
            { "t_tokens_generation",             t_tokens_generation },

            { "n_decode_total",                  n_decode_total },
            { "n_busy_slots_total",              n_busy_slots_total },

            { "kv_cache_tokens_count",           kv_cache_tokens_count },
            { "kv_cache_used_cells",             kv_cache_used_cells },

            { "slots",                           slots_data },
        };
    }
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override {
        if (is_save) {
            return json {
                { "id_slot",   id_slot },
                { "filename",  filename },
                { "n_saved",   n_tokens },
                { "n_written", n_bytes },
                { "timings", {
                    { "save_ms", t_ms }
                }},
            };
        } else {
            return json {
                { "id_slot",    id_slot },
                { "filename",   filename },
                { "n_restored", n_tokens },
                { "n_read",     n_bytes },
                { "timings", {
                    { "restore_ms", t_ms }
                }},
            };
        }
    }
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override {
        return json {
            { "id_slot",  id_slot },
            { "n_erased", n_erased },
        };
    }
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override {
        return json {{ "success", true }};
    }
};

struct server_slot {
    int id;
    int id_task = -1;

    // only used for completion/embedding/infill/rerank
    server_task_type task_type = SERVER_TASK_TYPE_COMPLETION;

    llama_batch batch_spec = {};

    llama_context * ctx = nullptr;
    llama_context * ctx_dft = nullptr;

    common_speculative * spec = nullptr;

    // the index relative to completion multi-task request
    size_t index = 0;

    struct slot_params params;

    slot_state state = SLOT_STATE_IDLE;

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_past      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;
    int32_t n_predict   = -1; // TODO: disambiguate from params.n_predict

    // n_prompt_tokens may not be equal to prompt_tokens.size(), because prompt maybe truncated
    int32_t n_prompt_tokens           = 0;
    int32_t n_prompt_tokens_processed = 0;

    // input prompt tokens
    llama_tokens prompt_tokens;

    size_t last_nl_pos = 0;

    std::string generated_text;
    llama_tokens cache_tokens;
    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;
    stop_type stop;

    std::string stopping_word;

    // sampling
    json json_schema;

    struct common_sampler * smpl = nullptr;

    llama_token sampled;

    // stats
    size_t n_sent_text        = 0; // number of sent text character
    size_t n_sent_token_probs = 0;

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing; // ms
    double t_token_generation;  // ms

    std::function<void(int)> callback_on_release;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens    = 0;
        last_nl_pos        = 0;
        generated_text     = "";
        has_new_line       = false;
        truncated          = false;
        stop               = STOP_TYPE_NONE;
        stopping_word      = "";
        n_past             = 0;
        n_sent_text        = 0;
        n_sent_token_probs = 0;
        task_type          = SERVER_TASK_TYPE_COMPLETION;

        generated_token_probs.clear();
    }

    bool is_non_causal() const {
        return task_type == SERVER_TASK_TYPE_EMBEDDING || task_type == SERVER_TASK_TYPE_RERANK;
    }

    bool has_budget(const common_params & global_params) {
        if (params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (params.n_predict != -1) {
            n_remaining = params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return ctx_dft && params.speculative.n_max > 0 && params.cache_prompt;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }
        generated_token_probs.push_back(token);
    }

    void release() {
        if (is_processing()) {
            SLT_INF(*this, "stop processing: n_past = %d, truncated = %d\n", n_past, truncated);

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
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n = n_decoded;
        timings.predicted_ms = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second = 1e3 / t_token_generation * n_decoded;

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        size_t stop_pos = std::string::npos;

        for (const std::string & word : params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = find_partial_stop_string(word, text);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "\n"
                "\rprompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "\r       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "\r      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second,
                t_token_generation, n_decoded, t_gen, n_gen_second,
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);
    }

    json to_json() const {
        return json {
            {"id",            id},
            {"id_task",       id_task},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
            {"non_causal",    is_non_causal()},
            {"params",        params.to_json()},
            {"prompt",        common_detokenize(ctx, prompt_tokens)},
            {"next_token",
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                    {"stopping_word",  stopping_word},
                }
            },
        };
    }
};

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
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
    std::function<void(void)>        callback_update_slots;

    // Add a new task to the end of the queue
    int post(server_task task, bool front = false) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        GGML_ASSERT(task.id != -1);
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
    int post(std::vector<server_task> & tasks, bool front = false) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        for (auto & task : tasks) {
            if (task.id == -1) {
                task.id = id++;
            }
            QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int) tasks.size(), front);
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
        callback_new_task = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Call when the state of one slot is changed, it will move one task from deferred to main queue
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
                if (queue_tasks.empty()) {
                    if (!running) {
                        QUE_DBG("%s", "terminate\n");
                        return;
                    }
                    condition_tasks.wait(lock, [&]{
                        return (!queue_tasks.empty() || !running);
                    });
                }
            }
        }
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
        SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());

        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.insert(id_task);
    }

    void add_waiting_tasks(const std::vector<server_task> & tasks) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (const auto & task : tasks) {
            SRV_DBG("add task %d to waiting list. current waiting = %d (before add)\n", task.id, (int) waiting_task_ids.size());
            waiting_task_ids.insert(task.id);
        }
    }

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task) {
        SRV_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());

        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.erase(id_task);
    }

    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (const auto & id_task : id_tasks) {
            SRV_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());
            waiting_task_ids.erase(id_task);
        }
    }

    // This function blocks the thread until there is a response for one of the id_tasks
    server_task_result_ptr recv(const std::unordered_set<int> & id_tasks) {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_results);
            condition_results.wait(lock, [&]{
                return !queue_results.empty();
            });

            for (int i = 0; i < (int) queue_results.size(); i++) {
                if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                    server_task_result_ptr res = std::move(queue_results[i]);
                    queue_results.erase(queue_results.begin() + i);
                    return res;
                }
            }
        }

        // should never reach here
    }

    // single-task version of recv()
    server_task_result_ptr recv(int id_task) {
        std::unordered_set<int> id_tasks = {id_task};
        return recv(id_tasks);
    }

    // Send a new result to a waiting id_task
    void send(server_task_result_ptr && result) {
        SRV_DBG("sending result for task id = %d\n", result->id);

        std::unique_lock<std::mutex> lock(mutex_results);
        for (const auto & id_task : waiting_task_ids) {
            if (result->id == id_task) {
                SRV_DBG("task id = %d pushed to result queue\n", result->id);

                queue_results.emplace_back(std::move(result));
                condition_results.notify_all();
                return;
            }
        }
    }
};

struct server_context {
    torch::nn::Embedding emb = nullptr;

    tcim::Stream stream;
    tcim::Module::WeightManager wm;
    tcim::Module::Option option;

    tcim::Module prefill_part1_model;
    std::map<std::string, tcim::Tensor> prefill_part1_input_map;
    std::map<std::string, tcim::Tensor> prefill_part1_output_map;
    
    tcim::Module prefill_part2_model;
    std::map<std::string, tcim::Tensor> prefill_part2_input_map;
    std::map<std::string, tcim::Tensor> prefill_part2_output_map;

    tcim::Module prefill_head_model;
    std::map<std::string, tcim::Tensor> prefill_head_input_map;
    std::map<std::string, tcim::Tensor> prefill_head_output_map;

    tcim::Module decode_part1_model;
    std::map<std::string, tcim::Tensor> decode_part1_input_map;
    std::map<std::string, tcim::Tensor> decode_part1_output_map;

    tcim::Module decode_part2_model;
    std::map<std::string, tcim::Tensor> decode_part2_input_map;
    std::map<std::string, tcim::Tensor> decode_part2_output_map;

    tcim::Module decode_head_model;
    std::map<std::string, tcim::Tensor> decode_head_input_map;
    std::map<std::string, tcim::Tensor> decode_head_output_map;

    int32_t context_length;

    llama_token  token;
    torch::Tensor input_emd;
    llama_pos decode_pos = 0;
    int32_t decode_n_tokens = 0;

    size_t n_finished = 0;

    std::vector<server_task> tasks;
    std::unordered_set<int> task_ids;

    common_params params_base;

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    std::vector<common_lora_adapter_container> loras;

    llama_model * model_dft = nullptr;
    llama_context_params cparams_dft;

    llama_batch batch = {};

    bool clean_kv_cache = true;
    bool add_bos_token  = true;
    bool has_eos_token  = false;

    int32_t n_ctx; // total context for all clients / slots

    // slots / clients
    std::vector<server_slot> slots;
    json default_generation_settings_for_props;

    server_queue    queue_tasks;
    server_response queue_results;

    server_metrics metrics;

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    ~server_context() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }

        if (model) {
            llama_free_model(model);
            model = nullptr;
        }

        if (model_dft) {
            llama_free_model(model_dft);
            model_dft = nullptr;
        }

        // Clear any sampling context
        for (server_slot & slot : slots) {
            common_sampler_free(slot.smpl);
            slot.smpl = nullptr;

            llama_free(slot.ctx_dft);
            slot.ctx_dft = nullptr;

            common_speculative_free(slot.spec);
            slot.spec = nullptr;

            llama_batch_free(slot.batch_spec);
        }

        llama_batch_free(batch);
    }

    bool load_model(const common_params & params) {
        SRV_INF("loading model '%s'\n", params.model.c_str());

        params_base = params;

        common_init_result llama_init = common_init_from_params(params_base);

        model = llama_init.model;
        ctx   = llama_init.context;
        loras = llama_init.lora_adapters;

        if (model == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.c_str());
            return false;
        }

        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_add_bos_token(model);
        has_eos_token = !llama_add_eos_token(model);

        if (!params_base.speculative.model.empty()) {
            SRV_INF("loading draft model '%s'\n", params_base.speculative.model.c_str());

            auto params_dft = params_base;

            params_dft.devices      = params_base.speculative.devices;
            params_dft.model        = params_base.speculative.model;
            params_dft.n_ctx        = params_base.speculative.n_ctx == 0 ? params_base.n_ctx / params_base.n_parallel : params_base.speculative.n_ctx;
            params_dft.n_gpu_layers = params_base.speculative.n_gpu_layers;
            params_dft.n_parallel   = 1;

            common_init_result llama_init_dft = common_init_from_params(params_dft);

            model_dft = llama_init_dft.model;

            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_base.speculative.model.c_str());
                return false;
            }

            if (!common_speculative_are_compatible(ctx, llama_init_dft.context)) {
                SRV_ERR("the draft model '%s' is not compatible with the target model '%s'\n", params_base.speculative.model.c_str(), params_base.model.c_str());

                llama_free      (llama_init_dft.context);
                llama_free_model(llama_init_dft.model);

                return false;
            }

            const int n_ctx_dft = llama_n_ctx(llama_init_dft.context);

            cparams_dft = common_context_params_to_llama(params_dft);
            cparams_dft.n_batch = n_ctx_dft;

            // force F16 KV cache for the draft model for extra performance
            cparams_dft.type_k = GGML_TYPE_F16;
            cparams_dft.type_v = GGML_TYPE_F16;

            // the context is not needed - we will create one for each slot
            llama_free(llama_init_dft.context);
        }

        return true;
    }

    bool validate_model_chat_template() const {
        std::vector<char> model_template(2048, 0); // longest known template is about 1200 bytes
        std::string template_key = "tokenizer.chat_template";
        int32_t res = llama_model_meta_val_str(model, template_key.c_str(), model_template.data(), model_template.size());
        if (res >= 0) {
            llama_chat_message chat[] = {{"user", "test"}};
            std::string tmpl = std::string(model_template.data(), model_template.size());
            int32_t chat_res = llama_chat_apply_template(model, tmpl.c_str(), chat, 1, true, nullptr, 0);
            return chat_res > 0;
        }
        return false;
    }

    void init() {

        torch::Tensor input;
        //std::string input_path = "./qembedding.pt";
        std::string input_path = "./quant_qembedding.pt";
        
        try {
            torch::load(input, input_path);
            input = input.view({-1, 3584});
            //std::cout << "input options: " << input.options() << "\n";
            //std::cout << "input dim: " << input.dim() << "\n";
            //std::cout << "input size: " << input.size(0) << "\n";
            //std::cout << "embedding content: " << std::endl;
            //for (int i = 0; i < 20; i++) {
            //    std::cout << input[0][i] << std::endl;
            //}
            //std::cout << std::endl;
            emb = torch::nn::Embedding::from_pretrained(input);
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n" << e.what_without_backtrace();
            return;
        }

        //std::cout << "options: " << input.options() << "\n";
        //std::cout << "dim: " << input.dim() << "\n";
        //std::cout << "size: " << input.size(0) << "\n";

        stream = tcim::Stream(true);
        wm = tcim::Module::WeightManager::CreateWeightManager(0);
        option = tcim::Module::Option(wm);

        // 1. load model
        //std::cout << "LoadFromFile prefill_part1" << std::endl;
        std::string prefill_part1_model_path = "./qwen2_prefill_part1.hmm";
        if (!fs::exists(prefill_part1_model_path)) {
            std::cerr << prefill_part1_model_path << " not exist. you should run build.py in prefill_part1 example first." << std::endl;
            exit(-1);
        }
        prefill_part1_model = tcim::Module::LoadFromFile(prefill_part1_model_path, option);
        if (!prefill_part1_model) {
            std::cerr << " load model " << prefill_part1_model_path << " fail." << std::endl;
            exit(-1);
        }
        //printf("model %s loaded.\n", prefill_part1_model_path.c_str());
        prefill_part1_model.SetStream(stream);

        // 2. get input info
        int input_num = prefill_part1_model.GetInputNum();
        //std::cout << "Count of Input: " << input_num << std::endl;
        for (int idx = 0; idx < input_num; idx++) {
            auto input_name = prefill_part1_model.GetInputName(idx);
            auto input_info = prefill_part1_model.GetInputInfo(input_name).AsContiguous();
            //std::cout << "Input[" << input_name << "] " << input_info << std::endl;
            auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
            prefill_part1_input_map.insert(std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get output info
        int output_num = prefill_part1_model.GetOutputNum();
        //std::cout << "Count of Output: " << output_num << std::endl;
        for (int idx = 0; idx < output_num; idx++) {
            auto output_name = prefill_part1_model.GetOutputName(idx);
            auto output_info = prefill_part1_model.GetOutputInfo(output_name).AsContiguous();
            //std::cout << "Output[" << output_name << "] " << output_info << std::endl;
            auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
            prefill_part1_output_map.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 5. set input
        for (const auto& input : prefill_part1_input_map) {
            prefill_part1_model.SetInput(input.first, input.second);
        }

        // 6. run and sync
        //prefill_part1_model.Run();
        //prefill_part1_model.Sync();

        // 7. get output
        for (auto& output : prefill_part1_output_map) {
            prefill_part1_model.GetOutput(output.first, output.second);
        }

        // 8. postprocess, with no softmax
        for (auto& output : prefill_part1_output_map) {
            //std::cout << "output name: " << output.first << "output data: " << output.second << std::endl;
            //std::cout << "output name: " << output.first << std::endl;
        }

        // 1. load model
        //std::cout << "LoadFromFile prefill_part2" << std::endl;
        std::string prefill_part2_model_path = "./qwen2_prefill_part2.hmm";
        if (!fs::exists(prefill_part2_model_path)) {
            std::cerr << prefill_part2_model_path << " not exist. you should run build.py in prefill_part2 example first." << std::endl;
            exit(-1);
        }
        prefill_part2_model = tcim::Module::LoadFromFile(prefill_part2_model_path, option);
        if (!prefill_part2_model) {
            std::cerr << " load model " << prefill_part2_model_path << " fail." << std::endl;
            exit(-1);
        }
        //printf("model %s loaded.\n", prefill_part2_model_path.c_str());
        prefill_part2_model.SetStream(stream);

        // 2. get input info
        int input_num2 = prefill_part2_model.GetInputNum();
        //std::cout << "Count of Input: " << input_num2 << std::endl;
        for (int idx = 0; idx < input_num2; idx++) {
            auto input_name = prefill_part2_model.GetInputName(idx);
            auto input_info = prefill_part2_model.GetInputInfo(input_name).AsContiguous();
            //std::cout << "Input[" << input_name << "] " << input_info << std::endl;
            auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
            prefill_part2_input_map.insert(std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get output info
        int output_num2 = prefill_part2_model.GetOutputNum();
        //std::cout << "Count of Output: " << output_num2 << std::endl;
        for (int idx = 0; idx < output_num2; idx++) {
            auto output_name = prefill_part2_model.GetOutputName(idx);
            auto output_info = prefill_part2_model.GetOutputInfo(output_name).AsContiguous();
            //std::cout << "Output[" << output_name << "] " << output_info << std::endl;
            auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
            prefill_part2_output_map.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 5. set input
        for (const auto& input : prefill_part2_input_map) {
            prefill_part2_model.SetInput(input.first, input.second);
        }

        // 6. run and sync
        //prefill_part2_model.Run();
        //prefill_part2_model.Sync();

        // 7. get output
        for (auto& output : prefill_part2_output_map) {
            prefill_part2_model.GetOutput(output.first, output.second);
        }

        // 8. postprocess, with no softmax
        for (auto& output : prefill_part2_output_map) {
            //std::cout << "output name: " << output.first << "output data: " << output.second << std::endl;
            //std::cout << "output name: " << output.first << std::endl;
        }

        // 1. load model
        //std::cout << "LoadFromFile prefill_head" << std::endl;
        std::string prefill_head_model_path = "./qwen2_prefill_head.hmm";
        if (!fs::exists(prefill_head_model_path)) {
            std::cerr << prefill_head_model_path << " not exist. you should run build.py in prefill_head example first." << std::endl;
            exit(-1);
        }
        prefill_head_model = tcim::Module::LoadFromFile(prefill_head_model_path, option);
        if (!prefill_head_model) {
            std::cerr << " load model " << prefill_head_model_path << " fail." << std::endl;
            exit(-1);
        }
        //printf("model %s loaded.\n", prefill_head_model_path.c_str());
        prefill_head_model.SetStream(stream);

        // 2. get input info
        int input_num_head = prefill_head_model.GetInputNum();
        //std::cout << "Count of Input: " << input_num_head << std::endl;
        for (int idx = 0; idx < input_num_head; idx++) {
            auto input_name = prefill_head_model.GetInputName(idx);
            auto input_info = prefill_head_model.GetInputInfo(input_name).AsContiguous();
            //std::cout << "Input[" << input_name << "] " << input_info << std::endl;
            auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
            prefill_head_input_map.insert(std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get output info
        int output_num_head = prefill_head_model.GetOutputNum();
        //std::cout << "Count of Output: " << output_num_head << std::endl;
        for (int idx = 0; idx < output_num_head; idx++) {
            auto output_name = prefill_head_model.GetOutputName(idx);
            auto output_info = prefill_head_model.GetOutputInfo(output_name).AsContiguous();
            //std::cout << "Output[" << output_name << "] " << output_info << std::endl;
            auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
            prefill_head_output_map.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 5. set input
        for (const auto& input : prefill_head_input_map) {
            prefill_head_model.SetInput(input.first, input.second);
        }

        // 6. run and sync
        //prefill_head_model.Run();
        //prefill_head_model.Sync();

        // 7. get output
        for (auto& output : prefill_head_output_map) {
            prefill_head_model.GetOutput(output.first, output.second);
        }

        // 8. postprocess, with no softmax
        for (auto& output : prefill_head_output_map) {
            //std::cout << "output name: " << output.first << "output data: " << output.second << std::endl;
            //std::cout << "output name: " << output.first << std::endl;
        }

        // 1. load model
        //std::cout << "LoadFromFile decode_part1" << std::endl;
        std::string decode_part1_model_path = "./qwen2_decode_part1.hmm";
        if (!fs::exists(decode_part1_model_path)) {
            std::cerr << decode_part1_model_path << " not exist. you should run build.py in decode_part1 example first." << std::endl;
            exit(-1);
        }
        decode_part1_model = tcim::Module::LoadFromFile(decode_part1_model_path, option);
        if (!decode_part1_model) {
            std::cerr << " load model " << decode_part1_model_path << " fail." << std::endl;
            exit(-1);
        }
        //printf("model %s loaded.\n", decode_part1_model_path.c_str());
        decode_part1_model.SetStream(stream);

        // 2. get input info
        int input_num_decode1 = decode_part1_model.GetInputNum();
        //std::cout << "Count of Input: " << input_num_decode1 << std::endl;
        for (int idx = 0; idx < input_num_decode1; idx++) {
            auto input_name = decode_part1_model.GetInputName(idx);
            auto input_info = decode_part1_model.GetInputInfo(input_name).AsContiguous();
            //std::cout << "Input[" << input_name << "] " << input_info << std::endl;
            auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
            decode_part1_input_map.insert(std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get output info
        int output_num_decode1 = decode_part1_model.GetOutputNum();
        //std::cout << "Count of Output: " << output_num_decode1 << std::endl;
        for (int idx = 0; idx < output_num_decode1; idx++) {
            auto output_name = decode_part1_model.GetOutputName(idx);
            auto output_info = decode_part1_model.GetOutputInfo(output_name).AsContiguous();
            //std::cout << "Output[" << output_name << "] " << output_info << std::endl;
            auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
            decode_part1_output_map.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 5. set input
        //for (const auto& input : decode_part1_input_map) {
        //    decode_part1_model.SetInput(input.first, input.second);
        //}

        // 6. run and sync
        //decode_part1_model.Run();
        //decode_part1_model.Sync();

        // 7. get output
        //for (auto& output : decode_part1_output_map) {
        //    decode_part1_model.GetOutput(output.first, output.second);
        //}

        // 8. postprocess, with no softmax
        //for (auto& output : decode_part1_output_map) {
        //    //std::cout << "output name: " << output.first << "output data: " << output.second << std::endl;
        //    std::cout << "output name: " << output.first << std::endl;
        //}

        // 1. load model
        //std::cout << "LoadFromFile decode_part2" << std::endl;
        std::string decode_part2_model_path = "./qwen2_decode_part2.hmm";
        if (!fs::exists(decode_part2_model_path)) {
            std::cerr << decode_part2_model_path << " not exist. you should run build.py in decode_part2 example first." << std::endl;
            exit(-1);
        }
        decode_part2_model = tcim::Module::LoadFromFile(decode_part2_model_path, option);
        if (!decode_part2_model) {
            std::cerr << " load model " << decode_part2_model_path << " fail." << std::endl;
            exit(-1);
        }
        //printf("model %s loaded.\n", decode_part2_model_path.c_str());
        decode_part2_model.SetStream(stream);

        // 2. get input info
        int input_num_decode_part2 = decode_part2_model.GetInputNum();
        //std::cout << "Count of Input: " << input_num_decode_part2 << std::endl;
        for (int idx = 0; idx < input_num_decode_part2; idx++) {
            auto input_name = decode_part2_model.GetInputName(idx);
            auto input_info = decode_part2_model.GetInputInfo(input_name).AsContiguous();
            //std::cout << "Input[" << input_name << "] " << input_info << std::endl;
            auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
            decode_part2_input_map.insert(std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get output info
        int output_num_decode_part2 = decode_part2_model.GetOutputNum();
        //std::cout << "Count of Output: " << output_num_decode_part2 << std::endl;
        for (int idx = 0; idx < output_num_decode_part2; idx++) {
            auto output_name = decode_part2_model.GetOutputName(idx);
            auto output_info = decode_part2_model.GetOutputInfo(output_name).AsContiguous();
            //std::cout << "Output[" << output_name << "] " << output_info << std::endl;
            auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
            decode_part2_output_map.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 5. set input
        //for (const auto& input : decode_part2_input_map) {
        //    decode_part2_model.SetInput(input.first, input.second);
        //}

        // 6. run and sync
        //decode_part2_model.Run();
        //decode_part2_model.Sync();

        // 7. get output
        //for (auto& output : decode_part2_output_map) {
        //    decode_part2_model.GetOutput(output.first, output.second);
        //}

        // 8. postprocess, with no softmax
        //for (auto& output : decode_part2_output_map) {
        //    //std::cout << "output name: " << output.first << "output data: " << output.second << std::endl;
        //    std::cout << "output name: " << output.first << std::endl;
        //}

        // 1. load model
        //std::cout << "LoadFromFile decode_head" << std::endl;
        std::string decode_head_model_path = "./qwen2_decode_head.hmm";
        if (!fs::exists(decode_head_model_path)) {
            std::cerr << decode_head_model_path << " not exist. you should run build.py in decode_head example first." << std::endl;
            exit(-1);
        }
        decode_head_model = tcim::Module::LoadFromFile(decode_head_model_path, option);
        if (!decode_head_model) {
            std::cerr << " load model " << decode_head_model_path << " fail." << std::endl;
            exit(-1);
        }
        //printf("model %s loaded.\n", decode_head_model_path.c_str());
        decode_head_model.SetStream(stream);

        // 2. get input info
        int input_num_decode_head = decode_head_model.GetInputNum();
        //std::cout << "Count of Input: " << input_num_decode_head << std::endl;
        for (int idx = 0; idx < input_num_decode_head; idx++) {
            auto input_name = decode_head_model.GetInputName(idx);
            auto input_info = decode_head_model.GetInputInfo(input_name).AsContiguous();
            //std::cout << "Input[" << input_name << "] " << input_info << std::endl;
            auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
            decode_head_input_map.insert(std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }

        // 4. get output info
        int output_num_decode_head = decode_head_model.GetOutputNum();
        std::cout << "Count of Output: " << output_num_decode_head << std::endl;
        for (int idx = 0; idx < output_num_decode_head; idx++) {
            auto output_name = decode_head_model.GetOutputName(idx);
            auto output_info = decode_head_model.GetOutputInfo(output_name).AsContiguous();
            //std::cout << "Output[" << output_name << "] " << output_info << std::endl;
            auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
            decode_head_output_map.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
        }

        // 5. set input
        //for (const auto& input : decode_head_input_map) {
        //    decode_head_model.SetInput(input.first, input.second);
        //}

        // 6. run and sync
        //decode_head_model.Run();
        //decode_head_model.Sync();

        // 7. get output
        //for (auto& output : decode_head_output_map) {
        //    decode_head_model.GetOutput(output.first, output.second);
        //}

        // 8. postprocess, with no softmax
        //for (auto& output : decode_head_output_map) {
        //    //std::cout << "output name: " << output.first << "output data: " << output.second << std::endl;
        //    std::cout << "output name: " << output.first << std::endl;
        //}

        // set kvcache input
        for (int idx = 0; idx < 14; idx++) {
            std::string k_cache_name = hm_string_format("model_layers_%d_self_attn_kcache_input", idx);
            std::string v_cache_name = hm_string_format("model_layers_%d_self_attn_vcache_input", idx);
            auto kcache = prefill_part1_model.GetInput(k_cache_name);
            auto vcache = prefill_part1_model.GetInput(v_cache_name);
            decode_part1_model.SetInput(k_cache_name, kcache);
            decode_part1_model.SetInput(v_cache_name, vcache);
        }

        for (int idx = 14; idx < 28; idx++) {
            std::string k_cache_name = hm_string_format("model_layers_%d_self_attn_kcache_input", idx);
            std::string v_cache_name = hm_string_format("model_layers_%d_self_attn_vcache_input", idx);
            auto kcache = prefill_part2_model.GetInput(k_cache_name);
            auto vcache = prefill_part2_model.GetInput(v_cache_name);
            decode_part2_model.SetInput(k_cache_name, kcache);
            decode_part2_model.SetInput(v_cache_name, vcache);
        }

        // set decode input
        std::vector<int16_t> current_length_input = {1};
        auto current_length_input_tensor_opts = torch::TensorOptions().dtype(torch::kInt16);
        auto current_length_input_tensor = torch::from_blob(current_length_input.data(), {int64_t(current_length_input.size())}, current_length_input_tensor_opts).clone();

        //std::cout << "current_length_input_tensor: " << current_length_input_tensor << std::endl;

        
        auto decode_part1_current_length = decode_part1_input_map.at("current_length");
        //int16toint16((int16_t *)decode_part1_current_length.Data(), (int16_t *)current_length_input.data(), current_length_input.size());
        //int16toint16((int16_t *)decode_part1_current_length.Data(), (int16_t *)current_length_input_tensor.data<int16_t>(), current_length_input_tensor.size(0));
        //int16toint16((int16_t *)decode_part1_current_length.Data(), (int16_t *)current_length_input_tensor.data<int16_t>(), current_length_input_tensor.numel());
        int16toint16((int16_t *)decode_part1_current_length.Data(), (int16_t *)current_length_input_tensor.data_ptr<int16_t>(), current_length_input_tensor.numel());
        auto decode_part2_current_length = decode_part2_input_map.at("current_length");
        //int16toint16((int16_t *)decode_part2_current_length.Data(), (int16_t *)current_length_input.data(), current_length_input.size());
        //int16toint16((int16_t *)decode_part2_current_length.Data(), (int16_t *)current_length_input_tensor.data<int16_t>(), current_length_input_tensor.numel());
        int16toint16((int16_t *)decode_part2_current_length.Data(), (int16_t *)current_length_input_tensor.data_ptr<int16_t>(), current_length_input_tensor.numel());
        decode_part1_model.SetInput("current_length", decode_part1_current_length);
        decode_part2_model.SetInput("current_length", decode_part2_current_length);
        //auto decode_part1_output = decode_part1_output_map.at("model_layers_13_resadd2");
        tcim::Tensor decode_part1_output = decode_part1_model.GetOutput("model_layers_13_resadd2", tcim::HDPL);
        decode_part2_model.SetInput("model_layers_13_resadd2", decode_part1_output);
        //auto decode_part2_output = decode_part2_output_map.at("model_layers_27_resadd2");
        tcim::Tensor decode_part2_output = decode_part2_model.GetOutput("model_layers_27_resadd2", tcim::HDPL);
        decode_head_model.SetInput("model_layers_27_resadd2", decode_part2_output);
        
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

                slot.ctx_dft = llama_new_context_with_model(model_dft, cparams_dft);
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

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx);

            // only a single seq_id per token is needed
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        metrics.init();
    }

    server_slot * get_slot_by_id(int id) {
        for (server_slot & slot : slots) {
            if (slot.id == id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            int lcs_len = 0;
            float similarity = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // skip the slot if it does not contains cached tokens
                if (slot.cache_tokens.empty()) {
                    continue;
                }

                // length of the Longest Common Subsequence between the current slot's prompt and the input prompt
                int cur_lcs_len = common_lcs(slot.cache_tokens, task.prompt_tokens);

                // fraction of the common subsequence length compared to the current slot's prompt length
                float cur_similarity = static_cast<float>(cur_lcs_len) / static_cast<int>(slot.cache_tokens.size());

                // select the current slot if the criteria match
                if (cur_lcs_len > lcs_len && cur_similarity > slot_prompt_similarity) {
                    lcs_len = cur_lcs_len;
                    similarity = cur_similarity;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_DBG(*ret, "selected slot by lcs similarity, lcs_len = %d, similarity = %f\n", lcs_len, similarity);
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = ggml_time_us();
            for (server_slot & slot : slots) {
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

    bool launch_slot_with_task(server_slot & slot, const server_task & task) {
        slot.reset();
        SRV_INF("d slot.n_past = %d\n", slot.n_past);
        SRV_INF("d slot.n_prompt_tokens = %d\n", slot.n_prompt_tokens);
        slot.id_task       = task.id;
        slot.index         = task.index;
        slot.task_type     = task.type;
        slot.params        = std::move(task.params);
        slot.prompt_tokens = std::move(task.prompt_tokens);

        SLT_INF(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        if (slot.n_predict > 0 && slot.params.n_predict > slot.n_predict) {
            // Might be better to reject the request with a 400 ?
            slot.params.n_predict = slot.n_predict;
            SLT_WRN(slot, "n_predict = %d exceeds server configuration, setting to %d", slot.n_predict, slot.n_predict);
        }

        if (slot.params.ignore_eos && has_eos_token) {
            slot.params.sampling.logit_bias.push_back({llama_token_eos(model), -INFINITY});
        }

        {
            if (slot.smpl != nullptr) {
                common_sampler_free(slot.smpl);
            }

            slot.smpl = common_sampler_init(model, slot.params.sampling);
            if (slot.smpl == nullptr) {
                // for now, the only error that may happen here is invalid grammar
                send_error(task, "Failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
        }

        if (slot.ctx_dft) {
            llama_batch_free(slot.batch_spec);

            slot.batch_spec = llama_batch_init(slot.params.speculative.n_max + 1, 0, 1);
        }

        slot.state = SLOT_STATE_STARTED;

        SLT_INF(slot, "%s", "processing task\n");

        return true;
    }

    void kv_cache_clear() {
        SRV_DBG("%s", "clearing KV cache\n");

        // clear the entire KV cache
        llama_kv_cache_clear(ctx);
        clean_kv_cache = false;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = common_token_to_piece(ctx, result.tok, params_base.special);
        slot.sampled = result.tok;

        // search stop word and delete it
        slot.generated_text += token_str;
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = false;
        for (unsigned i = 1; i < 5 && i <= slot.generated_text.size(); ++i) {
            unsigned char c = slot.generated_text[slot.generated_text.size() - i];
            if ((c & 0xC0) == 0x80) {
                // continuation byte: 10xxxxxx
                continue;
            }
            if ((c & 0xE0) == 0xC0) {
                // 2-byte character: 110xxxxx ...
                incomplete = i < 2;
            } else if ((c & 0xF0) == 0xE0) {
                // 3-byte character: 1110xxxx ...
                incomplete = i < 3;
            } else if ((c & 0xF8) == 0xF0) {
                // 4-byte character: 11110xxx ...
                incomplete = i < 4;
            }
            // else 1-byte character or invalid byte
            break;
        }

        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            }

            slot.add_token(result);
            if (slot.params.stream) {
                send_partial_response(slot, result);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.params.n_predict);
        }

        if (slot.has_new_line) {
            // if we have already seen a new line, we stop after a certain time limit
            if (slot.params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.params.t_max_predict_ms);
            }

            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
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
        }

        // if context shift is disabled, we stop when it reaches the context limit
        SLT_DBG(slot, "slot.n_past = %d slot.n_ctx = %d", slot.n_past, slot.n_ctx);
        if (slot.n_past >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, n_past = %d, n_prompt_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.n_decoded, slot.n_prompt_tokens, slot.n_past, slot.n_ctx);
        }

        if (llama_token_is_eog(model, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            decode_pos = 0;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        const auto n_ctx_train = llama_n_ctx_train(model);

        if (slot.params.n_predict < 1 && slot.n_predict < 1 && slot.n_prompt_tokens + slot.n_decoded >= n_ctx_train) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false; // stop prediction

            SLT_WRN(slot,
                    "n_predict (%d) is set for infinite generation. "
                    "Limiting generated tokens to n_ctx_train (%d) to avoid EOS-less generation infinite loop\n",
                    slot.params.n_predict, n_ctx_train);
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.id_task, error, type);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        auto res = std::make_unique<server_task_result_error>();
        res->id       = id_task;
        res->err_type = type;
        res->err_msg  = error;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id      = slot.id_task;
        res->index   = slot.index;
        res->content = tkn.text_to_send;

        res->n_decoded       = slot.n_decoded;
        res->n_prompt_tokens = slot.n_prompt_tokens;

        res->verbose           = slot.params.verbose;
        res->oaicompat         = slot.params.oaicompat;
        res->oaicompat_chat    = slot.params.oaicompat_chat;
        res->oaicompat_model   = slot.params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.params.sampling.n_probs > 0) {
            const llama_tokens to_send_toks = common_tokenize(ctx, tkn.text_to_send, false);

            const size_t probs_pos      = std::min(slot.n_sent_token_probs,                       slot.generated_token_probs.size());
            const size_t probs_stop_pos = std::min(slot.n_sent_token_probs + to_send_toks.size(), slot.generated_token_probs.size());

            std::vector<completion_token_output> probs_output;
            if (probs_pos < probs_stop_pos) {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin() + probs_pos,
                        slot.generated_token_probs.begin() + probs_stop_pos);
            }
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();
        res->id              = slot.id_task;
        res->id_slot         = slot.id;

        res->index           = slot.index;
        res->content         = slot.generated_text;
        res->timings         = slot.get_timings();
        res->prompt          = common_detokenize(ctx, slot.prompt_tokens, true);

        res->truncated       = slot.truncated;
        res->n_decoded       = slot.n_decoded;
        res->n_prompt_tokens = slot.n_prompt_tokens;
        res->n_tokens_cached = slot.n_past;
        res->has_new_line    = slot.has_new_line;
        res->stopping_word   = slot.stopping_word;
        res->stop            = slot.stop;

        res->verbose           = slot.params.verbose;
        res->stream            = slot.params.stream;
        res->oaicompat         = slot.params.oaicompat;
        res->oaicompat_chat    = slot.params.oaicompat_chat;
        res->oaicompat_model   = slot.params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.params.sampling.n_probs > 0) {
            if (!slot.params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id    = slot.id_task;
        res->index = slot.index;

        const int n_embd = llama_n_embd(model);

        std::vector<float> embd_res(n_embd, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding = std::vector<float>(n_embd, 0.0f);
                continue;
            }

            common_embd_normalize(embd, embd_res.data(), n_embd);
            res->embedding = embd_res;
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id    = slot.id_task;
        res->index = slot.index;

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

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

    void cancel_tasks(const std::unordered_set<int> & id_tasks) {
        std::vector<server_task> cancel_tasks;
        cancel_tasks.reserve(id_tasks.size());
        for (const auto & id_task : id_tasks) {
            SRV_WRN("cancel task, id_task = %d\n", id_task);

            server_task task(SERVER_TASK_TYPE_CANCEL);
            task.id_target = id_task;
            cancel_tasks.push_back(task);
            queue_results.remove_waiting_task_id(id_task);
        }
        // push to beginning of the queue, so it has highest priority
        queue_tasks.post(cancel_tasks, true);
        decode_pos = 0;
    }

    // receive the results from task(s)
    void receive_multi_results(
            const std::unordered_set<int> & id_tasks,
            const std::function<void(std::vector<server_task_result_ptr>&)> & result_handler,
            const std::function<void(json)> & error_handler) {
        std::vector<server_task_result_ptr> results(id_tasks.size());
        for (size_t i = 0; i < id_tasks.size(); i++) {
            server_task_result_ptr result = queue_results.recv(id_tasks);

            if (result->is_error()) {
                error_handler(result->to_json());
                cancel_tasks(id_tasks);
                return;
            }

            GGML_ASSERT(
                dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                || dynamic_cast<server_task_result_embd*>(result.get()) != nullptr
                || dynamic_cast<server_task_result_rerank*>(result.get()) != nullptr
            );
            const size_t idx = result->get_index();
            GGML_ASSERT(idx < results.size() && "index out of range");
            results[idx] = std::move(result);
        }
        result_handler(results);
    }

    // receive the results from task(s), in stream mode
    void receive_cmpl_results_stream(
            const std::unordered_set<int> & id_tasks,
            const std::function<bool(server_task_result_ptr&)> & result_handler,
            const std::function<void(json)> & error_handler) {
        size_t n_finished = 0;
        while (true) {
            server_task_result_ptr result = queue_results.recv(id_tasks);

            if (result->is_error()) {
                error_handler(result->to_json());
                cancel_tasks(id_tasks);
                return;
            }

            GGML_ASSERT(
                dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
            );
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

    int completion_generate(
            json & data,
            std::vector<server_task_result_ptr> & results) {
        auto type = SERVER_TASK_TYPE_COMPLETION;
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);
        

        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;
        bool oaicompat = false;
        bool oaicompat_chat = false;

        try {
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx, data.at("prompt"), true, true);
            tasks.reserve(tokenized_prompts.size());
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task = server_task(type);

                task.id    = queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens    = std::move(tokenized_prompts[i]);
                task.params           = server_task::params_from_json_cmpl(model, params_base, data);
                task.id_selected_slot = json_value(data, "id_slot", -1);

                // OAI-compat
                task.params.oaicompat           = oaicompat;
                task.params.oaicompat_chat      = oaicompat_chat;
                task.params.oaicompat_cmpl_id   = completion_id;
                // oaicompat_model is already populated by params_from_json_cmpl

                tasks.push_back(task);
            }
        } catch (const std::exception & e) {
            log_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return -1;
        }

        queue_results.add_waiting_tasks(tasks);
        queue_tasks.post(tasks);

        bool stream = json_value(data, "stream", false);
        const auto task_ids = server_task::get_list_id(tasks);
        results.resize(task_ids.size());

        if (!stream) {
            SRV_INF("mode: %s\n", "no stream");
            for (size_t i = 0; i < task_ids.size(); i++) {
                server_task_result_ptr result = queue_results.recv(task_ids);

                if (result->is_error()) {
                    cancel_tasks(task_ids);
                    return -1;
                }

                GGML_ASSERT(
                    dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    || dynamic_cast<server_task_result_embd*>(result.get()) != nullptr
                    || dynamic_cast<server_task_result_rerank*>(result.get()) != nullptr
                );
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

    int completion_generate_str(
            json & data,
            std::string & result_str) {
        auto type = SERVER_TASK_TYPE_COMPLETION;
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);
        

        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;
        bool oaicompat = false;
        bool oaicompat_chat = false;

        try {
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx, data.at("prompt"), true, true);
            tasks.reserve(tokenized_prompts.size());
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task = server_task(type);

                task.id    = queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens    = std::move(tokenized_prompts[i]);
                task.params           = server_task::params_from_json_cmpl(model, params_base, data);
                task.id_selected_slot = json_value(data, "id_slot", -1);

                // OAI-compat
                task.params.oaicompat           = oaicompat;
                task.params.oaicompat_chat      = oaicompat_chat;
                task.params.oaicompat_cmpl_id   = completion_id;
                // oaicompat_model is already populated by params_from_json_cmpl

                tasks.push_back(task);
            }
        } catch (const std::exception & e) {
            log_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return -1;
        }

        queue_results.add_waiting_tasks(tasks);
        queue_tasks.post(tasks);

        bool stream = json_value(data, "stream", false);
        const auto task_ids = server_task::get_list_id(tasks);
        std::vector<server_task_result_ptr> results(task_ids.size());

        if (!stream) {
            SRV_INF("mode: %s\n", "no stream");
            for (size_t i = 0; i < task_ids.size(); i++) {
                server_task_result_ptr result = queue_results.recv(task_ids);

                if (result->is_error()) {
                    cancel_tasks(task_ids);
                    return -1;
                }

                GGML_ASSERT(
                    dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    || dynamic_cast<server_task_result_embd*>(result.get()) != nullptr
                    || dynamic_cast<server_task_result_rerank*>(result.get()) != nullptr
                );
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
            return 0;
        } else {
            SRV_ERR("Error: %s\n", "This is not stream api!");
            queue_results.remove_waiting_task_ids(task_ids);
            return -1;
        }
        return 0;
    }

    int completion_stream_generate(
            json & data) {
        auto type = SERVER_TASK_TYPE_COMPLETION;
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

        auto completion_id = gen_chatcmplid();
        
        bool oaicompat = false;
        bool oaicompat_chat = false;

        try {
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx, data.at("prompt"), true, true);
            tasks.reserve(tokenized_prompts.size());
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task = server_task(type);

                task.id    = queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens    = std::move(tokenized_prompts[i]);
                task.params           = server_task::params_from_json_cmpl(model, params_base, data);
                task.id_selected_slot = json_value(data, "id_slot", -1);

                // OAI-compat
                task.params.oaicompat           = oaicompat;
                task.params.oaicompat_chat      = oaicompat_chat;
                task.params.oaicompat_cmpl_id   = completion_id;
                // oaicompat_model is already populated by params_from_json_cmpl

                tasks.push_back(task);
            }
        } catch (const std::exception & e) {
            log_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return -1;
        }

        queue_results.add_waiting_tasks(tasks);
        queue_tasks.post(tasks);

        bool stream = json_value(data, "stream", false);

        task_ids = server_task::get_list_id(tasks);
        

        if (!stream) {
            SRV_INF("mode: %s\n", "no stream");
            SRV_ERR("Error: %s\n", "This is stream api!");
            queue_results.remove_waiting_task_ids(task_ids);
            return -1;
        }
        
        return 0;
    };

    int completion_stream_generate_next_token(
            server_task_result_ptr & result) {
        
        result = queue_results.recv(task_ids);

        if (result->is_error()) {
            cancel_tasks(task_ids);
            return -1;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
            || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
        );

        if (result->is_stop()) {
            if (++n_finished == task_ids.size()) {
                SRV_INF("all task finished = %d\n", n_finished);
                queue_results.remove_waiting_task_ids(task_ids);
                return 1;
            }
        }
        return 0;
    }

    int completion_stream_generate_next_token_str(
            std::string & result_str) {
        
        auto result = queue_results.recv(task_ids);

        if (result->is_error()) {
            cancel_tasks(task_ids);
            return -1;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
            || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
        );
        result_str = dump_result(result->to_json());

        if (result->is_stop()) {
            if (++n_finished == task_ids.size()) {
                SRV_INF("all task finished = %d\n", n_finished);
                queue_results.remove_waiting_task_ids(task_ids);
                return 1;
            }
        }
        return 0;
    }

    // receive the results from task(s)
    int generate(
            const std::unordered_set<int> & id_tasks,
            std::vector<server_task_result_ptr> & results) {
        
        for (size_t i = 0; i < id_tasks.size(); i++) {
            server_task_result_ptr result = queue_results.recv(id_tasks);

            if (result->is_error()) {
                cancel_tasks(id_tasks);
                return -1;
            }

            GGML_ASSERT(
                dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                || dynamic_cast<server_task_result_embd*>(result.get()) != nullptr
                || dynamic_cast<server_task_result_rerank*>(result.get()) != nullptr
            );
            const size_t idx = result->get_index();
            GGML_ASSERT(idx < results.size() && "index out of range");
            results[idx] = std::move(result);
        }

        queue_results.remove_waiting_task_ids(id_tasks);
        return 0;
    }
    
    int generate_stream(
            const std::unordered_set<int> & id_tasks) {
        
        n_finished = 0;
        return 0;
    }

    int generate_stream_next_token(
            const std::unordered_set<int> & id_tasks,
            server_task_result_ptr & result) {
        
        result = queue_results.recv(id_tasks);

        if (result->is_error()) {
            cancel_tasks(id_tasks);
            return -1;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
            || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
        );

        if (result->is_stop()) {
            if (++n_finished == id_tasks.size()) {
                SRV_INF("all task finished = %d\n", n_finished);
                queue_results.remove_waiting_task_ids(id_tasks);
                return 1;
            }
        }
        return 0;
    }

    //
    // Functions to process the task
    //

    void process_single_task(server_task task) {
        //SRV_INF("%s: %d\n", "process_single_task", task.type);
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    const int id_slot = task.id_selected_slot;

                    server_slot * slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(task);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(task);
                        break;
                    }

                    if (!launch_slot_with_task(*slot, task)) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", task.id);
                        break;
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.id_task == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json();

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred.size();
                    res->t_start             = metrics.t_start;

                    res->kv_cache_tokens_count = llama_get_kv_cache_token_count(ctx);
                    res->kv_cache_used_cells   = llama_get_kv_cache_used_cells(ctx);

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    int id_slot = task.slot_action.slot_id;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(task);
                        break;
                    }

                    const size_t token_count = slot->cache_tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id, slot->cache_tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    int id_slot = task.slot_action.slot_id;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(task);
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    slot->cache_tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), slot->id, slot->cache_tokens.data(), slot->cache_tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->cache_tokens.resize(0);
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    slot->cache_tokens.resize(token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    int id_slot = task.slot_action.slot_id;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(task);
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->cache_tokens.size();
                    llama_kv_cache_seq_rm(ctx, slot->id, -1, -1);
                    slot->cache_tokens.clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    common_lora_adapters_apply(ctx, loras);
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    int houmo_decode(llama_batch batch) {

        if (decode_pos == 0) {
            //SRV_INF("%s", "prefill stage\n");
            std::vector<llama_token> tokens(batch.token, batch.token + batch.n_tokens);
            //for(const int& i : tokens) {
            //    std::cout << "i = " << i << std::endl;
            //}
            auto input_echo_len = tokens.size();
            //auto input_echo_len = test_tokens.size();
            

            //auto prefill_part1_output = prefill_part1_output_map.at("model_layers_13_resadd2");
            tcim::Tensor prefill_part1_output = prefill_part1_model.GetOutput("model_layers_13_resadd2", tcim::HDPL);
            prefill_part2_model.SetInput("model_layers_13_resadd2", prefill_part1_output);
            auto prefill_length = 128;
            //std::cout << "input_echo_len: " << input_echo_len << std::endl;
            //std::cout << "prefill_length: " << prefill_length << std::endl;

            auto prefill_loop_round = ceil(static_cast<float>(input_echo_len) / static_cast<float>(prefill_length));
            auto valid_length = 0;
            auto current_length = 0;
            auto gather_index = 0;
            //std::cout << "prefill_loop_round: " << prefill_loop_round << std::endl;
            std::vector<llama_token> input_ids;
            for (int round = 0; round < prefill_loop_round; round++) {
                    valid_length = round * prefill_length;
                    if (round == prefill_loop_round - 1) {
                        current_length = input_echo_len - round * prefill_length;
                        gather_index = current_length - 1;
                        std::vector<llama_token>::const_iterator first = tokens.begin() + round * prefill_length;
                        std::vector<llama_token>::const_iterator last = tokens.begin() + input_echo_len;
                        std::vector<llama_token> round_tokens(first, last);
                        input_ids = round_tokens;
                    } else {
                        current_length = prefill_length;
                        std::vector<llama_token>::const_iterator first = tokens.begin() + round * prefill_length;
                        std::vector<llama_token>::const_iterator last = tokens.begin() + (round + 1) * prefill_length;
                        std::vector<llama_token> round_tokens(first, last);
                        input_ids = round_tokens;
                    }
                    //std::cout << "input_ids: " << input_ids << std::endl;
                    torch::Tensor inputs_embeds;
                    auto opts = torch::TensorOptions().dtype(torch::kInt);
                    auto inp = torch::from_blob(input_ids.data(), {int64_t(input_ids.size())}, opts).clone();
                    //std::cout << "inp: " << inp << "\n";
                    try {
                        inputs_embeds = emb->forward(inp);
                    }
                    catch (const c10::Error& e) {
                        std::cerr << "error loading the model\n" << e.what_without_backtrace();
                        return -1;
                    }
                    //std::cout << "inputs_embeds options: " << inputs_embeds.options() << std::endl;
                    //std::cout << "inputs_embeds dim: " << inputs_embeds.dim() << std::endl;
                    //std::cout << "inputs_embeds size: " << inputs_embeds.size(0) << std::endl;
                    //std::cout << "output: " << output << "\n";
                    //for (int i = 0; i < 5; i++) {
                    //    std::cout << inputs_embeds[0][i] << " ";
                    //}
                    //std::cout << std::endl;
                    
                    
                    auto effective_length = input_ids.size();
                    auto pad_embeds = torch::zeros({prefill_length - effective_length, 3584}, torch::kInt16);
                    auto padded_embeds = torch::cat({inputs_embeds, pad_embeds});
                    padded_embeds = padded_embeds.view({4, prefill_length / 4, 3584}).to(torch::kInt16);
                    //std::cout << "padded_embeds: " << padded_embeds << std::endl;
                    //std::cout << "padded_embeds numel: " << padded_embeds.numel() << std::endl;

                    //for (int i = 0; i < 1; i++) {
                    //    std::cout << padded_embeds[0][i] << " ";
                    //}
                    //std::cout << std::endl;
                    //for (int i = 0; i < 1; i++) {
                    //    std::cout << padded_embeds[1][i] << " ";
                    //}
                    //std::cout << std::endl;
                    //for (int i = 0; i < 1; i++) {
                    //    std::cout << padded_embeds[2][i] << " ";
                    //}
                    //std::cout << std::endl;
                    //for (int i = 0; i < 1; i++) {
                    //    std::cout << padded_embeds[3][i] << " ";
                    //}
                    //std::cout << std::endl;
                    //std::this_thread::sleep_for(std::chrono::seconds(5));

                    auto input_data = prefill_part1_input_map.at("input_1");
                    int16toint16((int16_t *)input_data.Data(), (int16_t *)padded_embeds.data_ptr<int16_t>(), padded_embeds.numel());

                    std::vector<int16_t> valid_length_input = {valid_length};
                    auto valid_length_input_tensor_opts = torch::TensorOptions().dtype(torch::kInt16);
                    auto valid_length_input_tensor = torch::from_blob(valid_length_input.data(), {int64_t(valid_length_input.size())}, valid_length_input_tensor_opts).clone();

                    //std::cout << "valid_length_input_tensor: " << valid_length_input_tensor << std::endl;
                    auto prefill_part1_valid_length = prefill_part1_input_map.at("valid_length");
                    int16toint16((int16_t *)prefill_part1_valid_length.Data(), (int16_t *)valid_length_input_tensor.data_ptr<int16_t>(), valid_length_input_tensor.numel());
                    auto prefill_part2_valid_length = prefill_part2_input_map.at("valid_length");
                    int16toint16((int16_t *)prefill_part2_valid_length.Data(), (int16_t *)valid_length_input_tensor.data_ptr<int16_t>(), valid_length_input_tensor.numel());

                    std::vector<int16_t> current_length_input = {current_length};
                    auto current_length_input_tensor_opts = torch::TensorOptions().dtype(torch::kInt16);
                    auto current_length_input_tensor = torch::from_blob(current_length_input.data(), {int64_t(current_length_input.size())}, current_length_input_tensor_opts).clone();

                    //std::cout << "current_length_input_tensor: " << current_length_input_tensor << std::endl;
                    auto prefill_part1_current_length = prefill_part1_input_map.at("current_length");
                    int16toint16((int16_t *)prefill_part1_current_length.Data(), (int16_t *)current_length_input_tensor.data_ptr<int16_t>(), current_length_input_tensor.numel());
                    auto prefill_part2_current_length = prefill_part2_input_map.at("current_length");
                    int16toint16((int16_t *)prefill_part2_current_length.Data(), (int16_t *)current_length_input_tensor.data_ptr<int16_t>(), current_length_input_tensor.numel());
                    prefill_part1_model.SetInput("input_1", input_data);
                    prefill_part1_model.SetInput("valid_length", prefill_part1_valid_length);
                    prefill_part1_model.SetInput("current_length", prefill_part1_current_length);
                    prefill_part2_model.SetInput("valid_length", prefill_part2_valid_length);
                    prefill_part2_model.SetInput("current_length", prefill_part2_current_length);
                    prefill_part1_model.Run();
                    prefill_part2_model.Run();
                    prefill_part2_model.Sync();
                    //std::cout << "prefill run finished" << std::endl;
            }

            tcim::Tensor prefill_part2_output = prefill_part2_model.GetOutput("model_layers_27_resadd2", tcim::HDPL);
            prefill_head_model.SetInput("model_layers_27_resadd2", prefill_part2_output);
            std::vector<int16_t> seq_length_input = {gather_index};
            auto seq_length_input_tensor_opts = torch::TensorOptions().dtype(torch::kInt16);
            auto seq_length_input_tensor = torch::from_blob(seq_length_input.data(), {int64_t(seq_length_input.size())}, seq_length_input_tensor_opts).clone();

            //std::cout << "seq_length_input_tensor: " << seq_length_input_tensor << std::endl;
            auto prefill_head_seq_length = prefill_head_input_map.at("current_length");
            int16toint16((int16_t *)prefill_head_seq_length.Data(), (int16_t *)seq_length_input_tensor.data_ptr<int16_t>(), seq_length_input_tensor.numel());
            prefill_head_model.SetInput("current_length", prefill_head_seq_length);
            prefill_head_model.Run();
            prefill_head_model.Sync();
            tcim::Tensor input_data = prefill_head_model.GetOutput("lm_head_add_list_0");
            //std::cout << "input_data Size: " << input_data.Size() << std::endl;
            //std::this_thread::sleep_for(std::chrono::seconds(5));
            std::vector<int16_t> prefill_output;
            for (int idx = 0; idx < input_data.Size() / 2; idx++) {
                auto data = *((int16_t *)input_data.Data() + idx);
                //if (data != 0) {
                //    std::cout << "data at " << idx << ": " << data << std::endl;
                //}
                
                prefill_output.push_back(data);
            }

            auto prefill_output_opts = torch::TensorOptions().dtype(torch::kInt16);
            auto prefill_output_tensor = torch::from_blob(prefill_output.data(), {int64_t(prefill_output.size())}, prefill_output_opts).clone();
            //std::cout << "prefill_output options: " << prefill_output_tensor.options() << std::endl;
            //std::cout << "prefill_output dim: " << prefill_output_tensor.dim() << std::endl;
            //std::cout << "prefill_output size: " << prefill_output_tensor.size(0) << std::endl;
            //for (int i = 0; i < 3; i++) {
            //    std::cout << prefill_output_tensor[i] << " ";
            //}
            //std::cout << std::endl;
            //for (int i = 152060; i < 152064; i++) {
            //    std::cout << prefill_output_tensor[i] << " ";
            //}
            //std::cout << std::endl;
            auto selected_output = torch::argmax(prefill_output_tensor).to(torch::kInt32);
            //std::cout << "selected_output: " << selected_output << std::endl;
            auto prefill_token = *(int32_t *)selected_output.data_ptr<int32_t>();
            auto response = common_token_to_piece(ctx, prefill_token);
            //std::cout << "response: ";
            //auto invalid_utf8 = false;
            //write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
            //std::cout << std::endl;

            context_length = input_echo_len;

            std::vector<llama_token> prefill_input_ids = {prefill_token};
            torch::Tensor prefill_inputs_embeds;
            auto prefill_opts = torch::TensorOptions().dtype(torch::kInt);
            auto prefill_inp = torch::from_blob(prefill_input_ids.data(), {int64_t(prefill_input_ids.size())}, prefill_opts).clone();
            //std::cout << "prefill_inp: " << prefill_inp << "\n";
            try {
                prefill_inputs_embeds = emb->forward(prefill_inp);
            }
            catch (const c10::Error& e) {
                std::cerr << "error loading the model\n" << e.what_without_backtrace();
                return -1;
            }
            //std::cout << "prefill_inputs_embeds options: " << prefill_inputs_embeds.options() << std::endl;
            //std::cout << "prefill_inputs_embeds dim: " << prefill_inputs_embeds.dim() << std::endl;
            //std::cout << "prefill_inputs_embeds size: " << prefill_inputs_embeds.size(0) << std::endl;
            //for (int i = 0; i < 3; i++) {
            //    std::cout << prefill_inputs_embeds[0][i] << " ";
            //}
            //std::cout << std::endl;
            prefill_inputs_embeds = prefill_inputs_embeds.view({1, 1, -1}).to(torch::kInt16);
            //std::cout << "prefill_inputs_embeds options: " << prefill_inputs_embeds.options() << std::endl;
            //std::cout << "prefill_inputs_embeds dim: " << prefill_inputs_embeds.dim() << std::endl;
            //std::cout << "prefill_inputs_embeds size: " << prefill_inputs_embeds.size(0) << std::endl;

            input_emd = prefill_inputs_embeds;
            //write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
            //std::cout.flush();
            token = prefill_token;
            decode_pos = batch.n_tokens;
        } else {
            //SRV_INF("%s", "decode stage\n");
            auto invalid_utf8 = false;
            auto input_data = decode_part1_input_map.at("input_1");
            int16toint16((int16_t *)input_data.Data(), (int16_t *)input_emd.data_ptr<int16_t>(), input_emd.numel());
            
            std::vector<int16_t> valid_length_input = {context_length - 1};
            auto valid_length_input_tensor_opts = torch::TensorOptions().dtype(torch::kInt16);
            auto valid_length_input_tensor = torch::from_blob(valid_length_input.data(), {int64_t(valid_length_input.size())}, valid_length_input_tensor_opts).clone();

            //std::cout << "valid_length_input_tensor: " << valid_length_input_tensor << std::endl;
            auto decode_part1_valid_length =decode_part1_input_map.at("valid_length");
            int16toint16((int16_t *)decode_part1_valid_length.Data(), (int16_t *)valid_length_input_tensor.data_ptr<int16_t>(), valid_length_input_tensor.numel());
            auto decode_part2_valid_length = decode_part2_input_map.at("valid_length");
            int16toint16((int16_t *)decode_part2_valid_length.Data(), (int16_t *)valid_length_input_tensor.data_ptr<int16_t>(), valid_length_input_tensor.numel());

            decode_part1_model.SetInput("input_1", input_data);
            decode_part1_model.SetInput("valid_length", decode_part1_valid_length);
            decode_part2_model.SetInput("valid_length", decode_part2_valid_length);
            //clock_t infer_start,infer_end;
            //infer_start = clock();
            decode_part1_model.Run();
            decode_part2_model.Run();
            decode_head_model.Run();
            decode_head_model.Sync();
            //infer_end = clock();
            auto decode_data = decode_head_model.GetOutput("lm_head_add_list_0");

            //std::cout << "decode_data Size: " << decode_data.Size() << std::endl;
            //std::this_thread::sleep_for(std::chrono::seconds(5));
            std::vector<int16_t> decode_output;
            for (int idx = 0; idx < decode_data.Size() / 2; idx++) {
                auto data = *((int16_t *)decode_data.Data() + idx);
                //if (data != 0) {
                //    std::cout << "data at " << idx << ": " << data << std::endl;
                //}
                
                decode_output.push_back(data);
            }

            auto decode_output_opts = torch::TensorOptions().dtype(torch::kInt16);
            auto decode_output_tensor = torch::from_blob(decode_output.data(), {int64_t(decode_output.size())}, decode_output_opts).clone();
            //std::cout << "decode_output_tensor options: " << decode_output_tensor.options() << std::endl;
            //std::cout << "decode_output_tensor dim: " << decode_output_tensor.dim() << std::endl;
            //std::cout << "decode_output_tensor size: " << decode_output_tensor.size(0) << std::endl;
            //for (int i = 0; i < 3; i++) {
            //    std::cout << decode_output_tensor[i] << " ";
            //}
            //std::cout << std::endl;
            //for (int i = 152060; i < 152064; i++) {
            //    std::cout << decode_output_tensor[i] << " ";
            //}
            //std::cout << std::endl;
            auto selected_output = torch::argmax(decode_output_tensor).to(torch::kInt32);
            //std::cout << "selected_output: " << selected_output << std::endl;
            auto generate_token = *(int32_t *)selected_output.data_ptr<int32_t>();
            //auto response = common_token_to_piece(ctx, generate_token);
            //std::cout << "response: ";
            //auto invalid_utf8 = false;
            //write_utf8_cstr_to_stdout(response.c_str(), invalid_utf8);
            //std::cout.flush();
            //std::cout << std::endl;

            std::vector<llama_token> decode_input_ids = {generate_token};
            torch::Tensor decode_inputs_embeds;
            auto decode_opts = torch::TensorOptions().dtype(torch::kInt);
            auto decode_inp = torch::from_blob(decode_input_ids.data(), {int64_t(decode_input_ids.size())}, decode_opts).clone();
            //std::cout << "decode_inp: " << decode_inp << "\n";
            try {
                input_emd = emb->forward(decode_inp);
            }
            catch (const c10::Error& e) {
                std::cerr << "error loading the model\n" << e.what_without_backtrace();
                return -1;
            }
            //std::cout << "input_emd options: " << input_emd.options() << std::endl;
            //std::cout << "input_emd dim: " << input_emd.dim() << std::endl;
            //std::cout << "input_emd size: " << input_emd.size(0) << std::endl;
            //for (int i = 0; i < 3; i++) {
            //    std::cout << input_emd[0][i] << " ";
            //}
            //std::cout << std::endl;
            input_emd = input_emd.view({1, 1, -1}).to(torch::kInt16);
            //std::cout << "input_emd options: " << input_emd.options() << std::endl;
            //std::cout << "input_emd dim: " << input_emd.dim() << std::endl;
            //std::cout << "input_emd size: " << input_emd.size(0) << std::endl;
            context_length = context_length + 1;
            if (llama_token_is_eog(model, generate_token)) {
                std::cout << std::endl;
                std::cout << "llama_token_is_eog" << std::endl;
            }
            token = generate_token;
            decode_pos = decode_pos + batch.n_tokens;
        }

        return 0;
    }

    void update_slots() {
        //SRV_INF("a batch.n_tokens = %d\n", batch.n_tokens);
        //SRV_INF("%s", "update_slots\n");
        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_INF("%s", "all slots are idle\n");
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
        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.n_past + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    slot.release();
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    continue;
                }

                // Shift context
                const int n_keep    = slot.params.n_keep + add_bos_token;
                const int n_left    = slot.n_past - n_keep;
                const int n_discard = slot.params.n_discard ? slot.params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                llama_kv_cache_seq_rm (ctx, slot.id, n_keep            , n_keep + n_discard);
                llama_kv_cache_seq_add(ctx, slot.id, n_keep + n_discard, slot.n_past,        -n_discard);

                if (slot.params.cache_prompt) {
                    for (size_t i = n_keep + n_discard; i < slot.cache_tokens.size(); i++) {
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
        //SRV_INF("b batch.n_tokens = %d\n", batch.n_tokens);

        // frist, add sampled tokens from any ongoing sequences
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            slot.i_batch = batch.n_tokens;

            common_batch_add(batch, slot.sampled, slot.n_past, { slot.id }, true);

            slot.n_past += 1;

            if (slot.params.cache_prompt) {
                slot.cache_tokens.push_back(slot.sampled);
            }

            SLT_DBG(slot, "slot decode token, n_ctx = %d, n_past = %d, n_cache_tokens = %d, truncated = %d\n",
                    slot.n_ctx, slot.n_past, (int) slot.cache_tokens.size(), slot.truncated);
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx);
        int32_t n_ubatch = llama_n_ubatch(ctx);

        // track if this is an embedding or non-embedding batch
        // if we've added sampled tokens above, we are in non-embedding mode
        // -1: none, 0: non-embedding, 1: embedding
        // TODO: make enum
        int32_t batch_type = batch.n_tokens > 0 ? 0 : -1;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    decode_pos = 0;
                    auto & prompt_tokens = slot.prompt_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.n_past = 0;
                        slot.n_prompt_tokens = prompt_tokens.size();
                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_INF(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, n_prompt_tokens = %d\n", slot.n_ctx, slot.params.n_keep, slot.n_prompt_tokens);

                        // print prompt tokens (for debugging)
                        if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, prompt_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx, prompt_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) prompt_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx, prompt_tokens[i]).c_str());
                            }
                        }

                        // empty prompt passed -> release the slot and send empty response
                        if (prompt_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.release();
                            slot.print_timings();
                            send_final_response(slot);
                            continue;
                        }

                        if (slot.is_non_causal()) {
                            if (slot.n_prompt_tokens > n_ubatch) {
                                slot.release();
                                send_error(slot, "input is too large to process. increase the physical batch size", ERROR_TYPE_SERVER);
                                continue;
                            }

                            if (slot.n_prompt_tokens > slot.n_ctx) {
                                slot.release();
                                send_error(slot, "input is larger than the max context size. skipping", ERROR_TYPE_SERVER);
                                continue;
                            }
                        } else {
                            if (!params_base.ctx_shift) {
                                // if context shift is disabled, we make sure prompt size is smaller than KV size
                                // TODO: there should be a separate parameter that control prompt truncation
                                //       context shift should be applied only during the generation phase
                                if (slot.n_prompt_tokens >= slot.n_ctx) {
                                    slot.release();
                                    send_error(slot, "the request exceeds the available context size. try increasing the context size or enable context shift", ERROR_TYPE_INVALID_REQUEST);
                                    continue;
                                }
                            }
                            if (slot.params.n_keep < 0) {
                                slot.params.n_keep = slot.n_prompt_tokens;
                            }
                            slot.params.n_keep = std::min(slot.n_ctx - 4, slot.params.n_keep);

                            // if input prompt is too big, truncate it
                            if (slot.n_prompt_tokens >= slot.n_ctx) {
                                const int n_left = slot.n_ctx - slot.params.n_keep;

                                const int n_block_size = n_left / 2;
                                const int erased_blocks = (slot.n_prompt_tokens - slot.params.n_keep - n_block_size) / n_block_size;

                                llama_tokens new_tokens(
                                        prompt_tokens.begin(),
                                        prompt_tokens.begin() + slot.params.n_keep);

                                new_tokens.insert(
                                        new_tokens.end(),
                                        prompt_tokens.begin() + slot.params.n_keep + erased_blocks * n_block_size,
                                        prompt_tokens.end());

                                prompt_tokens = std::move(new_tokens);

                                slot.truncated = true;
                                slot.n_prompt_tokens = prompt_tokens.size();

                                SLT_WRN(slot, "input truncated, n_ctx = %d, n_keep = %d, n_left = %d, n_prompt_tokens = %d\n", slot.n_ctx, slot.params.n_keep, n_left, slot.n_prompt_tokens);

                                GGML_ASSERT(slot.n_prompt_tokens < slot.n_ctx);
                            }

                            if (slot.params.cache_prompt) {
                                //SRV_INF("e slot.n_past = %d\n", slot.n_past);
                                //SRV_INF("e slot.n_prompt_tokens = %d\n", slot.n_prompt_tokens);
                                // reuse any previously computed tokens that are common with the new prompt
                                slot.n_past = common_lcp(slot.cache_tokens, prompt_tokens);

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (params_base.n_cache_reuse > 0) {
                                    size_t head_c = slot.n_past; // cache
                                    size_t head_p = slot.n_past; // current prompt

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, slot.n_past = %d\n", params_base.n_cache_reuse, slot.n_past);

                                    while (head_c < slot.cache_tokens.size() &&
                                           head_p < prompt_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.cache_tokens.size() &&
                                               head_p + n_match < prompt_tokens.size()     &&
                                               slot.cache_tokens[head_c + n_match] == prompt_tokens[head_p + n_match]) {

                                            n_match++;
                                        }

                                        if (n_match >= (size_t) params_base.n_cache_reuse) {
                                            SLT_INF(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            llama_kv_cache_seq_rm (ctx, slot.id, head_p, head_c);
                                            llama_kv_cache_seq_add(ctx, slot.id, head_c, -1,     kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.cache_tokens[head_p + i] = slot.cache_tokens[head_c + i];
                                                slot.n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new slot.n_past = %d\n", slot.n_past);
                                }
                                //SRV_INF("f slot.n_past = %d\n", slot.n_past);
                                //SRV_INF("f slot.n_prompt_tokens = %d\n", slot.n_prompt_tokens);
                            }
                        }

                        if (slot.n_past == slot.n_prompt_tokens && slot.n_past > 0) {
                            // we have to evaluate at least 1 token to generate logits.
                            SLT_WRN(slot, "need to evaluate at least 1 token to generate logits, n_past = %d, n_prompt_tokens = %d\n", slot.n_past, slot.n_prompt_tokens);

                            slot.n_past--;
                        }

                        slot.n_prompt_tokens_processed = 0;
                    }

                    // non-causal tasks require to fit the entire prompt in the physical batch
                    if (slot.is_non_causal()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.n_prompt_tokens > n_batch) {
                            continue;
                        }
                    }

                    // check that we are in the right batch_type, if not defer the slot
                    int slot_type = slot.is_non_causal();
                    if (batch_type == -1) {
                        batch_type = slot_type;
                    } else if (batch_type != slot_type) {
                        continue;
                    }

                    // keep only the common part
                    if (!llama_kv_cache_seq_rm(ctx, slot.id, slot.n_past, -1)) {
                        // could not partially delete (likely using a non-Transformer model)
                        llama_kv_cache_seq_rm(ctx, slot.id, -1, -1);

                        // there is no common part left
                        slot.n_past = 0;
                    }

                    SLT_INF(slot, "kv cache rm [%d, end)\n", slot.n_past);

                    // remove the non-common part from the cache
                    slot.cache_tokens.resize(slot.n_past);
                    //SRV_INF("c batch.n_tokens = %d\n", batch.n_tokens);
                    //SRV_INF("c slot.n_past = %d\n", slot.n_past);
                    //SRV_INF("c slot.n_prompt_tokens = %d\n", slot.n_prompt_tokens);

                    // add prompt tokens for processing in the current batch
                    while (slot.n_past < slot.n_prompt_tokens && batch.n_tokens < n_batch) {
                        common_batch_add(batch, prompt_tokens[slot.n_past], slot.n_past, { slot.id }, false);

                        if (slot.params.cache_prompt) {
                            slot.cache_tokens.push_back(prompt_tokens[slot.n_past]);
                        }

                        slot.n_prompt_tokens_processed++;
                        slot.n_past++;
                    }

                    SLT_INF(slot, "prompt processing progress, n_past = %d, n_tokens = %d, progress = %f\n", slot.n_past, batch.n_tokens, (float) slot.n_prompt_tokens_processed / slot.n_prompt_tokens);

                    // entire prompt has been processed
                    if (slot.n_past == slot.n_prompt_tokens) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        common_sampler_reset(slot.smpl);

                        // Process all prompt tokens through sampler system
                        for (int i = 0; i < slot.n_prompt_tokens; ++i) {
                            common_sampler_accept(slot.smpl, prompt_tokens[i], false);
                        }

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        SLT_INF(slot, "prompt done, n_past = %d, n_tokens = %d\n", slot.n_past, batch.n_tokens);
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

        // make sure we're in the right embedding mode
        llama_set_embeddings(ctx, batch_type == 1);

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i += n_batch) {
            //SRV_INF("batch.n_tokens = %d\n", batch.n_tokens);
            //SRV_INF("decode_pos = %d\n", decode_pos);
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            const int ret = houmo_decode(batch);
            metrics.on_decoded(slots);

            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    SRV_ERR("failed to decode the batch: KV cache is full - try increasing it via the context size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);
                    for (auto & slot : slots) {
                        slot.release();
                        send_error(slot, "Input prompt is too big compared to KV size. Please try increasing KV size.");
                    }
                    break; // break loop of n_batch
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size - try increasing it via the context size or enable defragmentation, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            for (auto & slot : slots) {
                
                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }
                //SRV_INF("n_decoded = %d\n", slot.n_decoded);

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task_type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        //send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task_type == SERVER_TASK_TYPE_RERANK) {
                        //send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                llama_token id = token;

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl, id, true);

                slot.n_decoded += 1;

                const int64_t t_current = ggml_time_us();

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = (t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok = id;

                const auto * cur_p = common_sampler_get_candidates(slot.smpl);

                for (size_t i = 0; i < (size_t) slot.params.sampling.n_probs; ++i) {
                    auto tok_id = cur_p->data[i].id;
                    result.probs.push_back({
                        tok_id,
                        tokens_to_output_formatted_string(ctx, tok_id),
                        i >= cur_p->size ? 0.0f : cur_p->data[i].p,
                    });
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.release();
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    continue;
                }
            }

            // do speculative decoding
            for (auto & slot : slots) {
                if (!slot.is_processing() || !slot.can_speculate()) {
                    continue;
                }

                if (slot.state != SLOT_STATE_GENERATING) {
                    continue;
                }

                // determine the max draft that fits the current slot state
                int n_draft_max = slot.params.speculative.n_max;

                // note: n_past is not yet increased for the `id` token sampled above
                //       also, need to leave space for 1 extra token to allow context shifts
                n_draft_max = std::min(n_draft_max, slot.n_ctx - slot.n_past - 2);

                if (slot.n_remaining > 0) {
                    n_draft_max = std::min(n_draft_max, slot.n_remaining - 1);
                }

                SLT_DBG(slot, "max possible draft: %d\n", n_draft_max);

                if (n_draft_max < slot.params.speculative.n_min) {
                    SLT_DBG(slot, "the max possible draft is too small: %d < %d - skipping speculative decoding\n", n_draft_max, slot.params.speculative.n_min);

                    continue;
                }

                llama_token id = slot.sampled;

                struct common_speculative_params params_spec;
                params_spec.n_draft   = n_draft_max;
                params_spec.n_reuse   = llama_n_ctx(slot.ctx_dft) - slot.params.speculative.n_max;
                params_spec.p_min     = slot.params.speculative.p_min;

                llama_tokens draft = common_speculative_gen_draft(slot.spec, params_spec, slot.cache_tokens, id);

                // ignore small drafts
                if (slot.params.speculative.n_min > (int) draft.size()) {
                    SLT_DBG(slot, "ignoring small draft: %d < %d\n", (int) draft.size(), slot.params.speculative.n_min);

                    continue;
                }

                // construct the speculation batch
                common_batch_clear(slot.batch_spec);
                common_batch_add  (slot.batch_spec, id, slot.n_past, { slot.id }, true);

                for (size_t i = 0; i < draft.size(); ++i) {
                    common_batch_add(slot.batch_spec, draft[i], slot.n_past + 1 + i, { slot.id }, true);
                }

                SLT_DBG(slot, "decoding speculative batch, size = %d\n", slot.batch_spec.n_tokens);

                llama_decode(ctx, slot.batch_spec);

                // the accepted tokens from the speculation
                const auto ids = common_sampler_sample_and_accept_n(slot.smpl, ctx, draft);

                slot.n_past    += ids.size();
                slot.n_decoded += ids.size();

                slot.cache_tokens.push_back(id);
                slot.cache_tokens.insert(slot.cache_tokens.end(), ids.begin(), ids.end() - 1);

                llama_kv_cache_seq_rm(ctx, slot.id, slot.n_past, -1);

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok = ids[i];

                    if (!process_token(result, slot)) {
                        // release slot because of stop condition
                        slot.release();
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        break;
                    }
                }

                SLT_DBG(slot, "accepted %d/%d draft tokens, new n_past = %d\n", (int) ids.size() - 1, (int) draft.size(), slot.n_past);
            }
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    json model_meta() const {
        return json {
            {"vocab_type",  llama_vocab_type    (model)},
            {"n_vocab",     llama_n_vocab       (model)},
            {"n_ctx_train", llama_n_ctx_train   (model)},
            {"n_embd",      llama_n_embd        (model)},
            {"n_params",    llama_model_n_params(model)},
            {"size",        llama_model_size    (model)},
        };
    }
};

static void log_server_request(const httplib::Request & req, const httplib::Response & res) {
    // skip GH copilot requests when using default port
    if (req.path == "/v1/health" || req.path == "/v1/completions") {
        return;
    }

    LOG_INF("request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);

    LOG_DBG("request:  %s\n", req.body.c_str());
    LOG_DBG("response: %s\n", res.body.c_str());
}

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

int main(int argc, char ** argv) {
    // own arguments required by this example
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    common_init();

    // struct that contains llama context and inference
    server_context ctx_server;

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.cpuparams.n_threads, params.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    std::unique_ptr<httplib::Server> svr;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_INF("Running with SSL: key = %s, cert = %s\n", params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
        svr.reset(
            new httplib::SSLServer(params.ssl_file_cert.c_str(), params.ssl_file_key.c_str())
        );
    } else {
        LOG_INF("Running without SSL\n");
        svr.reset(new httplib::Server());
    }
#else
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_ERR("Server is built without SSL support\n");
        return 1;
    }
    svr.reset(new httplib::Server());
#endif

    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    svr->set_default_headers({{"Server", "llama.cpp"}});
    svr->set_logger(log_server_request);

    auto res_error = [](httplib::Response & res, const json & error_data) {
        json final_response {{"error", error_data}};
        res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
        res.status = json_value(error_data, "code", 500);
    };

    auto res_ok = [](httplib::Response & res, const json & data) {
        res.set_content(safe_json_to_str(data), MIMETYPE_JSON);
        res.status = 200;
    };

    svr->set_exception_handler([&res_error](const httplib::Request &, httplib::Response & res, const std::exception_ptr & ep) {
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
        LOG_WRN("got exception: %s\n", formatted_error.dump().c_str());
        res_error(res, formatted_error);
    });

    svr->set_error_handler([&res_error](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // for other error codes, we skip processing here because it's already done by res_error()
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout (params.timeout_read);
    svr->set_write_timeout(params.timeout_write);

    std::unordered_map<std::string, std::string> log_data;

    log_data["hostname"] = params.hostname;
    log_data["port"]     = std::to_string(params.port);

    if (params.api_keys.size() == 1) {
        auto key = params.api_keys[0];
        log_data["api_key"] = "api_key: ****" + key.substr(std::max((int)(key.length() - 4), 0));
    } else if (params.api_keys.size() > 1) {
        log_data["api_key"] = "api_key: " + std::to_string(params.api_keys.size()) + " keys loaded";
    }

    // Necessary similarity of prompt for slot selection
    ctx_server.slot_prompt_similarity = params.slot_prompt_similarity;

    //
    // Middlewares
    //

    auto middleware_validate_api_key = [&params, &res_error](const httplib::Request & req, httplib::Response & res) {
        static const std::unordered_set<std::string> public_endpoints = {
            "/health",
            "/models",
            "/v1/models",
        };

        // If API key is not set, skip validation
        if (params.api_keys.empty()) {
            return true;
        }

        // If path is public or is static file, skip validation
        if (public_endpoints.find(req.path) != public_endpoints.end() || req.path == "/") {
            return true;
        }

        // Check for API key in the header
        auto auth_header = req.get_header_value("Authorization");

        std::string prefix = "Bearer ";
        if (auth_header.substr(0, prefix.size()) == prefix) {
            std::string received_api_key = auth_header.substr(prefix.size());
            if (std::find(params.api_keys.begin(), params.api_keys.end(), received_api_key) != params.api_keys.end()) {
                return true; // API key is valid
            }
        }

        // API key is invalid or not provided
        res_error(res, format_error_response("Invalid API Key", ERROR_TYPE_AUTHENTICATION));

        LOG_WRN("Unauthorized: Invalid API Key\n");

        return false;
    };

    auto middleware_server_state = [&res_error, &state](const httplib::Request & req, httplib::Response & res) {
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_LOADING_MODEL) {
            auto tmp = string_split<std::string>(req.path, '.');
            if (req.path == "/" || tmp.back() == "html") {
                res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len, "text/html; charset=utf-8");
                res.status = 503;
            } else {
                res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            }
            return false;
        }
        return true;
    };

    // register server middlewares
    svr->set_pre_routing_handler([&middleware_validate_api_key, &middleware_server_state](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods",     "GET, POST");
            res.set_header("Access-Control-Allow-Headers",     "*");
            res.set_content("", "text/html"); // blank response, no data
            return httplib::Server::HandlerResponse::Handled; // skip further processing
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

    const auto handle_health = [&](const httplib::Request &, httplib::Response & res) {
        // error and loading states are handled by middleware
        json health = {{"status", "ok"}};
        res_ok(res, health);
    };

    const auto handle_slots = [&](const httplib::Request & req, httplib::Response & res) {
        if (!params.endpoint_slots) {
            res_error(res, format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        // request slots data using task queue
        server_task task(SERVER_TASK_TYPE_METRICS);
        task.id = ctx_server.queue_tasks.get_new_id();
        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(task, true); // high-priority task

        // get the result
        server_task_result_ptr result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        if (result->is_error()) {
            res_error(res, result->to_json());
            return;
        }

        // TODO: get rid of this dynamic_cast
        auto res_metrics = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_metrics != nullptr);

        // optionally return "fail_on_no_slot" error
        if (req.has_param("fail_on_no_slot")) {
            if (res_metrics->n_idle_slots == 0) {
                res_error(res, format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return;
            }
        }

        res_ok(res, res_metrics->slots_data);
    };

    const auto handle_metrics = [&](const httplib::Request &, httplib::Response & res) {
        if (!params.endpoint_metrics) {
            res_error(res, format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        // request slots data using task queue
        server_task task(SERVER_TASK_TYPE_METRICS);
        task.id = ctx_server.queue_tasks.get_new_id();
        task.metrics_reset_bucket = true;

        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(task, true); // high-priority task

        // get the result
        server_task_result_ptr result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        if (result->is_error()) {
            res_error(res, result->to_json());
            return;
        }

        // TODO: get rid of this dynamic_cast
        auto res_metrics = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_metrics != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_metrics->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_metrics->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_metrics->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_metrics->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_metrics->n_decode_total}
            }, {
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_metrics->n_busy_slots_total / (float) res_metrics->n_decode_total}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_metrics->n_prompt_tokens_processed ? 1.e3 / res_metrics->t_prompt_processing * res_metrics->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_metrics->n_tokens_predicted ? 1.e3 / res_metrics->t_tokens_generation * res_metrics->n_tokens_predicted : 0.}
            },{
                    {"name",  "kv_cache_usage_ratio"},
                    {"help",  "KV-cache usage. 1 means 100 percent usage."},
                    {"value",  1. * res_metrics->kv_cache_used_cells / params.n_ctx}
            },{
                    {"name",  "kv_cache_tokens"},
                    {"help",  "KV-cache tokens."},
                    {"value",  (uint64_t) res_metrics->kv_cache_tokens_count}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of request processing."},
                    {"value",  (uint64_t) res_metrics->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of request deferred."},
                    {"value",  (uint64_t) res_metrics->n_tasks_deferred}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res.set_header("Process-Start-Time-Unix", std::to_string(res_metrics->t_start));

        res.set_content(prometheus.str(), "text/plain; version=0.0.4");
        res.status = 200; // HTTP OK
    };

    const auto handle_slots_save = [&ctx_server, &res_error, &res_ok, &params](const httplib::Request & req, httplib::Response & res, int id_slot) {
        json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = ctx_server.queue_tasks.get_new_id();
        task.slot_action.slot_id  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;

        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(task);

        server_task_result_ptr result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        if (result->is_error()) {
            res_error(res, result->to_json());
            return;
        }

        res_ok(res, result->to_json());
    };

    const auto handle_slots_restore = [&ctx_server, &res_error, &res_ok, &params](const httplib::Request & req, httplib::Response & res, int id_slot) {
        json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = ctx_server.queue_tasks.get_new_id();
        task.slot_action.slot_id  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;

        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(task);

        server_task_result_ptr result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        if (result->is_error()) {
            res_error(res, result->to_json());
            return;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
        res_ok(res, result->to_json());
    };

    const auto handle_slots_erase = [&ctx_server, &res_error, &res_ok](const httplib::Request & /* req */, httplib::Response & res, int id_slot) {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = ctx_server.queue_tasks.get_new_id();
        task.slot_action.slot_id = id_slot;

        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(task);

        server_task_result_ptr result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        if (result->is_error()) {
            res_error(res, result->to_json());
            return;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
        res_ok(res, result->to_json());
    };

    const auto handle_slots_action = [&params, &res_error, &handle_slots_save, &handle_slots_restore, &handle_slots_erase](const httplib::Request & req, httplib::Response & res) {
        if (params.slot_save_path.empty()) {
            res_error(res, format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        std::string id_slot_str = req.path_params.at("id_slot");
        int id_slot;

        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res_error(res, format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
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
            res_error(res, format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        }
    };

    const auto handle_props = [&ctx_server, &res_ok](const httplib::Request &, httplib::Response & res) {
        // this endpoint is publicly available, please only return what is safe to be exposed
        json data = {
            { "default_generation_settings", ctx_server.default_generation_settings_for_props },
            { "total_slots",                 ctx_server.params_base.n_parallel },
            { "model_path",                  ctx_server.params_base.model },
            { "chat_template",               llama_get_chat_template(ctx_server.model) },
        };

        res_ok(res, data);
    };

    const auto handle_props_change = [&ctx_server, &res_error, &res_ok](const httplib::Request & req, httplib::Response & res) {
        if (!ctx_server.params_base.endpoint_props) {
            res_error(res, format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        json data = json::parse(req.body);

        // update any props here

        res_ok(res, {{ "success", true }});
    };

    // handle completion-like requests (completion, chat, infill)
    // we can optionally provide a custom format for partial results and final results
    const auto handle_completions_generic = [&ctx_server, &res_error, &res_ok](
            server_task_type type,
            json & data,
            httplib::Response & res,
            bool oaicompat = false,
            bool oaicompat_chat = false) {
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

        if (ctx_server.params_base.embedding) {
            res_error(res, format_error_response("This server does not support completions. Start it without `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;

        try {
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.ctx, data.at("prompt"), true, true);
            tasks.reserve(tokenized_prompts.size());
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task = server_task(type);

                task.id    = ctx_server.queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens    = std::move(tokenized_prompts[i]);
                task.params           = server_task::params_from_json_cmpl(ctx_server.model, ctx_server.params_base, data);
                task.id_selected_slot = json_value(data, "id_slot", -1);

                // OAI-compat
                task.params.oaicompat           = oaicompat;
                task.params.oaicompat_chat      = oaicompat_chat;
                task.params.oaicompat_cmpl_id   = completion_id;
                // oaicompat_model is already populated by params_from_json_cmpl

                tasks.push_back(task);
            }
        } catch (const std::exception & e) {
            res_error(res, format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        bool stream = json_value(data, "stream", false);
        const auto task_ids = server_task::get_list_id(tasks);

        if (!stream) {
            ctx_server.receive_multi_results(task_ids, [&](std::vector<server_task_result_ptr> & results) {
                if (results.size() == 1) {
                    // single result
                    res_ok(res, results[0]->to_json());
                } else {
                    // multiple results (multitask)
                    json arr = json::array();
                    for (auto & res : results) {
                        arr.push_back(res->to_json());
                    }
                    res_ok(res, arr);
                }
            }, [&](const json & error_data) {
                res_error(res, error_data);
            });

            ctx_server.queue_results.remove_waiting_task_ids(task_ids);
        } else {
            const auto chunked_content_provider = [task_ids, &ctx_server, oaicompat](size_t, httplib::DataSink & sink) {
                ctx_server.receive_cmpl_results_stream(task_ids, [&](server_task_result_ptr & result) -> bool {
                    json res_json = result->to_json();
                    if (res_json.is_array()) {
                        for (const auto & res : res_json) {
                            if (!server_sent_event(sink, "data", res)) {
                                return false;
                            }
                        }
                        return true;
                    } else {
                        return server_sent_event(sink, "data", res_json);
                    }
                }, [&](const json & error_data) {
                    server_sent_event(sink, "error", error_data);
                });
                if (oaicompat) {
                    static const std::string ev_done = "data: [DONE]\n\n";
                    sink.write(ev_done.data(), ev_done.size());
                }
                sink.done();
                return false;
            };

            auto on_complete = [task_ids, &ctx_server] (bool) {
                ctx_server.queue_results.remove_waiting_task_ids(task_ids);
            };

            res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
        }
    };

    auto generate_error = [](const json & error_data) {
        const std::string str =
        error_data.dump(-1, ' ', false, json::error_handler_t::replace) +
            "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).

        SRV_ERR("%s\n", str.c_str());
    };

    auto generate_ok = [](const json & data) {
        const std::string str =
        data.dump(-1, ' ', false, json::error_handler_t::replace) +
            "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
        SRV_INF("%s\n", str.c_str());
    };

    // handle completion-like requests (completion, chat, infill)
    // we can optionally provide a custom format for partial results and final results
    const auto handle_completions_generate = [&ctx_server, &generate_error, &generate_ok](
            server_task_type type,
            json & data,
            bool oaicompat = false,
            bool oaicompat_chat = false) {
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

        if (ctx_server.params_base.embedding) {
            generate_error(format_error_response("This server does not support completions. Start it without `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;

        try {
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.ctx, data.at("prompt"), true, true);
            tasks.reserve(tokenized_prompts.size());
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task = server_task(type);

                task.id    = ctx_server.queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens    = std::move(tokenized_prompts[i]);
                task.params           = server_task::params_from_json_cmpl(ctx_server.model, ctx_server.params_base, data);
                task.id_selected_slot = json_value(data, "id_slot", -1);

                // OAI-compat
                task.params.oaicompat           = oaicompat;
                task.params.oaicompat_chat      = oaicompat_chat;
                task.params.oaicompat_cmpl_id   = completion_id;
                // oaicompat_model is already populated by params_from_json_cmpl

                tasks.push_back(task);
            }
        } catch (const std::exception & e) {
            generate_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        bool stream = json_value(data, "stream", false);
        const auto task_ids = server_task::get_list_id(tasks);

        if (!stream) {
            SRV_INF("mode: %s\n", "no stream");
            ctx_server.receive_multi_results(task_ids, [&](std::vector<server_task_result_ptr> & results) {
                if (results.size() == 1) {
                    // single result
                    generate_ok(results[0]->to_json());
                } else {
                    // multiple results (multitask)
                    json arr = json::array();
                    for (auto & res : results) {
                        arr.push_back(res->to_json());
                    }
                    generate_ok(arr);
                }
            }, [&](const json & error_data) {
                generate_error(error_data);
            });

            ctx_server.queue_results.remove_waiting_task_ids(task_ids);
        } else {
            SRV_INF("mode: %s\n", "stream");
            ctx_server.receive_cmpl_results_stream(task_ids, [&](server_task_result_ptr & result) -> bool {
                json res_json = result->to_json();
                if (res_json.is_array()) {
                    for (const auto & res : res_json) {
                        generate_ok(res);
                    }
                    return true;
                } else {
                    generate_ok(res_json);
                }
            }, [&](const json & error_data) {
                generate_error(error_data);
            });

            ctx_server.queue_results.remove_waiting_task_ids(task_ids);
        }
    };

    const auto handle_completions_generate_v2 = [&ctx_server, &generate_error, &generate_ok](
            server_task_type type,
            json & data,
            bool oaicompat = false,
            bool oaicompat_chat = false) {
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

        auto completion_id = gen_chatcmplid();
        std::vector<server_task> tasks;

        try {
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.ctx, data.at("prompt"), true, true);
            tasks.reserve(tokenized_prompts.size());
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task = server_task(type);

                task.id    = ctx_server.queue_tasks.get_new_id();
                task.index = i;

                task.prompt_tokens    = std::move(tokenized_prompts[i]);
                task.params           = server_task::params_from_json_cmpl(ctx_server.model, ctx_server.params_base, data);
                task.id_selected_slot = json_value(data, "id_slot", -1);

                // OAI-compat
                task.params.oaicompat           = oaicompat;
                task.params.oaicompat_chat      = oaicompat_chat;
                task.params.oaicompat_cmpl_id   = completion_id;
                // oaicompat_model is already populated by params_from_json_cmpl

                tasks.push_back(task);
            }
        } catch (const std::exception & e) {
            generate_error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(tasks);

        bool stream = json_value(data, "stream", false);
        const auto task_ids = server_task::get_list_id(tasks);

        if (!stream) {
            SRV_INF("mode: %s\n", "no stream");
            std::vector<server_task_result_ptr> results(task_ids.size());
            ctx_server.generate(task_ids, results);
            if (results.size() == 1) {
                    // single result
                generate_ok(results[0]->to_json());
            } else {
                // multiple results (multitask)
                json arr = json::array();
                for (auto & res : results) {
                    arr.push_back(res->to_json());
                }
                generate_ok(arr);
            }
        } else {
            SRV_INF("mode: %s\n", "stream");
            ctx_server.generate_stream(task_ids);
            while (true) {
                server_task_result_ptr result;
                int ret = ctx_server.generate_stream_next_token(task_ids, result);
                if (ret == 1) {
                    SRV_INF("stop found: %s\n", dump_result(result->to_json()).c_str());
                    break;
                } else {
                    SRV_INF("content: %s\n", dump_result(result->to_json()).c_str());
                }
            }    
        }
    };

    const auto handle_completions = [&handle_completions_generic](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);
        return handle_completions_generic(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            res,
            /* oaicompat */ false,
            /* oaicompat_chat */ false);
    };

    const auto handle_infill = [&ctx_server, &res_error, &handle_completions_generic](const httplib::Request & req, httplib::Response & res) {
        // check model compatibility
        std::string err;
        if (llama_token_fim_pre(ctx_server.model) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_token_fim_suf(ctx_server.model) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_token_fim_mid(ctx_server.model) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res_error(res, format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        json data = json::parse(req.body);

        // validate input
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res_error(res, format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res_error(res, format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res_error(res, format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res_error(res, format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res_error(res, format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res_error(res, format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.ctx, prompt, true, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_infill(
            ctx_server.ctx,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            ctx_server.params_base.n_batch,
            ctx_server.params_base.n_predict,
            ctx_server.slots[0].n_ctx, // TODO: there should be a better way
            ctx_server.params_base.spm_infill,
            tokenized_prompts[0]
        );

        return handle_completions_generic(SERVER_TASK_TYPE_INFILL, data, res);
    };

    const auto handle_chat_completions = [&ctx_server, &params, &res_error, &handle_completions_generic](const httplib::Request & req, httplib::Response & res) {
        if (ctx_server.params_base.embedding) {
            res_error(res, format_error_response("This server does not support completions. Start it without `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        json data = oaicompat_completion_params_parse(ctx_server.model, json::parse(req.body), params.chat_template);
        return handle_completions_generic(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            res,
            /* oaicompat */ true,
            /* oaicompat_chat */ true);
    };

    const auto handle_models = [&params, &ctx_server, &res_ok](const httplib::Request &, httplib::Response & res) {
        json models = {
            {"object", "list"},
            {"data", {
                {
                    {"id",       params.model_alias},
                    {"object",   "model"},
                    {"created",  std::time(0)},
                    {"owned_by", "llamacpp"},
                    {"meta",     ctx_server.model_meta()}
                },
             }}
        };

        res_ok(res, models);
    };

    const auto handle_tokenize = [&ctx_server, &res_ok](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.ctx, body.at("content"), add_special, true);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.ctx, token);
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

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        const json data = format_tokenizer_response(tokens_response);
        res_ok(res, data);
    };

    const auto handle_detokenize = [&ctx_server, &res_ok](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.ctx, tokens.cbegin(), tokens.cend());
        }

        const json data = format_detokenized_response(content);
        res_ok(res, data);
    };

    const auto handle_embeddings = [&ctx_server, &res_error, &res_ok](const httplib::Request & req, httplib::Response & res) {
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
            res_error(res, format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        // create and queue the task
        json responses = json::array();
        bool error = false;
        {
            std::vector<server_task> tasks;
            std::vector<llama_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.ctx, prompt, /* add_special */ false, true);
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task   = server_task(SERVER_TASK_TYPE_EMBEDDING);
                task.id            = ctx_server.queue_tasks.get_new_id();
                task.index         = i;
                task.prompt_tokens = std::move(tokenized_prompts[i]);
                tasks.push_back(task);
            }

            ctx_server.queue_results.add_waiting_tasks(tasks);
            ctx_server.queue_tasks.post(tasks);

            // get the result
            std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

            ctx_server.receive_multi_results(task_ids, [&](std::vector<server_task_result_ptr> & results) {
                for (auto & res : results) {
                    GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
                    responses.push_back(res->to_json());
                }
            }, [&](const json & error_data) {
                res_error(res, error_data);
                error = true;
            });

            ctx_server.queue_results.remove_waiting_task_ids(task_ids);
        }

        if (error) {
            return;
        }

        // write JSON response
        json root = oaicompat
            ? format_embeddings_response_oaicompat(body, responses)
            : responses.size() == 1 ? responses[0] : json(responses);
        res_ok(res, root);
    };

    const auto handle_rerank = [&ctx_server, &res_error, &res_ok](const httplib::Request & req, httplib::Response & res) {
        if (!ctx_server.params_base.reranking || ctx_server.params_base.embedding) {
            res_error(res, format_error_response("This server does not support reranking. Start it with `--reranking` and without `--embedding`", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        const json body = json::parse(req.body);

        // TODO: implement
        //int top_n = 1;
        //if (body.count("top_n") != 1) {
        //    top_n = body.at("top_n");
        //} else {
        //    res_error(res, format_error_response("\"top_n\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        //    return;
        //}

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res_error(res, format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return;
            }
        } else {
            res_error(res, format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        std::vector<std::string> documents = json_value(body, "documents", std::vector<std::string>());
        if (documents.empty()) {
            res_error(res, format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        llama_tokens tokenized_query = tokenize_input_prompts(ctx_server.ctx, query, /* add_special */ false, true)[0];

        // create and queue the task
        json responses = json::array();
        bool error = false;
        {
            std::vector<server_task> tasks;
            std::vector<llama_tokens> tokenized_docs = tokenize_input_prompts(ctx_server.ctx, documents, /* add_special */ false, true);
            tasks.reserve(tokenized_docs.size());
            for (size_t i = 0; i < tokenized_docs.size(); i++) {
                server_task task   = server_task(SERVER_TASK_TYPE_RERANK);
                task.id            = ctx_server.queue_tasks.get_new_id();
                task.index         = i;
                task.prompt_tokens = format_rerank(ctx_server.model, tokenized_query, tokenized_docs[i]);
                tasks.push_back(task);
            }

            ctx_server.queue_results.add_waiting_tasks(tasks);
            ctx_server.queue_tasks.post(tasks);

            // get the result
            std::unordered_set<int> task_ids = server_task::get_list_id(tasks);

            ctx_server.receive_multi_results(task_ids, [&](std::vector<server_task_result_ptr> & results) {
                for (auto & res : results) {
                    GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                    responses.push_back(res->to_json());
                }
            }, [&](const json & error_data) {
                res_error(res, error_data);
                error = true;
            });
        }

        if (error) {
            return;
        }

        // write JSON response
        json root = format_response_rerank(body, responses);
        res_ok(res, root);
    };

    const auto handle_lora_adapters_list = [&](const httplib::Request &, httplib::Response & res) {
        json result = json::array();
        for (size_t i = 0; i < ctx_server.loras.size(); ++i) {
            auto & lora = ctx_server.loras[i];
            result.push_back({
                {"id", i},
                {"path", lora.path},
                {"scale", lora.scale},
            });
        }
        res_ok(res, result);
        res.status = 200; // HTTP OK
    };

    const auto handle_lora_adapters_apply = [&](const httplib::Request & req, httplib::Response & res) {
        const std::vector<json> body = json::parse(req.body);
        int max_idx = ctx_server.loras.size();

        // clear existing value
        for (auto & lora : ctx_server.loras) {
            lora.scale = 0.0f;
        }

        // set value
        for (auto entry : body) {
            int id      = entry.at("id");
            float scale = entry.at("scale");
            if (0 <= id && id < max_idx) {
                ctx_server.loras[id].scale = scale;
            } else {
                throw std::runtime_error("invalid adapter id");
            }
        }

        server_task task(SERVER_TASK_TYPE_SET_LORA);
        task.id = ctx_server.queue_tasks.get_new_id();
        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(task);

        server_task_result_ptr result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        if (result->is_error()) {
            res_error(res, result->to_json());
            return;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res_ok(res, result->to_json());
    };

    //
    // Router
    //

    if (!params.webui) {
        LOG_INF("Web UI is disabled\n");
    } else {
        // register static assets routes
        if (!params.public_path.empty()) {
            // Set the base directory for serving static files
            bool is_found = svr->set_mount_point("/", params.public_path);
            if (!is_found) {
                LOG_ERR("%s: static assets path not found: %s\n", __func__, params.public_path.c_str());
                return 1;
            }
        } else {
            // using embedded static index.html
            svr->Get("/", [](const httplib::Request &, httplib::Response & res) {
                res.set_content(reinterpret_cast<const char*>(index_html), index_html_len, "text/html; charset=utf-8");
                return false;
            });
        }
    }

    // register API routes
    svr->Get ("/health",              handle_health); // public endpoint (no API key check)
    svr->Get ("/metrics",             handle_metrics);
    svr->Get ("/props",               handle_props);
    svr->Post("/props",               handle_props_change);
    svr->Get ("/models",              handle_models); // public endpoint (no API key check)
    svr->Get ("/v1/models",           handle_models); // public endpoint (no API key check)
    svr->Post("/completion",          handle_completions); // legacy
    svr->Post("/completions",         handle_completions);
    svr->Post("/v1/completions",      handle_completions);
    svr->Post("/chat/completions",    handle_chat_completions);
    svr->Post("/v1/chat/completions", handle_chat_completions);
    svr->Post("/infill",              handle_infill);
    svr->Post("/embedding",           handle_embeddings); // legacy
    svr->Post("/embeddings",          handle_embeddings);
    svr->Post("/v1/embeddings",       handle_embeddings);
    svr->Post("/rerank",              handle_rerank);
    svr->Post("/reranking",           handle_rerank);
    svr->Post("/v1/rerank",           handle_rerank);
    svr->Post("/v1/reranking",        handle_rerank);
    svr->Post("/tokenize",            handle_tokenize);
    svr->Post("/detokenize",          handle_detokenize);
    // LoRA adapters hotswap
    svr->Get ("/lora-adapters",       handle_lora_adapters_list);
    svr->Post("/lora-adapters",       handle_lora_adapters_apply);
    // Save & load slots
    svr->Get ("/slots",               handle_slots);
    svr->Post("/slots/:id_slot",      handle_slots_action);

    //
    // Start the server
    //
    if (params.n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        params.n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    log_data["n_threads_http"] =  std::to_string(params.n_threads_http);
    svr->new_task_queue = [&params] { return new httplib::ThreadPool(params.n_threads_http); };

    // clean up function, to be called before exit
    auto clean_up = [&svr]() {
        svr->stop();
        llama_backend_free();
    };

    // bind HTTP listen port
    bool was_bound = false;
    if (params.port == 0) {
        int bound_port = svr->bind_to_any_port(params.hostname);
        if ((was_bound = (bound_port >= 0))) {
            params.port = bound_port;
        }
    } else {
        was_bound = svr->bind_to_port(params.hostname, params.port);
    }

    if (!was_bound) {
        //LOG_ERROR("couldn't bind HTTP server socket", {
        //    {"hostname", params.hostname},
        //    {"port", params.port},
        //});
        LOG_ERR("%s: couldn't bind HTTP server socket, hostname: %s, port: %d\n", __func__, params.hostname.c_str(), params.port);
        clean_up();
        return 1;
    }

    // run the HTTP server in a thread
    std::thread t([&]() { svr->listen_after_bind(); });
    svr->wait_until_ready();

    LOG_INF("%s: HTTP server is listening, hostname: %s, port: %d, http threads: %d\n", __func__, params.hostname.c_str(), params.port, params.n_threads_http);

    // load the model
    LOG_INF("%s: loading model\n", __func__);

    if (!ctx_server.load_model(params)) {
        clean_up();
        t.join();
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return 1;
    }

    ctx_server.init();
    state.store(SERVER_STATE_READY);

    LOG_INF("%s: model loaded\n", __func__);

    // if a custom chat template is not supplied, we will use the one that comes with the model (if any)
    if (params.chat_template.empty()) {
        if (!ctx_server.validate_model_chat_template()) {
            LOG_WRN("%s: The chat template that comes with this model is not yet supported, falling back to chatml. This may cause the model to output suboptimal responses\n", __func__);
            params.chat_template = "chatml";
        }
    }

    // print sample chat example to make it clear which template is used
    LOG_INF("%s: chat template, built_in: %d, chat_example: '%s'\n", __func__, params.chat_template.empty(), common_chat_format_example(ctx_server.model, params.chat_template).c_str());

    ctx_server.queue_tasks.on_new_task(std::bind(
                &server_context::process_single_task, &ctx_server, std::placeholders::_1));

    ctx_server.queue_tasks.on_update_slots(std::bind(
                &server_context::update_slots, &ctx_server));

    shutdown_handler = [&](int) {
        ctx_server.queue_tasks.terminate();
    };

    

    LOG_INF("%s: add signal_handler\n", __func__);

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    LOG_INF("%s: server is listening on http://%s:%d - starting the main loop\n", __func__, params.hostname.c_str(), params.port);

    //ctx_server.queue_tasks.start_loop();
    std::thread ctx_t([&]() { ctx_server.queue_tasks.start_loop(); });

    std::string req_body = "{\"messages\":[{\"content\":\"现在假设你是一名会议记录人员，需要你对一段会议记录的每一句话依次进行文本规整化，并直接返回规整化后的文本，回答的文本中不要增加任何额外内容。要求：1、规整化后的文本要忠于原始会议记录，保留会议记录中全部的有用信息；2、去除原文中明显的无意义表达；3、如果原文中存在明显的错别字，也需要进行改正。4、不要删除任何有用的信息。5、请用中文进行回答。6、不允许在回答中添加编造成分。7、如果原始会议记录中存在明显重复的表达，需用清晰通顺的方式改写。8、需注意原文中的数值不得擅自修改。原始会议记录为：“哦，喂啊。啊，你讲啊啊啊。”\",\"role\":\"user\"}],\"stream\":false,\"model\":\"Qwen2.5-32B-Instruct\"}";
    //std::string req_body = "{\"messages\":[{\"content\":\"现在假设你是一名会议记录人员，需要你对一段会议记录的每一句话依次进行文本规整化，并直接返回规整化后的文本，回答的文本中不要增加任何额外内容。要求：1、规整化后的文本要忠于原始会议记录，保留会议记录中全部的有用信息；2、去除原文中明显的无意义表达；3、如果原文中存在明显的错别字，也需要进行改正。4、不要删除任何有用的信息。5、请用中文进行回答。6、不允许在回答中添加编造成分。7、如果原始会议记录中存在明显重复的表达，需用清晰通顺的方式改写。8、需注意原文中的数值不得擅自修改。原始会议记录为：“哦，喂啊。啊，你讲啊啊啊。”\",\"role\":\"user\"}],\"stream\":true,\"model\":\"Qwen2.5-32B-Instruct\"}";
    json data = oaicompat_completion_params_parse(ctx_server.model, json::parse(req_body), params.chat_template);
    int ret = 0;
    const std::string str =
        data.dump(-1, ' ', false, json::error_handler_t::replace) +
            "\n\n";
    LOG_INF("data: %s\n", str.c_str());
    //handle_completions_generate(
    //    SERVER_TASK_TYPE_COMPLETION,
    //    data,
    //    /* oaicompat */ true,
    //    /* oaicompat_chat */ true);
//
    //handle_completions_generate_v2(
    //    SERVER_TASK_TYPE_COMPLETION,
    //    data,
    //    /* oaicompat */ true,
    //    /* oaicompat_chat */ true);
//
    //std::vector<server_task_result_ptr> results;
    //int ret = ctx_server.completion_generate(data, results);
    //if (ret == 0) {
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
    //ret = ctx_server.completion_stream_generate(data);
    //if (ret == 0) {
    //    while (true) {
    //        server_task_result_ptr result;
    //        int ret = ctx_server.completion_stream_generate_next_token(result);
    //        if (ret == 1) {
    //            SRV_INF("stop found: %s\n", dump_result(result->to_json()).c_str());
    //            break;
    //        } else {
    //            SRV_INF("content: %s\n", dump_result(result->to_json()).c_str());
    //        }
    //    }
    //} else {
    //    LOG_ERR("Error: %d\n", ret);
    //}

    LOG_INF("Start str generate\n");

    std::string result_json_str;
    ret = ctx_server.completion_generate_str(data, result_json_str);
    if (ret == 0) {
        SRV_INF("completion_generate_str content: %s\n", result_json_str.c_str());
    } else {
        LOG_ERR("Error: %d\n", ret);
    }
    

    ret = ctx_server.completion_stream_generate(data);
    if (ret == 0) {
        while (true) {
            std::string result_json_str;
            ret = ctx_server.completion_stream_generate_next_token_str(result_json_str);
            if (ret == 1) {
                SRV_INF("stop found str: %s\n", result_json_str.c_str());
                break;
            } else if (ret == -1) {
                LOG_ERR("Error: %d\n", ret);
            } else {
                SRV_INF("completion_stream_generate_next_token_str str: %s\n", result_json_str.c_str());
            }
        }
    } else {
        LOG_ERR("Error: %d\n", ret);
    }
    

    LOG_INF("%s: thead join\n", __func__);
    t.join();
    ctx_t.join();
    clean_up();

    return 0;
}

#ifndef HLIE_COMMON_DEFINE_H
#define HLIE_COMMON_DEFINE_H

#include "utils.hpp"

enum HttpType {
    GET,
    POST
};

enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
    ERROR_TYPE_EXCEED_CONTEXT_SIZE, // custom error
};

enum model_scale_type {
    MODEL_1DOT5B = 1,
    MODEL_3B,
    MODEL_7B,
    MODEL_8B,
    MODEL_14B,
    MODEL_32B,
    MODEL_72B,
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

enum server_state {
    SERVER_STATE_LOADING_MODEL, // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY, // Server is ready and model is loaded
};


enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

enum oaicompat_type {
    OAICOMPAT_TYPE_NONE,
    OAICOMPAT_TYPE_CHAT,
    OAICOMPAT_TYPE_COMPLETION,
    OAICOMPAT_TYPE_EMBEDDING,
};

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_STARTED, // TODO: this state is only used for setting up the
                        // initial prompt processing; maybe merge it with
                        // launch_slot_with_task in the future
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};


constexpr auto MIMETYPE_JSON = "application/json; charset=utf-8";
constexpr int HTTP_POLLING_SECONDS = 1;

inline void res_error(httplib::Response &res, const json &error_data) {
    const json final_response{{"error", error_data}};
    res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
    res.status = json_value(error_data, "code", 500);
}

inline void res_ok(httplib::Response &res, const json &data) {
    res.set_content(safe_json_to_str(data), MIMETYPE_JSON);
    res.status = 200;
}

inline json format_error_response(const std::string& message, const error_type type, const int ret = -1) {
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
    case ERROR_TYPE_EXCEED_CONTEXT_SIZE:
        type_str = "exceed_context_size_error";
        code = 400;
        break;
    }
    return json{
        {"code", code},
        {"message", message},
        {"type", type_str},
        {"ret", ret},
    };
}


#endif // HLIE_COMMON_DEFINE_H

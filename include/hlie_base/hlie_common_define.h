#ifndef HLIE_COMMON_DEFINE_H
#define HLIE_COMMON_DEFINE_H

#include <server/server-common.h>
#include <server/server-http.h>

enum HttpType {
    GET,
    POST
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

enum server_state {
    SERVER_STATE_LOADING_MODEL, // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY, // Server is ready and model is loaded
};


// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};


constexpr auto MIMETYPE_JSON = "application/json; charset=utf-8";
constexpr int HTTP_POLLING_SECONDS = 1;

inline json format_error_response_ex(const std::string& message, const error_type type, const int ret = -1) {
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

// generator-like API for HTTP response generation
struct server_res_generator_light : server_http_res {
    server_res_generator_light() = default;
    json response_data_saved;
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }

    void save(const json & response_data) {
        response_data_saved = response_data;
    }

    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

constexpr uint16_t FP16_ONE     = 1; // 1.0
constexpr uint16_t FP16_NEG_INF = 0xFC00; // -65504


#endif // HLIE_COMMON_DEFINE_H

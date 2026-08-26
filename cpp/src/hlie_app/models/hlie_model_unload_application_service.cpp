#include "hlie_app/models/hlie_model_unload_application_service.h"

#include <utility>

#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

namespace {

constexpr const char* LLM_MODEL_TYPE = "llm";
constexpr const char* DOCUMENT_MODEL_TYPE = "document";
constexpr const char* TTS_MODEL_TYPE = "tts";
constexpr const char* COSYVOICE_MODEL_TYPE = "cosyvoice";

server_http_res_ptr json_response(json payload, const int status = 200) {
    auto response = std::make_unique<server_http_res>();
    response->status = status;
    response->content_type = MIMETYPE_JSON;
    response->data = payload.dump();
    return response;
}

server_http_res_ptr error_response(const json& error) {
    return json_response({{"error", error}}, error.value("code", 500));
}

json legacy_error_response(const std::string& message,
                           const error_type type) {
    auto error = format_error_response_ex(message, type);
    error.erase("ret");
    return error;
}

std::string json_string_value(const json& body, const char* key) {
    if (!body.contains(key) || body.at(key).is_null()) {
        return {};
    }
    if (body.at(key).is_string()) {
        return body.at(key).get<std::string>();
    }
    return body.at(key).dump();
}

} // namespace

HLIEModelUnloadApplicationService::HLIEModelUnloadApplicationService(
    HLIEModelUnloadApplicationDependencies dependencies)
    : dependencies_(std::move(dependencies)) {}

server_http_res_ptr HLIEModelUnloadApplicationService::handle(
    const server_http_req& request) const {
    if (dependencies_.replicas_mode_provider &&
        dependencies_.replicas_mode_provider()) {
        return error_response(format_error_response_ex(
            "In the cluster or multi-instance mode, external calls to start model instances are not allowed.",
            ERROR_TYPE_NOT_SUPPORTED, 80005));
    }

    const auto body = json::parse(request.body);
    std::string requested_model = json_string_value(body, "model_id");
    if (requested_model.empty()) {
        requested_model = json_string_value(body, "model");
    }
    if (!requested_model.empty()) {
        const auto model_key = dependencies_.resolve_model_key
            ? dependencies_.resolve_model_key(requested_model)
            : std::nullopt;
        if (!model_key) {
            return error_response(legacy_error_response(
                "model not found", ERROR_TYPE_NOT_FOUND));
        }
        if (!dependencies_.evict_model ||
            !dependencies_.evict_model(*model_key, "api_unload")) {
            LOG_WARN("Model unload refused or not matched: requested={}, key={}",
                     json(requested_model).dump(), json(*model_key).dump());
            return error_response(format_error_response_ex(
                "model is busy or not loaded", ERROR_TYPE_UNAVAILABLE, 80004));
        }
        if (dependencies_.publish_unloaded) {
            dependencies_.publish_unloaded(*model_key);
        }
        LOG_DEBUG("Model unload HTTP command completed: requested={}, key={}",
                  json(requested_model).dump(), json(*model_key).dump());
        return json_response({{"code", 0}, {"model", *model_key}});
    }

    std::string model_type = body.value(
        "model_type", std::string{LLM_MODEL_TYPE});
    if (model_type == DOCUMENT_MODEL_TYPE) {
        model_type = LLM_MODEL_TYPE;
    }
    if (model_type == TTS_MODEL_TYPE) {
        model_type = COSYVOICE_MODEL_TYPE;
    }
    if (!dependencies_.is_supported_model_type ||
        !dependencies_.is_supported_model_type(model_type)) {
        return error_response(format_error_response_ex(
            "Invalid model type.", ERROR_TYPE_INVALID_REQUEST, 80006));
    }

    const auto model_key = dependencies_.find_loaded_model_by_type
        ? dependencies_.find_loaded_model_by_type(model_type)
        : std::nullopt;
    if (!model_key) {
        LOG_WARN("Model unload requested for a type with no loaded service: type={}",
                 model_type);
        return error_response(format_error_response_ex(
            "model is busy or not loaded", ERROR_TYPE_UNAVAILABLE, 80004));
    }
    if (!dependencies_.evict_model ||
        !dependencies_.evict_model(*model_key, "api_unload_by_type")) {
        LOG_WARN("Model unload refused because model is busy or unmatched: type={}, model={}",
                 model_type, json(*model_key).dump());
        return error_response(format_error_response_ex(
            "model is busy or not loaded", ERROR_TYPE_UNAVAILABLE, 80004));
    }
    if (dependencies_.publish_unloaded) {
        dependencies_.publish_unloaded(*model_key);
    }
    LOG_DEBUG("Model unload HTTP command completed: type={}, key={}",
              model_type, json(*model_key).dump());
    return json_response({{"code", 0}});
}

HLIE_NAMESPACE_END

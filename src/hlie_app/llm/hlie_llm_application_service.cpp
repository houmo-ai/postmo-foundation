#include "hlie_app/llm/hlie_llm_application_service.h"

#include <algorithm>
#include <cctype>
#include <utility>

#include "hlie_app/inference/hlie_admission_guarded_response.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_exception.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

namespace {

constexpr auto MODEL_LOAD_TIMEOUT = std::chrono::minutes(5);

server_http_res_ptr json_response(json payload, const int status = 200) {
    auto response = std::make_unique<server_http_res>();
    response->status = status;
    response->content_type = MIMETYPE_JSON;
    response->data = payload.dump();
    return response;
}

json legacy_error(const std::string& message, const error_type type) {
    auto error = format_error_response_ex(message, type);
    error.erase("ret");
    return error;
}

server_http_res_ptr error_response(json error) {
    return json_response({{"error", error}}, error.value("code", 500));
}

server_http_res_ptr raw_error_response(const std::string& message) {
    return json_response({{"error", message}}, 500);
}

std::string normalize_model_name(const std::string& name) {
    const auto first = std::find_if_not(
        name.begin(), name.end(), [](const unsigned char ch) {
            return std::isspace(ch);
        });
    const auto last = std::find_if_not(
        name.rbegin(), name.rend(), [](const unsigned char ch) {
            return std::isspace(ch);
        }).base();
    if (first >= last) {
        return {};
    }
    std::string normalized(first, last);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](const unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return normalized;
}

} // namespace

HLIELlmApplicationService::HLIELlmApplicationService(
    HLIEAdmissionController& admission,
    HLIELlmApplicationDependencies dependencies)
    : admission_(admission), dependencies_(std::move(dependencies)) {}

server_http_res_ptr HLIELlmApplicationService::handle(
    const server_http_req& request,
    const HLIEInferenceOperation operation,
    const std::string& method) const {
    auto permit = admission_.try_acquire();
    if (!permit) {
        throw InferenceQueueFullException("LLM admission limit reached");
    }
    auto response = route(request, operation, method);
    return attach_admission_permit(std::move(response), permit);
}

server_http_res_ptr HLIELlmApplicationService::handle_admitted(
    const server_http_req& request,
    const HLIEInferenceOperation operation,
    const std::string& method) const {
    return route(request, operation, method);
}

server_http_res_ptr HLIELlmApplicationService::handle_properties(
    const server_http_req& request) const {
    if (!request.get_param("model").empty()) {
        return handle(request, HLIEInferenceOperation::Properties, "GET");
    }
    const auto model_key = dependencies_.find_ready_embedded_llm
        ? dependencies_.find_ready_embedded_llm() : std::nullopt;
    if (model_key) {
        return dispatch(*model_key, HLIEInferenceOperation::Properties,
                        request, "GET");
    }
    return json_response({
        {"role", "router"},
        {"max_instances", 4},
        {"model_alias", "llama-server"},
        {"model_path", "none"},
        {"default_generation_settings", {
            {"params", json{}}, {"n_ctx", 0}}},
    });
}

server_http_res_ptr HLIELlmApplicationService::handle_slots(
    const server_http_req& request) const {
    const auto model_key = dependencies_.find_ready_embedded_llm
        ? dependencies_.find_ready_embedded_llm() : std::nullopt;
    if (!model_key) {
        return error_response(legacy_error(
            "Model is not available for now.", ERROR_TYPE_UNAVAILABLE));
    }
    return dispatch(*model_key, HLIEInferenceOperation::Slots,
                    request, "GET");
}

server_http_res_ptr HLIELlmApplicationService::dispatch(
    const std::string& model_key,
    const HLIEInferenceOperation operation,
    const server_http_req& request,
    const std::string& method) const {
    if (!dependencies_.dispatch) {
        return error_response(format_error_response_ex(
            "inference dispatcher is unavailable",
            ERROR_TYPE_UNAVAILABLE, 80004));
    }
    return dependencies_.dispatch(model_key, operation, request, method);
}

server_http_res_ptr HLIELlmApplicationService::dispatch_document_or_llm(
    const std::string& model_key,
    const HLIEInferenceOperation operation,
    const server_http_req& request,
    const nlohmann::json& body,
    const std::string& method) const {
    if (dependencies_.dispatch_document) {
        return dependencies_.dispatch_document(
            model_key, operation, request, body, method);
    }
    return dispatch(model_key, operation, request, method);
}

server_http_res_ptr HLIELlmApplicationService::route(
    const server_http_req& request,
    const HLIEInferenceOperation operation,
    const std::string& method) const {
    if (dependencies_.draining_provider &&
        dependencies_.draining_provider()) {
        return error_response(format_error_response_ex(
            "service is draining", ERROR_TYPE_UNAVAILABLE, 503));
    }

    nlohmann::json body = nlohmann::json::object();
    std::string name = request.get_param("model");
    if (!request.body.empty()) {
        body = nlohmann::json::parse(request.body);
        if (name.empty()) {
            name = body.value("model", std::string{});
        }
    }
    LOG_DEBUG("Routing LLM request: path={}, model={}, model_length={}, fields={}, files={}",
              request.path, json(name).dump(), name.size(),
              body.is_object() ? body.size() : 0, request.files.size());
    if (name.empty()) {
        return error_response(legacy_error(
            "model name is missing from the request",
            ERROR_TYPE_INVALID_REQUEST));
    }

    auto instance = dependencies_.resolve_model
        ? dependencies_.resolve_model(name) : std::nullopt;
    if (!instance) {
        if (dependencies_.refresh_discovery) {
            dependencies_.refresh_discovery();
        }
        instance = dependencies_.resolve_model
            ? dependencies_.resolve_model(name) : std::nullopt;
        if (!instance) {
            LOG_WARN("LLM request model lookup failed after discovery: path={}, model={}, normalized_model={}",
                     request.path, json(name).dump(),
                     json(normalize_model_name(name)).dump());
            return error_response(legacy_error(
                "model not found", ERROR_TYPE_INVALID_REQUEST));
        }
    }

    const int local_port = dependencies_.local_port_provider
        ? dependencies_.local_port_provider() : 0;
    const int master_port = dependencies_.master_port_provider
        ? dependencies_.master_port_provider() : 7901;
    LOG_DEBUG("LLM request route: model={}, status={}, port={}",
              json(name).dump(), static_cast<int>(instance->state),
              instance->port);
    if (instance->state == HLIELlmModelState::Loaded &&
        (instance->port == master_port || master_port != local_port)) {
        return dispatch_document_or_llm(
            instance->key, operation, request, body, method);
    }

    const bool started_unloaded =
        instance->state == HLIELlmModelState::Unloaded;
    std::optional<HLIELlmAutoLoadRequest> load_request;
    if (started_unloaded) {
        const bool auto_load = dependencies_.auto_load_provider &&
            dependencies_.auto_load_provider();
        if (!auto_load) {
            return error_response(legacy_error(
                "model not loaded", ERROR_TYPE_INVALID_REQUEST));
        }
        if (method != "POST") {
            return error_response(legacy_error(
                "model is not loaded; load it before using this GET endpoint",
                ERROR_TYPE_INVALID_REQUEST));
        }

        HLIELlmAutoLoadRequest candidate;
        candidate.model = name;
        candidate.model_path = body.value("model_path", std::string{});
        const std::string devices = body.value("devices", "undefined");
        if (devices != "undefined") {
            const auto parsed = dependencies_.parse_devices
                ? dependencies_.parse_devices(devices) : std::nullopt;
            if (!parsed) {
                return error_response(format_error_response_ex(
                    "The devices parsed failed.",
                    ERROR_TYPE_UNAVAILABLE, 80004));
            }
            candidate.requested_device_ids = *parsed;
        }
        if (body.contains("params") && body.at("params").is_object()) {
            candidate.mmproj_path = body.at("params").value(
                "mm_proj_path", std::string{});
        }
        load_request = std::move(candidate);
    }

    if (started_unloaded ||
        instance->state == HLIELlmModelState::Loading) {
        if (!dependencies_.ensure_ready) {
            return error_response(format_error_response_ex(
                "model readiness service is unavailable",
                ERROR_TYPE_SERVER, 80001));
        }
        const auto readiness = dependencies_.ensure_ready(
            instance->key, std::move(load_request),
            std::chrono::steady_clock::now() + MODEL_LOAD_TIMEOUT,
            [&request] { return request.should_stop(); });
        switch (readiness) {
        case HLIELlmReadinessOutcome::Ready:
            break;
        case HLIELlmReadinessOutcome::Missing:
            return error_response(legacy_error(
                started_unloaded
                    ? "model disappeared before automatic load"
                    : "model not found while loading",
                ERROR_TYPE_NOT_FOUND));
        case HLIELlmReadinessOutcome::Unavailable:
            return error_response(format_error_response_ex(
                started_unloaded
                    ? "model runtime is unavailable for automatic load"
                    : "model stopped becoming ready",
                ERROR_TYPE_UNAVAILABLE, 80004));
        case HLIELlmReadinessOutcome::InvalidLoadRequest:
            return error_response(format_error_response_ex(
                "model load request is unavailable",
                ERROR_TYPE_UNAVAILABLE, 80004));
        case HLIELlmReadinessOutcome::ScheduleFailed:
            return error_response(format_error_response_ex(
                "failed to schedule model loading",
                ERROR_TYPE_UNAVAILABLE, 80004));
        case HLIELlmReadinessOutcome::TimedOut:
            LOG_WARN("Timed out waiting for model load: model={}, timeout_sec={}",
                     json(name).dump(),
                     std::chrono::duration_cast<std::chrono::seconds>(
                         MODEL_LOAD_TIMEOUT).count());
            return error_response(format_error_response_ex(
                "timed out waiting for model to load",
                ERROR_TYPE_UNAVAILABLE, 80004));
        case HLIELlmReadinessOutcome::Cancelled:
            if (dependencies_.draining_provider &&
                dependencies_.draining_provider()) {
                return error_response(format_error_response_ex(
                    "service started draining while waiting for model load",
                    ERROR_TYPE_UNAVAILABLE, 503));
            }
            LOG_INFO("Stopped waiting for model load after client cancellation: model={}",
                     json(name).dump());
            {
                auto error = legacy_error(
                    "request cancelled while waiting for model to load",
                    ERROR_TYPE_UNAVAILABLE);
                error["code"] = 499;
                return error_response(std::move(error));
            }
        case HLIELlmReadinessOutcome::Orphaned:
            return error_response(format_error_response_ex(
                "model runtime ownership is uncertain",
                ERROR_TYPE_UNAVAILABLE, 80004));
        case HLIELlmReadinessOutcome::Failed:
        case HLIELlmReadinessOutcome::Unloaded:
            return raw_error_response("failed to load model: " + name);
        }
    }

    const auto loaded = dependencies_.resolve_model
        ? dependencies_.resolve_model(name) : std::nullopt;
    if (!loaded) {
        return error_response(legacy_error(
            "model not found after load", ERROR_TYPE_NOT_FOUND));
    }
    if (loaded->state == HLIELlmModelState::Unloaded) {
        return raw_error_response("failed to load model: " + name);
    }
    if (loaded->port == local_port) {
        return dispatch_document_or_llm(
            loaded->key, operation, request, body, method);
    }
    return dispatch(loaded->key, operation, request, method);
}

HLIE_NAMESPACE_END

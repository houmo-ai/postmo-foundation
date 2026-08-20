#include "hlie_app/speech/hlie_speech_application_service.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <utility>

#include "hlie_base/hlie_logger.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

namespace {

using json = nlohmann::ordered_json;

constexpr const char* COSYVOICE_KIND = "cosyvoice";
constexpr const char* TTS_KIND = "tts";
constexpr const char* QWEN3_TTS_KIND = "qwen3_tts";

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

void set_error(httplib::Response& response, const std::string& message,
               const char* type, const int status,
               const std::optional<int> ret = 80004) {
    json error{
        {"code", status},
        {"message", message},
        {"type", type},
    };
    if (ret) {
        error["ret"] = *ret;
    }
    response.status = status;
    response.set_content(json{{"error", error}}.dump(),
                         "application/json; charset=utf-8");
}

bool is_speech_kind(const std::string& service_kind) {
    return service_kind == COSYVOICE_KIND || service_kind == TTS_KIND ||
        service_kind == QWEN3_TTS_KIND;
}

} // namespace

HLIESpeechApplicationService::HLIESpeechApplicationService(
    HLIEAdmissionController& qwen_admission,
    HLIESpeechApplicationDependencies dependencies)
    : qwen_admission_(qwen_admission),
      dependencies_(std::move(dependencies)) {}

void HLIESpeechApplicationService::handle(
    const httplib::Request& request, httplib::Response& response) const {
    const auto body = json::parse(request.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        set_error(response, "request body must be a JSON object",
                  "invalid_request_error", 400);
        return;
    }
    if (body.contains("model") && !body.at("model").is_null() &&
        !body.at("model").is_string()) {
        set_error(response, "model must be a string",
                  "invalid_request_error", 400);
        return;
    }
    const std::string requested_model =
        body.contains("model") && body.at("model").is_string()
        ? body.at("model").get<std::string>() : std::string{};

    std::optional<HLIESpeechModelSnapshot> instance;
    if (!requested_model.empty() && dependencies_.resolve_model) {
        instance = dependencies_.resolve_model(requested_model);
        if (instance && instance->state == HLIESpeechModelState::Loading) {
            const auto readiness = dependencies_.wait_until_ready
                ? dependencies_.wait_until_ready(
                      instance->key, request.is_connection_closed)
                : HLIESpeechReadinessStatus::Unavailable;
            if (readiness != HLIESpeechReadinessStatus::Ready) {
                switch (readiness) {
                case HLIESpeechReadinessStatus::Cancelled:
                    set_error(
                        response,
                        "request cancelled while waiting for speech model to load",
                        "unavailable_error", 499, std::nullopt);
                    break;
                case HLIESpeechReadinessStatus::TimedOut:
                    set_error(response,
                              "timed out waiting for speech model to load",
                              "unavailable_error", 503);
                    break;
                case HLIESpeechReadinessStatus::Missing:
                    set_error(response, "speech model not found while loading",
                              "not_found_error", 404);
                    break;
                case HLIESpeechReadinessStatus::Orphaned:
                    set_error(response,
                              "speech model runtime ownership is uncertain",
                              "unavailable_error", 503);
                    break;
                case HLIESpeechReadinessStatus::Unavailable:
                case HLIESpeechReadinessStatus::Ready:
                    set_error(response, "failed to load speech model",
                              "unavailable_error", 503);
                    break;
                }
                return;
            }
            instance = dependencies_.resolve_model(requested_model);
        }
    }

    std::string model_key;
    std::string service_kind;
    if (instance) {
        service_kind = instance->service_kind;
        if (!is_speech_kind(service_kind) ||
            instance->state != HLIESpeechModelState::Loaded) {
            set_error(response, "requested speech model is not loaded",
                      "invalid_request_error", 400);
            return;
        }
        model_key = instance->key;
    } else if (!requested_model.empty()) {
        const auto normalized = normalize_model_name(requested_model);
        if ((normalized == COSYVOICE_KIND || normalized == TTS_KIND) &&
            dependencies_.find_ready_model) {
            const auto cosy = dependencies_.find_ready_model(
                COSYVOICE_KIND, HLIERuntimeKind::Embedded);
            if (cosy) {
                model_key = *cosy;
                service_kind = COSYVOICE_KIND;
            }
        } else if ((normalized == QWEN3_TTS_KIND ||
                    normalized == "qwen3-tts") &&
                   dependencies_.find_ready_model) {
            const auto qwen = dependencies_.find_ready_model(
                QWEN3_TTS_KIND, HLIERuntimeKind::Embedded);
            if (qwen) {
                model_key = *qwen;
                service_kind = QWEN3_TTS_KIND;
            }
        }
        if (model_key.empty()) {
            set_error(response, "requested speech model is not loaded",
                      "invalid_request_error", 400);
            return;
        }
    } else {
        const auto cosy = dependencies_.find_ready_model
            ? dependencies_.find_ready_model(
                  COSYVOICE_KIND, HLIERuntimeKind::Embedded)
            : std::nullopt;
        if (cosy) {
            model_key = *cosy;
            service_kind = COSYVOICE_KIND;
        } else {
            const auto qwen = dependencies_.find_ready_model
                ? dependencies_.find_ready_model(
                      QWEN3_TTS_KIND, HLIERuntimeKind::Embedded)
                : std::nullopt;
            if (qwen) {
                set_error(response, "model is required when using Qwen3-TTS",
                          "invalid_request_error", 400);
            } else {
                set_error(response, "No speech model is available",
                          "unavailable_error", 503);
            }
            return;
        }
    }

    if (!dependencies_.acquire_target) {
        set_error(response, "inference dispatcher is unavailable",
                  "unavailable_error", 503);
        return;
    }
    auto acquired = dependencies_.acquire_target(model_key);
    if (!acquired.target) {
        set_error(response,
                  acquired.detail.empty() ? "speech model is unavailable"
                                          : acquired.detail,
                  "unavailable_error", 503);
        return;
    }

    auto target = std::move(*acquired.target);
    std::shared_ptr<void> request_guard = std::move(target.lifetime_guard);
    if (service_kind == QWEN3_TTS_KIND &&
        target.identity.kind == HLIERuntimeKind::Embedded) {
        auto permit = qwen_admission_.try_acquire();
        if (!permit) {
            set_error(response, "Qwen3-TTS admission limit reached",
                      "rate_limit_error", 429, 80008);
            return;
        }
        auto permit_guard =
            std::make_shared<HLIEAdmissionController::Permit>(
                std::move(*permit));
        request_guard = std::make_shared<std::pair<
            std::shared_ptr<void>,
            std::shared_ptr<HLIEAdmissionController::Permit>>>(
                std::move(request_guard), std::move(permit_guard));
    }

    const char* backend_name = service_kind == QWEN3_TTS_KIND
        ? QWEN3_TTS_KIND : COSYVOICE_KIND;
    response.set_header("X-HLIE-TTS-Backend", backend_name);
    LOG_DEBUG("Speech request route: backend={}, model={}, generation={}, runtime_kind={}",
              backend_name, json(model_key).dump(),
              target.identity.launch_ticket.generation,
              static_cast<int>(target.identity.kind));
    try {
        const bool dispatched = target.dispatch &&
            target.dispatch(request, response, std::move(request_guard));
        if (dependencies_.observe_dispatch) {
            dependencies_.observe_dispatch(dispatched);
        }
        if (!dispatched) {
            set_error(response,
                      "speech endpoint does not support this request",
                      "not_supported_error", 501);
        }
    } catch (const std::exception& error) {
        if (dependencies_.observe_dispatch) {
            dependencies_.observe_dispatch(false);
        }
        LOG_ERROR("Speech endpoint failed: model={}, generation={}, error={}",
                  json(model_key).dump(),
                  target.identity.launch_ticket.generation,
                  json(error.what()).dump());
        set_error(response, "failed to connect to speech model instance",
                  "unavailable_error", 503);
    }
}

HLIE_NAMESPACE_END

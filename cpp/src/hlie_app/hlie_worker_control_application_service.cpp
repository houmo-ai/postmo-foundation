#include "hlie_app/control/hlie_worker_control_application_service.h"

#include <stdexcept>
#include <utility>

#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

namespace {

template <typename T>
T compatibility_value(const json& body, const char* key,
                      const T& fallback) {
    if (!body.contains(key) || body.at(key).is_null()) {
        return fallback;
    }
    try {
        return body.at(key).get<T>();
    } catch (const nlohmann::json::type_error&) {
        LOG_WARN("Worker control request field has the wrong type: field={}",
                 key);
        return fallback;
    }
}

server_http_res_ptr json_response(json payload, const int status = 200) {
    auto response = std::make_unique<server_http_res>();
    response->status = status;
    response->content_type = MIMETYPE_JSON;
    response->data = payload.dump();
    return response;
}

server_http_res_ptr error_response(const std::string& message,
                                   const error_type type,
                                   const int ret) {
    auto error = format_error_response_ex(message, type, ret);
    return json_response({{"error", error}}, error.value("code", 500));
}

HLIEWorkerModelStatus worker_status_from_string(const std::string& value) {
    if (value == "unloaded") {
        return HLIEWorkerModelStatus::Unloaded;
    }
    if (value == "loading") {
        return HLIEWorkerModelStatus::Loading;
    }
    if (value == "loaded") {
        return HLIEWorkerModelStatus::Loaded;
    }
    if (value == "sleeping") {
        return HLIEWorkerModelStatus::Sleeping;
    }
    throw std::runtime_error("invalid server model status");
}

} // namespace

HLIEWorkerControlApplicationService::HLIEWorkerControlApplicationService(
    HLIEWorkerControlService& worker_control,
    HLIEWorkerControlApplicationDependencies dependencies)
    : worker_control_(worker_control),
      dependencies_(std::move(dependencies)) {}

server_http_res_ptr
HLIEWorkerControlApplicationService::handle_get_status() const {
    const auto snapshot = worker_control_.snapshot();
    json processors = json::array();
    processors.push_back({
        {"params", ""},
        {"instance_id", 0},
        {"port", snapshot.local_port},
        {"uuid", ""},
        {"process_id", 0},
        {"generation", 0},
        {"start_time", snapshot.local_start_time},
        {"models", snapshot.local_models},
    });
    for (const auto& worker : snapshot.workers) {
        processors.push_back({
            {"params", worker.params},
            {"instance_id", worker.instance_id},
            {"port", worker.port},
            // Preserve the legacy DTO's instance-id uuid field.
            {"uuid", worker.instance_id},
            {"process_id", worker.process_id},
            {"generation", worker.generation},
            {"start_time", worker.start_time},
            {"models", worker.models},
        });
    }
    return json_response({{"code", 0},
                          {"processor", std::move(processors)}});
}

server_http_res_ptr
HLIEWorkerControlApplicationService::handle_post_status(
    const server_http_req& request) const {
    const auto body = json::parse(request.body);
    std::string model =
        compatibility_value(body, "model", std::string{});
    const std::string requested_model = model;
    const std::string value =
        compatibility_value(body, "value", std::string{});
    const int port = compatibility_value(body, "port", 0);
    const auto generation = body.value("generation", HLIEModelGeneration{0});
    std::string launch_token = compatibility_value(
        body, "launch_token", std::string{});
    const auto status = worker_status_from_string(value);

    LOG_DEBUG("Received model status update: model={}, model_length={}, value={}, port={}, generation={}",
              json(model).dump(), model.size(), value, port, generation);
    const auto result = worker_control_.apply_status({
        std::move(model), status, port, generation, std::move(launch_token)});
    if (result == HLIEWorkerStatusApplyResult::Applied) {
        LOG_INFO("Model status updated: request_model={}, generation={}, status={}, port={}",
                 json(requested_model).dump(), generation,
                 static_cast<int>(status), port);
    } else if (result == HLIEWorkerStatusApplyResult::LookupFailed) {
        // Preserve the legacy acknowledgement while surfacing the mismatch.
        LOG_WARN("Model status update lookup failed: model={}, value={}, port={}, generation={}",
                 json(requested_model).dump(), value, port, generation);
    } else {
        LOG_DEBUG("Rejected stale model status: generation={}, value={}, port={}",
                  generation, value, port);
        return error_response("stale worker status update",
                              ERROR_TYPE_INVALID_REQUEST, 409);
    }
    return json_response({{"success", true}});
}

server_http_res_ptr HLIEWorkerControlApplicationService::handle_shutdown(
    const server_http_req& request) const {
    const auto body = json::parse(request.body);
    const auto generation =
        body.value("generation", HLIEModelGeneration{0});
    const std::string launch_token = compatibility_value(
        body, "launch_token", std::string{});
    const auto result = worker_control_.request_shutdown(
        generation, launch_token);
    if (result == HLIEWorkerShutdownResult::NotDynamicWorker) {
        return error_response(
            "worker shutdown is only available to dynamic workers",
            ERROR_TYPE_PERMISSION, 80009);
    }
    if (result == HLIEWorkerShutdownResult::CallbackUnavailable) {
        return error_response("worker shutdown callback is unavailable",
                              ERROR_TYPE_SERVER, 80001);
    }
    if (result == HLIEWorkerShutdownResult::InvalidTicket) {
        LOG_WARN("Rejected worker shutdown request: requested_generation={}",
                 generation);
        return error_response("invalid worker shutdown ticket",
                              ERROR_TYPE_PERMISSION, 80009);
    }

    const auto ticket = dependencies_.launch_ticket_provider
        ? dependencies_.launch_ticket_provider()
        : HLIEModelLaunchTicket{};
    const bool already_requested =
        result == HLIEWorkerShutdownResult::AlreadyRequested;
    if (!already_requested) {
        LOG_INFO("Graceful worker shutdown requested: model={}, generation={}",
                 json(ticket.model_key).dump(), ticket.generation);
    }
    return json_response({
        {"success", true},
        {"generation", ticket.generation},
        {"already_requested", already_requested},
    });
}

HLIE_NAMESPACE_END

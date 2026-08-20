#include "hlie_app/models/hlie_stop_service_application_service.h"

#include <unordered_set>
#include <utility>

#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

namespace {

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

server_http_res_ptr worker_stop_error_response() {
    // Preserve the legacy array-shaped error body for this compatibility API.
    const json error = {{"ret", 80001},
                        "Failed to stop process for instance_id"};
    return json_response({{"error", error}}, 500);
}

server_http_res_ptr success_response(const int instance_id,
                                     const int process_id) {
    return json_response({
        {"code", 0},
        {"instance_id", instance_id},
        {"process_id", process_id},
    });
}

std::vector<std::string> unique_model_keys(
    const std::vector<std::string>& model_keys) {
    std::vector<std::string> result;
    std::unordered_set<std::string> seen;
    result.reserve(model_keys.size());
    for (const auto& key : model_keys) {
        if (!key.empty() && seen.insert(key).second) {
            result.push_back(key);
        }
    }
    return result;
}

} // namespace

HLIEStopServiceApplicationService::HLIEStopServiceApplicationService(
    HLIEStopServiceApplicationDependencies dependencies)
    : dependencies_(std::move(dependencies)) {}

server_http_res_ptr HLIEStopServiceApplicationService::handle(
    const server_http_req& request) const {
    if (dependencies_.replicas_mode_provider &&
        dependencies_.replicas_mode_provider()) {
        return error_response(format_error_response_ex(
            "In the cluster or multi-instance mode, external calls to start model instances are not allowed.",
            ERROR_TYPE_NOT_SUPPORTED, 80005));
    }

    const auto body = json::parse(request.body);
    const int instance_id = body.at("instance_id").get<int>();
    const int process_id = body.at("process_id").get<int>();
    if (instance_id == 0) {
        return stop_local_models(instance_id, process_id);
    }
    if (process_id != 0) {
        return stop_worker(instance_id, process_id);
    }
    LOG_DEBUG("Stop service compatibility no-op: instance_id={}, process_id={}",
              instance_id, process_id);
    return success_response(instance_id, process_id);
}

server_http_res_ptr HLIEStopServiceApplicationService::stop_local_models(
    const int instance_id, const int process_id) const {
    const auto local_model_keys = unique_model_keys(
        dependencies_.local_model_keys_provider
            ? dependencies_.local_model_keys_provider()
            : std::vector<std::string>{});
    std::vector<std::string> claimed;
    std::vector<std::string> busy;
    claimed.reserve(local_model_keys.size());

    for (const auto& model_key : local_model_keys) {
        if (dependencies_.try_begin_eviction &&
            dependencies_.try_begin_eviction(model_key)) {
            claimed.push_back(model_key);
        } else {
            busy.push_back(model_key);
        }
    }

    if (!busy.empty()) {
        if (dependencies_.abort_eviction) {
            for (const auto& model_key : claimed) {
                dependencies_.abort_eviction(model_key);
            }
        }
        LOG_WARN("Local stop refused because models are busy: models={}",
                 json(busy).dump());
        return error_response(format_error_response_ex(
            "model is busy or not loaded", ERROR_TYPE_UNAVAILABLE, 80004));
    }

    for (std::size_t index = 0; index < claimed.size(); ++index) {
        if (dependencies_.evict_claimed_model &&
            dependencies_.evict_claimed_model(claimed[index])) {
            continue;
        }
        if (dependencies_.abort_eviction) {
            for (std::size_t remaining = index + 1;
                 remaining < claimed.size(); ++remaining) {
                dependencies_.abort_eviction(claimed[remaining]);
            }
        }
        LOG_ERROR("Local stop failed after eviction claim: model={}",
                  json(claimed[index]).dump());
        return error_response(format_error_response_ex(
            "failed to stop one or more model runtimes",
            ERROR_TYPE_SERVER, 80001));
    }

    LOG_DEBUG("Local stop HTTP command completed: model_count={}",
              claimed.size());
    return success_response(instance_id, process_id);
}

server_http_res_ptr HLIEStopServiceApplicationService::stop_worker(
    const int instance_id, const int process_id) const {
    const auto worker = dependencies_.worker_provider
        ? dependencies_.worker_provider(static_cast<std::uint64_t>(process_id))
        : std::nullopt;
    if (!worker) {
        LOG_DEBUG("Worker stop is already converged: instance_id={}, process_id={}",
                  instance_id, process_id);
        return success_response(instance_id, process_id);
    }

    bool stop_confirmed = false;
    if (worker->launch_ticket.valid() &&
        dependencies_.stop_ticketed_worker) {
        stop_confirmed =
            dependencies_.stop_ticketed_worker(worker->launch_ticket);
    }
    if (!stop_confirmed && dependencies_.stop_worker_fallback) {
        stop_confirmed = dependencies_.stop_worker_fallback(*worker);
    }
    if (!stop_confirmed) {
        LOG_ERROR("Failed to stop Worker process: instance_id={}, process_id={}",
                  instance_id, process_id);
        return worker_stop_error_response();
    }

    if (worker->launch_ticket.valid()) {
        if (dependencies_.publish_confirmed_stop) {
            dependencies_.publish_confirmed_stop(worker->launch_ticket);
        }
    } else if (dependencies_.publish_confirmed_legacy_stop) {
        std::unordered_set<std::string> published;
        for (const auto& model : worker->model_names) {
            const auto model_key = dependencies_.resolve_model_key
                ? dependencies_.resolve_model_key(model)
                : std::nullopt;
            if (model_key && published.insert(*model_key).second) {
                dependencies_.publish_confirmed_legacy_stop(*model_key);
            }
        }
    }

    LOG_DEBUG("Worker stop HTTP command completed: instance_id={}, process_id={}",
              instance_id, process_id);
    return success_response(instance_id, process_id);
}

HLIE_NAMESPACE_END

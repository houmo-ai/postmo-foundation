#include "hlie_app/startup/hlie_bootstrap_application_service.h"

#include <exception>
#include <utility>

#include "hlie_base/hlie_logger.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

namespace {

const std::string* argument(
    const HLIEBootstrapRequest& request, const char* name) {
    const auto it = request.arguments.find(name);
    return it == request.arguments.end() ? nullptr : &it->second;
}

std::optional<HLIEModelLaunchTicket> parse_launch_ticket(
    const HLIEBootstrapRequest& request) {
    const auto* generation = argument(request, "--launch_generation");
    if (!generation) {
        return HLIEModelLaunchTicket{};
    }

    HLIEModelLaunchTicket ticket;
    try {
        ticket.generation = std::stoull(*generation);
    } catch (const std::exception&) {
        ticket.generation = 0;
    }
    if (const auto* token = argument(request, "--launch_token")) {
        ticket.token = *token;
    }
    if (const auto* model_key = argument(request, "--launch_model_key")) {
        ticket.model_key = *model_key;
    }
    if (!ticket.valid()) {
        return std::nullopt;
    }
    return ticket;
}

HLIEBootstrapStatus to_bootstrap_status(
    const HLIEDefaultModelInitializationStatus status) {
    switch (status) {
    case HLIEDefaultModelInitializationStatus::Initialized:
        return HLIEBootstrapStatus::Ready;
    case HLIEDefaultModelInitializationStatus::ZeroLoad:
        return HLIEBootstrapStatus::ZeroLoad;
    case HLIEDefaultModelInitializationStatus::Failed:
        return HLIEBootstrapStatus::Failed;
    }
    return HLIEBootstrapStatus::Failed;
}

} // namespace

HLIEBootstrapApplicationService::HLIEBootstrapApplicationService(
    HLIEBootstrapApplicationDependencies dependencies)
    : dependencies_(std::move(dependencies)) {}

HLIEBootstrapResult HLIEBootstrapApplicationService::initialize(
    const HLIEBootstrapRequest& request) const {
    if (!dependencies_.prepare_launch_configuration ||
        !dependencies_.prepare_launch_configuration()) {
        LOG_ERROR("Failed to prepare bootstrap launch configuration");
        return {};
    }

    const auto launch_ticket = parse_launch_ticket(request);
    if (!launch_ticket) {
        LOG_ERROR("Invalid worker launch ticket arguments");
        return {};
    }
    if (launch_ticket->valid()) {
        LOG_INFO("Worker launch ticket accepted: model={}, generation={}",
                 nlohmann::json(launch_ticket->model_key).dump(),
                 launch_ticket->generation);
    }

    const auto* one_time_manifest = argument(request, "--launch_manifest");
    const auto* legacy_manifest = argument(request, "--json_info");
    const auto* manifest = one_time_manifest
        ? one_time_manifest : legacy_manifest;
    if (manifest) {
        auto model = dependencies_.parse_manifest
            ? dependencies_.parse_manifest(*manifest) : std::nullopt;

        // A launch manifest transfers ownership to this process. Consume it
        // even when parsing fails so stale credentials cannot be retried.
        if (one_time_manifest &&
            dependencies_.discard_one_time_manifest) {
            dependencies_.discard_one_time_manifest(*manifest);
        }
        if (!model) {
            LOG_ERROR("Failed to parse model launch manifest: {}", *manifest);
            return {HLIEBootstrapStatus::Failed,
                    HLIEBootstrapProcessMode::Worker, *launch_ticket};
        }

        auto worker_model = std::move(*model);
        if (!dependencies_.resolve_llm_launch_params ||
            !dependencies_.resolve_llm_launch_params(worker_model)) {
            LOG_ERROR("Failed to resolve llama params from launch manifest: {}",
                      *manifest);
            return {HLIEBootstrapStatus::Failed,
                    HLIEBootstrapProcessMode::Worker, *launch_ticket};
        }
        LOG_INFO("Successfully parsed model launch manifest: {}", *manifest);
        if (!dependencies_.register_model) {
            LOG_ERROR("Worker model registration adapter is unavailable");
            return {HLIEBootstrapStatus::Failed,
                    HLIEBootstrapProcessMode::Worker, *launch_ticket};
        }
        dependencies_.register_model(worker_model);
        LOG_INFO("Initializing Worker model Service: key={}, service_kind={}",
                 worker_model.model_key_, worker_model.get_service_kind());
        if (!dependencies_.initialize_worker_model ||
            !dependencies_.initialize_worker_model(worker_model)) {
            LOG_ERROR("Failed to initialize Worker model Service: key={}, service_kind={}",
                      worker_model.model_key_,
                      worker_model.get_service_kind());
            return {HLIEBootstrapStatus::Failed,
                    HLIEBootstrapProcessMode::Worker, *launch_ticket};
        }
        return {HLIEBootstrapStatus::Ready,
                HLIEBootstrapProcessMode::Worker, *launch_ticket};
    }

    if (!dependencies_.discover_models || !dependencies_.register_model ||
        !dependencies_.initialize_discovery_baseline ||
        !dependencies_.initialize_defaults) {
        LOG_ERROR("Main-process bootstrap adapters are incomplete");
        return {HLIEBootstrapStatus::Failed,
                HLIEBootstrapProcessMode::Main, *launch_ticket};
    }

    auto models = dependencies_.discover_models();
    for (auto& model : models) {
        dependencies_.register_model(model);
    }
    dependencies_.initialize_discovery_baseline();

    HLIEDefaultModelInitializationOptions options;
    options.model_directory = request.model_directory;
    options.default_model = request.default_model;
    options.default_mmproj = request.default_mmproj;
    options.enable_sd3 = request.enable_sd3;
    options.zero_load_allowed =
        argument(request, "--instance_id") == nullptr;

    // This phase only initializes concrete Service slots. Runtime start and
    // generation publication deliberately happen later in the run phase.
    const auto initialized = dependencies_.initialize_defaults(models, options);
    return {to_bootstrap_status(initialized.status),
            HLIEBootstrapProcessMode::Main, *launch_ticket};
}

HLIE_NAMESPACE_END

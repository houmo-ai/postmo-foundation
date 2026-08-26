#include "hlie_app/composition/hlie_application_composition_root.h"

#include <chrono>
#include <exception>
#include <stdexcept>
#include <utility>

#include "hlie_app/asr/hlie_default_asr_realtime_transport.h"
#include "hlie_app/hlie_model_service_factory.h"
#include "hlie_app/hlie_model_service_manager.h"
#include "hlie_app/inference/hlie_inference_http_error_mapper.h"
#include "hlie_app/runtime/hlie_runtime_directory.h"
#include "hlie_app/runtime/hlie_worker_supervisor.h"
#include "hlie_base/hlie_util.h"
#include "hlie_base/hlie_logger.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

namespace {

server_model_status to_legacy_status(const HLIEWorkerModelStatus status) {
    switch (status) {
    case HLIEWorkerModelStatus::Unloaded:
        return SERVER_MODEL_STATUS_UNLOADED;
    case HLIEWorkerModelStatus::Loading:
        return SERVER_MODEL_STATUS_LOADING;
    case HLIEWorkerModelStatus::Loaded:
        return SERVER_MODEL_STATUS_LOADED;
    case HLIEWorkerModelStatus::Sleeping:
        return SERVER_MODEL_STATUS_SLEEPING;
    }
    return SERVER_MODEL_STATUS_UNLOADED;
}

HLIELlmModelState to_llm_model_state(const server_model_status status) {
    switch (status) {
    case SERVER_MODEL_STATUS_UNLOADED:
        return HLIELlmModelState::Unloaded;
    case SERVER_MODEL_STATUS_LOADING:
        return HLIELlmModelState::Loading;
    case SERVER_MODEL_STATUS_LOADED:
        return HLIELlmModelState::Loaded;
    case SERVER_MODEL_STATUS_SLEEPING:
        return HLIELlmModelState::Other;
    }
    return HLIELlmModelState::Other;
}

HLIELlmReadinessOutcome to_llm_readiness_outcome(
    const HLIEModelReadinessStatus status) {
    switch (status) {
    case HLIEModelReadinessStatus::Ready:
        return HLIELlmReadinessOutcome::Ready;
    case HLIEModelReadinessStatus::Missing:
        return HLIELlmReadinessOutcome::Missing;
    case HLIEModelReadinessStatus::Unavailable:
        return HLIELlmReadinessOutcome::Unavailable;
    case HLIEModelReadinessStatus::InvalidLoadRequest:
        return HLIELlmReadinessOutcome::InvalidLoadRequest;
    case HLIEModelReadinessStatus::ScheduleFailed:
        return HLIELlmReadinessOutcome::ScheduleFailed;
    case HLIEModelReadinessStatus::TimedOut:
        return HLIELlmReadinessOutcome::TimedOut;
    case HLIEModelReadinessStatus::Cancelled:
        return HLIELlmReadinessOutcome::Cancelled;
    case HLIEModelReadinessStatus::Orphaned:
        return HLIELlmReadinessOutcome::Orphaned;
    case HLIEModelReadinessStatus::Failed:
        return HLIELlmReadinessOutcome::Failed;
    case HLIEModelReadinessStatus::Unloaded:
        return HLIELlmReadinessOutcome::Unloaded;
    }
    return HLIELlmReadinessOutcome::Unavailable;
}

} // namespace

std::optional<HLIEApplicationComponents>
HLIEApplicationCompositionRoot::compose(
    HLIEApplicationCompositionContext context) {
    try {
        HLIEApplicationComponents components;
        components.device_service = std::make_shared<HLIEDeviceService>();
        if (components.device_service->init() < 0) {
            LOG_ERROR("Failed to initialize the HLIE device service");
            return std::nullopt;
        }

        components.model_catalog = std::make_shared<HLIEModelCatalog>();
        components.resource_scheduler = std::make_unique<HLIEResourceScheduler>(
            components.device_service,
            std::move(context.initial_residency_policy));
        components.runtime_factory =
            std::make_unique<HLIEDefaultModelRuntimeFactory>(
                context.service_manager,
                context.supervisor_provider,
                std::move(context.embedded_identity_allocator));
        auto embedded_startup_local_port_provider =
            context.local_port_provider;
        auto llm_local_port_provider = context.local_port_provider;
        auto llm_master_port_provider =
            std::move(context.master_port_provider);
        auto llm_auto_load_provider =
            std::move(context.auto_load_provider);
        auto model_list_local_port_provider = context.local_port_provider;
        auto stop_service_local_model_provider =
            context.model_list_local_provider;
        auto stop_service_replicas_mode_provider =
            context.replicas_mode_provider;
        auto stop_service_supervisor_provider = context.supervisor_provider;
        auto observability_supervisor_provider = context.supervisor_provider;
        auto lifecycle_supervisor_provider = context.supervisor_provider;
        auto lifecycle_supervisor_ensurer = context.supervisor_ensurer;
        auto lifecycle_dynamic_worker_provider =
            context.worker_control_dependencies.dynamic_worker_provider;
        auto lifecycle_replicas_mode_provider =
            context.worker_control_dependencies.replicas_mode_provider;
        auto lifecycle_launch_ticket_provider =
            context.worker_control_dependencies.launch_ticket_provider;
        auto asr_local_port_provider = context.local_port_provider;
        auto speech_local_port_provider = context.local_port_provider;
        components.backend_launcher =
            std::make_unique<HLIEDefaultModelBackendLauncher>(
                context.service_manager, components.device_service,
                context.model_directory, context.program_path,
                context.supervisor_provider,
                std::move(context.local_port_provider),
                std::move(context.worker_spawn_policy));
        components.model_controller = std::make_unique<HLIEModelController>(
            *components.resource_scheduler, context.runtime_directory,
            *components.runtime_factory,
            HLIEModelController::StateChangedCallback{},
            components.backend_launcher.get(), components.model_catalog);
        auto default_model_llm_parameter_resolver =
            context.llm_parameter_resolver;
        auto bootstrap_llm_parameter_resolver =
            context.llm_parameter_resolver;
        auto default_model_device_service = components.device_service;
        const auto default_model_directory = context.model_directory;
        components.default_model_initialization_service =
            std::make_unique<HLIEDefaultModelInitializationService>(
                HLIEDefaultModelInitializationDependencies{
                    [](const HLIEModelInfo& model_info) {
                        return HLIEModelServiceFactory::
                            uses_llm_primary_service(model_info);
                    },
                    [](HLIEModelInfo& model_info,
                       const std::string& mmproj_path) {
                        model_info.set_mm_proj_path(mmproj_path, true);
                    },
                    std::move(default_model_llm_parameter_resolver),
                    [controller = components.model_controller.get()](
                        const std::string& model_key) {
                        return controller && controller->contains(model_key);
                    },
                    [controller = components.model_controller.get()](
                        HLIEModelInfo& model_info) {
                        if (controller) {
                            controller->register_model(
                                model_info, model_info.get_estimated_mem());
                        }
                    },
                    [&service_manager = context.service_manager,
                     device_service = default_model_device_service,
                     model_directory = default_model_directory](
                        const HLIEModelInfo& model_info) {
                        return service_manager.load_by_model_info(
                            model_directory, model_info, device_service);
                    },
                    [&service_manager = context.service_manager,
                     device_service = std::move(default_model_device_service),
                     model_directory = default_model_directory](
                        const HLIEModelInfo& model_info) {
                        return service_manager.load_sd(
                            model_directory, model_info, device_service);
                    },
                });
        components.embedded_startup_target_provider =
            std::make_unique<HLIEDefaultEmbeddedStartupTargetProvider>(
                context.service_manager,
                std::move(embedded_startup_local_port_provider));
        components.embedded_startup_service =
            std::make_unique<HLIEEmbeddedStartupService>(
                *components.model_controller,
                *components.embedded_startup_target_provider,
                std::move(context.embedded_startup_ready),
                [&service_manager = context.service_manager](
                    const HLIEEmbeddedStartupTarget& target) {
                    return service_manager.stop_and_reset_by_type(
                        target.service_kind,
                        target.companion_service != nullptr);
                });
        components.inference_endpoint_resolver =
            std::make_unique<HLIEDefaultInferenceEndpointResolver>(
                context.service_manager,
                std::move(context.proxy_timeout_provider));
        auto readiness_draining_provider = context.draining_provider;
        auto llm_draining_provider = context.draining_provider;
        components.inference_dispatcher =
            std::make_unique<HLIEInferenceDispatcher>(
                *components.model_controller, context.runtime_directory,
                *components.inference_endpoint_resolver,
                std::move(context.draining_provider));
        components.model_list_application_service =
            std::make_unique<HLIEModelListApplicationService>(
                HLIEModelListApplicationDependencies{
                    [controller = components.model_controller.get()] {
                        return controller
                            ? controller->get_all_metas()
                            : std::vector<server_model_meta>{};
                    },
                    std::move(context.model_list_local_provider),
                    std::move(context.model_list_process_provider),
                    [controller = components.model_controller.get()](
                        const std::string& requested)
                        -> std::optional<std::string> {
                        if (!controller) {
                            return std::nullopt;
                        }
                        const auto instance =
                            controller->get_instance_snapshot(requested);
                        return instance
                            ? std::optional<std::string>{instance->key}
                            : std::nullopt;
                    },
                    std::move(model_list_local_port_provider),
                });
        components.model_unload_application_service =
            std::make_unique<HLIEModelUnloadApplicationService>(
                HLIEModelUnloadApplicationDependencies{
                    std::move(context.replicas_mode_provider),
                    [controller = components.model_controller.get()](
                        const std::string& requested)
                        -> std::optional<std::string> {
                        if (!controller) {
                            return std::nullopt;
                        }
                        const auto instance =
                            controller->get_instance_snapshot(requested);
                        return instance
                            ? std::optional<std::string>{instance->key}
                            : std::nullopt;
                    },
                    std::move(context.loaded_model_by_type_provider),
                    [](const std::string& model_type) {
                        return HLIEModelServiceFactory::
                            is_supported_model_type(model_type);
                    },
                    [controller = components.model_controller.get()](
                        const std::string& model_key,
                        const std::string& reason) {
                        return controller && controller->evict_runtime(
                            model_key, reason).evicted();
                    },
                    std::move(context.model_unloaded_callback),
                });
        components.stop_service_application_service =
            std::make_unique<HLIEStopServiceApplicationService>(
                HLIEStopServiceApplicationDependencies{
                    std::move(stop_service_replicas_mode_provider),
                    [provider = std::move(stop_service_local_model_provider)] {
                        std::vector<std::string> result;
                        if (!provider) {
                            return result;
                        }
                        for (const auto& instance : provider()) {
                            result.push_back(instance.model_key);
                        }
                        return result;
                    },
                    [controller = components.model_controller.get()](
                        const std::string& model_key) {
                        return controller &&
                            controller->try_begin_eviction(model_key);
                    },
                    [controller = components.model_controller.get()](
                        const std::string& model_key) {
                        if (controller) {
                            controller->abort_eviction(model_key);
                        }
                    },
                    [controller = components.model_controller.get()](
                        const std::string& model_key) {
                        return controller && controller->evict_runtime(
                            model_key, "api_stop_service", true).evicted();
                    },
                    std::move(context.stop_service_worker_provider),
                    [controller = components.model_controller.get()](
                        const HLIEModelLaunchTicket& ticket) {
                        return controller && controller->stop_and_detach_runtime(
                            ticket, HLIERuntimeStopReason::ManualUnload);
                    },
                    [supervisor_provider =
                         std::move(stop_service_supervisor_provider)](
                        const HLIEStopServiceWorkerSnapshot& worker) {
                        auto* supervisor = supervisor_provider
                            ? supervisor_provider() : nullptr;
                        if (!supervisor) {
                            return false;
                        }
                        const HLIEWorkerRuntime runtime{
                            worker.process_id,
                            worker.instance_id,
                            worker.port,
                            worker.model_names,
                            worker.launch_ticket,
                        };
                        return supervisor->request_stop(
                            runtime,
                            "api_stop_instance_compatibility").confirmed;
                    },
                    [controller = components.model_controller.get()](
                        const HLIEModelLaunchTicket& ticket) {
                        if (controller) {
                            (void)controller->publish_confirmed_stop(ticket);
                        }
                    },
                    [controller = components.model_controller.get()](
                        const std::string& requested)
                        -> std::optional<std::string> {
                        if (!controller) {
                            return std::nullopt;
                        }
                        const auto instance =
                            controller->get_instance_snapshot(requested);
                        return instance
                            ? std::optional<std::string>{instance->key}
                            : std::nullopt;
                    },
                    [controller = components.model_controller.get()](
                        const std::string& model_key) {
                        if (controller) {
                            (void)controller->publish_confirmed_legacy_stop(
                                model_key);
                        }
                    },
                });
        auto model_load_residency_policy_provider =
            context.residency_policy_provider;
        components.control_plane_reconciler =
            std::make_unique<HLIEControlPlaneReconciler>(
                *components.resource_scheduler, *components.model_controller,
                components.inference_dispatcher.get(),
                std::move(context.residency_policy_provider),
                std::move(context.external_live_owner_provider));
        components.model_discovery_service =
            std::make_unique<HLIEModelDiscoveryService>(
                context.model_directory,
                std::move(context.discovery_options_provider),
                std::move(context.prepare_catalog_update));
        components.bootstrap_application_service =
            std::make_unique<HLIEBootstrapApplicationService>(
                HLIEBootstrapApplicationDependencies{
                    std::move(
                        context.bootstrap_prepare_launch_configuration),
                    [](const std::string& path)
                        -> std::optional<HLIEModelInfo> {
                        HLIEModelInfo model;
                        if (model.parse_from_file(path) != 0) {
                            return std::nullopt;
                        }
                        return model;
                    },
                    [](const std::string& path) {
                        HLIEWorkerSupervisor::discard_launch_manifest(
                            path, "worker_consumed");
                    },
                    std::move(context.bootstrap_model_provider),
                    std::move(bootstrap_llm_parameter_resolver),
                    [controller = components.model_controller.get()](
                        HLIEModelInfo& model) {
                        controller->register_model(
                            model, model.get_estimated_mem());
                    },
                    [&service_manager = context.service_manager,
                     device_service = components.device_service,
                     model_directory = context.model_directory](
                        const HLIEModelInfo& model) {
                        return service_manager.load_by_model_info(
                            model_directory, model, device_service);
                    },
                    [discovery =
                         components.model_discovery_service.get()] {
                        (void)discovery->initialize_baseline();
                    },
                    [initializer =
                         components.default_model_initialization_service.get()](
                        const std::vector<HLIEModelInfo>& models,
                        const HLIEDefaultModelInitializationOptions& options) {
                        return initializer->initialize(models, options);
                    },
                });
        std::function<HLIEObservabilityModelSummary()> model_summary_provider =
            [controller = components.model_controller.get()] {
                HLIEObservabilityModelSummary summary;
                if (!controller) {
                    return summary;
                }
                for (const auto& model :
                     controller->get_all_runtime_snapshots()) {
                    summary.loaded += hlie_model_state_is_ready(model.state)
                        ? 1 : 0;
                    summary.loading += hlie_model_state_is_loading(model.state)
                        ? 1 : 0;
                    summary.failed +=
                        model.state == HLIEModelLifecycleState::Failed ? 1 : 0;
                }
                return summary;
            };
        auto full_model_summary_provider = model_summary_provider;
        components.observability_application_service =
            std::make_unique<HLIEObservabilityApplicationService>(
                HLIEObservabilityApplicationDependencies{
                    std::move(model_summary_provider),
                    [model_provider =
                         std::move(full_model_summary_provider),
                     supervisor_provider =
                         std::move(observability_supervisor_provider),
                     scheduler = components.resource_scheduler.get(),
                     runtime_directory = &context.runtime_directory,
                     dispatcher = components.inference_dispatcher.get(),
                     reconciler = components.control_plane_reconciler.get(),
                     discovery = components.model_discovery_service.get(),
                     llm_admission = context.llm_admission,
                     qwen_admission = context.qwen3_tts_admission,
                     cancellation_provider = std::move(
                         context.observability_cancellation_lag_provider)] {
                        HLIEObservabilitySnapshot snapshot;
                        snapshot.models = model_provider
                            ? model_provider()
                            : HLIEObservabilityModelSummary{};

                        if (llm_admission) {
                            const auto metrics = llm_admission->snapshot();
                            snapshot.llm_admission = {
                                metrics.max_in_flight, metrics.in_flight,
                                metrics.accepted, metrics.rejected};
                        }
                        if (qwen_admission) {
                            const auto metrics = qwen_admission->snapshot();
                            snapshot.qwen3_tts_admission = {
                                metrics.max_in_flight, metrics.in_flight,
                                metrics.accepted, metrics.rejected};
                        }

                        auto* supervisor = supervisor_provider
                            ? supervisor_provider() : nullptr;
                        if (supervisor) {
                            const auto metrics = supervisor->metrics();
                            snapshot.workers = {
                                supervisor->running_runtimes().size(),
                                metrics.unexpected_exits,
                                metrics.ready_timeouts,
                                metrics.graceful_shutdowns,
                                metrics.forced_shutdowns,
                                metrics.port_conflicts,
                            };
                        }

                        if (runtime_directory) {
                            const auto runtimes =
                                runtime_directory->health_snapshots();
                            snapshot.runtimes.total =
                                runtime_directory->size();
                            for (const auto& runtime : runtimes) {
                                snapshot.runtimes.embedded +=
                                    runtime.identity.kind ==
                                        HLIERuntimeKind::Embedded ? 1 : 0;
                                snapshot.runtimes.process +=
                                    runtime.identity.kind ==
                                        HLIERuntimeKind::Process ? 1 : 0;
                                snapshot.runtimes.ready +=
                                    runtime.state ==
                                        HLIERuntimeHealthState::Ready ? 1 : 0;
                                snapshot.runtimes.starting +=
                                    runtime.state ==
                                        HLIERuntimeHealthState::Starting ? 1 : 0;
                                snapshot.runtimes.draining +=
                                    runtime.state ==
                                        HLIERuntimeHealthState::Draining ? 1 : 0;
                                snapshot.runtimes.failed +=
                                    runtime.state ==
                                        HLIERuntimeHealthState::Failed ||
                                    runtime.state ==
                                        HLIERuntimeHealthState::Orphaned
                                        ? 1 : 0;
                            }
                        }

                        if (dispatcher) {
                            const auto metrics = dispatcher->metrics();
                            snapshot.inference = {
                                metrics.active_sessions,
                                metrics.dispatched,
                                metrics.draining_rejected,
                                metrics.model_not_found,
                                metrics.model_not_ready,
                                metrics.runtime_unavailable,
                                metrics.endpoint_unavailable,
                                metrics.unsupported_operation,
                                metrics.endpoint_failures,
                                metrics.cached_endpoints,
                            };
                        }
                        if (reconciler) {
                            const auto metrics = reconciler->metrics();
                            snapshot.reconciliation = {
                                metrics.running,
                                metrics.reconciliation_ticks,
                                metrics.reconciliation_failures,
                                metrics.policy_refreshes,
                                metrics.idle_ttl_evictions,
                            };
                        }
                        if (discovery) {
                            const auto metrics = discovery->metrics();
                            snapshot.discovery = {
                                metrics.running,
                                metrics.pending_stability,
                                metrics.scan_attempts,
                                metrics.parse_attempts,
                                metrics.scan_failures,
                                metrics.published_models,
                                metrics.requested_generation,
                                metrics.completed_generation,
                            };
                        }
                        if (scheduler) {
                            const auto metrics = scheduler->metrics();
                            snapshot.resources = {
                                metrics.expired_pending_rollbacks,
                                metrics.expired_pending_protected,
                                metrics.tracked_pending,
                                metrics.tracked_committed,
                            };
                            for (const auto& device :
                                 scheduler->ledger_snapshots()) {
                                snapshot.devices.push_back({
                                    device.device_id,
                                    device.total_mb,
                                    device.observed_used_mb,
                                    device.pending_reserved_mb,
                                    device.committed_reserved_mb,
                                    device.committed_unobserved_mb,
                                    device.effective_used_mb,
                                    device.admissible_mb,
                                });
                            }
                        }
                        if (cancellation_provider) {
                            snapshot.cancellation_lag =
                                cancellation_provider();
                        }
                        return snapshot;
                    },
                });
        components.model_load_service =
            std::make_unique<HLIEModelLoadService>(
                *components.model_controller,
                *components.resource_scheduler,
                components.model_discovery_service.get(),
                context.model_directory, context.replicas_mode,
                std::move(model_load_residency_policy_provider),
                std::move(context.llm_parameter_resolver),
                HLIEModelServiceFactory::default_registry());
        auto* model_load_service = components.model_load_service.get();
        components.model_readiness_service =
            std::make_unique<HLIEModelReadinessService>(
                *components.model_controller,
                [model_load_service](
                    HLIEModelLoadRequest request,
                    const int local_port,
                    const std::optional<HLIEModelLaunchTicket>& preclaimed) {
                    return model_load_service->load(
                        std::move(request), local_port, preclaimed);
                },
                std::move(readiness_draining_provider),
                std::move(context.model_load_succeeded));
        components.document_application_service =
            std::make_unique<HLIEDocumentApplicationService>(
                HLIEDocumentApplicationDependencies{
                    [dispatcher = components.inference_dispatcher.get()](
                        const std::string& model_key,
                        const HLIEInferenceOperation operation,
                        const server_http_req& request,
                        const std::string& method) {
                        return HLIEInferenceHttpErrorMapper::map(
                            dispatcher->dispatch(
                                model_key, operation, request, method));
                    },
                    [&service_manager = context.service_manager] {
                        (void)service_manager.stop_and_reset_document_pipeline();
                    },
                    [](const HLIEDocumentRouteObservation& observation) {
                        LOG_DEBUG("Routing PDF request to document pipeline: model={}, source={}, filename={}, content_type={}",
                                  nlohmann::json(observation.model_key).dump(),
                                  nlohmann::json(observation.source).dump(),
                                  nlohmann::json(observation.filename).dump(),
                                  nlohmann::json(observation.content_type).dump());
                    },
                });
        if (!context.llm_admission) {
            LOG_ERROR("Application composition is missing LLM admission controller");
            return std::nullopt;
        }
        components.llm_application_service =
            std::make_unique<HLIELlmApplicationService>(
                *context.llm_admission,
                HLIELlmApplicationDependencies{
                    std::move(llm_draining_provider),
                    [controller = components.model_controller.get()](
                        const std::string& requested)
                        -> std::optional<HLIELlmModelSnapshot> {
                        if (!controller) {
                            return std::nullopt;
                        }
                        const auto instance =
                            controller->get_instance_snapshot(requested);
                        if (!instance) {
                            return std::nullopt;
                        }
                        return HLIELlmModelSnapshot{
                            instance->key,
                            to_llm_model_state(instance->meta.status),
                            instance->meta.port,
                        };
                    },
                    [discovery = components.model_discovery_service.get()] {
                        if (discovery) {
                            const auto generation = discovery->request_scan();
                            (void)discovery->wait_for(generation);
                        }
                    },
                    std::move(llm_auto_load_provider),
                    [](const std::string& devices)
                        -> std::optional<std::vector<int>> {
                        std::vector<int> parsed;
                        if (HlieUtil::parse_device(devices, parsed) < 0) {
                            return std::nullopt;
                        }
                        return parsed;
                    },
                    [readiness = components.model_readiness_service.get(),
                     local_port = llm_local_port_provider](
                        const std::string& model_key,
                        std::optional<HLIELlmAutoLoadRequest> request,
                        const std::chrono::steady_clock::time_point deadline,
                        const std::function<bool()>& should_stop) {
                        if (!readiness) {
                            return HLIELlmReadinessOutcome::Unavailable;
                        }
                        std::optional<HLIEModelLoadRequest> load_request;
                        if (request) {
                            HLIEModelLoadRequest converted;
                            converted.model_type = LLM;
                            converted.model = std::move(request->model);
                            converted.model_path =
                                std::move(request->model_path);
                            converted.mmproj_path =
                                std::move(request->mmproj_path);
                            converted.requested_device_ids =
                                std::move(request->requested_device_ids);
                            load_request = std::move(converted);
                        }
                        const auto result = readiness->ensure_ready({
                            model_key,
                            std::move(load_request),
                            local_port ? local_port() : 0,
                            deadline,
                            should_stop,
                        });
                        return to_llm_readiness_outcome(result.status);
                    },
                    llm_local_port_provider,
                    std::move(llm_master_port_provider),
                    [controller = components.model_controller.get(),
                     runtime_directory = &context.runtime_directory]()
                        -> std::optional<std::string> {
                        if (!controller || !runtime_directory) {
                            return std::nullopt;
                        }
                        for (const auto& instance :
                             controller->get_all_instances()) {
                            if (instance.meta.status !=
                                    SERVER_MODEL_STATUS_LOADED ||
                                instance.model_info.get_service_kind() != LLM) {
                                continue;
                            }
                            const auto ticket = controller->get_launch_ticket(
                                instance.key);
                            const auto runtime = ticket
                                ? runtime_directory->find(*ticket) : nullptr;
                            if (runtime && runtime->identity().kind ==
                                    HLIERuntimeKind::Embedded) {
                                return instance.key;
                            }
                        }
                        return std::nullopt;
                    },
                    [dispatcher = components.inference_dispatcher.get()](
                        const std::string& model_key,
                        const HLIEInferenceOperation operation,
                        const server_http_req& request,
                        const std::string& method) {
                        return HLIEInferenceHttpErrorMapper::map(
                            dispatcher->dispatch(
                                model_key, operation, request, method));
                    },
                    [document = components.document_application_service.get()](
                        const std::string& model_key,
                        const HLIEInferenceOperation operation,
                        const server_http_req& request,
                        const nlohmann::json& body,
                        const std::string& method) {
                        return document->dispatch_or_route(
                            model_key, operation, request, body, method);
                    },
                });
        auto asr_route_llm =
            [llm = components.llm_application_service.get()](
                const server_http_req& request,
                const HLIEInferenceOperation operation) {
                return llm->handle_admitted(request, operation);
            };
        auto asr_listener_allowed =
            std::move(context.asr_realtime_listener_allowed);
        auto asr_find_ready_model =
            [&controller = *components.model_controller,
             &runtime_directory = context.runtime_directory](
                const std::string& service_kind,
                const HLIERuntimeKind runtime_kind)
                -> std::optional<std::string> {
                for (const auto& instance : controller.get_all_instances()) {
                    if (instance.meta.status != SERVER_MODEL_STATUS_LOADED ||
                        instance.model_info.get_service_kind() != service_kind) {
                        continue;
                    }
                    const auto ticket = controller.get_launch_ticket(instance.key);
                    const auto runtime = ticket
                        ? runtime_directory.find(*ticket) : nullptr;
                    if (runtime && runtime->identity().kind == runtime_kind) {
                        return instance.key;
                    }
                }
                return std::nullopt;
            };
        components.asr_application_service =
            std::make_unique<HLIEAsrApplicationService>(
                *context.llm_admission,
                HLIEAsrApplicationDependencies{
                    std::move(asr_find_ready_model),
                    [&controller = *components.model_controller](
                        const std::string& requested)
                        -> std::optional<HLIEAsrModelCandidate> {
                        const auto instance =
                            controller.get_instance_snapshot(requested);
                        if (!instance) {
                            return std::nullopt;
                        }
                        return HLIEAsrModelCandidate{
                            instance->key,
                            instance->model_info.get_model_name(),
                        };
                    },
                    [&controller = *components.model_controller] {
                        std::vector<HLIEAsrModelCandidate> candidates;
                        for (const auto& instance : controller.get_all_instances()) {
                            candidates.push_back({
                                instance.key,
                                instance.model_info.get_model_name(),
                            });
                        }
                        return candidates;
                    },
                    [dispatcher = components.inference_dispatcher.get()](
                        const std::string& model_key,
                        const HLIEInferenceOperation operation,
                        const server_http_req& request,
                        const std::string& method) {
                        return HLIEInferenceHttpErrorMapper::map(
                            dispatcher->dispatch(
                                model_key, operation, request, method));
                    },
                    std::move(asr_route_llm),
                    [dispatcher = components.inference_dispatcher.get()](
                        const std::string& model_key,
                        const WhisperStreamRequest& request,
                        const HLIEAsrTranscriptCallback& callback) {
                        auto acquired = dispatcher->acquire(
                            model_key, HLIEInferenceOperation::WhisperRealtime);
                        if (!acquired.acquired()) {
                            return std::string{};
                        }
                        const auto endpoint = acquired.session.endpoint();
                        if (!endpoint) {
                            return std::string{};
                        }
                        const auto transcript = endpoint->dispatch_realtime_transcription(
                            request, callback);
                        dispatcher->observe_native_dispatch(!transcript.empty());
                        return transcript;
                    },
                    std::move(asr_listener_allowed),
                    [&service_manager = context.service_manager] {
                        return service_manager.whisper() != nullptr ||
                            service_manager.llm() != nullptr;
                    },
                    std::move(asr_local_port_provider),
                    [] {
                        return std::make_unique<
                            HLIEDefaultAsrRealtimeTransport>();
                    },
                });
        if (!context.qwen3_tts_admission) {
            LOG_ERROR("Application composition is missing Qwen3-TTS admission controller");
            return std::nullopt;
        }
        auto speech_find_ready_model =
            [&controller = *components.model_controller,
             &runtime_directory = context.runtime_directory](
                const std::string& service_kind,
                const HLIERuntimeKind runtime_kind)
                -> std::optional<std::string> {
                for (const auto& instance : controller.get_all_instances()) {
                    const auto& actual_kind =
                        instance.model_info.get_service_kind();
                    const bool kind_matches = actual_kind == service_kind ||
                        ((service_kind == "cosyvoice" || service_kind == "tts") &&
                         (actual_kind == "cosyvoice" || actual_kind == "tts"));
                    if (instance.meta.status != SERVER_MODEL_STATUS_LOADED ||
                        !kind_matches) {
                        continue;
                    }
                    const auto ticket = controller.get_launch_ticket(instance.key);
                    const auto runtime = ticket
                        ? runtime_directory.find(*ticket) : nullptr;
                    if (runtime && runtime->identity().kind == runtime_kind) {
                        return instance.key;
                    }
                }
                return std::nullopt;
            };
        components.speech_application_service =
            std::make_unique<HLIESpeechApplicationService>(
                *context.qwen3_tts_admission,
                HLIESpeechApplicationDependencies{
                    [&controller = *components.model_controller](
                        const std::string& requested)
                        -> std::optional<HLIESpeechModelSnapshot> {
                        const auto instance =
                            controller.get_instance_snapshot(requested);
                        if (!instance) {
                            return std::nullopt;
                        }
                        HLIESpeechModelState state = HLIESpeechModelState::Other;
                        if (instance->meta.status == SERVER_MODEL_STATUS_LOADING) {
                            state = HLIESpeechModelState::Loading;
                        } else if (instance->meta.status == SERVER_MODEL_STATUS_LOADED) {
                            state = HLIESpeechModelState::Loaded;
                        }
                        return HLIESpeechModelSnapshot{
                            instance->key,
                            instance->model_info.get_service_kind(),
                            state,
                        };
                    },
                    std::move(speech_find_ready_model),
                    [readiness = components.model_readiness_service.get(),
                     local_port = std::move(speech_local_port_provider)](
                        const std::string& model_key,
                        const std::function<bool()>& should_stop) {
                        if (!readiness) {
                            return HLIESpeechReadinessStatus::Unavailable;
                        }
                        const auto result = readiness->ensure_ready({
                            model_key, std::nullopt,
                            local_port ? local_port() : 0,
                            std::chrono::steady_clock::now() +
                                std::chrono::minutes(5),
                            should_stop,
                        });
                        switch (result.status) {
                        case HLIEModelReadinessStatus::Ready:
                            return HLIESpeechReadinessStatus::Ready;
                        case HLIEModelReadinessStatus::Cancelled:
                            return HLIESpeechReadinessStatus::Cancelled;
                        case HLIEModelReadinessStatus::TimedOut:
                            return HLIESpeechReadinessStatus::TimedOut;
                        case HLIEModelReadinessStatus::Missing:
                            return HLIESpeechReadinessStatus::Missing;
                        case HLIEModelReadinessStatus::Orphaned:
                            return HLIESpeechReadinessStatus::Orphaned;
                        default:
                            return HLIESpeechReadinessStatus::Unavailable;
                        }
                    },
                    [dispatcher = components.inference_dispatcher.get()](
                        const std::string& model_key) {
                        HLIESpeechAcquireResult result;
                        auto acquired = dispatcher->acquire(
                            model_key, HLIEInferenceOperation::Speech);
                        if (!acquired.acquired()) {
                            result.detail = std::move(acquired.detail);
                            return result;
                        }
                        auto endpoint = acquired.session.endpoint();
                        if (!endpoint) {
                            result.detail = "speech endpoint is unavailable";
                            return result;
                        }
                        HLIESpeechDispatchTarget target;
                        target.identity = acquired.session.identity();
                        target.lifetime_guard =
                            acquired.session.lifetime_guard();
                        target.dispatch =
                            [endpoint = std::move(endpoint)](
                                const httplib::Request& request,
                                httplib::Response& response,
                                std::shared_ptr<void> lifetime_guard) {
                                return endpoint->dispatch_speech(
                                    request, response,
                                    std::move(lifetime_guard));
                            };
                        result.target = std::move(target);
                        return result;
                    },
                    [dispatcher = components.inference_dispatcher.get()](
                        const bool successful) {
                        dispatcher->observe_native_dispatch(successful);
                    },
                });
        if (!context.worker_status_publisher) {
            LOG_ERROR("Application composition is missing Worker status publisher");
            return std::nullopt;
        }
        components.worker_status_publisher =
            std::move(context.worker_status_publisher);
        auto worker_dependencies =
            std::move(context.worker_control_dependencies);
        auto worker_launch_ticket_provider =
            worker_dependencies.launch_ticket_provider;
        worker_dependencies.resolve_model_key =
            [controller = components.model_controller.get()](
                const std::string& requested) -> std::optional<std::string> {
                if (!controller) {
                    return std::nullopt;
                }
                const auto instance = controller->get_instance_snapshot(requested);
                return instance ? std::optional<std::string>{instance->key}
                                : std::nullopt;
            };
        worker_dependencies.apply_status =
            [controller = components.model_controller.get()](
                const HLIEModelLaunchTicket& ticket,
                const HLIEWorkerModelStatus status,
                const int port) {
                return controller && controller->apply_worker_status(
                    ticket, to_legacy_status(status), port);
            };
        components.worker_control_service =
            std::make_unique<HLIEWorkerControlService>(
                *components.worker_status_publisher,
                std::move(worker_dependencies));
        components.worker_control_application_service =
            std::make_unique<HLIEWorkerControlApplicationService>(
                *components.worker_control_service,
                HLIEWorkerControlApplicationDependencies{
                    std::move(worker_launch_ticket_provider)});
        components.application_lifecycle_service =
            std::make_unique<HLIEApplicationLifecycleService>(
                HLIEApplicationLifecycleDependencies{
                    [startup = components.embedded_startup_service.get()](
                        const HLIEEmbeddedStartupGroup group) {
                        return startup->start_group(group);
                    },
                    [admission = context.llm_admission](
                        const std::size_t limit) {
                        admission->configure(limit);
                    },
                    [admission = context.qwen3_tts_admission](
                        const std::size_t limit) {
                        admission->configure(limit);
                    },
                    [asr = components.asr_application_service.get()] {
                        asr->ensure_realtime();
                    },
                    [asr = components.asr_application_service.get()] {
                        asr->stop_realtime();
                    },
                    [reconciler = components.control_plane_reconciler.get()] {
                        reconciler->start();
                    },
                    [reconciler = components.control_plane_reconciler.get()] {
                        reconciler->stop();
                    },
                    [discovery = components.model_discovery_service.get()] {
                        discovery->start();
                    },
                    [discovery = components.model_discovery_service.get()] {
                        discovery->stop();
                    },
                    [ensure_supervisor =
                         std::move(lifecycle_supervisor_ensurer)] {
                        auto* supervisor = ensure_supervisor
                            ? ensure_supervisor() : nullptr;
                        if (!supervisor) {
                            throw std::runtime_error(
                                "Worker Supervisor is unavailable");
                        }
                        supervisor->start();
                    },
                    [controller = components.model_controller.get()] {
                        controller->begin_draining();
                    },
                    [readiness = components.model_readiness_service.get()] {
                        readiness->stop_and_join();
                    },
                    [controller = components.model_controller.get()] {
                        return controller->stop_all(
                            HLIERuntimeStopReason::Shutdown);
                    },
                    [supervisor_provider = lifecycle_supervisor_provider,
                     controller = components.model_controller.get()](
                        const std::string& reason) {
                        auto* supervisor = supervisor_provider
                            ? supervisor_provider() : nullptr;
                        if (!supervisor) {
                            return true;
                        }
                        bool confirmed = true;
                        for (const auto& stopped :
                             supervisor->shutdown_all(reason)) {
                            if (!stopped.confirmed) {
                                LOG_ERROR("Failed to stop dynamic Worker during application shutdown: pid={}, generation={}",
                                          stopped.runtime.runtime_id,
                                          stopped.runtime.launch_ticket.generation);
                                confirmed = false;
                                continue;
                            }
                            (void)controller->publish_confirmed_stop(
                                stopped.runtime.launch_ticket);
                        }
                        return confirmed;
                    },
                    [supervisor_provider = lifecycle_supervisor_provider] {
                        if (auto* supervisor = supervisor_provider
                                ? supervisor_provider() : nullptr) {
                            supervisor->stop();
                        }
                    },
                    [&service_manager = context.service_manager,
                     worker_control = components.worker_control_service.get(),
                     dynamic_worker =
                         std::move(lifecycle_dynamic_worker_provider),
                     replicas_mode =
                         std::move(lifecycle_replicas_mode_provider),
                     launch_ticket =
                         std::move(lifecycle_launch_ticket_provider)] {
                        const bool publish_worker_unloaded =
                            dynamic_worker && dynamic_worker() &&
                            !(replicas_mode && replicas_mode());
                        HLIEResidualServiceCleanupResult result;
                        const auto notify_unloaded =
                            [publish_worker_unloaded, worker_control,
                             &launch_ticket,
                             &result](const std::string& model_key) {
                                if (publish_worker_unloaded && worker_control &&
                                    launch_ticket) {
                                    result.status_publications_confirmed =
                                        worker_control->publish_unloaded(
                                            model_key, launch_ticket()) &&
                                        result.status_publications_confirmed;
                                }
                            };

                        if (const auto key = service_manager.model_key_by_type(
                                HLIEModelType::SD)) {
                            notify_unloaded(*key);
                            result.services_confirmed =
                                service_manager.stop_and_reset_sd() &&
                                result.services_confirmed;
                        }
                        if (const auto key = service_manager.model_key_by_type(
                                HLIEModelType::WHISPER)) {
                            notify_unloaded(*key);
                            result.services_confirmed =
                                service_manager.stop_and_reset_whisper() &&
                                result.services_confirmed;
                        }
                        if (const auto key = service_manager.model_key_by_type(
                                HLIEModelType::COSYVOICE)) {
                            notify_unloaded(*key);
                            result.services_confirmed =
                                service_manager.stop_and_reset_cosyvoice() &&
                                result.services_confirmed;
                        }
                        if (const auto key = service_manager.model_key_by_type(
                                HLIEModelType::QWEN3_TTS)) {
                            notify_unloaded(*key);
                            result.services_confirmed =
                                service_manager.stop_and_reset_qwen3_tts() &&
                                result.services_confirmed;
                        }
                        if (const auto key = service_manager.model_key_by_type(
                                HLIEModelType::LLM)) {
                            notify_unloaded(*key);
                            result.services_confirmed =
                                service_manager.stop_and_reset_document_pipeline() &&
                                result.services_confirmed;
                            result.services_confirmed =
                                service_manager.stop_and_reset_llm() &&
                                result.services_confirmed;
                        }
                        return result;
                    },
                    [worker_control = components.worker_control_service.get()] {
                        worker_control->stop_and_join();
                    },
                });
        return std::optional<HLIEApplicationComponents>{
            std::move(components)};
    } catch (const std::exception& ex) {
        LOG_ERROR("Application composition failed: {}", ex.what());
    } catch (...) {
        LOG_ERROR("Application composition failed with an unknown exception");
    }
    return std::nullopt;
}

HLIE_NAMESPACE_END

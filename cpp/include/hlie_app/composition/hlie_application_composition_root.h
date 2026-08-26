#ifndef HLIE_APP_APPLICATION_COMPOSITION_ROOT_H
#define HLIE_APP_APPLICATION_COMPOSITION_ROOT_H

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "hlie_app/catalog/hlie_model_catalog.h"
#include "hlie_app/control/hlie_control_plane_reconciler.h"
#include "hlie_app/control/hlie_model_controller.h"
#include "hlie_app/control/hlie_model_load_service.h"
#include "hlie_app/control/hlie_model_readiness_service.h"
#include "hlie_app/control/hlie_worker_control_service.h"
#include "hlie_app/control/hlie_worker_control_application_service.h"
#include "hlie_app/discovery/hlie_model_discovery_service.h"
#include "hlie_app/asr/hlie_asr_application_service.h"
#include "hlie_app/speech/hlie_speech_application_service.h"
#include "hlie_app/document/hlie_document_application_service.h"
#include "hlie_app/hlie_device_service.h"
#include "hlie_app/inference/hlie_default_inference_endpoint_resolver.h"
#include "hlie_app/inference/hlie_inference_dispatcher.h"
#include "hlie_app/llm/hlie_llm_application_service.h"
#include "hlie_app/lifecycle/hlie_application_lifecycle_service.h"
#include "hlie_app/models/hlie_model_list_application_service.h"
#include "hlie_app/models/hlie_model_unload_application_service.h"
#include "hlie_app/models/hlie_stop_service_application_service.h"
#include "hlie_app/observability/hlie_observability_application_service.h"
#include "hlie_app/resource/hlie_resource_scheduler.h"
#include "hlie_app/runtime/hlie_default_model_backend_launcher.h"
#include "hlie_app/runtime/hlie_default_model_runtime_factory.h"
#include "hlie_app/startup/hlie_default_model_initialization_service.h"
#include "hlie_app/startup/hlie_bootstrap_application_service.h"
#include "hlie_app/startup/hlie_default_embedded_startup_target_provider.h"
#include "hlie_app/startup/hlie_embedded_startup_service.h"

HLIE_NAMESPACE_BEGIN

class HLIEModelServiceManager;
class HLIERuntimeDirectory;
class HLIEWorkerSupervisor;

struct HLIEApplicationCompositionContext {
    HLIEModelServiceManager& service_manager;
    HLIERuntimeDirectory& runtime_directory;
    std::string model_directory;
    std::string program_path;
    HLIEModelResidencyPolicy initial_residency_policy;

    std::function<HLIEWorkerSupervisor*()> supervisor_provider;
    std::function<HLIEWorkerSupervisor*()> supervisor_ensurer;
    std::function<HLIERuntimeIdentity(const HLIEModelLaunchTicket&)>
        embedded_identity_allocator;
    std::function<int()> local_port_provider;
    std::function<int()> master_port_provider;
    std::function<bool()> auto_load_provider;
    std::function<std::vector<HLIEModelListProcessSnapshot>()>
        model_list_process_provider;
    std::function<std::vector<HLIEModelListLocalInstance>()>
        model_list_local_provider;
    std::function<bool()> replicas_mode_provider;
    std::function<std::optional<std::string>(const std::string&)>
        loaded_model_by_type_provider;
    std::function<void(const std::string&)> model_unloaded_callback;
    std::function<std::optional<HLIEStopServiceWorkerSnapshot>(std::uint64_t)>
        stop_service_worker_provider;
    std::function<std::vector<HLIEObservabilityCancellationLag>()>
        observability_cancellation_lag_provider;
    std::function<bool()> worker_spawn_policy;
    std::function<ProxyTimeoutOptions()> proxy_timeout_provider;
    std::function<bool()> draining_provider;
    HLIEControlPlaneReconciler::ResidencyPolicyProvider
        residency_policy_provider;
    HLIEControlPlaneReconciler::ExternalLiveOwnerProvider
        external_live_owner_provider;
    HLIEModelDiscoveryService::OptionsProvider discovery_options_provider;
    HLIEModelDiscoveryService::PrepareCatalogUpdate
        prepare_catalog_update;
    std::function<bool()> bootstrap_prepare_launch_configuration;
    std::function<std::vector<HLIEModelInfo>()> bootstrap_model_provider;
    bool replicas_mode = false;
    HLIEModelLoadService::LlmParameterResolver llm_parameter_resolver;
    HLIEModelReadinessService::LoadSucceededCallback
        model_load_succeeded;
    HLIEAdmissionController* llm_admission = nullptr;
    HLIEAdmissionController* qwen3_tts_admission = nullptr;
    std::function<bool()> asr_realtime_listener_allowed;
    HLIEEmbeddedStartupService::ReadyCallback embedded_startup_ready;
    std::unique_ptr<IHLIEWorkerStatusPublisher> worker_status_publisher;
    HLIEWorkerControlDependencies worker_control_dependencies;
};

struct HLIEApplicationComponents {
    std::shared_ptr<HLIEDeviceService> device_service;
    std::shared_ptr<HLIEModelCatalog> model_catalog;
    std::unique_ptr<HLIEResourceScheduler> resource_scheduler;
    std::unique_ptr<IHLIEModelRuntimeFactory> runtime_factory;
    std::unique_ptr<IHLIEModelBackendLauncher> backend_launcher;
    std::unique_ptr<HLIEModelController> model_controller;
    std::unique_ptr<HLIEDefaultInferenceEndpointResolver>
        inference_endpoint_resolver;
    std::unique_ptr<HLIEInferenceDispatcher> inference_dispatcher;
    std::unique_ptr<HLIEControlPlaneReconciler> control_plane_reconciler;
    std::unique_ptr<HLIEModelDiscoveryService> model_discovery_service;
    std::unique_ptr<HLIEObservabilityApplicationService>
        observability_application_service;
    std::unique_ptr<HLIEModelLoadService> model_load_service;
    std::unique_ptr<HLIEModelReadinessService> model_readiness_service;
    std::unique_ptr<HLIEDocumentApplicationService>
        document_application_service;
    std::unique_ptr<HLIELlmApplicationService> llm_application_service;
    std::unique_ptr<HLIEAsrApplicationService> asr_application_service;
    std::unique_ptr<HLIESpeechApplicationService> speech_application_service;
    std::unique_ptr<HLIEModelListApplicationService>
        model_list_application_service;
    std::unique_ptr<HLIEModelUnloadApplicationService>
        model_unload_application_service;
    std::unique_ptr<HLIEStopServiceApplicationService>
        stop_service_application_service;
    std::unique_ptr<HLIEDefaultModelInitializationService>
        default_model_initialization_service;
    // Bootstrap callbacks reference default initialization and discovery;
    // declaring it later guarantees it is destroyed first.
    std::unique_ptr<HLIEBootstrapApplicationService>
        bootstrap_application_service;
    std::unique_ptr<IHLIEEmbeddedStartupTargetProvider>
        embedded_startup_target_provider;
    std::unique_ptr<HLIEEmbeddedStartupService> embedded_startup_service;
    // Declared before the service so the service is destroyed first and can
    // never publish through a destroyed transport adapter.
    std::unique_ptr<IHLIEWorkerStatusPublisher> worker_status_publisher;
    std::unique_ptr<HLIEWorkerControlService> worker_control_service;
    std::unique_ptr<HLIEWorkerControlApplicationService>
        worker_control_application_service;
    // Lifecycle callbacks reference the complete graph; declaring this last
    // guarantees the coordinator is destroyed before every dependency.
    std::unique_ptr<HLIEApplicationLifecycleService>
        application_lifecycle_service;
};

// The only production constructor for the P1 control-plane graph. It wires the
// current HOUMO TCIM implementation; it is not a multi-SDK plugin abstraction.
class HLIEApplicationCompositionRoot {
public:
    [[nodiscard]] static std::optional<HLIEApplicationComponents> compose(
        HLIEApplicationCompositionContext context);
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_APPLICATION_COMPOSITION_ROOT_H

#ifndef HLIE_WORKFLOW_H
#define HLIE_WORKFLOW_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <server/server-models.h>
#include <unordered_map>
#include <utility>
#include <vector>


#include "hlie_base/hlie_base.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_app/control/hlie_model_controller.h"
#include "hlie_app/control/hlie_worker_control_service.h"
#include "hlie_app/http/hlie_http_route_provider.h"
#include "hlie_app/http/hlie_api_router.h"
#include "hlie_app/inference/hlie_default_inference_endpoint_resolver.h"
#include "hlie_app/inference/hlie_inference_dispatcher.h"
#include "hlie_app/resource/hlie_resource_scheduler.h"
#include "hlie_app/runtime/hlie_runtime_directory.h"
#include "hlie_app/runtime/hlie_worker_supervisor.h"
#include "hlie_core/control/hlie_admission_controller.h"
#include "hlie_model_info.h"
#include "hlie_llm_preset_resolver.h"
#include "hlie_model_service.h"
#include "hlie_model_service_manager.h"
#include "httplib.h"

HLIE_NAMESPACE_BEGIN

class HLIEDeviceService;
class HLIEConfigService;
class HLIEControlPlaneReconciler;
class HLIEModelDiscoveryService;
class HLIEModelCatalog;
class HLIEModelLoadService;
class HLIEModelReadinessService;
class HLIEWorkerControlService;
class HLIEWorkerControlApplicationService;
class IHLIEEmbeddedStartupTargetProvider;
class HLIEEmbeddedStartupService;
class HLIEDocumentApplicationService;
class HLIEAsrApplicationService;
class HLIESpeechApplicationService;
class HLIEDefaultModelInitializationService;
class HLIEModelListApplicationService;
class HLIEModelUnloadApplicationService;
class HLIEStopServiceApplicationService;
class HLIEObservabilityApplicationService;
class HLIELlmApplicationService;
class HLIEBootstrapApplicationService;
class HLIEApplicationLifecycleService;

class HLIEWorkflow : public IHLIEHttpRouteProvider,
                     public IHLIEWebServiceStateProvider,
                     public IHLIEAuthPolicy {
public:
    HLIEWorkflow();

    ~HLIEWorkflow();

    int initialize(const std::string& model_dir, const std::string& program_path);

    [[nodiscard]] bool run();

    [[nodiscard]] bool publish_serving(
        const std::function<bool()>& publisher);

    void begin_draining();

    void stop();

    [[nodiscard]] bool is_draining() const override;

    [[nodiscard]] int get_model_service_state() const override;
    [[nodiscard]] int calculate_http_thread_count() const;

    // 路由配置结构
    /**
     * Get the main route configs
     *
     * @return the vector of RouteConfig including the Method,  the Path, the Handler, and the Model Type
     */
    std::vector<HLIEHttpRouteConfig> get_route_configs() override;

    void set_server_port(int port) override;

    [[nodiscard]] bool authentication_enabled() const override {
        return !api_keys_.empty();
    }
    [[nodiscard]] bool authorize_api_key(
        const std::string& api_key) const override;

    void set_process_shutdown_callback(std::function<void()> callback) {
        process_shutdown_callback_ = std::move(callback);
    }

private:
    using ModelInstance = HLIEModelController::ModelInstance;

    server_http_res_ptr start_service(
        const server_http_req& req,
        const std::optional<HLIEModelLaunchTicket>& preclaimed_launch = std::nullopt);

    bool router_validate_model(const std::string& name, const std::unique_ptr<server_res_generator_light>& res);

    server_http_res_ptr dispatch_inference(
        const std::string& model_key,
        HLIEInferenceOperation operation,
        const server_http_req& req,
        const std::string& method = "POST");

    [[nodiscard]] std::optional<std::string> find_ready_model_key(
        const std::string& service_kind,
        std::optional<HLIERuntimeKind> runtime_kind = std::nullopt) const;
    void initialize_api_router();

    server_http_res_ptr handle_model_discovery_refresh(const server_http_req& req);

    std::optional<server_model_meta> get_model_meta(const std::string& name);
    std::optional<ModelInstance> get_model_inst(const std::string& model_name);
    void notify_local_model_started(
        const std::string& notify_model_key,
        bool enable_asr_websocket);
    void schedule_slave_loaded_status(const std::string& model_key);
    [[nodiscard]] std::vector<std::string> get_public_local_model_names();
    [[nodiscard]] std::vector<ModelInstance>
    get_embedded_model_instances() const;
    [[nodiscard]] std::vector<HLIEWorkerProcessorSnapshot>
    get_worker_process_snapshots();
    bool initialize_llm_preset_catalogs();
    bool resolve_llm_launch_params(HLIEModelInfo& model_info);

    HLIEModelServiceManager model_service_manager_;

    std::shared_ptr<HLIEDeviceService> device_service_;
    std::shared_ptr<HLIEModelCatalog> model_catalog_;
    std::unique_ptr<HLIEResourceScheduler> resource_scheduler_;
    HLIEAdmissionController llm_admission_{64};
    HLIEAdmissionController qwen3_tts_admission_{1};

    std::vector<std::string> api_keys_;
    std::string program_path_;
    std::string model_dir_;
    int port_{};
    bool is_slave_{false};
    bool replicas_mode_{false};
    HLIEModelLaunchTicket worker_launch_ticket_;
    std::function<void()> process_shutdown_callback_;
    int master_port_{};
    time_t start_time_{};

    std::mutex mutex_;

    // Controller is the sole Registry owner and must outlive Supervisor event
    // callbacks. Runtime handles are destroyed before Supervisor/controller.
    std::unique_ptr<IHLIEModelRuntimeFactory> runtime_factory_;
    std::unique_ptr<IHLIEModelBackendLauncher> backend_launcher_;
    std::unique_ptr<HLIEModelController> model_controller_;
    std::unique_ptr<HLIEWorkerSupervisor> worker_supervisor_;
    // Runtime handles are destroyed before WorkerSupervisor and service slots.
    // Registry remains the lifecycle source of truth.
    HLIERuntimeDirectory runtime_directory_;
    std::unique_ptr<HLIEDefaultInferenceEndpointResolver> inference_endpoint_resolver_;
    std::unique_ptr<HLIEInferenceDispatcher> inference_dispatcher_;
    std::unique_ptr<HLIEApiRouter> api_router_;
    std::unique_ptr<HLIEControlPlaneReconciler> control_plane_reconciler_;
    std::unique_ptr<HLIEModelDiscoveryService> model_discovery_service_;
    std::unique_ptr<HLIEObservabilityApplicationService>
        observability_application_service_;
    std::unique_ptr<HLIEModelLoadService> model_load_service_;
    std::unique_ptr<HLIEModelReadinessService> model_readiness_service_;
    std::unique_ptr<HLIEDocumentApplicationService>
        document_application_service_;
    std::unique_ptr<HLIELlmApplicationService> llm_application_service_;
    std::unique_ptr<HLIEAsrApplicationService> asr_application_service_;
    std::unique_ptr<HLIESpeechApplicationService>
        speech_application_service_;
    std::unique_ptr<HLIEModelListApplicationService>
        model_list_application_service_;
    std::unique_ptr<HLIEModelUnloadApplicationService>
        model_unload_application_service_;
    std::unique_ptr<HLIEStopServiceApplicationService>
        stop_service_application_service_;
    std::unique_ptr<HLIEDefaultModelInitializationService>
        default_model_initialization_service_;
    std::unique_ptr<HLIEBootstrapApplicationService>
        bootstrap_application_service_;
    std::unique_ptr<IHLIEEmbeddedStartupTargetProvider>
        embedded_startup_target_provider_;
    std::unique_ptr<HLIEEmbeddedStartupService> embedded_startup_service_;
    std::unique_ptr<IHLIEWorkerStatusPublisher> worker_status_publisher_;
    std::unique_ptr<HLIEWorkerControlService> worker_control_service_;
    std::unique_ptr<HLIEWorkerControlApplicationService>
        worker_control_application_service_;
    // Declared after its lifecycle dependencies so it is destroyed first.
    std::unique_ptr<HLIEApplicationLifecycleService>
        application_lifecycle_service_;

    HLIELlmPresetCatalog llm_preset_catalog_;
    HLIELlmPresetCatalog llm_override_catalog_;
    bool llm_presets_enabled_{true};
    bool llm_preset_strict_{true};

};

HLIE_NAMESPACE_END

#endif // HLIE_WORKFLOW_H

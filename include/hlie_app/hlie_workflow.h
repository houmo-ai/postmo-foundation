#ifndef HLIE_WORKFLOW_H
#define HLIE_WORKFLOW_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <server/server-models.h>
#include <thread>
#include <vector>


#include "hlie_base/hlie_base.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_app/util/hlie_background_task_group.h"
#include "hlie_model_info.h"
#include "hlie_model_discovery_sync.h"
#include "hlie_llm_preset_resolver.h"
#include "hlie_model_registry.h"
#include "hlie_model_service.h"
#include "hlie_model_service_manager.h"
#include "hlie_ws_service.h"
#include "httplib.h"

HLIE_NAMESPACE_BEGIN

class HLIEDeviceService;
class LLModelService;
class SDModelService;
class WhisperModelService;
class HLIEConfigService;
class CosyVoiceModelService;
class Qwen3TTSModelService;

class HLIEWorkflow {
public:
    HLIEWorkflow() = default;

    ~HLIEWorkflow() = default;

    int initialize(const std::string& model_dir, const std::string& program_path);

    void run();

    void stop();

    [[nodiscard]] std::shared_ptr<IHLIEModelService> get_model_service(const std::string& type) const;

    [[nodiscard]] int cacl_thread_num() const;
    [[nodiscard]] int get_model_service_state() const;
    [[nodiscard]] std::vector<std::string> get_api_keys() const;

    // 定义处理函数类型
    using OriRequestHandler = std::function<void(const httplib::Request&, httplib::Response&)>;
    using RequestHandler = std::function<server_http_res_ptr(const server_http_req& req)>;

    // 路由配置结构
    struct RouteConfig {
        std::vector<std::string> paths;
        HttpType method; // "GET" or "POST"
        std::function<server_http_res_ptr(const server_http_req& req)> handler = nullptr;
        bool require_model = false; // 是否需要特定模型服务
        std::string model_type = LLM; // 需要的模型类型
        OriRequestHandler ori_handler = nullptr;
    };

    using ModelInstance = HLIEModelRegistry::ModelInstance;

    /**
     * Get the main route configs
     *
     * @return the vector of RouteConfig including the Method,  the Path, the Handler, and the Model Type
     */
    std::vector<RouteConfig> get_route_configs();

    void set_server_port(int port);

private:
    [[nodiscard]] server_http_res_ptr handle_health(const server_http_req& req) const;

    server_http_res_ptr handle_models(const server_http_req& req);

    server_http_res_ptr start_service(const server_http_req& req,
                                      const std::string& preclaimed_model_key = {});

    int start_whisper(std::optional<server_model_meta>& meta, std::unique_ptr<server_res_generator_light>& res, const server_http_req& req,
                       const std::string& model_name, const std::string& model_path, const std::string& devices);

    int start_cosyvoice(std::optional<server_model_meta>& meta, std::unique_ptr<server_res_generator_light>& res, const server_http_req& req,
                       const std::string& model_name, const std::string& model_path, const std::string& devices);

    int start_qwen3_tts(std::optional<server_model_meta>& meta, std::unique_ptr<server_res_generator_light>& res, const server_http_req& req,
                        const std::string& model_name, const std::string& model_path, const std::string& devices);

    bool router_validate_model(const std::string& name, const std::unique_ptr<server_res_generator_light>& res);

    static server_http_res_ptr proxy_request(const server_http_req& req, const std::string& method, const std::string& name, int port);

    server_http_res_ptr route_llm_request(const server_http_req& req, const RequestHandler& handler,
                                          const std::string& method = "POST");

    void handle_speech(const httplib::Request& req, httplib::Response& res);

    server_http_res_ptr handle_transcriptions_pipeline(const server_http_req& req, const json& body);

    server_http_res_ptr stop_service(const server_http_req& req);

    [[nodiscard]] server_http_res_ptr handle_get_status(const server_http_req& req);

    [[nodiscard]] server_http_res_ptr handle_post_status(const server_http_req& req);

    server_http_res_ptr handle_unload_model(const server_http_req& req);

    server_http_res_ptr handle_model_discovery_refresh(const server_http_req& req);

    void ensure_asr_websocket();
    std::string handle_asr_stream(const WhisperStreamRequest& request, const std::function<void(const std::string&)>& callback);

    std::vector<server_model_meta> get_all_model_metas();

    std::optional<server_model_meta> get_model_meta(const std::string& name);
    std::optional<ModelInstance> get_model_inst(const std::string& model_name);
    std::string register_model_instance(HLIEModelInfo& model_info);
    void rollback_model_start(const std::string& model_key, const std::string& reason);
    void mark_local_model_started(const std::string& model_key, const std::string& notify_model_key, bool enable_asr_websocket);
    bool evict_model(const ModelInstance& model_inst, const std::string& reason);
    void start_residency_ttl_worker();
    void stop_residency_ttl_worker();
    void residency_ttl_loop();
    void initialize_model_discovery_snapshot();
    int discover_new_models(bool force = false);
    uint64_t request_model_discovery_scan(bool force = false);
    bool wait_for_model_discovery(uint64_t generation);
    void start_model_discovery_worker();
    void stop_model_discovery_worker();
    void model_discovery_loop();
    bool initialize_llm_preset_catalogs();
    bool resolve_llm_launch_params(HLIEModelInfo& model_info);

    static int generate_instance_id() {
        static std::atomic counter{0};
        return ++counter;
    }

    HLIEModelServiceManager model_service_manager_;
    std::unique_ptr<HLIEWebSocketService> asr_websocket_service_;
    HLIEModelRegistry model_registry_;

    std::shared_ptr<HLIEDeviceService> device_service_;

    std::vector<std::string> api_keys_;
    std::string program_path_;
    std::string model_dir_;
    int port_{};
    bool is_slave_{false};
    bool replicas_mode_{false};
    int master_port_{};
    time_t start_time_{};

    std::mutex mutex_;

    std::condition_variable slave_cv_;
    std::mutex slave_mutex_;

    std::unique_ptr<std::thread> residency_ttl_thread_;
    std::atomic<bool> residency_ttl_stop_{false};

    std::unique_ptr<std::thread> model_discovery_thread_;
    std::atomic<bool> model_discovery_stop_{false};
    std::atomic<bool> model_discovery_requested_{false};
    HLIEModelDiscoverySync model_discovery_sync_;
    std::condition_variable model_discovery_cv_;
    std::mutex model_discovery_wait_mutex_;
    std::mutex model_discovery_mutex_;
    std::string model_discovery_snapshot_;
    bool model_discovery_pending_{false};
    std::chrono::steady_clock::time_point model_discovery_pending_since_{};
    std::chrono::steady_clock::time_point model_discovery_last_request_{};

    HLIELlmPresetCatalog llm_preset_catalog_;
    HLIELlmPresetCatalog llm_override_catalog_;
    bool llm_presets_enabled_{true};
    bool llm_preset_strict_{true};

    // Declared last so its destructor joins callbacks before other workflow state
    // is destroyed.
    BackgroundTaskGroup auto_load_tasks_;
};

HLIE_NAMESPACE_END

#endif // HLIE_WORKFLOW_H

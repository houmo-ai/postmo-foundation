#ifndef HLIE_WORKFLOW_H
#define HLIE_WORKFLOW_H

#include <memory>
#include <server/server-models.h>
#include <vector>


#include "hlie_base/hlie_base.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_model_info.h"
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

    server_http_res_ptr start_service(const server_http_req& req);

    int start_whisper(std::optional<server_model_meta>& meta, std::unique_ptr<server_res_generator_light>& res, const server_http_req& req,
                       const std::string& model_name, const std::string& model_path, const std::string& devices);

    int start_cosyvoice(std::optional<server_model_meta>& meta, std::unique_ptr<server_res_generator_light>& res, const server_http_req& req,
                       const std::string& model_name, const std::string& model_path, const std::string& devices);

    bool router_validate_model(const std::string& name, const std::unique_ptr<server_res_generator_light>& res);

    static server_http_res_ptr proxy_request(const server_http_req& req, const std::string& method, const std::string& name, int port);

    server_http_res_ptr route_llm_request(const server_http_req& req, const RequestHandler& handler,
                                          const std::string& method = "POST");

    server_http_res_ptr handle_transcriptions_pipeline(const server_http_req& req, const json& body);

    server_http_res_ptr stop_service(const server_http_req& req);

    [[nodiscard]] server_http_res_ptr handle_get_status(const server_http_req& req);

    [[nodiscard]] server_http_res_ptr handle_post_status(const server_http_req& req);

    server_http_res_ptr handle_unload_model(const server_http_req& req);

    void ensure_asr_websocket();
    std::string handle_asr_stream(const WhisperStreamRequest& request, const std::function<void(const std::string&)>& callback);

    std::vector<server_model_meta> get_all_model_metas();

    std::optional<server_model_meta> get_model_meta(const std::string& name);
    std::optional<std::reference_wrapper<ModelInstance>> get_model_inst(const std::string& model_name);
    void register_model_instance(HLIEModelInfo& model_info);
    void rollback_model_start(const std::string& model_key, const std::string& reason);
    void mark_local_model_started(const std::string& model_key, const std::string& notify_model_key, bool enable_asr_websocket);
    bool evict_model(const ModelInstance& model_inst, const std::string& reason);
    void start_residency_ttl_worker();
    void stop_residency_ttl_worker();
    void residency_ttl_loop();

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
};

HLIE_NAMESPACE_END

#endif // HLIE_WORKFLOW_H

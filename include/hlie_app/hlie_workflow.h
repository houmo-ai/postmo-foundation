#ifndef HLIE_WORKFLOW_H
#define HLIE_WORKFLOW_H

#include <memory>
#include <vector>
#include <server/server-models.h>


#include "hlie_base/hlie_base.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_model_info.h"
#include "hlie_model_service.h"
#include "httplib.h"

HLIE_NAMESPACE_BEGIN

class HLIEDeviceService;
class LLModelService;
class SDModelService;
class HLIEConfigService;

class HLIEWorkflow {
public:
     HLIEWorkflow() = default;

    ~HLIEWorkflow() = default;

    int initialize(const std::string& model_dir, const std::string& program_path);

    void run();

    void stop() const;

    [[nodiscard]] std::shared_ptr<IHLIEModelService> get_model_service(const std::string& type) const;

    [[nodiscard]] int cacl_thread_num() const;
    [[nodiscard]] int get_model_service_state() const;
    [[nodiscard]] std::vector<std::string> get_api_keys() const;

    // 定义处理函数类型
    using RequestHandler = std::function<void(const httplib::Request&, httplib::Response&)>;

    // 路由配置结构
    struct RouteConfig {
        std::vector<std::string> paths;
        HttpType method;  // "GET" or "POST"
        // RequestHandler handler;
        std::function<server_http_res_ptr(const server_http_req& req)> handler;
        bool require_model = false;  // 是否需要特定模型服务
        std::string model_type = LLM;  // 需要的模型类型
    };

    struct ModelInstance {
        int port;
        server_model_meta meta;
        HLIEModelInfo model_info;
    };

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
    server_http_res_ptr stop_service(const server_http_req& req);
    [[nodiscard]] server_http_res_ptr handle_get_status(const server_http_req& req) const;
    [[nodiscard]] server_http_res_ptr handle_post_status(const server_http_req& req);
    server_http_res_ptr handle_unload_model(const server_http_req& req);

    std::vector<server_model_meta> get_all_model_metas();

    std::optional<server_model_meta> get_model_meta(const std::string& name);

    static int generate_instance_id() {
        static std::atomic counter{0};
        return ++counter;
    }

    std::shared_ptr<LLModelService> llm_model_service_;
    std::shared_ptr<SDModelService> sd_model_service_;
    std::map<std::string, ModelInstance> model_instances_;

    std::shared_ptr<HLIEDeviceService> device_service_;

    std::vector<std::string> api_keys_;
    std::string program_path_;
    std::string model_dir_;
    int port_;
    bool is_slave_{false};
    time_t start_time_;

    std::mutex mutex_;
};

HLIE_NAMESPACE_END

#endif //HLIE_WORKFLOW_H

#ifndef HLIE_WORKFLOW_H
#define HLIE_WORKFLOW_H

#include <memory>
#include <vector>


#include "hlie_base/hlie_base.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_model_service.h"
#include "httplib.h"

HLIE_NAMESPACE_BEGIN

class LLModelService;
class SDModelService;
class HLIEConfigService;

class HLIEWorkflow {
public:
     HLIEWorkflow() = default;

    ~HLIEWorkflow() = default;

    int initialize(const std::string& model_dir);

    void run() const;

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
        RequestHandler handler;
        bool require_model = false;  // 是否需要特定模型服务
        std::string model_type = LLM;  // 需要的模型类型
    };

    /**
     * Get the main route configs
     *
     * @return the vector of RouteConfig including the Method,  the Path, the Handler, and the Model Type
     */
    [[nodiscard]] const std::vector<RouteConfig>& get_route_configs() const;

private:
    void handle_health(httplib::Response& res) const;
    void handle_models(httplib::Response& res) const;

    std::shared_ptr<LLModelService> llm_model_service_;
    std::shared_ptr<SDModelService> sd_model_service_;

    std::vector<std::string> api_keys_;
};

HLIE_NAMESPACE_END

#endif //HLIE_WORKFLOW_H

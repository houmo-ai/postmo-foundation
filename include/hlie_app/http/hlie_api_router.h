#ifndef HLIE_APP_HTTP_API_ROUTER_H
#define HLIE_APP_HTTP_API_ROUTER_H

#include <functional>

#include "hlie_app/http/hlie_http_route_provider.h"
#include "hlie_app/inference/hlie_inference_endpoint.h"

HLIE_NAMESPACE_BEGIN

struct HLIEApiRouteHandlers {
    using Handler = std::function<server_http_res_ptr(const server_http_req&)>;
    using LlmHandler = std::function<server_http_res_ptr(
        const server_http_req&, HLIEInferenceOperation, const std::string&)>;
    using OperationHandler = std::function<server_http_res_ptr(
        const server_http_req&, HLIEInferenceOperation)>;
    using NativeHandler = std::function<void(
        const httplib::Request&, httplib::Response&)>;

    Handler live;
    Handler ready;
    Handler metrics;
    Handler health;
    Handler models;
    Handler refresh_models;
    LlmHandler llm;
    Handler image_generations;
    Handler transcriptions;
    NativeHandler speech;
    Handler devices;
    Handler properties;
    Handler slots;
    Handler start_model;
    Handler stop_model;
    Handler get_status;
    Handler post_status;
    Handler worker_shutdown;
    OperationHandler qwen3_tts_debug;
    Handler unload_model;
};

class HLIEApiRouter final : public IHLIEHttpRouteProvider {
public:
    explicit HLIEApiRouter(HLIEApiRouteHandlers handlers);

    [[nodiscard]] std::vector<HLIEHttpRouteConfig>
    get_route_configs() override;

private:
    HLIEApiRouteHandlers handlers_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_HTTP_API_ROUTER_H

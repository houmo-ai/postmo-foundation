#ifndef HLIE_APP_DEFAULT_INFERENCE_ENDPOINT_RESOLVER_H
#define HLIE_APP_DEFAULT_INFERENCE_ENDPOINT_RESOLVER_H

#include <functional>

#include "hlie_app/inference/hlie_inference_endpoint.h"
#include "hlie_app/util/hlie_proxy_timeout.h"

HLIE_NAMESPACE_BEGIN

class HLIEModelServiceManager;

// Current HOUMO implementation of the backend-neutral endpoint resolver.
// Concrete Service snapshots and Worker HTTP proxy construction stay in the
// .cpp so neither Dispatcher nor Router includes model-specific headers.
class HLIEDefaultInferenceEndpointResolver final
    : public IHLIEInferenceEndpointResolver {
public:
    using ProxyTimeoutProvider = std::function<ProxyTimeoutOptions()>;

    HLIEDefaultInferenceEndpointResolver(
        HLIEModelServiceManager& service_manager,
        ProxyTimeoutProvider proxy_timeout_provider);

    [[nodiscard]] std::shared_ptr<IHLIEInferenceEndpoint> resolve(
        const HLIEInferenceEndpointRequest& request) override;

private:
    HLIEModelServiceManager& service_manager_;
    ProxyTimeoutProvider proxy_timeout_provider_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_DEFAULT_INFERENCE_ENDPOINT_RESOLVER_H

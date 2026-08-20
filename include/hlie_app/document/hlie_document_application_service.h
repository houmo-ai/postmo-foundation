#ifndef HLIE_DOCUMENT_APPLICATION_SERVICE_H
#define HLIE_DOCUMENT_APPLICATION_SERVICE_H

#include <functional>
#include <optional>
#include <string>

#include "hlie_app/inference/hlie_inference_endpoint.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

// Request-boundary orchestration for document inputs. PDF rendering, layout,
// region cropping, and OCR remain owned by DocumentPipelineService/endpoints.
struct HLIEDocumentRouteObservation {
    std::string model_key;
    std::string source;
    std::string filename;
    std::string content_type;
};

struct HLIEDocumentApplicationDependencies {
    using DispatchCallback = std::function<server_http_res_ptr(
        const std::string&, HLIEInferenceOperation, const server_http_req&,
        const std::string&)>;

    DispatchCallback dispatch;
    // The document companion shares the primary LLM runtime lifecycle. The
    // composition root supplies the existing stop/reset operation here.
    std::function<void()> stop_pipeline;
    std::function<void(const HLIEDocumentRouteObservation&)> route_observer;
};

class HLIEDocumentApplicationService final {
public:
    explicit HLIEDocumentApplicationService(
        HLIEDocumentApplicationDependencies dependencies);

    HLIEDocumentApplicationService(const HLIEDocumentApplicationService&) = delete;
    HLIEDocumentApplicationService& operator=(const HLIEDocumentApplicationService&) = delete;

    [[nodiscard]] std::optional<HLIEInferenceOperation> operation_for(
        const server_http_req& request, const nlohmann::json& body) const;

    [[nodiscard]] server_http_res_ptr dispatch_or_route(
        const std::string& model_key, HLIEInferenceOperation operation,
        const server_http_req& request, const nlohmann::json& body,
        const std::string& method = "POST") const;

    void stop_document_pipeline() const;

private:
    HLIEDocumentApplicationDependencies dependencies_;
};

HLIE_NAMESPACE_END

#endif // HLIE_DOCUMENT_APPLICATION_SERVICE_H

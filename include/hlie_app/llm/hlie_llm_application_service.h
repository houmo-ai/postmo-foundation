#ifndef HLIE_APP_LLM_APPLICATION_SERVICE_H
#define HLIE_APP_LLM_APPLICATION_SERVICE_H

#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "hlie_app/inference/hlie_inference_endpoint.h"
#include "hlie_core/control/hlie_admission_controller.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

enum class HLIELlmModelState {
    Other,
    Unloaded,
    Loading,
    Loaded,
};

enum class HLIELlmReadinessOutcome {
    Ready,
    Missing,
    Unavailable,
    InvalidLoadRequest,
    ScheduleFailed,
    TimedOut,
    Cancelled,
    Orphaned,
    Failed,
    Unloaded,
};

struct HLIELlmModelSnapshot {
    std::string key;
    HLIELlmModelState state = HLIELlmModelState::Other;
    int port = 0;
};

struct HLIELlmAutoLoadRequest {
    std::string model;
    std::string model_path;
    std::string mmproj_path;
    std::optional<std::vector<int>> requested_device_ids;
};

struct HLIELlmApplicationDependencies {
    using Dispatch = std::function<server_http_res_ptr(
        const std::string&, HLIEInferenceOperation,
        const server_http_req&, const std::string&)>;
    using DocumentDispatch = std::function<server_http_res_ptr(
        const std::string&, HLIEInferenceOperation,
        const server_http_req&, const nlohmann::json&, const std::string&)>;

    std::function<bool()> draining_provider;
    std::function<std::optional<HLIELlmModelSnapshot>(const std::string&)>
        resolve_model;
    std::function<void()> refresh_discovery;
    std::function<bool()> auto_load_provider;
    std::function<std::optional<std::vector<int>>(const std::string&)>
        parse_devices;
    std::function<HLIELlmReadinessOutcome(
        const std::string&, std::optional<HLIELlmAutoLoadRequest>,
        std::chrono::steady_clock::time_point,
        const std::function<bool()>&)> ensure_ready;
    std::function<int()> local_port_provider;
    std::function<int()> master_port_provider;
    std::function<std::optional<std::string>()> find_ready_embedded_llm;
    Dispatch dispatch;
    DocumentDispatch dispatch_document;
};

// Owns LLM model selection, auto-load compatibility, readiness error mapping,
// admission lifetime, document routing, and /props + /slots behavior.
class HLIELlmApplicationService final {
public:
    HLIELlmApplicationService(
        HLIEAdmissionController& admission,
        HLIELlmApplicationDependencies dependencies);

    [[nodiscard]] server_http_res_ptr handle(
        const server_http_req& request,
        HLIEInferenceOperation operation =
            HLIEInferenceOperation::ChatCompletions,
        const std::string& method = "POST") const;
    // Composite request flows such as ASR own one permit across several LLM
    // dispatches and enter here after admission has already succeeded.
    [[nodiscard]] server_http_res_ptr handle_admitted(
        const server_http_req& request,
        HLIEInferenceOperation operation,
        const std::string& method = "POST") const;
    [[nodiscard]] server_http_res_ptr handle_properties(
        const server_http_req& request) const;
    [[nodiscard]] server_http_res_ptr handle_slots(
        const server_http_req& request) const;

private:
    [[nodiscard]] server_http_res_ptr route(
        const server_http_req& request,
        HLIEInferenceOperation operation,
        const std::string& method) const;
    [[nodiscard]] server_http_res_ptr dispatch(
        const std::string& model_key,
        HLIEInferenceOperation operation,
        const server_http_req& request,
        const std::string& method) const;
    [[nodiscard]] server_http_res_ptr dispatch_document_or_llm(
        const std::string& model_key,
        HLIEInferenceOperation operation,
        const server_http_req& request,
        const nlohmann::json& body,
        const std::string& method) const;

    HLIEAdmissionController& admission_;
    HLIELlmApplicationDependencies dependencies_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_LLM_APPLICATION_SERVICE_H

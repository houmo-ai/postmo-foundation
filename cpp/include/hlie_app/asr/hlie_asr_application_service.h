#ifndef HLIE_ASR_APPLICATION_SERVICE_H
#define HLIE_ASR_APPLICATION_SERVICE_H

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "hlie_app/inference/hlie_inference_endpoint.h"
#include "hlie_core/control/hlie_admission_controller.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

struct WhisperStreamRequest;

using HLIEAsrTranscriptCallback =
    std::function<void(const std::string&)>;
using HLIEAsrRealtimeHandler = std::function<std::string(
    const WhisperStreamRequest&, const HLIEAsrTranscriptCallback&)>;

class IHLIEAsrRealtimeTransport {
public:
    virtual ~IHLIEAsrRealtimeTransport() = default;
    virtual void start(HLIEAsrRealtimeHandler handler, int port) = 0;
    virtual void stop() = 0;
};

struct HLIEAsrModelCandidate {
    std::string key;
    std::string model_name;
};

struct HLIEAsrApplicationDependencies {
    using FindReadyModel = std::function<std::optional<std::string>(
        const std::string&, HLIERuntimeKind)>;
    using ResolveModel = std::function<std::optional<HLIEAsrModelCandidate>(
        const std::string&)>;
    using Dispatch = std::function<server_http_res_ptr(
        const std::string&, HLIEInferenceOperation, const server_http_req&,
        const std::string&)>;
    using RouteLlm = std::function<server_http_res_ptr(
        const server_http_req&, HLIEInferenceOperation)>;
    using DispatchRealtime = std::function<std::string(
        const std::string&, const WhisperStreamRequest&,
        const HLIEAsrTranscriptCallback&)>;

    FindReadyModel find_ready_model;
    ResolveModel resolve_model;
    std::function<std::vector<HLIEAsrModelCandidate>()> model_candidates;
    Dispatch dispatch;
    RouteLlm route_llm;
    DispatchRealtime dispatch_realtime;
    std::function<bool()> realtime_listener_allowed;
    std::function<bool()> realtime_model_available;
    std::function<int()> local_port;
    std::function<std::unique_ptr<IHLIEAsrRealtimeTransport>()>
        realtime_transport_factory;
};

// Owns ASR request orchestration and realtime listener lifecycle. Concrete
// Whisper/LLM services and uWebSockets remain behind injected app adapters.
class HLIEAsrApplicationService final {
public:
    HLIEAsrApplicationService(
        HLIEAdmissionController& admission,
        HLIEAsrApplicationDependencies dependencies);
    ~HLIEAsrApplicationService();

    HLIEAsrApplicationService(const HLIEAsrApplicationService&) = delete;
    HLIEAsrApplicationService& operator=(const HLIEAsrApplicationService&) = delete;

    [[nodiscard]] server_http_res_ptr handle_transcriptions(
        const server_http_req& request);
    [[nodiscard]] std::string handle_realtime(
        const WhisperStreamRequest& request,
        const HLIEAsrTranscriptCallback& callback);

    void ensure_realtime();
    void stop_realtime();

private:
    [[nodiscard]] server_http_res_ptr handle_word_timestamp_pipeline(
        const server_http_req& request, const nlohmann::ordered_json& body);

    HLIEAdmissionController& admission_;
    HLIEAsrApplicationDependencies dependencies_;
    std::mutex realtime_mutex_;
    std::unique_ptr<IHLIEAsrRealtimeTransport> realtime_transport_;
};

HLIE_NAMESPACE_END

#endif // HLIE_ASR_APPLICATION_SERVICE_H

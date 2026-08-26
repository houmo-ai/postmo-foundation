#ifndef HLIE_APP_INFERENCE_ENDPOINT_H
#define HLIE_APP_INFERENCE_ENDPOINT_H

#include <memory>
#include <string>
#include <functional>

#include "hlie_base/hlie_base.h"
#include "hlie_core/runtime/hlie_model_runtime.h"
#include "server/server-http.h"

namespace httplib {
class Request;
class Response;
}

HLIE_NAMESPACE_BEGIN

struct WhisperStreamRequest;

// Stable request-boundary operations. Concrete model services remain behind
// endpoint adapters; this enum is intentionally outside the token/decode loop.
enum class HLIEInferenceOperation {
    ChatCompletions,
    Completions,
    CompletionsOpenAI,
    Responses,
    Embeddings,
    EmbeddingsOpenAI,
    Rerank,
    Properties,
    Slots,
    ImageGenerations,
    LlmTranscription,
    WhisperTranscription,
    DocumentPreprocess,
    DocumentOcr,
    Qwen3TtsPrefillDebug,
    Qwen3TtsCodecFramesDebug,
    Qwen3TtsReferenceDebug,
    Qwen3TtsSpeechDebug,
    Qwen3TtsStatefulDecoderDebug,
    WhisperRealtime,
    Speech,
};

struct HLIEInferenceEndpointRequest {
    HLIERuntimeIdentity identity;
    std::string service_kind;

    [[nodiscard]] bool valid() const noexcept {
        return identity.valid() && !service_kind.empty();
    }
};

// App-layer data-plane endpoint. It is neutral across embedded/process
// backends but intentionally adapts the existing HTTP request/response value
// objects so migration does not copy multipart payloads or add a local hop.
class IHLIEInferenceEndpoint {
public:
    virtual ~IHLIEInferenceEndpoint() = default;

    [[nodiscard]] virtual HLIERuntimeIdentity identity() const = 0;
    [[nodiscard]] virtual bool supports(HLIEInferenceOperation operation) const = 0;
    [[nodiscard]] virtual server_http_res_ptr dispatch(
        HLIEInferenceOperation operation,
        const server_http_req& request,
        const std::string& method) = 0;

    [[nodiscard]] virtual std::string dispatch_realtime_transcription(
        const WhisperStreamRequest&,
        const std::function<void(const std::string&)>&) {
        return {};
    }

    [[nodiscard]] virtual bool dispatch_speech(
        const httplib::Request&,
        httplib::Response&,
        std::shared_ptr<void>) {
        return false;
    }
};

class IHLIEInferenceEndpointResolver {
public:
    virtual ~IHLIEInferenceEndpointResolver() = default;

    [[nodiscard]] virtual std::shared_ptr<IHLIEInferenceEndpoint> resolve(
        const HLIEInferenceEndpointRequest& request) = 0;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_INFERENCE_ENDPOINT_H

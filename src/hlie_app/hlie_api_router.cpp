#include "hlie_app/http/hlie_api_router.h"

#include <utility>

#include "hlie_app/hlie_model_service.h"

HLIE_NAMESPACE_BEGIN

HLIEApiRouter::HLIEApiRouter(HLIEApiRouteHandlers handlers)
    : handlers_(std::move(handlers)) {}

std::vector<HLIEHttpRouteConfig> HLIEApiRouter::get_route_configs() {
    const auto llm = [this](const HLIEInferenceOperation operation,
                            std::string method = "POST") {
        return [this, operation, method = std::move(method)](
                   const server_http_req& request) {
            return handlers_.llm(request, operation, method);
        };
    };
    const auto qwen_debug = [this](const HLIEInferenceOperation operation) {
        return [this, operation](const server_http_req& request) {
            return handlers_.qwen3_tts_debug(request, operation);
        };
    };

    return {
        {{"/live", "/livez"}, GET, handlers_.live, false},
        {{"/ready", "/readyz"}, GET, handlers_.ready, false},
        {{"/metrics", "/hlie/v1/metrics"}, GET, handlers_.metrics, false},
        {{"/health", "/v1/health"}, GET, handlers_.health, false},
        {{"/v1/models", "/models", "/api/tags"}, GET, handlers_.models, false},
        {{"/hlie/v1/models/refresh", "/models/refresh"}, POST,
         handlers_.refresh_models, false},
        {{"/v1/chat/completions", "/chat/completions", "/api/chat"}, POST,
         llm(HLIEInferenceOperation::ChatCompletions), true,
         HLIEModelType::LLM},
        {{"/v1/images/generations", "/images/generations"}, POST,
         handlers_.image_generations, true, HLIEModelType::SD},
        {{"/v1/audio/transcriptions"}, POST, handlers_.transcriptions, false,
         HLIEModelType::WHISPER},
        {{"/v1/audio/speech"}, POST, nullptr, false, HLIEModelType::TTS,
         handlers_.speech},
        {{"/hlie/v1/devices"}, GET, handlers_.devices, false},
        {{"/v1/completions"}, POST,
         llm(HLIEInferenceOperation::CompletionsOpenAI), true,
         HLIEModelType::LLM},
        {{"/completion", "/completions"}, POST,
         llm(HLIEInferenceOperation::Completions), true,
         HLIEModelType::LLM},
        {{"/v1/responses", "/responses"}, POST,
         llm(HLIEInferenceOperation::Responses), true, HLIEModelType::LLM},
        {{"/embedding", "/embeddings"}, POST,
         llm(HLIEInferenceOperation::Embeddings), true, HLIEModelType::LLM},
        {{"/v1/embeddings"}, POST,
         llm(HLIEInferenceOperation::EmbeddingsOpenAI), true,
         HLIEModelType::LLM},
        {{"/rerank", "/reranking", "/v1/rerank", "/v1/reranking"}, POST,
         llm(HLIEInferenceOperation::Rerank), true, HLIEModelType::LLM},
        {{"/props"}, GET, handlers_.properties, true, HLIEModelType::LLM},
        {{"/slots"}, GET, handlers_.slots, true, HLIEModelType::LLM},
        {{"/hlie/v1/start", "/models/load"}, POST, handlers_.start_model,
         false},
        {{"/hlie/v1/stop"}, POST, handlers_.stop_model, false},
        {{"/hlie/v1/status"}, GET, handlers_.get_status, false},
        {{"/models/status"}, POST, handlers_.post_status, false},
        {{"/internal/worker/shutdown"}, POST, handlers_.worker_shutdown,
         false},
        {{"/hlie/v1/qwen3_tts/prefill_debug"}, POST,
         qwen_debug(HLIEInferenceOperation::Qwen3TtsPrefillDebug), true,
         HLIEModelType::QWEN3_TTS},
        {{"/hlie/v1/qwen3_tts/codec_frames_debug"}, POST,
         qwen_debug(HLIEInferenceOperation::Qwen3TtsCodecFramesDebug), true,
         HLIEModelType::QWEN3_TTS},
        {{"/hlie/v1/qwen3_tts/reference_debug"}, POST,
         qwen_debug(HLIEInferenceOperation::Qwen3TtsReferenceDebug), true,
         HLIEModelType::QWEN3_TTS},
        {{"/hlie/v1/qwen3_tts/speech_debug"}, POST,
         qwen_debug(HLIEInferenceOperation::Qwen3TtsSpeechDebug), true,
         HLIEModelType::QWEN3_TTS},
        {{"/hlie/v1/qwen3_tts/stateful_decoder_debug"}, POST,
         qwen_debug(HLIEInferenceOperation::Qwen3TtsStatefulDecoderDebug), true,
         HLIEModelType::QWEN3_TTS},
        {{"/hlie/v1/unload_model", "/models/unload"}, POST,
         handlers_.unload_model, false},
    };
}

HLIE_NAMESPACE_END

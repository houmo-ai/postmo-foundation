#include "hlie_app/inference/hlie_default_inference_endpoint_resolver.h"

#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "hlie_app/document/hlie_document_pipeline_service.h"
#include "hlie_app/hlie_cosyvoice_model_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_model_service_manager.h"
#include "hlie_app/hlie_qwen3_tts_model_service.h"
#include "hlie_app/hlie_sd_model_service.h"
#include "hlie_app/hlie_server_http_proxy.h"
#include "hlie_app/hlie_whisper_model_service.h"
#include "hlie_app/hlie_ws_service.h"
#include "httplib.h"

HLIE_NAMESPACE_BEGIN

namespace {

template <typename ServiceT>
bool ready(const std::shared_ptr<ServiceT>& service) {
    return service && service->get_state() == SERVER_STATE_READY;
}

class EmbeddedInferenceEndpoint final : public IHLIEInferenceEndpoint {
public:
    EmbeddedInferenceEndpoint(
        HLIERuntimeIdentity identity,
        std::shared_ptr<LLModelService> llm,
        std::shared_ptr<SDModelService> sd,
        std::shared_ptr<WhisperModelService> whisper,
        std::shared_ptr<CosyVoiceModelService> cosyvoice,
        std::shared_ptr<Qwen3TTSModelService> qwen3_tts,
        std::shared_ptr<DocumentPipelineService> document)
        : identity_(std::move(identity)), llm_(std::move(llm)),
          sd_(std::move(sd)), whisper_(std::move(whisper)),
          cosyvoice_(std::move(cosyvoice)),
          qwen3_tts_(std::move(qwen3_tts)), document_(std::move(document)) {}

    HLIERuntimeIdentity identity() const override { return identity_; }

    bool supports(const HLIEInferenceOperation operation) const override {
        switch (operation) {
        case HLIEInferenceOperation::ChatCompletions:
        case HLIEInferenceOperation::Completions:
        case HLIEInferenceOperation::CompletionsOpenAI:
        case HLIEInferenceOperation::Responses:
        case HLIEInferenceOperation::Embeddings:
        case HLIEInferenceOperation::EmbeddingsOpenAI:
        case HLIEInferenceOperation::Rerank:
        case HLIEInferenceOperation::Properties:
        case HLIEInferenceOperation::Slots:
        case HLIEInferenceOperation::LlmTranscription:
            return ready(llm_);
        case HLIEInferenceOperation::ImageGenerations:
            return ready(sd_);
        case HLIEInferenceOperation::WhisperTranscription:
            return ready(whisper_);
        case HLIEInferenceOperation::DocumentPreprocess:
            return ready(document_);
        case HLIEInferenceOperation::DocumentOcr:
            return ready(document_) && ready(llm_);
        case HLIEInferenceOperation::Qwen3TtsPrefillDebug:
        case HLIEInferenceOperation::Qwen3TtsCodecFramesDebug:
        case HLIEInferenceOperation::Qwen3TtsReferenceDebug:
        case HLIEInferenceOperation::Qwen3TtsSpeechDebug:
        case HLIEInferenceOperation::Qwen3TtsStatefulDecoderDebug:
            return ready(qwen3_tts_);
        case HLIEInferenceOperation::WhisperRealtime:
            return ready(whisper_);
        case HLIEInferenceOperation::Speech:
            return ready(qwen3_tts_) || ready(cosyvoice_);
        }
        return false;
    }

    server_http_res_ptr dispatch(const HLIEInferenceOperation operation,
                                 const server_http_req& request,
                                 const std::string&) override {
        if (!supports(operation)) {
            return {};
        }
        switch (operation) {
        case HLIEInferenceOperation::ChatCompletions:
            return llm_->handle_chat_completions(request);
        case HLIEInferenceOperation::Completions:
            return llm_->handle_completions(request);
        case HLIEInferenceOperation::CompletionsOpenAI:
            return llm_->handle_completions_oai(request);
        case HLIEInferenceOperation::Responses:
            return llm_->handle_responses(request);
        case HLIEInferenceOperation::Embeddings:
            return llm_->handle_embeddings(request);
        case HLIEInferenceOperation::EmbeddingsOpenAI:
            return llm_->handle_embeddings_oai(request);
        case HLIEInferenceOperation::Rerank:
            return llm_->handle_rerank(request);
        case HLIEInferenceOperation::Properties:
            return llm_->handle_props(request);
        case HLIEInferenceOperation::Slots:
            return llm_->handle_slots(request);
        case HLIEInferenceOperation::LlmTranscription:
            return llm_->handle_transcriptions(request);
        case HLIEInferenceOperation::ImageGenerations:
            return sd_->handle_image_generations(request);
        case HLIEInferenceOperation::WhisperTranscription:
            return whisper_->handle_transcriptions(request);
        case HLIEInferenceOperation::DocumentPreprocess:
            return document_->handle_document_request(request);
        case HLIEInferenceOperation::DocumentOcr:
            return document_->handle_document_ocr_request(request, *llm_);
        case HLIEInferenceOperation::Qwen3TtsPrefillDebug:
            return qwen3_tts_->handle_prefill_debug(request);
        case HLIEInferenceOperation::Qwen3TtsCodecFramesDebug:
            return qwen3_tts_->handle_codec_frames_debug(request);
        case HLIEInferenceOperation::Qwen3TtsReferenceDebug:
            return qwen3_tts_->handle_reference_debug(request);
        case HLIEInferenceOperation::Qwen3TtsSpeechDebug:
            return qwen3_tts_->handle_speech_debug(request);
        case HLIEInferenceOperation::Qwen3TtsStatefulDecoderDebug:
            return qwen3_tts_->handle_stateful_decoder_debug(request);
        case HLIEInferenceOperation::WhisperRealtime:
        case HLIEInferenceOperation::Speech:
            return {};
        }
        return {};
    }

    std::string dispatch_realtime_transcription(
        const WhisperStreamRequest& request,
        const std::function<void(const std::string&)>& callback) override {
        return ready(whisper_) ? whisper_->handle_stream(request, callback)
                               : std::string{};
    }

    bool dispatch_speech(const httplib::Request& request,
                         httplib::Response& response,
                         std::shared_ptr<void> lifetime_guard) override {
        if (ready(qwen3_tts_)) {
            qwen3_tts_->handle_speech(
                request, response, std::move(lifetime_guard));
            return true;
        }
        if (ready(cosyvoice_)) {
            cosyvoice_->handle_speech(request, response);
            return true;
        }
        return false;
    }

private:
    HLIERuntimeIdentity identity_;
    std::shared_ptr<LLModelService> llm_;
    std::shared_ptr<SDModelService> sd_;
    std::shared_ptr<WhisperModelService> whisper_;
    std::shared_ptr<CosyVoiceModelService> cosyvoice_;
    std::shared_ptr<Qwen3TTSModelService> qwen3_tts_;
    std::shared_ptr<DocumentPipelineService> document_;
};

class ProcessInferenceEndpoint final : public IHLIEInferenceEndpoint {
public:
    ProcessInferenceEndpoint(HLIERuntimeIdentity identity,
                             ProxyTimeoutOptions timeout_options)
        : identity_(std::move(identity)),
          timeout_options_(timeout_options.normalized()) {}

    HLIERuntimeIdentity identity() const override { return identity_; }
    bool supports(HLIEInferenceOperation) const override { return true; }

    server_http_res_ptr dispatch(HLIEInferenceOperation,
                                 const server_http_req& request,
                                 const std::string& method) override {
        std::string proxy_path = request.path;
        if (!request.query_string.empty()) {
            proxy_path += "?" + request.query_string;
        }
        return std::make_unique<ServerHttpProxy>(
            method, "127.0.0.1", identity_.endpoint_port, proxy_path,
            request.headers, request.body, request.files, request.should_stop,
            timeout_options_);
    }

    bool dispatch_speech(const httplib::Request& request,
                         httplib::Response& response,
                         std::shared_ptr<void> lifetime_guard) override {
        bool stream = false;
        const auto body = nlohmann::json::parse(request.body, nullptr, false);
        if (body.is_object() && body.contains("stream")) {
            const auto& value = body.at("stream");
            if (value.is_boolean()) {
                stream = value.get<bool>();
            } else if (value.is_string()) {
                const auto text = value.get<std::string>();
                stream = text == "true" || text == "1" || text == "yes";
            }
        }

        std::map<std::string, std::string> headers;
        for (const auto& [key, value] : request.headers) {
            headers[key] = value;
        }
        std::string proxy_path = request.path;
        if (!request.params.empty()) {
            proxy_path += "?";
            bool first = true;
            for (const auto& [key, value] : request.params) {
                if (!first) {
                    proxy_path += "&";
                }
                first = false;
                proxy_path += httplib::encode_query_component(key) + "=" +
                    httplib::encode_query_component(value);
            }
        }

        auto proxy = std::make_shared<ServerHttpProxy>(
            "POST", "127.0.0.1", identity_.endpoint_port, proxy_path, headers,
            request.body, std::map<std::string, uploaded_file>{},
            request.is_connection_closed, timeout_options_);
        response.status = proxy->status;
        for (const auto& [key, value] : proxy->headers) {
            response.set_header(key, value);
        }

        if (!stream) {
            std::string complete_body;
            std::string chunk;
            while (proxy->next(chunk)) {
                complete_body += chunk;
                chunk.clear();
            }
            proxy->cancel();
            response.set_content(std::move(complete_body), proxy->content_type);
            return true;
        }

        response.set_chunked_content_provider(
            proxy->content_type,
            [proxy, lifetime_guard](size_t, httplib::DataSink& sink) {
                (void)lifetime_guard;
                if (!sink.is_writable()) {
                    proxy->cancel();
                    return false;
                }
                std::string chunk;
                const bool has_next = proxy->next(chunk);
                if (!chunk.empty() && !sink.write(chunk.data(), chunk.size())) {
                    proxy->cancel();
                    return false;
                }
                if (!has_next) {
                    if (!sink.is_writable()) {
                        proxy->cancel();
                        return false;
                    }
                    const auto trailers = proxy->response_trailers();
                    if (trailers.empty()) {
                        sink.done();
                    } else {
                        httplib::Headers converted;
                        for (const auto& [key, value] : trailers) {
                            converted.emplace(key, value);
                        }
                        sink.done_with_trailer(converted);
                    }
                    proxy->cancel();
                }
                return true;
            },
            [proxy, lifetime_guard](bool) {
                (void)lifetime_guard;
                proxy->cancel();
            });
        return true;
    }

private:
    HLIERuntimeIdentity identity_;
    ProxyTimeoutOptions timeout_options_;
};

} // namespace

HLIEDefaultInferenceEndpointResolver::HLIEDefaultInferenceEndpointResolver(
    HLIEModelServiceManager& service_manager,
    ProxyTimeoutProvider proxy_timeout_provider)
    : service_manager_(service_manager),
      proxy_timeout_provider_(std::move(proxy_timeout_provider)) {}

std::shared_ptr<IHLIEInferenceEndpoint>
HLIEDefaultInferenceEndpointResolver::resolve(
    const HLIEInferenceEndpointRequest& request) {
    if (!request.valid()) {
        return {};
    }
    if (request.identity.kind == HLIERuntimeKind::Process) {
        const auto options = proxy_timeout_provider_
            ? proxy_timeout_provider_() : ProxyTimeoutOptions{};
        return std::make_shared<ProcessInferenceEndpoint>(
            request.identity, options);
    }

    const auto& kind = request.service_kind;
    const auto& model_key = request.identity.launch_ticket.model_key;
    auto llm = kind == HLIEModelType::LLM ? service_manager_.llm() : nullptr;
    if (llm && llm->get_model_key() != model_key) {
        llm.reset();
    }
    auto sd = kind == HLIEModelType::SD && model_key == HLIEModelType::SD3
        ? service_manager_.sd() : nullptr;
    auto whisper = kind == HLIEModelType::WHISPER
        ? service_manager_.whisper() : nullptr;
    if (whisper && whisper->get_model_key() != model_key) {
        whisper.reset();
    }
    auto cosyvoice = (kind == HLIEModelType::COSYVOICE ||
                      kind == HLIEModelType::TTS)
        ? service_manager_.cosyvoice() : nullptr;
    if (cosyvoice && cosyvoice->get_model_key() != model_key) {
        cosyvoice.reset();
    }
    auto qwen3_tts = kind == HLIEModelType::QWEN3_TTS
        ? service_manager_.qwen3_tts() : nullptr;
    if (qwen3_tts && qwen3_tts->get_model_key() != model_key) {
        qwen3_tts.reset();
    }
    auto document = kind == HLIEModelType::LLM
        ? service_manager_.document_pipeline() : nullptr;
    if (document && document->get_model_key() != model_key) {
        document.reset();
    }

    const bool has_primary = ready(llm) || ready(sd) || ready(whisper) ||
        ready(cosyvoice) || ready(qwen3_tts);
    if (!has_primary) {
        return {};
    }
    return std::make_shared<EmbeddedInferenceEndpoint>(
        request.identity, std::move(llm), std::move(sd), std::move(whisper),
        std::move(cosyvoice), std::move(qwen3_tts), std::move(document));
}

HLIE_NAMESPACE_END

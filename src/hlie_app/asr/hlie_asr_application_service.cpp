#include "hlie_app/asr/hlie_asr_application_service.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <utility>

#include "hlie_app/inference/hlie_admission_guarded_response.h"
#include "hlie_app/hlie_qwen_asr_stream_decoder.h"
#include "hlie_base/hlie_exception.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"
#include "hlie_app/asr/hlie_asr_realtime_contract.h"

HLIE_NAMESPACE_BEGIN

namespace {

using json = nlohmann::ordered_json;
constexpr const char* WHISPER_SERVICE_KIND = "whisper";

std::string normalize_model_name(std::string value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(), [](const unsigned char ch) {
            return std::isspace(ch);
        });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(), [](const unsigned char ch) {
            return std::isspace(ch);
        }).base();
    if (first >= last) {
        return {};
    }
    std::string normalized(first, last);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](const unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return normalized;
}

bool is_whisper_transcription_model(const std::string& model) {
    return model.empty() || HlieUtil::contains_ignore_case(model, "whisper");
}

bool json_boolish(const json& body, const char* key) {
    if (!body.contains(key) || body.at(key).is_null()) {
        return false;
    }
    const auto& value = body.at(key);
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_string()) {
        const auto normalized = normalize_model_name(value.get<std::string>());
        return normalized == "true" || normalized == "1" ||
            normalized == "yes";
    }
    return false;
}

std::string json_string_value(const json& body, const char* key) {
    if (!body.contains(key) || body.at(key).is_null()) {
        return {};
    }
    const auto& value = body.at(key);
    return value.is_string() ? value.get<std::string>() : value.dump();
}

std::string json_string_or_default(
    const json& body, const char* key,
    const std::string& fallback = {}) {
    if (!body.contains(key) || body.at(key).is_null() ||
        !body.at(key).is_string()) {
        return fallback;
    }
    return body.at(key).get<std::string>();
}

bool json_array_or_string_contains(
    const json& body, const char* key, const std::string& expected) {
    if (!body.contains(key) || body.at(key).is_null()) {
        return false;
    }
    const auto& value = body.at(key);
    if (value.is_array()) {
        return std::any_of(value.begin(), value.end(), [&](const auto& item) {
            return item.is_string() &&
                normalize_model_name(item.template get<std::string>()) == expected;
        });
    }
    return value.is_string() &&
        normalize_model_name(value.get<std::string>()) == expected;
}

bool requests_word_timestamps(const json& body) {
    return json_array_or_string_contains(
               body, "timestamp_granularities", "word") ||
        json_array_or_string_contains(
            body, "timestamp_granularities[]", "word");
}

bool is_force_aligner_name(const std::string& name) {
    return HlieUtil::contains_ignore_case(name, "forcealigner") ||
        HlieUtil::contains_ignore_case(name, "forcedaligner") ||
        HlieUtil::contains_ignore_case(name, "force-aligner") ||
        HlieUtil::contains_ignore_case(name, "forced-aligner") ||
        HlieUtil::contains_ignore_case(name, "force aligner");
}

bool is_qwen3_asr_name(const std::string& name) {
    return HlieUtil::contains_ignore_case(name, "qwen3_asr") ||
        HlieUtil::contains_ignore_case(name, "qwen3-asr") ||
        HlieUtil::contains_ignore_case(name, "qwen3asr");
}

struct AsrResponse final : server_http_res {
    void set_error(const json& error) {
        status = error.value("code", 500);
        data = json{{"error", error}}.dump();
    }

    void set_ok(const json& value) {
        status = 200;
        data = value.dump();
    }
};

json error_payload(const std::string& message, const char* type,
                   const int code,
                   const std::optional<int> ret = std::nullopt) {
    json error = {
        {"code", code},
        {"message", message},
        {"type", type},
    };
    if (ret) {
        error["ret"] = *ret;
    }
    return error;
}

server_http_res_ptr error_response(
    const std::string& message, const char* type, const int code,
    const std::optional<int> ret = std::nullopt) {
    auto response = std::make_unique<AsrResponse>();
    response->set_error(error_payload(message, type, code, ret));
    return response;
}

std::string trim_copy(const std::string& value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(), [](const unsigned char ch) {
            return std::isspace(ch);
        });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(), [](const unsigned char ch) {
            return std::isspace(ch);
        }).base();
    return first >= last ? std::string{} : std::string(first, last);
}

std::string parse_asr_text_response(const std::string& data) {
    const auto body = json::parse(data, nullptr, false);
    if (body.is_discarded()) {
        return data;
    }
    if (body.contains("text") && body["text"].is_string()) {
        return body["text"].get<std::string>();
    }
    if (body.contains("error")) {
        return "HLIECpp-FAILED-888";
    }
    return body.dump();
}

std::string clean_asr_pipeline_text(std::string text) {
    if (const auto pos = text.find("<asr_text>"); pos != std::string::npos) {
        text = text.substr(pos + std::string("<asr_text>").size());
    }
    if (const auto pos = text.find("</asr_text>"); pos != std::string::npos) {
        text = text.substr(0, pos);
    }
    return trim_copy(text);
}

std::string read_http_response_body(const server_http_res_ptr& response) {
    if (!response) {
        return {};
    }
    if (!response->is_stream()) {
        return response->data;
    }
    std::string data;
    std::string chunk;
    while (response->next(chunk)) {
        data += chunk;
        chunk.clear();
    }
    return data;
}

void normalize_openai_timestamp_granularities(json& body) {
    if (!body.contains("timestamp_granularities") &&
        body.contains("timestamp_granularities[]")) {
        body["timestamp_granularities"] = body["timestamp_granularities[]"];
    }
}

void append_le_u16(raw_buffer& out, const uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
}

void append_le_u32(raw_buffer& out, const uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
}

raw_buffer pcmf32_to_wav_bytes(
    const std::vector<float>& pcmf32, const int sample_rate) {
    constexpr uint16_t channels = 1;
    constexpr uint16_t bits_per_sample = 16;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t byte_rate = static_cast<uint32_t>(std::max(sample_rate, 1)) * block_align;
    const uint32_t data_size = static_cast<uint32_t>(pcmf32.size() * sizeof(int16_t));

    raw_buffer out;
    out.reserve(44 + data_size);
    out.insert(out.end(), {'R', 'I', 'F', 'F'});
    append_le_u32(out, 36 + data_size);
    out.insert(out.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    append_le_u32(out, 16);
    append_le_u16(out, 1);
    append_le_u16(out, channels);
    append_le_u32(out, static_cast<uint32_t>(std::max(sample_rate, 1)));
    append_le_u32(out, byte_rate);
    append_le_u16(out, block_align);
    append_le_u16(out, bits_per_sample);
    out.insert(out.end(), {'d', 'a', 't', 'a'});
    append_le_u32(out, data_size);
    for (const float sample : pcmf32) {
        const auto clipped = std::clamp(sample, -1.0f, 1.0f);
        append_le_u16(out, static_cast<uint16_t>(
            static_cast<int16_t>(clipped * 32767.0f)));
    }
    return out;
}

server_http_res_ptr unavailable_response(const std::string& message) {
    return error_response(message, "unavailable_error", 503, 80004);
}

} // namespace

HLIEAsrApplicationService::HLIEAsrApplicationService(
    HLIEAdmissionController& admission,
    HLIEAsrApplicationDependencies dependencies)
    : admission_(admission), dependencies_(std::move(dependencies)) {}

HLIEAsrApplicationService::~HLIEAsrApplicationService() {
    stop_realtime();
}

server_http_res_ptr HLIEAsrApplicationService::handle_transcriptions(
    const server_http_req& request) {
    const json body = json::parse(request.body);
    const std::string model_name = json_string_or_default(body, "model");
    const bool force_align_direct = json_boolish(body, "asr_force_align") ||
        json_boolish(body, "force_align");
    const bool pipeline_requested = requests_word_timestamps(body) &&
        !force_align_direct &&
        (!is_whisper_transcription_model(model_name) ||
         body.contains("force_aligner_model") ||
         body.contains("aligner_model") ||
         body.contains("timestamp_model"));
    if (pipeline_requested) {
        return handle_word_timestamp_pipeline(request, body);
    }

    if (model_name.empty() ||
        HlieUtil::contains_ignore_case(model_name, WHISPER_SERVICE_KIND)) {
        const auto model_key = dependencies_.find_ready_model
            ? dependencies_.find_ready_model(WHISPER_SERVICE_KIND,
                                              HLIERuntimeKind::Embedded)
            : std::nullopt;
        if (!model_key || !dependencies_.dispatch) {
            return unavailable_response(
                "Whisper model is not available for now.");
        }
        return dependencies_.dispatch(
            *model_key, HLIEInferenceOperation::WhisperTranscription,
            request, "POST");
    }

    auto permit = admission_.try_acquire();
    if (!permit) {
        throw InferenceQueueFullException("LLM admission limit reached");
    }
    if (!dependencies_.route_llm) {
        return unavailable_response("LLM transcription route is unavailable");
    }
    auto response = dependencies_.route_llm(
        request, HLIEInferenceOperation::LlmTranscription);
    return attach_admission_permit(std::move(response), permit);
}

server_http_res_ptr HLIEAsrApplicationService::handle_word_timestamp_pipeline(
    const server_http_req& request, const nlohmann::ordered_json& body) {
    auto response = std::make_unique<AsrResponse>();
    auto permit = admission_.try_acquire();
    if (!permit) {
        response->status = 429;
        response->data = json{{"error", error_payload(
            "LLM admission limit reached", "rate_limit_error", 429, 80008)}}.dump();
        return response;
    }

    if (!requests_word_timestamps(body)) {
        return error_response(
            "ASR pipeline requires timestamp_granularities[]=word",
            "invalid_request_error", 400);
    }
    if (json_string_value(body, "response_format") != "verbose_json") {
        return error_response(
            "timestamp_granularities requires response_format to be verbose_json",
            "invalid_request_error", 400);
    }
    if (request.files.find("file") == request.files.end()) {
        return error_response(
            "No input file found for transcription", "invalid_request_error", 400);
    }

    const std::string requested_model = json_string_value(body, "model");
    std::string asr_model;
    if (!requested_model.empty() && !is_force_aligner_name(requested_model) &&
        !is_whisper_transcription_model(requested_model) &&
        dependencies_.resolve_model && dependencies_.resolve_model(requested_model)) {
        asr_model = requested_model;
    }
    const auto candidates = dependencies_.model_candidates
        ? dependencies_.model_candidates() : std::vector<HLIEAsrModelCandidate>{};
    if (asr_model.empty()) {
        for (const auto& candidate : candidates) {
            if (is_qwen3_asr_name(candidate.key) ||
                is_qwen3_asr_name(candidate.model_name)) {
                asr_model = candidate.key;
                break;
            }
        }
    }

    std::string aligner_model = json_string_value(body, "force_aligner_model");
    if (aligner_model.empty()) {
        aligner_model = json_string_value(body, "aligner_model");
    }
    if (aligner_model.empty()) {
        aligner_model = json_string_value(body, "timestamp_model");
    }
    if (aligner_model.empty()) {
        for (const auto& candidate : candidates) {
            if (is_force_aligner_name(candidate.key) ||
                is_force_aligner_name(candidate.model_name)) {
                aligner_model = candidate.key;
                break;
            }
        }
    }

    if (asr_model.empty()) {
        return error_response(
            "ASR model is missing; pass a Qwen3 ASR model in the model field for word timestamp transcription",
            "invalid_request_error", 400);
    }
    if (aligner_model.empty()) {
        return error_response(
            "ForceAligner model is missing; pass aligner_model or load a Qwen3 ForceAligner model",
            "invalid_request_error", 400);
    }

    LOG_DEBUG("ASR word timestamp pipeline selected: asr_model={}, aligner_model={}, file_count={}",
              json(asr_model).dump(), json(aligner_model).dump(), request.files.size());
    if (!dependencies_.route_llm) {
        return unavailable_response("ASR transcription route is unavailable");
    }

    json asr_body = body;
    asr_body["model"] = asr_model;
    asr_body["response_format"] = "json";
    asr_body["stream"] = "false";
    for (const auto* field : {"timestamp_granularities", "timestamp_granularities[]",
                              "asr_model", "transcription_model", "force_aligner_model",
                              "aligner_model", "timestamp_model", "asr_force_align",
                              "force_align", "text"}) {
        asr_body.erase(field);
    }
    server_http_req asr_request{
        request.params, request.headers, request.path, request.query_string,
        asr_body.dump(), request.files, request.should_stop};
    const auto asr_response = dependencies_.route_llm(
        asr_request, HLIEInferenceOperation::LlmTranscription);
    if (!asr_response) {
        return error_response(
            "ASR transcription returned no response", "server_error", 500);
    }
    const std::string asr_data = read_http_response_body(asr_response);
    if (asr_response->status < 200 || asr_response->status >= 300) {
        LOG_ERROR("ASR pipeline transcription failed: status={}, body_bytes={}",
                  asr_response->status, asr_data.size());
        auto failed = std::make_unique<server_http_res>();
        failed->status = asr_response->status;
        failed->content_type = asr_response->content_type;
        failed->data = asr_data;
        return failed;
    }

    const std::string transcript = clean_asr_pipeline_text(
        parse_asr_text_response(asr_data));
    if (transcript.empty() || transcript == "HLIECpp-FAILED-888") {
        return error_response(
            "ASR transcription did not return text", "server_error", 500);
    }

    json align_body = body;
    normalize_openai_timestamp_granularities(align_body);
    align_body["model"] = aligner_model;
    align_body["response_format"] = "verbose_json";
    align_body["stream"] = "false";
    align_body["asr_force_align"] = "true";
    align_body["text"] = transcript;
    if (!align_body.contains("timestamp_granularities")) {
        align_body["timestamp_granularities"] = json::array({"word"});
    }
    for (const auto* field : {"timestamp_granularities[]", "asr_model",
                              "transcription_model", "force_aligner_model",
                              "aligner_model", "timestamp_model"}) {
        align_body.erase(field);
    }
    server_http_req align_request{
        request.params, request.headers, request.path, request.query_string,
        align_body.dump(), request.files, request.should_stop};
    const auto align_response = dependencies_.route_llm(
        align_request, HLIEInferenceOperation::LlmTranscription);
    if (!align_response) {
        return error_response(
            "ForceAligner returned no response", "server_error", 500);
    }
    const std::string align_data = read_http_response_body(align_response);
    if (align_response->status < 200 || align_response->status >= 300) {
        LOG_ERROR("ASR pipeline force alignment failed: status={}, body_bytes={}",
                  align_response->status, align_data.size());
        auto failed = std::make_unique<server_http_res>();
        failed->status = align_response->status;
        failed->content_type = align_response->content_type;
        failed->data = align_data;
        return failed;
    }

    json final_body = json::parse(align_data, nullptr, false);
    if (final_body.is_discarded() || !final_body.is_object()) {
        return error_response(
            "ForceAligner returned invalid JSON", "server_error", 500);
    }
    if (final_body.contains("error")) {
        auto failed = std::make_unique<server_http_res>();
        failed->status = 500;
        failed->data = final_body.dump();
        return failed;
    }
    final_body["text"] = transcript;
    if (!final_body.contains("task")) {
        final_body["task"] = "transcribe";
    }
    if (body.contains("language") && body.at("language").is_string() &&
        !final_body.contains("language")) {
        final_body["language"] = body.at("language");
    }
    response->set_ok(final_body);
    return response;
}

std::string HLIEAsrApplicationService::handle_realtime(
    const WhisperStreamRequest& request,
    const HLIEAsrTranscriptCallback& callback) {
    const std::string model = request.model.empty() ? "whisper-1" : request.model;
    if (is_whisper_transcription_model(request.model)) {
        const auto model_key = dependencies_.find_ready_model
            ? dependencies_.find_ready_model(WHISPER_SERVICE_KIND,
                                              HLIERuntimeKind::Embedded)
            : std::nullopt;
        if (!model_key || !dependencies_.dispatch_realtime) {
            LOG_WARN("ASR websocket Whisper target is not ready");
            return "HLIECpp-FAILED-888";
        }
        LOG_DEBUG("ASR websocket route: backend=whisper, model={}, samples={}",
                  model, request.pcmf32.size());
        const auto transcript = dependencies_.dispatch_realtime(
            *model_key, request, callback);
        return transcript.empty() ? "HLIECpp-FAILED-888" : transcript;
    }

    LOG_DEBUG("ASR websocket route: backend=llm, model={}, samples={}, sample_rate={}",
              model, request.pcmf32.size(), request.sample_rate);
    bool filter_qwen_markers = is_qwen3_asr_name(model);
    if (!filter_qwen_markers && dependencies_.resolve_model) {
        if (const auto candidate = dependencies_.resolve_model(model)) {
            filter_qwen_markers = is_qwen3_asr_name(candidate->key) ||
                is_qwen3_asr_name(candidate->model_name);
        }
    }

    const raw_buffer wav = pcmf32_to_wav_bytes(request.pcmf32, request.sample_rate);
    json request_body{
        {"model", model},
        {"response_format", request.response_format.empty()
             ? "json" : request.response_format},
        {"stream", std::string(LLAMA_TRANSCRIPTION_SDK_STREAM_ENABLED)},
    };
    if (!request.prompt.empty()) {
        request_body["prompt"] = request.prompt;
    }
    if (!request.language.empty()) {
        request_body["language"] = request.language;
    }
    const std::function<bool()> should_stop = request.should_stop
        ? request.should_stop : std::function<bool()>([] { return false; });
    server_http_req transcription_request{
        {}, {}, "/v1/audio/transcriptions", {}, request_body.dump(),
        {{"file", uploaded_file{wav, "audio.wav", "audio/wav"}}},
        should_stop};

    auto permit = admission_.try_acquire();
    if (!permit) {
        LOG_WARN("ASR websocket LLM admission rejected: model={}", model);
        return "HLIECpp-FAILED-888";
    }
    if (!dependencies_.route_llm) {
        LOG_ERROR("ASR websocket LLM route is unavailable");
        return "HLIECpp-FAILED-888";
    }
    const auto response = dependencies_.route_llm(
        transcription_request, HLIEInferenceOperation::LlmTranscription);
    if (!response) {
        LOG_ERROR("ASR websocket LLM transcription returned null response");
        return "HLIECpp-FAILED-888";
    }
    if (response->status < 200 || response->status >= 300) {
        LOG_ERROR("ASR websocket LLM transcription failed: status={}, body_bytes={}",
                  response->status, response->data.size());
        return "HLIECpp-FAILED-888";
    }
    if (!response->is_stream()) {
        LOG_ERROR("ASR websocket backend ignored stream=true: model={}, content_type={}",
                  model, response->content_type);
        return "HLIECpp-FAILED-888";
    }

    const bool sse_transport =
        response->content_type.find("text/event-stream") != std::string::npos;
    HLIEQwenAsrStreamDecoder decoder(
        sse_transport, callback, filter_qwen_markers);
    std::string chunk;
    while (true) {
        const bool has_next = response->next(chunk);
        if (!chunk.empty()) {
            decoder.push(chunk);
        }
        chunk.clear();
        if (!has_next) {
            break;
        }
    }
    const auto transcript = decoder.finish();
    if (!decoder.saw_stream_delta()) {
        LOG_ERROR("ASR websocket stream returned no transcript delta: model={}, content_type={}",
                  model, response->content_type);
        return "HLIECpp-FAILED-888";
    }
    if (decoder.final_text_mismatch()) {
        LOG_WARN("ASR websocket final text differs from emitted deltas: model={}, chars={}",
                 model, transcript.size());
    }
    LOG_DEBUG("ASR websocket LLM transcription completed: chars={}", transcript.size());
    return transcript == "HLIECpp-FAILED-888"
        ? "HLIECpp-FAILED-888" : transcript;
}

void HLIEAsrApplicationService::ensure_realtime() {
    if (dependencies_.realtime_listener_allowed &&
        !dependencies_.realtime_listener_allowed()) {
        return;
    }
    if (!dependencies_.realtime_model_available ||
        !dependencies_.realtime_model_available() ||
        !dependencies_.realtime_transport_factory) {
        return;
    }

    std::lock_guard lock(realtime_mutex_);
    if (realtime_transport_) {
        return;
    }
    auto transport = dependencies_.realtime_transport_factory();
    if (!transport) {
        LOG_ERROR("Failed to create ASR realtime transport");
        return;
    }
    const int port = dependencies_.local_port ? dependencies_.local_port() : 0;
    const int websocket_port = port >= 1 && port <= 65437 ? port + 98 : 7999;
    transport->start(
        [this](const WhisperStreamRequest& request,
               const HLIEAsrTranscriptCallback& callback) {
            return handle_realtime(request, callback);
        },
        websocket_port);
    realtime_transport_ = std::move(transport);
    LOG_INFO("ASR websocket router started on port {}", websocket_port);
}

void HLIEAsrApplicationService::stop_realtime() {
    std::unique_ptr<IHLIEAsrRealtimeTransport> transport;
    {
        std::lock_guard lock(realtime_mutex_);
        transport = std::move(realtime_transport_);
    }
    if (transport) {
        transport->stop();
        LOG_INFO("ASR websocket router stopped");
    }
}

HLIE_NAMESPACE_END

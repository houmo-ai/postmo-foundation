#include "hlie_app/hlie_qwen3_tts_model_service.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

#include "hlie_app/hlie_device_service.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_app/models/hlie_qwen3_tts_model.h"
#include "hlie_base/hlie_audio.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

#ifdef WIN32
#include <windows.h>
#endif

HLIE_NAMESPACE_BEGIN

namespace {
constexpr const char* kDefaultCloneReferenceText =
    "希望你以后能够做的比我还好呦。";
constexpr const char* kDefaultCloneReferenceFile = "zero_shot_prompt.wav";

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](const unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

struct WavHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size = 0;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1;
    uint16_t num_channels = 1;
    uint32_t sample_rate = 0;
    uint32_t byte_rate = 0;
    uint16_t block_align = 2;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size = 0;
};
static_assert(sizeof(WavHeader) == 44, "PCM WAV header must be 44 bytes");

std::string pcm16_wav(const std::vector<float>& audio, const int sample_rate) {
    if (audio.size() > (std::numeric_limits<uint32_t>::max() - 36U) / sizeof(int16_t)) {
        throw std::overflow_error("Qwen3-TTS audio is too large for a WAV response");
    }
    WavHeader header;
    header.sample_rate = static_cast<uint32_t>(sample_rate);
    header.byte_rate = header.sample_rate * header.block_align;
    header.data_size = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
    header.chunk_size = 36U + header.data_size;
    std::string bytes(sizeof(header) + header.data_size, '\0');
    std::memcpy(bytes.data(), &header, sizeof(header));
    for (size_t index = 0; index < audio.size(); ++index) {
        const int16_t pcm = static_cast<int16_t>(std::clamp(
            audio[index] * 32767.0F, -32768.0F, 32767.0F));
        std::memcpy(bytes.data() + sizeof(header) + index * sizeof(pcm),
                    &pcm, sizeof(pcm));
    }
    return bytes;
}

std::string pcm16_bytes(const std::vector<float>& audio) {
    std::string bytes(audio.size() * sizeof(int16_t), '\0');
    for (size_t index = 0; index < audio.size(); ++index) {
        const int16_t pcm = static_cast<int16_t>(std::clamp(
            audio[index] * 32767.0F, -32768.0F, 32767.0F));
        std::memcpy(bytes.data() + index * sizeof(pcm), &pcm, sizeof(pcm));
    }
    return bytes;
}

std::string default_clone_reference_path() {
    const std::string working_directory_path = HlieUtil::join_paths(
        "samples", kDefaultCloneReferenceFile);
#ifdef WIN32
    std::array<char, MAX_PATH> executable_path{};
    const DWORD length = GetModuleFileNameA(
        nullptr, executable_path.data(), static_cast<DWORD>(executable_path.size()));
    if (length > 0 && length < executable_path.size()) {
        const std::filesystem::path executable(
            std::string(executable_path.data(), length));
        const std::string packaged_path = HlieUtil::join_paths(
            executable.parent_path().string(), "..", "samples",
            kDefaultCloneReferenceFile);
        std::error_code ec;
        if (std::filesystem::exists(packaged_path, ec)) {
            return packaged_path;
        }
    }
#endif
    return working_directory_path;
}

std::string safe_trailer_value(std::string value) {
    std::replace(value.begin(), value.end(), '\r', ' ');
    std::replace(value.begin(), value.end(), '\n', ' ');
    if (value.size() > 512) {
        value.resize(512);
    }
    return value;
}

std::string precise_float(const double value) {
    std::ostringstream stream;
    stream << std::setprecision(10) << value;
    return stream.str();
}

bool load_qwen3_reference_audio(const std::string& path, const int sample_rate,
                                std::vector<float>& pcm, std::string& error) {
    if (path.empty()) {
        error = "prompt_wav_path must not be empty";
        return false;
    }
    HLIEAudio audio;
    if (!audio.LoadAudio(path, sample_rate)) {
        error = "failed to load reference audio: " + path;
        return false;
    }
    pcm = audio.GetPcmData();
    if (pcm.empty()) {
        error = "reference audio contains no samples";
        return false;
    }
    return true;
}

void set_oai_error(httplib::Response& response, const std::string& message,
                   const error_type type) {
    const auto error = format_error_response_ex(message, type);
    response.status = json_value(error, "code", 500);
    response.set_content(safe_json_to_str({{"error", error}}), MIMETYPE_JSON);
}
}

Qwen3TTSModelService::~Qwen3TTSModelService() {
    stop();
}

int Qwen3TTSModelService::init(const std::string& model_dir,
                               const HLIEModelInfo& model_info,
                               const std::shared_ptr<HLIEDeviceService>& device_service) {
    std::lock_guard lock(model_mutex_);
    state_.store(SERVER_STATE_LOADING_MODEL);
    stopped_ = false;
    device_service_ = device_service;
    model_path_ = model_info.get_model_path();
    if (HlieUtil::is_relative_path(model_path_)) {
        model_path_ = HlieUtil::join_paths(model_dir, model_path_);
    }
    model_key_ = model_info.model_key_.empty()
                     ? HLIEModelInfo::build_model_key(model_info)
                     : model_info.model_key_;

    if (model_path_.empty() || !device_service_) {
        LOG_ERROR("Qwen3TTSModelService requires a model path and device service");
        return -1;
    }

    const int required_devices = model_info.get_device_num();
    const int required_mem_mb = model_info.get_estimated_mem();
    reservation_id_ = model_info.get_device_reservation_id();
    reservation_owned_externally_ = reservation_id_ != 0;
    std::string device_spec = model_info.get_devices();
    if (!device_spec.empty()) {
        if (!HlieUtil::is_valid_devices(device_spec) ||
            HlieUtil::parse_device(device_spec, devices_) < 0 || devices_.empty()) {
            LOG_ERROR("Qwen3TTSModelService requires valid devices: {}", device_spec);
            return -1;
        }
        if (devices_.size() != static_cast<size_t>(required_devices)) {
            LOG_ERROR("Qwen3-TTS model {} requires {} devices, but {} were configured",
                      model_info.get_model_name(), required_devices, devices_.size());
            devices_.clear();
            return -1;
        }
        if (reservation_id_ != 0) {
            if (!device_service_->reservation_matches(reservation_id_, devices_, required_mem_mb, model_key_)) {
                LOG_ERROR("Qwen3-TTS reservation {} does not match model/devices", reservation_id_);
                devices_.clear();
                return -1;
            }
        } else {
            const auto reservation = device_service_->reserve_mem_on_devices(
                devices_, required_mem_mb, false, model_key_);
            if (!reservation) {
                devices_.clear();
                return -1;
            }
            reservation_id_ = reservation->id;
        }
    } else {
        const auto reservation = device_service_->reserve_mem_on_device_count(
            required_devices, required_mem_mb, model_key_);
        if (!reservation) {
            return -1;
        }
        reservation_id_ = reservation->id;
        devices_ = reservation->devices;
    }
    acquired_mem_mb_ = required_mem_mb;

    model_ = std::make_shared<HLIEQwen3TTSModel>();
    model_->set_devices(devices_);
    if (model_->load_model(model_path_) < 0 || !model_->ready()) {
        LOG_ERROR("Qwen3TTSModelService failed to load {}", model_path_);
        model_.reset();
        if (!reservation_owned_externally_) {
            (void)device_service_->release_reservation(reservation_id_);
        }
        reservation_id_ = 0;
        reservation_owned_externally_ = false;
        acquired_mem_mb_ = 0;
        devices_.clear();
        return -1;
    }
    if (reservation_id_ == 0 ||
        (!reservation_owned_externally_ &&
         !device_service_->commit_reservation(reservation_id_))) {
        LOG_ERROR("Qwen3TTSModelService failed to commit reservation {}", reservation_id_);
        model_.reset();
        if (reservation_id_ != 0 && !reservation_owned_externally_) {
            (void)device_service_->release_reservation(reservation_id_);
        }
        reservation_id_ = 0;
        reservation_owned_externally_ = false;
        acquired_mem_mb_ = 0;
        devices_.clear();
        return -1;
    }
    LOG_INFO("Qwen3TTSModelService loaded model {} on devices {}",
             model_key_, HlieUtil::join_devices(devices_));
    return 0;
}

void Qwen3TTSModelService::run() {
    std::lock_guard lock(model_mutex_);
    if (!model_ || !model_->ready()) {
        LOG_ERROR("Qwen3TTSModelService cannot run before model initialization");
        return;
    }
    stopped_ = false;
    state_.store(SERVER_STATE_READY);
}

void Qwen3TTSModelService::process() {
    // Synthesis is intentionally not routed until the codec generation path is complete.
}

void Qwen3TTSModelService::stop() {
    std::lock_guard lock(model_mutex_);
    if (stopped_) {
        return;
    }
    stopped_ = true;
    state_.store(SERVER_STATE_LOADING_MODEL);
    model_.reset();
    if (device_service_ && reservation_id_ != 0 &&
        !reservation_owned_externally_) {
        LOG_INFO("Qwen3TTSModelService release devices mem: devices={}, mem={}MB",
                 HlieUtil::join_devices(devices_), acquired_mem_mb_);
        (void)device_service_->release_reservation(reservation_id_);
    }
    reservation_id_ = 0;
    reservation_owned_externally_ = false;
    devices_.clear();
    acquired_mem_mb_ = 0;
}

int Qwen3TTSModelService::get_state() {
    return state_.load();
}

server_http_res_ptr Qwen3TTSModelService::handle_prefill_debug(const server_http_req& req) {
    std::lock_guard lock(model_mutex_);
    auto res = std::make_unique<server_res_generator_light>();
    if (!model_ || !model_->ready() || get_state() != SERVER_STATE_READY) {
        res->error(format_error_response_ex("Qwen3-TTS model is not ready",
                                            ERROR_TYPE_UNAVAILABLE));
        return res;
    }

    const auto body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        res->error(format_error_response_ex("request body must be a JSON object",
                                            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    const std::string text = body.value(
        "input", body.value("text", std::string()));
    const std::string language = body.value("language", "Chinese");
    const std::string speaker = body.value("speaker", "vivian");

    Qwen3TTSPrefillDebugResult result;
    std::string error;
    if (!model_->debug_talker_prefill(text, language, speaker, result, error)) {
        res->error(format_error_response_ex(error, ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    res->ok({
        {"object", "qwen3_tts.prefill_debug"},
        {"model_size", model_->config().model_size},
        {"language", language},
        {"speaker", speaker},
        {"input_ids", result.input_ids},
        {"prompt_length", result.prompt_length},
        {"prompt_sum", result.prompt_sum},
        {"prompt_abs_mean", result.prompt_abs_mean},
        {"prefill_chunks", result.prefill_chunks},
        {"logits_shape", result.logits_shape},
        {"first_codec_token", result.first_codec_token},
        {"first_codec_logit", result.first_codec_logit},
        {"last_logits_head", result.last_logits_head},
    });
    return res;
}

server_http_res_ptr Qwen3TTSModelService::handle_codec_frames_debug(const server_http_req& req) {
    std::lock_guard lock(model_mutex_);
    auto res = std::make_unique<server_res_generator_light>();
    if (!model_ || !model_->ready() || get_state() != SERVER_STATE_READY) {
        res->error(format_error_response_ex("Qwen3-TTS model is not ready",
                                            ERROR_TYPE_UNAVAILABLE));
        return res;
    }

    const auto body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        res->error(format_error_response_ex("request body must be a JSON object",
                                            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    const std::string text = body.value(
        "input", body.value("text", std::string()));
    const std::string language = body.value("language", "Chinese");
    const std::string speaker = body.value("speaker", "vivian");
    const int max_frames = body.value("max_frames", 3);

    Qwen3TTSCodecFramesDebugResult result;
    std::string error;
    bool success = false;
    if (model_->is_base()) {
        const std::string reference_text = body.value(
            "prompt_text", body.value("reference_text", std::string()));
        const std::string reference_path = body.value(
            "prompt_wav_path", body.value("reference_audio_path", std::string()));
        std::vector<float> reference_pcm;
        success = load_qwen3_reference_audio(
            reference_path, model_->config().reference_sample_rate,
            reference_pcm, error) &&
            model_->debug_generate_clone_codec_frames(
                text, language, reference_text, reference_pcm, max_frames,
                result, error);
    } else {
        success = model_->debug_generate_codec_frames(
            text, language, speaker, max_frames, result, error);
    }
    if (!success) {
        res->error(format_error_response_ex(error, ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    res->ok({
        {"object", "qwen3_tts.codec_frames_debug"},
        {"model_size", model_->config().model_size},
        {"language", language},
        {"speaker", speaker},
        {"prompt_length", result.prefill.prompt_length},
        {"prompt_sum", result.prefill.prompt_sum},
        {"prompt_abs_mean", result.prefill.prompt_abs_mean},
        {"first_codec_token", result.prefill.first_codec_token},
        {"talker_tokens", result.talker_tokens},
        {"codec_frames", result.codec_frames},
        {"generated_frames", result.codec_frames.size()},
        {"reached_eos", result.reached_eos},
    });
    return res;
}

server_http_res_ptr Qwen3TTSModelService::handle_reference_debug(
    const server_http_req& req) {
    std::lock_guard lock(model_mutex_);
    auto res = std::make_unique<server_res_generator_light>();
    if (!model_ || !model_->ready() || get_state() != SERVER_STATE_READY) {
        res->error(format_error_response_ex("Qwen3-TTS model is not ready",
                                            ERROR_TYPE_UNAVAILABLE));
        return res;
    }
    if (!model_->is_base()) {
        res->error(format_error_response_ex(
            "reference_debug requires a Qwen3-TTS Base model",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    const auto body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        res->error(format_error_response_ex("request body must be a JSON object",
                                            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    const std::string path = body.value(
        "prompt_wav_path", body.value("reference_audio_path", std::string()));
    std::vector<float> pcm;
    std::string error;
    if (!load_qwen3_reference_audio(path, model_->config().reference_sample_rate,
                                    pcm, error)) {
        res->error(format_error_response_ex(error, ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    Qwen3TTSReferenceDebugResult result;
    if (!model_->debug_reference_audio(pcm, result, error)) {
        res->error(format_error_response_ex(error, ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    res->ok({
        {"object", "qwen3_tts.reference_debug"},
        {"sample_rate", model_->config().reference_sample_rate},
        {"samples", pcm.size()},
        {"valid_frames", result.valid_frames},
        {"codec_shape", result.codec_shape},
        {"codec_head", result.codec_head},
        {"speaker_embedding_shape", result.speaker_embedding_shape},
        {"speaker_embedding_sum", result.speaker_embedding_sum},
        {"speaker_embedding_abs_mean", result.speaker_embedding_abs_mean},
        {"speaker_embedding_head", result.speaker_embedding_head},
    });
    return res;
}

server_http_res_ptr Qwen3TTSModelService::handle_speech_debug(const server_http_req& req) {
    std::lock_guard lock(model_mutex_);
    auto res = std::make_unique<server_res_generator_light>();
    if (!model_ || !model_->ready() || get_state() != SERVER_STATE_READY) {
        res->error(format_error_response_ex("Qwen3-TTS model is not ready",
                                            ERROR_TYPE_UNAVAILABLE));
        return res;
    }
    const auto body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        res->error(format_error_response_ex("request body must be a JSON object",
                                            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    const std::string text = body.value("input", body.value("text", std::string()));
    const std::string language = body.value("language", "Chinese");
    const std::string speaker = body.value("voice", body.value("speaker", "vivian"));
    const int max_frames = body.value("max_frames", 512);
    if (max_frames <= 0 || max_frames > model_->context_length()) {
        res->error(format_error_response_ex(
            "max_frames must be between 1 and the model context length",
            ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    Qwen3TTSSynthesisResult result;
    std::string error;
    bool success = false;
    if (model_->is_base()) {
        const std::string reference_text = body.value(
            "prompt_text", body.value("reference_text", std::string()));
        const std::string reference_path = body.value(
            "prompt_wav_path", body.value("reference_audio_path", std::string()));
        std::vector<float> reference_pcm;
        success = load_qwen3_reference_audio(
            reference_path, model_->config().reference_sample_rate,
            reference_pcm, error) &&
            model_->synthesize_clone(text, language, reference_text,
                                     reference_pcm, max_frames, result, error);
    } else {
        success = model_->synthesize(
            text, language, speaker, max_frames, result, error);
    }
    if (!success) {
        res->error(format_error_response_ex(error, ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    try {
        double sum = 0.0;
        double absolute_sum = 0.0;
        for (const float sample : result.audio) {
            sum += sample;
            absolute_sum += std::abs(sample);
        }
        res->data = pcm16_wav(result.audio, model_->config().sample_rate);
        res->content_type = "audio/wav";
        res->status = 200;
        res->headers["Content-Disposition"] = "attachment; filename=\"qwen3_tts.wav\"";
        res->headers["X-Qwen3-TTS-Frames"] = std::to_string(result.codec_frames.size());
        res->headers["X-Qwen3-TTS-Samples"] = std::to_string(result.audio.size());
        res->headers["X-Qwen3-TTS-Reached-EOS"] = result.reached_eos ? "true" : "false";
        res->headers["X-Qwen3-TTS-Variant"] = model_->is_base() ? "base" : "custom_voice";
        res->headers["X-Qwen3-TTS-Audio-Sum"] = precise_float(sum);
        res->headers["X-Qwen3-TTS-Audio-Abs-Mean"] = precise_float(
            result.audio.empty() ? 0.0 : absolute_sum / result.audio.size());
    } catch (const std::exception& ex) {
        res->error(format_error_response_ex(ex.what(), ERROR_TYPE_SERVER));
    }
    return res;
}

server_http_res_ptr Qwen3TTSModelService::handle_stateful_decoder_debug(
    const server_http_req& req) {
    std::lock_guard lock(model_mutex_);
    auto res = std::make_unique<server_res_generator_light>();
    if (!model_ || !model_->ready() || get_state() != SERVER_STATE_READY) {
        res->error(format_error_response_ex("Qwen3-TTS model is not ready",
                                            ERROR_TYPE_UNAVAILABLE));
        return res;
    }
    const auto body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object() || !body.contains("codec_frames")) {
        res->error(format_error_response_ex(
            "request body must contain codec_frames", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::vector<std::vector<int32_t>> frames;
    try {
        frames = body.at("codec_frames").get<std::vector<std::vector<int32_t>>>();
    } catch (const json::exception& ex) {
        res->error(format_error_response_ex(
            std::string("invalid codec_frames: ") + ex.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::vector<float> audio;
    std::string error;
    if (!model_->debug_decode_codec_frames(frames, audio, error)) {
        res->error(format_error_response_ex(error, ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    double sum = 0.0;
    double absolute_sum = 0.0;
    for (const float sample : audio) {
        sum += sample;
        absolute_sum += std::abs(sample);
    }
    const auto minmax = std::minmax_element(audio.begin(), audio.end());
    const size_t head_size = std::min<size_t>(16, audio.size());
    res->ok({
        {"object", "qwen3_tts.stateful_decoder_debug"},
        {"codec_frames", frames.size()},
        {"samples", audio.size()},
        {"sum", sum},
        {"abs_mean", audio.empty() ? 0.0 : absolute_sum / audio.size()},
        {"min", audio.empty() ? 0.0F : *minmax.first},
        {"max", audio.empty() ? 0.0F : *minmax.second},
        {"head", std::vector<float>(audio.begin(), audio.begin() + head_size)},
    });
    return res;
}

void Qwen3TTSModelService::handle_speech(
    const httplib::Request& req, httplib::Response& res,
    std::shared_ptr<void> request_guard) {
    std::shared_ptr<HLIEQwen3TTSModel> model;
    {
        std::lock_guard lock(model_mutex_);
        if (!model_ || !model_->ready() || get_state() != SERVER_STATE_READY) {
            set_oai_error(res, "Qwen3-TTS model is not ready", ERROR_TYPE_UNAVAILABLE);
            return;
        }
        model = model_;
    }

    const auto body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        set_oai_error(res, "request body must be a JSON object", ERROR_TYPE_INVALID_REQUEST);
        return;
    }
    if (!body.contains("input") || !body.at("input").is_string() ||
        body.at("input").get_ref<const std::string&>().empty()) {
        set_oai_error(res, "input must be a non-empty string", ERROR_TYPE_INVALID_REQUEST);
        return;
    }
    const auto string_option = [&](const char* name, const std::string& fallback,
                                   std::string& value) {
        if (!body.contains(name) || body.at(name).is_null()) {
            value = fallback;
            return true;
        }
        if (!body.at(name).is_string()) {
            set_oai_error(res, std::string(name) + " must be a string",
                          ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        value = body.at(name).get<std::string>();
        return true;
    };
    const auto alias_present = [&](const char* primary, const char* secondary) {
        return (body.contains(primary) && !body.at(primary).is_null()) ||
               (body.contains(secondary) && !body.at(secondary).is_null());
    };
    const auto string_alias = [&](const char* primary, const char* secondary,
                                  std::string& value) {
        const char* selected = nullptr;
        if (body.contains(primary) && !body.at(primary).is_null()) {
            selected = primary;
        } else if (body.contains(secondary) && !body.at(secondary).is_null()) {
            selected = secondary;
        }
        if (!selected) {
            value.clear();
            return true;
        }
        return string_option(selected, std::string(), value);
    };

    bool stream_output = false;
    if (body.contains("stream") && !body.at("stream").is_null()) {
        if (!body.at("stream").is_boolean()) {
            set_oai_error(res, "stream must be a boolean",
                          ERROR_TYPE_INVALID_REQUEST);
            return;
        }
        stream_output = body.at("stream").get<bool>();
    }
    std::string language;
    std::string speaker;
    std::string response_format;
    if (!string_option("language", "Chinese", language) ||
        !string_option("voice", "vivian", speaker) ||
        !string_option("response_format", stream_output ? "pcm" : "wav",
                       response_format)) {
        return;
    }
    std::string reference_text;
    std::string reference_path;
    bool default_reference = false;
    if (model->is_base()) {
        const bool has_reference_text = alias_present(
            "prompt_text", "reference_text");
        const bool has_reference_audio = alias_present(
            "prompt_wav_path", "reference_audio_path");
        if (has_reference_text != has_reference_audio) {
            set_oai_error(
                res,
                "prompt_text and prompt_wav_path must be provided together",
                ERROR_TYPE_INVALID_REQUEST);
            return;
        }
        if (!has_reference_text) {
            reference_text = kDefaultCloneReferenceText;
            reference_path = default_clone_reference_path();
            default_reference = true;
        } else if (!string_alias(
                       "prompt_text", "reference_text", reference_text) ||
                   !string_alias(
                       "prompt_wav_path", "reference_audio_path",
                       reference_path)) {
            return;
        }
        if (reference_text.empty() || reference_path.empty()) {
            set_oai_error(
                res,
                "prompt_text and prompt_wav_path must be non-empty",
                ERROR_TYPE_INVALID_REQUEST);
            return;
        }
    }
    response_format = lower_ascii(response_format);
    if ((stream_output && response_format != "pcm") ||
        (!stream_output && response_format != "wav")) {
        set_oai_error(
            res,
            stream_output
                ? "Qwen3-TTS streaming currently supports response_format=pcm only"
                : "Qwen3-TTS non-streaming currently supports response_format=wav only",
            ERROR_TYPE_INVALID_REQUEST);
        return;
    }
    if (body.contains("speed") && !body.at("speed").is_null()) {
        if (!body.at("speed").is_number()) {
            set_oai_error(res, "speed must be a number", ERROR_TYPE_INVALID_REQUEST);
            return;
        }
        const double speed = body.at("speed").get<double>();
        if (std::abs(speed - 1.0) > 1e-6) {
            set_oai_error(res, "Qwen3-TTS currently supports speed=1.0 only",
                          ERROR_TYPE_INVALID_REQUEST);
            return;
        }
    }
    int max_frames = 512;
    if (body.contains("max_frames") && !body.at("max_frames").is_null()) {
        if (!body.at("max_frames").is_number_integer()) {
            set_oai_error(res, "max_frames must be an integer", ERROR_TYPE_INVALID_REQUEST);
            return;
        }
        const auto& value = body.at("max_frames");
        if (value.is_number_unsigned()) {
            const uint64_t unsigned_value = value.get<uint64_t>();
            if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                set_oai_error(res, "max_frames is out of range", ERROR_TYPE_INVALID_REQUEST);
                return;
            }
            max_frames = static_cast<int>(unsigned_value);
        } else {
            const int64_t signed_value = value.get<int64_t>();
            if (signed_value < std::numeric_limits<int>::min() ||
                signed_value > std::numeric_limits<int>::max()) {
                set_oai_error(res, "max_frames is out of range", ERROR_TYPE_INVALID_REQUEST);
                return;
            }
            max_frames = static_cast<int>(signed_value);
        }
    }
    if (max_frames <= 0 || max_frames >= model->context_length()) {
        set_oai_error(res, "max_frames must be between 1 and the model context length",
                      ERROR_TYPE_INVALID_REQUEST);
        return;
    }

    int64_t requested_seed = 1024;
    if (body.contains("seed") && !body.at("seed").is_null()) {
        if (!body.at("seed").is_number_integer()) {
            set_oai_error(res, "seed must be an integer", ERROR_TYPE_INVALID_REQUEST);
            return;
        }
        const auto& value = body.at("seed");
        if (value.is_number_unsigned()) {
            const uint64_t unsigned_value = value.get<uint64_t>();
            if (unsigned_value > std::numeric_limits<uint32_t>::max()) {
                set_oai_error(res, "seed must be -1 or a 32-bit unsigned integer",
                              ERROR_TYPE_INVALID_REQUEST);
                return;
            }
            requested_seed = static_cast<int64_t>(unsigned_value);
        } else {
            requested_seed = value.get<int64_t>();
            if (requested_seed < -1 ||
                requested_seed > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
                set_oai_error(res, "seed must be -1 or a 32-bit unsigned integer",
                              ERROR_TYPE_INVALID_REQUEST);
                return;
            }
        }
    }
    const uint32_t sampling_seed = requested_seed == -1
        ? std::random_device{}() : static_cast<uint32_t>(requested_seed);

    const std::string speaker_key = lower_ascii(speaker);
    const std::string language_key = lower_ascii(language);
    if (!model->is_base() &&
        !model->config().speaker_ids.contains(speaker_key)) {
        set_oai_error(res, "unsupported Qwen3-TTS speaker: " + speaker_key,
                      ERROR_TYPE_INVALID_REQUEST);
        return;
    }
    if (language_key != "auto" &&
        !model->config().codec_language_ids.contains(language_key)) {
        set_oai_error(res, "unsupported Qwen3-TTS language: " + language_key,
                      ERROR_TYPE_INVALID_REQUEST);
        return;
    }
    std::vector<float> reference_pcm;
    std::string error;
    if (model->is_base()) {
        if (!load_qwen3_reference_audio(
                reference_path, model->config().reference_sample_rate,
                reference_pcm, error)) {
            set_oai_error(res, error, ERROR_TYPE_INVALID_REQUEST);
            return;
        }
    }

    const std::string input = body.at("input").get<std::string>();
    if (!stream_output) {
        Qwen3TTSSynthesisResult result;
        bool success = false;
        if (model->is_base()) {
            success = model->synthesize_clone(
                input, language, reference_text, reference_pcm, max_frames,
                result, error, true, true, sampling_seed);
        } else {
            success = model->synthesize(
                input, language, speaker, max_frames, result, error, true, true,
                sampling_seed);
        }
        if (!success) {
            const error_type type =
                error.find("max_frames before EOS") != std::string::npos
                    ? ERROR_TYPE_INVALID_REQUEST
                    : error.find("context length") != std::string::npos
                        ? ERROR_TYPE_EXCEED_CONTEXT_SIZE
                        : ERROR_TYPE_SERVER;
            set_oai_error(res, error, type);
            return;
        }
        try {
            auto wav = pcm16_wav(result.audio, model->config().sample_rate);
            res.set_header("Content-Disposition",
                           "attachment; filename=\"speech.wav\"");
            res.set_header("X-Qwen3-TTS-Frames",
                           std::to_string(result.codec_frames.size()));
            res.set_header("X-Qwen3-TTS-Samples",
                           std::to_string(result.audio.size()));
            res.set_header("X-Qwen3-TTS-Reached-EOS",
                           result.reached_eos ? "true" : "false");
            res.set_header("X-Qwen3-TTS-Seed",
                           std::to_string(sampling_seed));
            res.set_header("X-Qwen3-TTS-Variant",
                           model->is_base() ? "base" : "custom_voice");
            if (default_reference) {
                res.set_header("X-Qwen3-TTS-Reference-Profile",
                               "default-zero-shot");
            }
            res.set_content(std::move(wav), "audio/wav");
        } catch (const std::exception& ex) {
            set_oai_error(res, ex.what(), ERROR_TYPE_SERVER);
        }
        return;
    }

    res.set_header("Cache-Control", "no-store");
    res.set_header("Content-Disposition", "inline; filename=\"speech.pcm\"");
    res.set_header("X-Audio-Bits-Per-Sample", "16");
    res.set_header("X-Audio-Channels", "1");
    res.set_header("X-Audio-Format", "s16le");
    res.set_header("X-Audio-Sample-Rate",
                   std::to_string(model->config().sample_rate));
    res.set_header("X-Qwen3-TTS-Seed", std::to_string(sampling_seed));
    res.set_header("X-Qwen3-TTS-Stream", "true");
    res.set_header("X-Qwen3-TTS-Variant",
                   model->is_base() ? "base" : "custom_voice");
    res.set_header(
        "Trailer",
        "X-Qwen3-TTS-Frames, X-Qwen3-TTS-Samples, "
        "X-Qwen3-TTS-Reached-EOS, X-Qwen3-TTS-Finish-Reason, "
        "X-Qwen3-TTS-Error");
    if (default_reference) {
        res.set_header("X-Qwen3-TTS-Reference-Profile", "default-zero-shot");
    }

    const bool base_variant = model->is_base();
    res.set_chunked_content_provider(
        "audio/pcm",
        [model = std::move(model), input, language, speaker, reference_text,
         reference_pcm = std::move(reference_pcm), max_frames, sampling_seed,
         base_variant, request_guard = std::move(request_guard),
         started = false](size_t, httplib::DataSink& sink) mutable {
            if (started) {
                sink.done();
                return true;
            }
            started = true;
            (void)request_guard;

            Qwen3TTSStreamResult result;
            std::string stream_error;
            const auto on_audio_chunk = [&](const std::vector<float>& audio) {
                const std::string bytes = pcm16_bytes(audio);
                return bytes.empty() || sink.write(bytes.data(), bytes.size());
            };
            const auto should_stop = [&]() {
                return !sink.is_writable();
            };

            bool success = false;
            try {
                if (base_variant) {
                    success = model->synthesize_clone_stream(
                        input, language, reference_text, reference_pcm,
                        max_frames, result, on_audio_chunk, should_stop,
                        stream_error, true, true, sampling_seed);
                } else {
                    success = model->synthesize_stream(
                        input, language, speaker, max_frames, result,
                        on_audio_chunk, should_stop, stream_error, true, true,
                        sampling_seed);
                }
            } catch (const std::exception& ex) {
                stream_error = ex.what();
            }

            if (!sink.is_writable()) {
                return false;
            }
            std::string finish_reason = "stop";
            if (!success) {
                if (stream_error.find("max_frames before EOS") !=
                    std::string::npos) {
                    finish_reason = "length";
                } else if (stream_error.find("cancel") != std::string::npos ||
                           stream_error.find("consumer stopped") !=
                               std::string::npos) {
                    finish_reason = "cancelled";
                } else {
                    finish_reason = "error";
                }
                LOG_ERROR("Qwen3-TTS HTTP audio stream failed: {}",
                          stream_error);
            }

            httplib::Headers trailers{
                {"X-Qwen3-TTS-Frames",
                 std::to_string(result.codec_frames)},
                {"X-Qwen3-TTS-Samples",
                 std::to_string(result.audio_samples)},
                {"X-Qwen3-TTS-Reached-EOS",
                 result.reached_eos ? "true" : "false"},
                {"X-Qwen3-TTS-Finish-Reason", finish_reason},
            };
            if (!success && !stream_error.empty()) {
                trailers.emplace("X-Qwen3-TTS-Error",
                                 safe_trailer_value(stream_error));
            }
            sink.done_with_trailer(trailers);
            return true;
        });
}

HLIE_NAMESPACE_END

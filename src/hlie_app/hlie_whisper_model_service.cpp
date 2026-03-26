#include "hlie_app/hlie_whisper_model_service.h"

#include "hlie_app/hlie_device_service.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

HLIE_NAMESPACE_BEGIN


namespace {
    void log_cb(const int level, const char* log, void* data) {
        // if (const auto* params = static_cast<SDParams*>(data); !log || (!params->verbose && level <= SD_LOG_DEBUG)) {
        //     return;
        // }

        spdlog::level::level_enum log_level;
        switch (level) {
        case 0:
            log_level = spdlog::level::level_enum::info;
            break;
        case 1:
            log_level = spdlog::level::level_enum::warn;
            break;
        case 2:
            log_level = spdlog::level::level_enum::err;
            break;
        case 3:
        default:
            log_level = spdlog::level::level_enum::debug;
            break;
        }

        spdlog::log(log_level, log);
    }

    //  500 -> 00:05.000
    // 6000 -> 01:00.000
    std::string to_timestamp(int64_t t, bool comma = false) {
        int64_t msec = t * 10;
        int64_t hr = msec / (1000 * 60 * 60);
        msec = msec - hr * (1000 * 60 * 60);
        int64_t min = msec / (1000 * 60);
        msec = msec - min * (1000 * 60);
        int64_t sec = msec / 1000;
        msec = msec - sec * 1000;

        char buf[32];
        snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int)hr, (int)min, (int)sec, comma ? "," : ".", (int)msec);

        return std::string(buf);
    }

    int timestamp_to_sample(int64_t t, int n_samples, int whisper_sample_rate) {
        return std::max(0, std::min((int)n_samples - 1, (int)((t * whisper_sample_rate) / 100)));
    }

    std::string estimate_diarization_speaker(std::vector<std::vector<float>> pcmf32s, int64_t t0, int64_t t1, bool id_only = false) {
        std::string speaker = "";
        const int64_t n_samples = pcmf32s[0].size();

        const int64_t is0 = timestamp_to_sample(t0, n_samples, WHISPER_SAMPLE_RATE);
        const int64_t is1 = timestamp_to_sample(t1, n_samples, WHISPER_SAMPLE_RATE);

        double energy0 = 0.0f;
        double energy1 = 0.0f;

        for (int64_t j = is0; j < is1; j++) {
            energy0 += fabs(pcmf32s[0][j]);
            energy1 += fabs(pcmf32s[1][j]);
        }

        if (energy0 > 1.1 * energy1) {
            speaker = "0";
        } else if (energy1 > 1.1 * energy0) {
            speaker = "1";
        } else {
            speaker = "?";
        }

        // printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, speaker = %s\n", is0, is1, energy0, energy1, speaker.c_str());

        if (!id_only) {
            speaker.insert(0, "(speaker ");
            speaker.append(")");
        }

        return speaker;
    }

    std::string output_str(struct whisper_context* whisper_ctx_, const whisper_params& params, std::vector<std::vector<float>> pcmf32s) {
        std::stringstream result;
        const int n_segments = whisper_full_n_segments(whisper_ctx_);
        for (int i = 0; i < n_segments; ++i) {
            const char* text = whisper_full_get_segment_text(whisper_ctx_, i);
            std::string speaker = "";

            if (params.diarize && pcmf32s.size() == 2) {
                const int64_t t0 = whisper_full_get_segment_t0(whisper_ctx_, i);
                const int64_t t1 = whisper_full_get_segment_t1(whisper_ctx_, i);
                speaker = estimate_diarization_speaker(pcmf32s, t0, t1);
            }
            printf("n_segments i %d text: %s speaker: %s \n", i, text, speaker.c_str());


            result << speaker << text << "\n";
        }
        return result.str();
    }

    bool parse_str_to_bool(const std::string& s) {
        if (s == "true" || s == "1" || s == "yes" || s == "y") {
            return true;
        }
        return false;
    }

    void get_req_parameters(const httplib::Request& req, whisper_params& params) {
        if (req.form.has_file("offset_t")) {
            params.offset_t_ms = std::stoi(req.form.get_file("offset_t").content);
        }
        if (req.form.has_file("offset_n")) {
            params.offset_n = std::stoi(req.form.get_file("offset_n").content);
        }
        if (req.form.has_file("duration")) {
            params.duration_ms = std::stoi(req.form.get_file("duration").content);
        }
        if (req.form.has_file("max_context")) {
            params.max_context = std::stoi(req.form.get_file("max_context").content);
        }
        if (req.form.has_file("max_len")) {
            params.max_len = std::stoi(req.form.get_file("max_len").content);
        }
        if (req.form.has_file("best_of")) {
            params.best_of = std::stoi(req.form.get_file("best_of").content);
        }
        if (req.form.has_file("beam_size")) {
            params.beam_size = std::stoi(req.form.get_file("beam_size").content);
        }
        if (req.form.has_file("audio_ctx")) {
            params.audio_ctx = std::stof(req.form.get_file("audio_ctx").content);
        }
        if (req.form.has_file("word_thold")) {
            params.word_thold = std::stof(req.form.get_file("word_thold").content);
        }
        if (req.form.has_file("entropy_thold")) {
            params.entropy_thold = std::stof(req.form.get_file("entropy_thold").content);
        }
        if (req.form.has_file("logprob_thold")) {
            params.logprob_thold = std::stof(req.form.get_file("logprob_thold").content);
        }
        if (req.form.has_file("debug_mode")) {
            params.debug_mode = parse_str_to_bool(req.form.get_file("debug_mode").content);
        }
        if (req.form.has_file("translate")) {
            params.translate = parse_str_to_bool(req.form.get_file("translate").content);
        }
        if (req.form.has_file("diarize")) {
            params.diarize = parse_str_to_bool(req.form.get_file("diarize").content);
        }
        if (req.form.has_file("tinydiarize")) {
            params.tinydiarize = parse_str_to_bool(req.form.get_file("tinydiarize").content);
        }
        if (req.form.has_file("split_on_word")) {
            params.split_on_word = parse_str_to_bool(req.form.get_file("split_on_word").content);
        }
        if (req.form.has_file("no_timestamps")) {
            params.no_timestamps = parse_str_to_bool(req.form.get_file("no_timestamps").content);
        }
        if (req.form.has_file("language")) {
            params.language = req.form.get_file("language").content;
        }
        if (req.form.has_file("detect_language")) {
            params.detect_language = parse_str_to_bool(req.form.get_file("detect_language").content);
        }
        if (req.form.has_file("prompt")) {
            params.prompt = req.form.get_file("prompt").content;
        }
        if (req.form.has_file("response_format")) {
            params.response_format = req.form.get_file("response_format").content;
        }
        if (req.form.has_file("temperature")) {
            params.temperature = std::stof(req.form.get_file("temperature").content);
        }
        if (req.form.has_file("temperature_inc")) {
            params.temperature_inc = std::stof(req.form.get_file("temperature_inc").content);
        }
        if (req.form.has_file("suppress_non_speech")) {
            params.suppress_nst = parse_str_to_bool(req.form.get_file("suppress_non_speech").content);
        }
        if (req.form.has_file("suppress_nst")) {
            params.suppress_nst = parse_str_to_bool(req.form.get_file("suppress_nst").content);
        }
        if (req.form.has_file("vad")) {
            params.vad = parse_str_to_bool(req.form.get_file("vad").content);
        }
        if (req.form.has_file("vad_threshold")) {
            params.vad_threshold = std::stof(req.form.get_file("vad_threshold").content);
        }
        if (req.form.has_file("vad_min_speech_duration_ms")) {
            params.vad_min_speech_duration_ms = std::stof(req.form.get_file("vad_min_speech_duration_ms").content);
        }
        if (req.form.has_file("vad_min_silence_duration_ms")) {
            params.vad_min_silence_duration_ms = std::stof(req.form.get_file("vad_min_silence_duration_ms").content);
        }
        if (req.form.has_file("vad_max_speech_duration_s")) {
            params.vad_max_speech_duration_s = std::stof(req.form.get_file("vad_max_speech_duration_s").content);
        }
        if (req.form.has_file("vad_speech_pad_ms")) {
            params.vad_speech_pad_ms = std::stoi(req.form.get_file("vad_speech_pad_ms").content);
        }
        if (req.form.has_file("vad_samples_overlap")) {
            params.vad_samples_overlap = std::stof(req.form.get_file("vad_samples_overlap").content);
        }
        if (req.form.has_file("no_language_probabilities")) {
            params.no_language_probabilities = parse_str_to_bool(req.form.get_file("no_language_probabilities").content);
        }
    }


    bool read_audio_data(const std::string& fname, std::vector<float>& pcmf32, std::vector<std::vector<float>>& pcmf32s, bool stereo) {
        std::vector<uint8_t> audio_data; // used for pipe input from stdin or ffmpeg decoding output

        ma_result result;
        ma_decoder_config decoder_config;
        ma_decoder decoder;

        decoder_config = ma_decoder_config_init(ma_format_f32, stereo ? 2 : 1, WHISPER_SAMPLE_RATE);

        if (fname == "-") {
#ifdef _WIN32
            _setmode(_fileno(stdin), _O_BINARY);
#endif

            uint8_t buf[1024];
            while (true) {
                const size_t n = fread(buf, 1, sizeof(buf), stdin);
                if (n == 0) {
                    break;
                }
                audio_data.insert(audio_data.end(), buf, buf + n);
            }

            if ((result = ma_decoder_init_memory(audio_data.data(), audio_data.size(), &decoder_config, &decoder)) != MA_SUCCESS) {

                fprintf(stderr, "Error: failed to open audio data from stdin (%s)\n", ma_result_description(result));

                return false;
            }

            fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, audio_data.size());
        } else if (((result = ma_decoder_init_file(fname.c_str(), &decoder_config, &decoder)) != MA_SUCCESS)) {
#if defined(WHISPER_FFMPEG)
            if (ffmpeg_decode_audio(fname, audio_data) != 0) {
                fprintf(stderr, "error: failed to ffmpeg decode '%s'\n", fname.c_str());

                return false;
            }

            if ((result = ma_decoder_init_memory(audio_data.data(), audio_data.size(), &decoder_config, &decoder)) != MA_SUCCESS) {
                fprintf(stderr, "error: failed to read audio data as wav (%s)\n", ma_result_description(result));

                return false;
            }
#else
            if ((result = ma_decoder_init_memory(fname.c_str(), fname.size(), &decoder_config, &decoder)) != MA_SUCCESS) {
                fprintf(stderr, "error: failed to read audio data as wav (%s)\n", ma_result_description(result));

                return false;
            }
#endif
        }

        ma_uint64 frame_count;
        ma_uint64 frames_read;

        if ((result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count)) != MA_SUCCESS) {
            fprintf(stderr, "error: failed to retrieve the length of the audio data (%s)\n", ma_result_description(result));

            return false;
        }

        pcmf32.resize(stereo ? frame_count * 2 : frame_count);

        if ((result = ma_decoder_read_pcm_frames(&decoder, pcmf32.data(), frame_count, &frames_read)) != MA_SUCCESS) {
            fprintf(stderr, "error: failed to read the frames of the audio data (%s)\n", ma_result_description(result));

            return false;
        }

        if (stereo) {
            std::vector<float> stereo_data = pcmf32;
            pcmf32.resize(frame_count);

            for (uint64_t i = 0; i < frame_count; i++) {
                pcmf32[i] = (stereo_data[2 * i] + stereo_data[2 * i + 1]);
            }

            pcmf32s.resize(2);
            pcmf32s[0].resize(frame_count);
            pcmf32s[1].resize(frame_count);
            for (uint64_t i = 0; i < frame_count; i++) {
                pcmf32s[0][i] = stereo_data[2 * i];
                pcmf32s[1][i] = stereo_data[2 * i + 1];
            }
        }

        ma_decoder_uninit(&decoder);

        return true;
    }
} // namespace


int WhisperModelService::init(const std::string& model_dir, const HLIEModelInfo& model_info,
                              const std::shared_ptr<HLIEDeviceService>& device_service) {
    blocking_queue_ = std::make_shared<HLIEBlockingQueue<QueueItem>>();

    // whisper init
    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu = params_.use_gpu;
    cparams.gpu_device = params_.gpu_device;
    cparams.flash_attn = params_.flash_attn;

    cparams.dtw_aheads_preset = WHISPER_AHEADS_MEDIUM;

    device_service_ = device_service;

    auto devices = model_info.get_devices();

    if (const int replicas = Singleton<HLIEConfigService>::instance()->get_int("server", "replicas", 1); replicas > 1) {
        is_multi_instance_ = true;
        if (const std::optional<int> val = Singleton<HLIEDeviceHelper>::instance()->acquire_device(SD); !val) {
            LOG_ERROR("Whisper cannot acquire available device");
            return -1;
        } else {
            HlieUtil::set_env("HLIEWHISPER_DEVICE_ID", std::to_string(val.value()));
            devices_.push_back(*val);
            LOG_INFO("Whisper using device: {}", val.value());
        }
    } else { // 单实例模式可以定制Model Devices
        if (!devices.empty()) {
            if (!HlieUtil::is_valid_devices(devices)) {
                LOG_ERROR("Whisper cannot acquire model specific device \"{}\"", devices);
                return -1;
            }

            if (HlieUtil::parse_device(devices, devices_) < 0) {
                LOG_ERROR("Whisper parse_device specific device \"{}\" failed", devices);
                return -1;
            }

            if (device_service_->acquire_mem_on_devices(devices_, 1200) < 0) {
                return -1;
            }

            HlieUtil::set_env("HLIEWHISPER_DEVICE_ID", std::to_string(devices_.at(0)));
            LOG_INFO("Whisper using model specific device: {}", devices_.at(0));
        } else {
            uint32_t device;
            if (device_service_->acquire_mem(11000, device) < 0) {
                return -1;
            }
            devices_.push_back(device);
            HlieUtil::set_env("HLIEWHISPER_DEVICE_ID", std::to_string(device));
            LOG_INFO("Whisper using model specific device: {}", device);
        }
    }

    // sd_set_log_callback(log_cb, params_.get());

    const auto& model = model_info.get_model_path();
    whisper_ctx_ = whisper_init_from_file_with_params(model.c_str(), cparams);
    LOG_INFO("whisper_init_from_file_with_params done");
    if (whisper_ctx_ == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 3;
    }

    default_params_ = params_;
    model_key_ = model_info.model_key_;
    websocket_service_ = std::make_unique<HLIEWebSocketService>();

    websocket_service_->init();

    return 0;
}

void WhisperModelService::run() {
    if (worker_thread_) {
        LOG_WARN("WhisperModelService thread is already running.");
        return;
    }

    // Todo 判断配置是否支持WS
    websocket_service_->run([this](const WhisperStreamRequest& req) { return this->handle_stream(req); });

    state_.store(SERVER_STATE_READY);
    should_stop_ = false;
    worker_thread_ = std::make_unique<std::thread>(&WhisperModelService::process, this);
    LOG_INFO("WhisperModelService thread started.");
}

void WhisperModelService::process() {
    try {
        while (!should_stop_) {
            const auto [request, promise] = blocking_queue_->pop(); // 阻塞直到有数据或队列关闭
            if (should_stop_) {
                break; // 防止 stop() 后仍有残留任务
            }

            try {
                auto start = std::chrono::steady_clock::now();
                auto transcription_result = transcription(request);
                auto end = std::chrono::steady_clock::now();
                // 构造结果
                WhisperResult result;
                result.result = transcription_result;
                result.success = transcription_result != "HLIECpp-FAILED-888";
                result.predicted_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                LOG_INFO("transcription time: {} ms", result.predicted_ms);

                // 通知调用者
                promise->set_value(result);
            } catch (const std::exception& e) {
                // 错误处理
                WhisperResult result;
                result.success = false;
                result.error_message = e.what();
                promise->set_value(result);
            }
        }
    } catch (const std::runtime_error&) {
        LOG_WARN("WhisperModelService process exited");
    }
}

void WhisperModelService::stop() {
    if (!worker_thread_) {
        return;
    }

    should_stop_ = true;
    if (blocking_queue_) {
        blocking_queue_->close(); // 通知队列关闭，唤醒可能阻塞的 pop()
    }

    if (worker_thread_->joinable()) {
        worker_thread_->join();
    }
    worker_thread_.reset();
    whisper_free(whisper_ctx_);

    if (websocket_service_) {
        websocket_service_->stop();
    }

    if (is_multi_instance_) {
        Singleton<HLIEDeviceHelper>::instance()->release_devices(devices_);
    } else {
        device_service_->release_mem_on_devices(devices_, 2000);
    }
    devices_.clear();
}

int WhisperModelService::get_state() {
    return state_;
}

int WhisperModelService::handle_health(nlohmann::ordered_json& resp) {
    return 0;
}

void WhisperModelService::handle_models(nlohmann::ordered_json& resp) const {
    resp["models"].push_back({
        {"name", "Whisper"},
        {"model", "whisper"},
    });

    resp["data"].push_back({
        {"id", "whisper"},
        {"object", "model"},
        {"created", std::time(0)},
        {"owned_by", "HLIECpp"},
    });
}

void WhisperModelService::fill_params(whisper_full_params& wparams) const {
    wparams.strategy = params_.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

    wparams.print_realtime = false;
    wparams.print_progress = params_.print_progress;
    wparams.print_timestamps = !params_.no_timestamps;
    wparams.print_special = params_.print_special;
    wparams.translate = params_.translate;
    wparams.language = params_.language.c_str();
    wparams.detect_language = params_.detect_language;
    wparams.n_threads = params_.n_threads;
    wparams.n_max_text_ctx = params_.max_context >= 0 ? params_.max_context : wparams.n_max_text_ctx;
    wparams.offset_ms = params_.offset_t_ms;
    wparams.duration_ms = params_.duration_ms;

    wparams.thold_pt = params_.word_thold;
    wparams.max_len = params_.max_len == 0 ? 60 : params_.max_len;
    wparams.split_on_word = params_.split_on_word;
    wparams.audio_ctx = params_.audio_ctx;

    wparams.debug_mode = params_.debug_mode;

    wparams.tdrz_enable = params_.tinydiarize; // [TDRZ]

    wparams.initial_prompt = params_.prompt.c_str();

    wparams.greedy.best_of = params_.best_of;
    wparams.beam_search.beam_size = params_.beam_size;

    wparams.temperature = params_.temperature;
    wparams.no_speech_thold = params_.no_speech_thold;
    wparams.temperature_inc = params_.temperature_inc;
    wparams.entropy_thold = params_.entropy_thold;
    wparams.logprob_thold = params_.logprob_thold;

    wparams.no_timestamps = params_.no_timestamps;
    wparams.token_timestamps = !params_.no_timestamps && params_.response_format == vjson_format;
    wparams.no_context = params_.no_context;

    wparams.suppress_nst = params_.suppress_nst;

    wparams.vad = params_.vad;
    wparams.vad_model_path = params_.vad_model.c_str();

    wparams.vad_params.threshold = params_.vad_threshold;
    wparams.vad_params.min_speech_duration_ms = params_.vad_min_speech_duration_ms;
    wparams.vad_params.min_silence_duration_ms = params_.vad_min_silence_duration_ms;
    wparams.vad_params.max_speech_duration_s = params_.vad_max_speech_duration_s;
    wparams.vad_params.speech_pad_ms = params_.vad_speech_pad_ms;
    wparams.vad_params.samples_overlap = params_.vad_samples_overlap;
}

std::string WhisperModelService::handle_stream(const WhisperStreamRequest& request) const {
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    fill_params(wparams);

    WhisperRequest req;
    req.filename = "stream";
    req.params = wparams;
    req.pcmf32 = request.pcmf32;
    req.n_processors = 1;

    const auto start = std::chrono::steady_clock::now();
    auto transcription_result = transcription(req);
    const auto end = std::chrono::steady_clock::now();
    auto predicted_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOG_INFO("transcription time: {} ms", predicted_ms);
    return transcription_result;
}

std::future<WhisperResult> WhisperModelService::transcription_async(const WhisperRequest& request) const {
    const auto promise = std::make_shared<std::promise<WhisperResult>>();
    auto future = promise->get_future();

    // 创建队列项
    QueueItem item;
    item.request = request;
    item.promise = promise;

    if (blocking_queue_) {
        blocking_queue_->push(item);
    }

    return future;
}

WhisperResult WhisperModelService::transcription_sync(const WhisperRequest& request) const {
    auto future = transcription_async(request);
    return future.get();
}

std::string WhisperModelService::transcription(const WhisperRequest& request) const {
    std::string results;
    try {
        LOG_INFO("whisper.cpp full parallel ready to run {}", request.filename);
        if (whisper_full_parallel(whisper_ctx_, request.params, request.pcmf32.data(), request.pcmf32.size(), request.n_processors) != 0) {
            return "HLIECpp-FAILED-888";
        }

        results = output_str(whisper_ctx_, params_, request.pcmf32s);
    } catch (const std::exception& e) {
        LOG_ERROR("transcription exception: {}", e.what());
        return "HLIECpp-FAILED-888";
    }
    LOG_INFO("transcription done");
    return results;
}

void WhisperModelService::handle_transcriptions(const httplib::Request& req, httplib::Response& res) {
    try {
        // first check user requested fields of the request
        auto file_it = req.form.files.find("file");
        if (file_it == req.form.files.end()) {
            fprintf(stderr, "error: no 'file' field in the request\n");
            const std::string error_resp = R"({"error":"no 'file' field in the request"})";
            res.set_content(error_resp, "application/json");
            return;
        }

        auto audio_file = req.form.get_file("file");

        // check non-required fields
        get_req_parameters(req, params_);

        std::string filename{audio_file.filename};
        printf("Received request: %s\n", filename.c_str());

        // audio arrays
        std::vector<float> pcmf32; // mono-channel F32 PCM
        std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM

        if (!read_audio_data(audio_file.content, pcmf32, pcmf32s, params_.diarize)) {
            fprintf(stderr, "error: failed to read audio data\n");
            const std::string error_resp = "{\"error\":\"failed to read audio data\"}";
            res.set_content(error_resp, "application/json");
            return;
        }

        printf("Successfully loaded %s\n", filename.c_str());

        // print system information
        {
            LOG_INFO("system_info: n_threads = {} / {} | {}", params_.n_threads * params_.n_processors, std::thread::hardware_concurrency(),
                     whisper_print_system_info());
        }

        // print some info about the processing
        {
            if (!whisper_is_multilingual(whisper_ctx_)) {
                if (params_.language != "en" || params_.translate) {
                    params_.language = "en";
                    params_.translate = false;
                    LOG_WARN("WARNING: model is not multilingual, ignoring language and translation options");
                }
            }
            if (params_.detect_language) {
                params_.language = "auto";
            }

            LOG_WARN("processing '{}' ({} samples, {} sec), {} threads, {} processors, lang = {}, task = {}, {}timestamps = {} ...\n",
                     filename.c_str(), int(pcmf32.size()), float(pcmf32.size()) / WHISPER_SAMPLE_RATE, params_.n_threads,
                     params_.n_processors, params_.language.c_str(), params_.translate ? "translate" : "transcribe",
                     params_.tinydiarize ? "tdrz = 1, " : "", params_.no_timestamps ? 0 : 1);
        }

        // run the inference
        {
            LOG_INFO("Running whisper.cpp inference on {}", filename);
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            fill_params(wparams);

            /*
            whisper_print_user_data user_data = {&params_, &pcmf32s, 0};

            // this callback is called on each new segment
            if (params_.print_realtime) {
                wparams.new_segment_callback = whisper_print_segment_callback;
                wparams.new_segment_callback_user_data = &user_data;
            }

            if (wparams.print_progress) {
                wparams.progress_callback = whisper_print_progress_callback;
                wparams.progress_callback_user_data = &user_data;
            }*/

            // tell whisper to abort if the HTTP connection closed
            wparams.abort_callback = [](void* user_data) {
                // user_data is a pointer to our Request
                auto req_ptr = static_cast<const httplib::Request*>(user_data);
                return req_ptr->is_connection_closed();
            };

            wparams.abort_callback_user_data = (void*)&req;

            WhisperRequest whisper_request;
            whisper_request.params = wparams;
            whisper_request.filename = filename;
            whisper_request.pcmf32 = pcmf32;
            whisper_request.pcmf32s = pcmf32s;
            whisper_request.n_processors = params_.n_processors;

            if (auto [results, success, error_message, predicted_ms] = transcription_sync(whisper_request); !success) {
                // handle failure or early abort
                if (req.is_connection_closed()) {
                    // log client disconnect
                    fprintf(stderr, "client disconnected, aborted processing\n");
                    res.status = 499; // Client Closed Request (nginx convention)
                    res.set_content(R"({"error":"client disconnected"})", "application/json");
                    return;
                }
                LOG_ERROR("failed to process audio");
                res.status = 500; // Internal Server Error
                const std::string error_resp = R"({"error":"failed to process audio"})";
                res.set_content(error_resp, "application/json");
                return;
            } else {
                auto jres = json{{"text", results}};
                res.set_content(jres.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
            }
        }

        LOG_INFO("process audio done");

        // reset params to their defaults
        params_ = default_params_;
    } catch (const std::exception& ex) {
        LOG_ERROR("handle_transcriptions ex: {}", ex.what());
        throw;
    }
}


HLIE_NAMESPACE_END

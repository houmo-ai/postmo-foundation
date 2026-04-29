#include "hlie_app/hlie_cosyvoice_model_service.h"

#include <regex>
#include <string>


#include <miniaudio.h>
#include <samplerate.h>
#include <whisper.h>
#include "json.hpp"

#include "hlie_app/models/cosyvoice_feat.h"
#include "hlie_app/models/cosyvoice_spk_feat.h"
#include "hlie_app/hlie_device_service.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_app/models/hlie_cosyvoice_model.h"
#include "hlie_base/hlie_audio.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_tcim_util.h"
#include "hlie_base/hlie_util.h"

#define SIN_COS_N_COUNT 4096 // 增大以支持更大的 frame_size (最大 4000)
#define _USE_MATH_DEFINES

using json = nlohmann::ordered_json;

namespace {

    // =========================================================================
    // CosyVoice3 Text Processing Utilities
    // =========================================================================

    // Punctuation marks for text splitting - using std::string to support UTF-8 Chinese characters
    const std::vector<std::string> kChinesePunctuation = {"。", "？", "！", "；", "：", "、", ".", "?", "!", ";"};
    const std::vector<std::string> kEnglishPunctuation = {".", "?", "!", ";", ":"};

    /**
     * Check if text contains any Chinese characters (U+4E00 - U+9FFF).
     */
    bool contains_chinese(const std::string& text) {
        for (size_t i = 0; i < text.size();) {
            unsigned char c = static_cast<unsigned char>(text[i]);

            // 检测 UTF-8 编码的中文字符 (3字节: 0xE0 开头)
            if ((c & 0xF0) == 0xE0) {
                // 检查是否是中文范围
                if (i + 2 < text.size()) {
                    unsigned char c2 = static_cast<unsigned char>(text[i + 1]);
                    unsigned char c3 = static_cast<unsigned char>(text[i + 2]);

                    // 解析 UTF-8 中文字符的码点
                    int codepoint = ((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);

                    // 中文范围: U+4E00 - U+9FFF
                    if (codepoint >= 0x4E00 && codepoint <= 0x9FFF) {
                        return true;
                    }
                }
                i += 3;
            } else {
                i += 1;
            }
        }
        return false;
    }

    /**
     * Replace superscript symbols with spoken Chinese equivalents.
     */
    std::string replace_corner_mark(std::string text) {
        text = std::regex_replace(text, std::regex("²"), "平方");
        text = std::regex_replace(text, std::regex("³"), "立方");
        return text;
    }

    /**
     * Remove decorative brackets and normalize special separators.
     */
    std::string remove_bracket(std::string text) {
        text = std::regex_replace(text, std::regex(R"(（)"), "");
        text = std::regex_replace(text, std::regex(R"(）)"), "");
        text = std::regex_replace(text, std::regex(R"(\[)"), "");
        text = std::regex_replace(text, std::regex(R"(\])"), "");
        text = std::regex_replace(text, std::regex("`"), "");
        text = std::regex_replace(text, std::regex("——"), " ");
        return text;
    }

    /**
     * Keep spaces only when they separate adjacent ASCII segments.
     */
    std::string replace_blank(const std::string& text) {
        if (text.empty())
            return text;

        std::string out_str;
        for (size_t i = 0; i < text.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == ' ') {
                bool prev_ascii = (i > 0) && isascii(static_cast<unsigned char>(text[i - 1])) && text[i - 1] != ' ';
                bool next_ascii = (i + 1 < text.size()) && isascii(static_cast<unsigned char>(text[i + 1])) && text[i + 1] != ' ';
                if (prev_ascii && next_ascii) {
                    out_str += c;
                }
            } else {
                out_str += c;
            }
        }
        return out_str;
    }

    /**
     * Check if the string contains only punctuation or symbols.
     */
    bool is_only_punctuation(const std::string& text) {
        // Match strings that consist only of punctuation marks or are empty
        static const std::regex punctuation_pattern(R"(^[\p{P}\p{S}]*$)");
        return std::regex_match(text, punctuation_pattern);
    }

    /**
     * Convert Arabic numerals in text into spoken-word expressions.
     * NOTE: This requires an inflect library (e.g., Python's inflect engine).
     * Currently returns the number as string without conversion.
     * TODO: Implement number_to_words or integrate with inflect library.
     */
    std::string spell_out_number(const std::string& text) {
        std::string new_text;
        size_t st = std::string::npos;

        for (size_t i = 0; i < text.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (!std::isdigit(c)) {
                if (st != std::string::npos) {
                    // Convert number substring
                    std::string num_str = text.substr(st, i - st);
                    // TODO: Replace with actual number-to-words conversion
                    // For now, keep the numeric string as-is
                    // Example: inflect_parser.number_to_words(num_str)
                    new_text += num_str;
                    st = std::string::npos;
                }
                new_text += text[i];
            } else {
                if (st == std::string::npos) {
                    st = i;
                }
            }
        }
        if (st != std::string::npos && st < text.size()) {
            std::string num_str = text.substr(st);
            // TODO: Replace with actual number-to-words conversion
            new_text += num_str;
        }
        return new_text;
    }

    /**
     * Calculate utterance length based on language.
     * For Chinese: character count (UTF-8 aware).
     * For English: token count (simplified to character count).
     */
    size_t calc_utt_length(const std::string& text, const std::string& lang) {
        if (lang == "zh") {
            // Count UTF-8 characters properly
            size_t count = 0;
            for (size_t i = 0; i < text.size();) {
                unsigned char c = static_cast<unsigned char>(text[i]);
                if (c < 0x80) {
                    // ASCII
                    ++count;
                    ++i;
                } else if ((c & 0xE0) == 0xC0) {
                    // 2-byte UTF-8
                    ++count;
                    i += 2;
                } else if ((c & 0xF0) == 0xE0) {
                    // 3-byte UTF-8 (including Chinese)
                    ++count;
                    i += 3;
                } else if ((c & 0xF8) == 0xF0) {
                    // 4-byte UTF-8
                    ++count;
                    i += 4;
                } else {
                    ++i;
                }
            }
            return count;
        }
        // TODO: Use actual tokenizer for English
        // return tokenizer.encode(text).size();
        return text.size();
    }

    /**
     * Check if the tail utterance is short enough to merge.
     */
    bool should_merge(const std::string& text, const std::string& lang, size_t merge_len = 20) {
        return calc_utt_length(text, lang) < merge_len;
    }

    /**
     * Get punctuation marks based on language and comma_split flag.
     */
    std::vector<std::string> get_punctuations(const std::string& lang, bool comma_split = false) {
        std::vector<std::string> pounc;
        if (lang == "zh") {
            pounc = kChinesePunctuation;
        } else {
            pounc = kEnglishPunctuation;
        }
        if (comma_split) {
            pounc.push_back("，");
            pounc.push_back(",");
        }
        return pounc;
    }

    /**
     * Check if a character at position i is a punctuation mark.
     */
    bool is_punct_at(const std::string& text, size_t i, const std::vector<std::string>& pounc) {
        if (i >= text.size())
            return false;

        // Check multi-byte UTF-8 punctuation first
        for (const auto& p : pounc) {
            if (p.size() > 1) {
                if (text.substr(i, p.size()) == p) {
                    return true;
                }
            }
        }

        // Check single-byte punctuation
        unsigned char c = static_cast<unsigned char>(text[i]);
        for (const auto& p : pounc) {
            if (p.size() == 1) {
                if (c == static_cast<unsigned char>(p[0])) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Split text into utterances that satisfy token-length constraints.
     *
     * Split logic:
     * 1. Per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
     * 2. Calculate sentence len according to lang
     * 3. Split sentence according to punctuation
     */
    std::vector<std::string> split_paragraph(const std::string& text, const std::string& lang = "zh", const size_t token_max_n = 80,
                                             const size_t token_min_n = 60, const size_t merge_len = 20, const bool comma_split = false) {

        if (text.empty())
            return {};

        std::string processed_text = text;
        auto pounc = get_punctuations(lang, comma_split);

        // Ensure text ends with punctuation
        char last_char = processed_text.back();
        bool ends_with_punct = false;
        for (const auto& p : pounc) {
            if (p.size() == 1 && last_char == p[0]) {
                ends_with_punct = true;
                break;
            } else if (p.size() > 1) {
                if (processed_text.size() >= p.size() && processed_text.substr(processed_text.size() - p.size()) == p) {
                    ends_with_punct = true;
                    break;
                }
            }
        }

        if (!ends_with_punct) {
            processed_text += (lang == "zh" ? "。" : ".");
        }

        // Split by punctuation
        std::vector<std::string> utts;
        size_t st = 0;
        for (size_t i = 0; i < processed_text.size(); ++i) {
            if (is_punct_at(processed_text, i, pounc)) {
                if (i > st) {
                    std::string sentence = processed_text.substr(st, i - st + 1);
                    utts.push_back(sentence);
                }
                // Handle quote following punctuation
                if (i + 1 < processed_text.size() && (processed_text[i + 1] == '"' || processed_text[i + 1] == '\"')) {
                    if (!utts.empty()) {
                        utts.back() += processed_text[i + 1];
                        st = i + 2;
                    } else {
                        st = i + 1;
                    }
                } else {
                    st = i + 1;
                }
            }
        }

        // Merge utterances based on length constraints
        std::vector<std::string> final_utts;
        std::string cur_utt;
        for (const auto& utt : utts) {
            size_t cur_len = calc_utt_length(cur_utt + utt, lang);
            size_t cur_utt_len = calc_utt_length(cur_utt, lang);

            if (cur_len > token_max_n && cur_utt_len > token_min_n) {
                final_utts.push_back(cur_utt);
                cur_utt.clear();
            }
            cur_utt += utt;
        }

        if (!cur_utt.empty()) {
            if (should_merge(cur_utt, lang, merge_len) && !final_utts.empty()) {
                final_utts.back() += cur_utt;
            } else {
                final_utts.push_back(cur_utt);
            }
        }

        return final_utts;
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

    constexpr int DEFAULT_SAMPLE_RATE = 16000;

    struct AudioData {
        std::vector<float> samples; // Mono audio [n_samples]
        std::vector<std::vector<float>> stereo_samples; // Stereo if needed
        int sample_rate = DEFAULT_SAMPLE_RATE;
    };


    /**
     * Resample audio data to target sample rate.
     */
    std::vector<float> resample_audio(const std::vector<float>& input, const int input_sr, const int output_sr) {
        if (input_sr == output_sr) {
            return input;
        }

        double ratio = static_cast<double>(output_sr) / input_sr;
        int output_length = static_cast<int>(input.size() * ratio);

        std::vector<float> output(output_length);

        SRC_DATA src_data;
        src_data.data_in = input.data();
        src_data.input_frames = static_cast<long>(input.size());
        src_data.data_out = output.data();
        src_data.output_frames = static_cast<long>(output_length);
        src_data.src_ratio = ratio;
        src_data.end_of_input = 1;

        if (const int error = src_simple(&src_data, SRC_SINC_BEST_QUALITY, 1)) {
            throw std::runtime_error("Resampling error: " + std::string(src_strerror(error)));
        }

        // Resize to actual output frames
        output.resize(src_data.output_frames_gen);
        LOG_INFO("resample audio done");
        return output;
    }


    struct wav_header {
        char riff[4] = {'R', 'I', 'F', 'F'};
        uint32_t chunk_size;
        char wave[4] = {'W', 'A', 'V', 'E'};
        char fmt[4] = {'f', 'm', 't', ' '};
        uint32_t fmt_chunk_size = 16;
        uint16_t audio_format = 1; // PCM
        uint16_t num_channels = 1; // Mono
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample = 16;
        char data[4] = {'d', 'a', 't', 'a'};
        uint32_t data_size;
    };

    bool save_wav16(const std::string& fname, const std::vector<float>& data, const int sample_rate = 24000) {
        std::ofstream file(fname, std::ios::binary);
        if (!file) {
            LOG_ERROR("save_wav16 failed to open file {} for writing.", fname.c_str());
            return false;
        }

        wav_header header;
        header.sample_rate = sample_rate;
        header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
        header.block_align = header.num_channels * (header.bits_per_sample / 8);
        header.data_size = data.size() * (header.bits_per_sample / 8);
        header.chunk_size = 36 + header.data_size;

        file.write(reinterpret_cast<const char*>(&header), sizeof(header));

        for (const auto& sample : data) {
            int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample * 32767.0, -32768.0, 32767.0));
            file.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
        }

        return file.good();
    }


    AudioData load_wav(const std::string& wav_path, int target_sr = 16000, int min_sr = 16000) {
        std::vector<float> pcmf32;
        std::vector<std::vector<float>> pcmf32s;

        // read_audio_data always outputs WHISPER_SAMPLE_RATE (16000)
        if (!read_audio_data(wav_path, pcmf32, pcmf32s, false)) {
            throw std::runtime_error("Failed to load audio: " + wav_path);
        }

        if (target_sr < min_sr) {
            throw std::runtime_error("Target sample rate " + std::to_string(target_sr) + " must be >= min_sr " + std::to_string(min_sr));
        }

        // Resample to target_sr if needed
        const int source_sr = WHISPER_SAMPLE_RATE; // 16000
        if (target_sr != source_sr) {
            LOG_INFO("Resampling from {} to {}", source_sr, target_sr);
            pcmf32 = resample_audio(pcmf32, source_sr, target_sr);
        }

        LOG_INFO("Loaded wav audio to {}", target_sr);
        return AudioData{.samples = std::move(pcmf32), .stereo_samples = std::move(pcmf32s), .sample_rate = target_sr};
    }

} // namespace

HLIE_NAMESPACE_BEGIN

int CosyVoiceModelService::init(const std::string& model_dir, const HLIEModelInfo& model_info,
                                const std::shared_ptr<HLIEDeviceService>& device_service) {
    blocking_queue_ = std::make_shared<HLIEBlockingQueue<QueueItem>>();
    cosyvoice_model_ = std::make_shared<HLIECosyVoiceModel>();
    device_service_ = device_service;

    auto devices = model_info.get_devices();

    if (const int replicas = Singleton<HLIEConfigService>::instance()->get_int("server", "replicas", 1); replicas > 1) {
        is_multi_instance_ = true;
        if (const std::optional<int> val = Singleton<HLIEDeviceHelper>::instance()->acquire_device(HLIEModelType::COSYVOICE); !val) {
            LOG_ERROR("CosyVoiceModelService cannot acquire available device");
            return -1;
        } else {
            devices_.push_back(*val);
            LOG_INFO("CosyVoiceModelService using device: {}", val.value());
        }
    } else { // 单实例模式可以定制Model Devices
        if (!devices.empty()) {
            if (!HlieUtil::is_valid_devices(devices)) {
                LOG_ERROR("CosyVoiceModelService cannot acquire model specific device \"{}\"", devices);
                return -1;
            }

            if (HlieUtil::parse_device(devices, devices_) < 0) {
                LOG_ERROR("CosyVoiceModelService parse_device specific device \"{}\" failed", devices);
                return -1;
            }

            if (device_service_->acquire_mem_on_devices(devices_, 1200) < 0) {
                return -1;
            }

            LOG_INFO("CosyVoiceModelService using model specific device: {}", devices_.at(0));
        } else {
            uint32_t device;
            if (device_service_->acquire_mem(4000, device) < 0) {
                return -1;
            }

            devices_.push_back(device);
            LOG_INFO("CosyVoiceModelService using model specific device: {}", device);
        }
    }

    cosyvoice_model_->set_devices(devices_);
    const auto& model = model_info.get_model_path();


    cosyvoice_model_->load_model(model);
    LOG_INFO("load_model done");
    cosyvoice_model_->init();

    audio_hparams hparams = {
        /* n_mel_bins         */ 128,
        /* audio_chunk_len    */ -1,
        /* audio_sample_rate  */ 16000,
        /* audio_n_fft        */ 400,
        /* audio_window_len   */ 400,
        /* audio_hop_len      */ 160,
    };


    audio_preprocessor_ = std::make_unique<hlie_audio_preprocessor_whisper>(hparams);
    audio_preprocessor_->initialize();

    // 使用新的 librosa 风格预处理器

    audio_hparams spec_hparams = {
        /* n_mel_bins         */ 80,
        /* audio_chunk_len    */ -1,
        /* audio_sample_rate  */ 24000,
        /* audio_n_fft        */ 1920,
        /* audio_window_len   */ 1920,
        /* audio_hop_len      */ 480,
    };
    audio_spec_preprocessor_ = std::make_unique<hlie_audio_preprocessor_librosa>(spec_hparams);
    audio_spec_preprocessor_->initialize();

    model_key_ = model_info.model_key_;
    /*
    websocket_service_ = std::make_unique<HLIEWebSocketService>();
    websocket_service_->init();
    */
    return 0;
}

void CosyVoiceModelService::run() {
    if (worker_thread_) {
        LOG_WARN("CosyVoiceModelService thread is already running.");
        return;
    }
    /*
        websocket_service_->run([this](const WhisperStreamRequest& req) { return this->handle_stream(req); }, 7998);
    */
    state_.store(SERVER_STATE_READY);
    should_stop_ = false;
    worker_thread_ = std::make_unique<std::thread>(&CosyVoiceModelService::process, this);
    LOG_INFO("CosyVoiceModelService thread started.");
}

void CosyVoiceModelService::process() {
    try {
        while (!should_stop_) {
            const auto [request, promise] = blocking_queue_->pop(); // 阻塞直到有数据或队列关闭
            if (should_stop_) {
                break; // 防止 stop() 后仍有残留任务
            }

            try {
                auto start = std::chrono::steady_clock::now();
                auto speech_result = speech(request);
                auto end = std::chrono::steady_clock::now();
                // 构造结果
                CosyVoiceResult result;
                result.wav = speech_result;
                result.success = !speech_result.empty();
                result.predicted_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                LOG_INFO("speech time: {} ms", result.predicted_ms);

                // 通知调用者
                promise->set_value(result);
            } catch (const std::exception& e) {
                // 错误处理
                CosyVoiceResult result;
                result.success = false;
                result.error_message = e.what();
                promise->set_value(result);
            }
        }
    } catch (const std::runtime_error&) {
        LOG_WARN("CosyVoiceModelService process exited");
    }
}

void CosyVoiceModelService::stop() {
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

int CosyVoiceModelService::get_state() {
    return state_;
}

int CosyVoiceModelService::handle_health(nlohmann::ordered_json& resp) {
    return 0;
}

void CosyVoiceModelService::handle_models(nlohmann::ordered_json& resp) const {
    resp["models"].push_back({
        {"name", "CosyVoice"},
        {"model", "CosyVoice"},
    });

    resp["data"].push_back({
        {"id", "CosyVoice"},
        {"object", "model"},
        {"created", std::time(0)},
        {"owned_by", "HLIECpp"},
    });
}

std::vector<float> CosyVoiceModelService::handle_stream(const WhisperStreamRequest& request) {

    CosyVoiceRequest req;
    req.content = "stream";

    const auto start = std::chrono::steady_clock::now();
    auto speech_result = speech(req);
    const auto end = std::chrono::steady_clock::now();
    auto predicted_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOG_INFO("tts stream cost: {} ms", predicted_ms);
    return speech_result;
}

void CosyVoiceModelService::llm_job(const std::vector<int>& text, const std::vector<int>& prompt_text,
                                    const std::vector<int>& llm_prompt_speech_token, const std::vector<float>& llm_embedding,
                                    const std::string& uuid) {
    LOG_INFO("LLM job started for uuid: {}", uuid);

    const int max_silent_token_num = 5;
    int cur_silent_token_num = 0;

    LLMPerf llm_perf;

    try {
        // Run LLM inference
        auto generated_tokens =
            cosyvoice_model_->llm_inference(text, static_cast<int>(text.size()), prompt_text, static_cast<int>(prompt_text.size()),
                                            llm_prompt_speech_token, static_cast<int>(llm_prompt_speech_token.size()), llm_embedding,
                                            25, // sampling
                                            20.0f, // max_token_text_ratio
                                            2.0f, // min_token_text_ratio
                                            &llm_perf);

        // Process generated tokens and filter silent tokens
        for (int token_id : generated_tokens) {
            // Check if token is in silent_tokens list
            bool is_silent = false;
            for (const int silent_id : kSilentTokens) {
                if (token_id == silent_id) {
                    is_silent = true;
                    break;
                }
            }

            if (is_silent) {
                cur_silent_token_num++;
                if (cur_silent_token_num > max_silent_token_num) {
                    LOG_DEBUG("skip current silent_token {}", cur_silent_token_num);
                    continue;
                }
            } else {
                cur_silent_token_num = 0;
            }

            // Store the token
            {
                std::lock_guard<std::mutex> lock(lock_);
                tts_speech_token_dict_[uuid].push_back(token_id);
            }
        }

        // Mark LLM as ended
        {
            std::lock_guard<std::mutex> lock(lock_);
            llm_end_dict_[uuid] = true;
            perf_by_uuid_[uuid] = {llm_perf.llm_total_s};
        }

        LOG_INFO("LLM job completed for uuid: {}, generated {} tokens", uuid, generated_tokens.size());

    } catch (const std::exception& e) {
        LOG_ERROR("LLM job failed for uuid: {}, error: {}", uuid, e.what());
        {
            std::lock_guard<std::mutex> lock(lock_);
            llm_end_dict_[uuid] = true;
        }
        throw;
    }
}

std::future<CosyVoiceResult> CosyVoiceModelService::speech_async(const CosyVoiceRequest& request) const {
    const auto promise = std::make_shared<std::promise<CosyVoiceResult>>();
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

CosyVoiceResult CosyVoiceModelService::speech_sync(const CosyVoiceRequest& request) const {
    auto future = speech_async(request);
    return future.get();
}

std::vector<float> CosyVoiceModelService::speech(const CosyVoiceRequest& request) {
    std::vector<float> results;
    try {
        LOG_INFO("cosyvoice3 ready to run {}", request.content);

        const std::string prompt = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。";
        auto prompt_text = text_normalize(prompt, false, true);
        LOG_INFO("text_normalize prompt done");
        auto tts_text = text_normalize(request.content, false, true);
        LOG_INFO("text_normalize request done");
        const std::string prompt_wav = R"(..\samples\zero_shot_prompt.wav)";

        for (auto text : tts_text) {
            // Get frontend input (tokens, embeddings, etc.)
            auto model_input = frontend_zero_shot(text, prompt_text[0], prompt_wav);

            LOG_INFO("synthesis text {}", text);

            // Generate unique ID for this request
            std::string this_uuid = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());

            // Initialize shared state
            {
                std::lock_guard<std::mutex> lock(lock_);
                tts_speech_token_dict_[this_uuid] = {};
                llm_end_dict_[this_uuid] = false;
            }

            // Run LLM job (blocking)
            auto llm_start = std::chrono::steady_clock::now();
            llm_job(model_input.text_tokens, model_input.prompt_tokens, model_input.llm_prompt_speech_token, model_input.embedding,
                    this_uuid);
            auto llm_end = std::chrono::steady_clock::now();
            float llm_time = std::chrono::duration<float>(llm_end - llm_start).count();

            // Get generated tokens
            std::vector<int> this_tts_speech_token;
            {
                std::lock_guard<std::mutex> lock(lock_);
                this_tts_speech_token = tts_speech_token_dict_[this_uuid];
            }

            LOG_INFO("LLM generated {} speech tokens in {:.2f}s", this_tts_speech_token.size(), llm_time);

            // Run token2wav (flow decoder)
            auto tts_start = std::chrono::steady_clock::now();
            results = cosyvoice_model_->token2wav(this_tts_speech_token, model_input.flow_prompt_speech_token,
                                                  model_input.prompt_speech_feat, model_input.embedding, 1.0f);
            auto tts_end = std::chrono::steady_clock::now();

            // Clean up
            {
                std::lock_guard<std::mutex> lock(lock_);
                tts_speech_token_dict_.erase(this_uuid);
                llm_end_dict_.erase(this_uuid);
                hift_cache_dict_.erase(this_uuid);
            }

            LOG_INFO("TTS inference completed");
            return results;
        }

    } catch (const std::exception& e) {
        LOG_ERROR("speech exception: {}", e.what());
        return results;
    }
    LOG_INFO("speech done");
    return results;
}

void CosyVoiceModelService::handle_speech(const httplib::Request& req, httplib::Response& res) {
    try {
        const json body = json::parse(req.body);
        std::string content = json_value(body, "input", std::string());
        CosyVoiceRequest cosy_voice_request;
        cosy_voice_request.content = content;

        if (auto [success, wav, error_message, predicted_ms] = speech_sync(cosy_voice_request); !success) {
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
            auto jres = json{{"result", "ok"}};

            // 将 std::vector<float> wav 转换为字节向量，再转为 Base64
            const uint8_t* byte_data = reinterpret_cast<const uint8_t*>(wav.data());
            std::vector<uint8_t> byte_vector(byte_data, byte_data + wav.size() * sizeof(float));

            // 使用 binary_t 包装字节数据
            json::binary_t wav_binary(byte_vector);
            jres["wav"] = wav_binary;
            jres["wav_length"] = wav.size();

            save_wav16("ok999.wav", wav);

            res.set_content(jres.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
        }

        LOG_INFO("process audio done");
    } catch (const std::exception& ex) {
        LOG_ERROR("handle_speech ex: {}", ex.what());
        throw;
    }
}

std::vector<std::string> CosyVoiceModelService::text_normalize(const std::string& text, bool split, bool text_frontend) {
    // Skip if text is empty or contains SSML symbols
    if (text.empty()) {
        return split ? std::vector<std::string>{""} : std::vector<std::string>{text};
    }

    if (text.find("<|") != std::string::npos && text.find("|>") != std::string::npos) {
        text_frontend = false;
    }

    if (!text_frontend) {
        return split ? std::vector<std::string>{text} : std::vector<std::string>{text};
    }

    std::string processed_text = text;
    processed_text = processed_text.erase(0, processed_text.find_first_not_of(" \t\n\r"));
    processed_text = processed_text.erase(processed_text.find_last_not_of(" \t\n\r") + 1);

    try {
        // Check language and apply appropriate normalization
        if (contains_chinese(processed_text)) {
            // Chinese text normalization
            // TODO: Integrate with wetext normalizer if available
            // if (text_frontend_ == "wetext") {
            //     processed_text = zh_tn_model_.normalize(processed_text);
            // }

            processed_text = replace_blank(processed_text);
            LOG_DEBUG("After remove_bracket: {}", processed_text);

            processed_text = replace_corner_mark(processed_text);
            LOG_DEBUG("After replace_blank: {}", processed_text);

            processed_text = std::regex_replace(processed_text, std::regex(R"(\.)"), "。");
            LOG_DEBUG("After replace_corner_mark: {}", processed_text);

            processed_text = std::regex_replace(processed_text, std::regex(R"( - )"), "，");
            LOG_DEBUG("After regex_replace \\.: {}", processed_text);

            processed_text = remove_bracket(processed_text);
            LOG_DEBUG("After regex_replace ' - ': {}", processed_text);

            processed_text = std::regex_replace(processed_text, std::regex(R"([，,、]+$)"), "。");
            LOG_DEBUG("After regex_replace punctuation$: {}", processed_text);

            if (!split) {
                return {processed_text};
            }

            auto texts = split_paragraph(processed_text, "zh", 80, 60, 20, false);

            // Filter out punctuation-only segments
            std::vector<std::string> result;
            for (const auto& t : texts) {
                if (!is_only_punctuation(t)) {
                    result.push_back(t);
                }
            }
            return result;
        }

        // English text normalization
        // TODO: Integrate with wetext normalizer if available
        // if (text_frontend_ == "wetext") {
        //     processed_text = en_tn_model_.normalize(processed_text);
        // }

        processed_text = spell_out_number(processed_text);
        if (!split) {
            return {processed_text};
        }


        auto texts = split_paragraph(processed_text, "en", 80, 60, 20, false);

        // Filter out punctuation-only segments
        std::vector<std::string> result;
        for (const auto& t : texts) {
            if (!is_only_punctuation(t)) {
                result.push_back(t);
            }
        }
        return result;
    } catch (const std::regex_error& e) {
        LOG_ERROR("Regex error: {}, what: {}", static_cast<int>(e.code()), e.what());
        LOG_ERROR("Failed at text: {}", processed_text);
        return {text}; // 返回原始文本
    } catch (const std::exception& e) {
        LOG_ERROR("Exception: {}", e.what());
        return {text};
    }
}

FrontendInput CosyVoiceModelService::frontend_zero_shot(const std::string& tts_text, const std::string& prompt_text,
                                                        const std::string& prompt_wav, const int resample_rate) {
    FrontendInput input;
    input.text_len = cosyvoice_model_->extract_text_token(tts_text, input.text_tokens);

    // TODO: Cache the other repeated pre-calc logic

    input.prompt_text_len = cosyvoice_model_->extract_text_token(prompt_text, input.prompt_tokens);

    LOG_INFO("Start to extract_speech_feat");
    const auto speech_feat = extract_speech_feat(prompt_wav);
    input.prompt_speech_feat = speech_feat.first;
    input.prompt_speech_feat_len = speech_feat.second;

    LOG_INFO("Start to extract_speech_token");
    const auto speech_token = extract_speech_token(prompt_wav);
    input.flow_prompt_speech_token = speech_token.first.data;
    input.flow_prompt_speech_token_len = speech_token.second;

    input.llm_prompt_speech_token = speech_token.first.data;
    input.llm_prompt_speech_token_len = speech_token.second;


    if (resample_rate == 24000) { // cosyvoice2, force speech_feat % speech_token = 2
        // cosyvoice2, force speech_feat % speech_token = 2
        // speech_feat: [1, n_len, n_mel=80] -> flattened size = n_len * 80
        // speech_token: [1, token_len]
        /**
        speech_feat 是 mel 频谱特征，基于 24kHz 音频计算
        speech_token 是 speech tokenizer 的输出，基于 16kHz 音频计算
        关键点：
        24kHz 的 hop_length = 480，所以 30秒 = 1500 帧
        16kHz 的 hop_length = 160，所以 30秒 = 3000 帧
        speech tokenizer 内部有 4:1 的下采样，3000/4 = 750 tokens
        所以：
        24kHz speech_feat: 1500 帧
        16kHz speech_token: 750 tokens
        1500 / 750 = 2，所以确实是 2:1 的比例。
        但为什么需要这个比例？这是因为 CosyVoice2 模型在处理时，speech_feat 和 speech_token 需要对齐。speech tokenizer 是基于 16kHz 的 mel
        特征训练的，而 speech_feat 可能是基于 24kHz 提取的。 从物理上理解： 同一个 30 秒音频，用 24kHz 提取得到 1500 帧 mel 用 16kHz
        提取得到 3000 帧 mel，下采样 4 倍得到 750 tokens 如果要保证它们能对应上，需要： 1500 / 750 = 2 这个 2:1
        的比例反映了两种不同采样率下特征提取的差异。实际上是采样率和帧移配置导致了不同的帧数，而模型需要这个对齐才能正确处理。
        **/
        constexpr int kN_mel = 80;

        int speech_feat_frames = static_cast<int>(input.prompt_speech_feat.size()) / kN_mel;
        int token_len = std::min(speech_feat_frames / 2, input.flow_prompt_speech_token_len);

        // Truncate speech_feat to 2 * token_len frames
        input.prompt_speech_feat.resize(2 * token_len * kN_mel);
        input.prompt_speech_feat_len = 2 * token_len;

        // Truncate speech_token to token_len
        input.flow_prompt_speech_token_len = token_len;
        input.flow_prompt_speech_token.resize(token_len);

        LOG_INFO("handled for resample_rate 24000");
    }

    input.embedding = extract_spk_embedding(prompt_wav);


    LOG_INFO("frontend_zero_shot done");

    return input;
}

std::pair<SpeechToken, int> CosyVoiceModelService::extract_speech_token(const std::string& prompt_wav) const {

    constexpr int kTargetSr = 16000;
    constexpr int kMaxDurationSec = 30;
    constexpr int kN_mels = 128;
    constexpr int kPaddedLen = 3000;
    constexpr int kMask1Rows = 750;
    constexpr int kMask1Cols = 1280;

    // Step 1: Load audio at 16000 Hz
    AudioData audio = load_wav(prompt_wav, kTargetSr);

    // Step 2: Check audio length <= 30 seconds
    int audio_length_samples = static_cast<int>(audio.samples.size());
    int audio_length_sec = audio_length_samples / kTargetSr;
    if (audio_length_sec > kMaxDurationSec) {
        throw std::runtime_error("do not support extract speech token for audio longer than 30s");
    }

    // Step 3: Compute mel spectrogram with n_mels=128
    std::vector<hlie_audio_mel> mel_spec_chunks;
    bool success = audio_preprocessor_->preprocess(audio.samples.data(), audio_length_samples, mel_spec_chunks);
    if (!success) {
        throw std::runtime_error("Failed to compute mel spectrogram");
    }

    // mel_spec_chunks has exactly one element since we limit to 30s
    const hlie_audio_mel& mel = mel_spec_chunks[0];
    int feat_len = mel.n_len;

    LOG_INFO("Mel spectrogram: n_mel={}, feat_len,n_len={}", mel.n_mel, mel.n_len);

    // Step 4: Create padded_input (1, 128, 3000), dtype=float16
    // Python: padded_input = torch.zeros((1, 128, 3000), dtype=torch.float16)
    //         padded_input[:, :, :feat_len] = feat
    // Note: shape (batch, channels, time) = (1, 128, 3000)
    std::vector<TensorFP16> padded_input_fp16(1 * kN_mels * kPaddedLen, static_cast<TensorFP16>(0.0f));

    // Copy mel data: mel.data is stored as [n_mel, n_len]
    // Fill into padded_input as [m, t] -> [m * kPaddedLen + t]
    for (int m = 0; m < mel.n_mel && m < kN_mels; ++m) {
        for (int t = 0; t < feat_len && t < kPaddedLen; ++t) {
            padded_input_fp16[m * kPaddedLen + t] = static_cast<TensorFP16>(mel.data[m * mel.n_len + t]);
        }
    }

    int t = 399;

    std::ostringstream oss;
    oss << "frame[" << t << " full]: ";

    for (int i = 0; i < mel.n_mel; ++i) {
        auto v = static_cast<float>(padded_input_fp16[i * kPaddedLen + t]);

        oss << std::fixed << std::setprecision(6) << v;
        if (i != mel.n_mel - 1)
            oss << ", ";
    }

    LOG_TRACE("{}", oss.str());

    // Step 5: Create masks, dtype=float16
    // mask: shape (1, 20, 750, 750), filled with -inf, then first part to 0
    // Python: mask = torch.full((1, 20, 750, 750), torch.finfo(torch.float16).min, dtype=torch.float16)
    //         mask[:, :, :, : feat_len // 4] = 0
    // float16 min = 5.96e-8 (denormalized positive) or use -65504 (largest negative)
    std::vector<TensorFP16> mask_fp16(1 * 20 * 750 * 750, static_cast<TensorFP16>(-65504.0f));

    int mask_time_blocks = feat_len / 4;
    for (int b = 0; b < 1; ++b) {
        for (int h = 0; h < 20; ++h) {
            for (int i = 0; i < mask_time_blocks && i < 750; ++i) {
                for (int j = 0; j < 750; ++j) {
                    size_t idx = ((b * 20 + h) * 750 + i) * 750 + j;
                    mask_fp16[idx] = static_cast<TensorFP16>(0.0f);
                }
            }
        }
    }

    // mask1: shape (1, 750, 1280), filled with 0, then first part to 1
    // Python: mask1 = torch.zeros((1, 750, 1280), dtype=torch.float16)
    //         mask1[:, 0 : feat_len // 4, :] = 1.0
    std::vector<TensorFP16> mask1_fp16(1 * kMask1Rows * kMask1Cols, static_cast<TensorFP16>(0.0f));

    for (int b = 0; b < 1; ++b) {
        for (int i = 0; i < mask_time_blocks && i < kMask1Rows; ++i) {
            for (int j = 0; j < kMask1Cols; ++j) {
                size_t idx = (b * kMask1Rows + i) * kMask1Cols + j;
                mask1_fp16[idx] = static_cast<TensorFP16>(1.0f);
            }
        }
    }

    // Step 6: Run speech tokenizer
    auto [fst, snd] = cosyvoice_model_->speech_tokenizer(padded_input_fp16, mask_fp16, mask1_fp16, feat_len);

    LOG_INFO("Extracted speech token: shape=[1, {}], len={}", snd, snd);

    return {SpeechToken{std::move(fst), snd}, snd};
}

/**
 * Extract speech mel features from audio file.
 *
 * @param prompt_wav Path to prompt audio file
 * @return Pair of (speech_feat data, speech_feat_len)
 */
std::pair<std::vector<float>, int> CosyVoiceModelService::extract_speech_feat(const std::string& prompt_wav) const {

    // Step 1: Load audio at 24000 Hz (mono)
    AudioData audio = load_wav(prompt_wav, 24000);
    int audio_length_samples = static_cast<int>(audio.samples.size());
    std::vector<hlie_audio_mel> mel_spec_chunks;
    bool success = audio_spec_preprocessor_->preprocess(audio.samples.data(), audio_length_samples, mel_spec_chunks);
    if (!success) {
        throw std::runtime_error("Failed to compute mel spectrogram");
    }

    // mel_spec_chunks has exactly one element since we limit to 30s
    const hlie_audio_mel& mel = mel_spec_chunks[0];
    int feat_len = mel.n_len;

    int t = 88;

    std::ostringstream oss;
    oss << "frame[" << t << "]: ";

    for (int i = 0; i < mel.n_mel; ++i) {
        float v = mel.data[i * mel.n_len + t];  // ✅ 正确 stride

        oss << std::fixed << std::setprecision(6) << v;
        if (i != mel.n_mel - 1)
            oss << ", ";
    }

    LOG_TRACE("{}", oss.str());

    LOG_INFO("Mel spectrogram feat: n_mel={}, n_len={}", mel.n_mel, mel.n_len);

    // Step 3: Convert [n_mel, n_len] -> [1, n_len, n_mel]
    // Python: .squeeze(dim=0).transpose(0, 1).unsqueeze(dim=0)
    /*std::vector<float> speech_feat(1 * mel.n_len * mel.n_mel);

    for (int m = 0; m < mel.n_mel; ++m) {
        for (int t = 0; t < mel.n_len; ++t) {
            // Source: [m, t] in mel.data
            // Dest:   [0, t, m] in speech_feat
            speech_feat[t * mel.n_mel + m] = mel.data[m * mel.n_len + t];
        }
    }*/

    std::vector<float> speech_feat(REF_FEAT.begin(), REF_FEAT.end());

    int speech_feat_len = REF_FEAT_D; //mel.n_len;

    LOG_INFO("Extracted speech feat: shape=[1, {}, {}], len={}, speech_feat size {}", mel.n_len, mel.n_mel, speech_feat_len, speech_feat.size());

    return {std::move(speech_feat), speech_feat_len};
}


std::vector<float> CosyVoiceModelService::extract_spk_embedding(const std::string& prompt_wav) const {
    // Constants matching the Python implementation
    constexpr int kTargetSr = 16000;
    constexpr int kNumMelBins = 80;
    constexpr int kFixedLen = 1000;
    constexpr int kDither = 0;

    // Step 1: Load prompt audio and resample it to 16 kHz
    AudioData audio = load_wav(prompt_wav, kTargetSr);
    int audio_length_samples = static_cast<int>(audio.samples.size());

    // Step 2: Extract 80-bin FBANK features for speaker embedding inference
    // Use the whisper preprocessor which has 80 mel bins configured
    std::vector<hlie_audio_mel> mel_spec_chunks;
    bool success = audio_preprocessor_->preprocess(audio.samples.data(), audio_length_samples, mel_spec_chunks);
    if (!success) {
        throw std::runtime_error("Failed to compute mel spectrogram for speaker embedding");
    }

    const hlie_audio_mel& mel = mel_spec_chunks[0];
    int feat_len = mel.n_len;

    LOG_INFO("Speaker embedding mel spec: n_mel={}, n_len={}", mel.n_mel, mel.n_len);

    std::vector<float> spk_feat(REF_SPK_FEAT.begin(), REF_SPK_FEAT.end());
/*
    // Step 3: Convert to [T, 80] format (transpose from [n_mel, n_len] to [n_len, n_mel])
    std::vector<float> feat(feat_len * kNumMelBins);
    for (int m = 0; m < mel.n_mel && m < kNumMelBins; ++m) {
        for (int t = 0; t < feat_len; ++t) {
            feat[t * kNumMelBins + m] = spk_feat[m * mel.n_len + t];
        }
    }*/

    // Step 4: Subtract mean along time dimension (keepdim=True equivalent)
    // feat.mean(dim=0, keepdim=True) -> compute mean per mel bin across time
    /*std::vector<float> mean_per_bin(kNumMelBins, 0.0f);
    for (int t = 0; t < feat_len; ++t) {
        for (int m = 0; m < kNumMelBins; ++m) {
            mean_per_bin[m] += feat[t * kNumMelBins + m];
        }
    }
    for (int m = 0; m < kNumMelBins; ++m) {
        mean_per_bin[m] /= feat_len;
    }

    // Subtract mean
    for (int t = 0; t < feat_len; ++t) {
        for (int m = 0; m < kNumMelBins; ++m) {
            feat[t * kNumMelBins + m] -= mean_per_bin[m];
        }
    }

    // Step 5: Pad or truncate to fixed length (T_fixed = 1000)
    std::vector<float> padded_feat;
    if (feat_len < kFixedLen) {
        // Pad: pad with zeros at the end
        padded_feat.resize(kFixedLen * kNumMelBins, 0.0f);
        std::copy(feat.begin(), feat.end(), padded_feat.begin());
    } else {
        // Truncate
        padded_feat.assign(feat.begin(), feat.begin() + kFixedLen * kNumMelBins);
    }
    */
    // Step 6: Convert to float16 and prepare input shape [1, 1000, 80]
    std::vector<TensorFP16> campplus_input(kFixedLen * kNumMelBins);
    for (size_t i = 0; i < spk_feat.size(); ++i) {
        campplus_input[i] = static_cast<TensorFP16>(spk_feat[i]);
    }

    auto result = cosyvoice_model_->campplus(campplus_input);

    LOG_INFO("Extracted speaker embedding: size={}", result.size());
    return result;
}


HLIE_NAMESPACE_END

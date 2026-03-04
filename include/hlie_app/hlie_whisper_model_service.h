#ifndef HLIE_WHISPER_MODEL_SERVICE_H
#define HLIE_WHISPER_MODEL_SERVICE_H

#include <future>
#include <memory>
#include <string>
#include <cfloat>

#include "hlie_base/hlie_blocking_queue.h"
#include "hlie_model_service.h"
#include "httplib.h"
#include "json.hpp"
#include "whisper.h"



// output formats
const std::string json_format = "json";
const std::string text_format = "text";
const std::string srt_format = "srt";
const std::string vjson_format = "verbose_json";
const std::string vtt_format = "vtt";


struct whisper_params {
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t progress_step = 5;
    int32_t max_context = -1;
    int32_t max_len = 0;
    int32_t best_of = 2;
    int32_t beam_size = -1;
    int32_t audio_ctx = 0;

    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    float logprob_thold = -1.00f;
    float temperature = 0.00f;
    float temperature_inc = 0.20f;
    float no_speech_thold = 0.6f;

    bool debug_mode = false;
    bool translate = false;
    bool detect_language = false;
    bool diarize = false;
    bool tinydiarize = false;
    bool split_on_word = false;
    bool no_fallback = false;
    bool print_special = false;
    bool print_colors = false;
    bool print_realtime = false;
    bool print_progress = false;
    bool no_timestamps = false;
    bool use_gpu = true;
    bool flash_attn = true;
    int32_t gpu_device = 0;
    bool suppress_nst = false;
    bool no_context = true;
    bool no_language_probabilities = false;

    std::string language = "en";
    std::string prompt = "";
    std::string font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
    std::string model = "models/ggml-base.en.bin";

    std::string response_format = json_format;

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]"; // TODO: set from command line

    std::string openvino_encode_device = "CPU";

    std::string dtw = "";

    // Voice Activity Detection (VAD) parameters
    bool vad = false;
    std::string vad_model = "";
    float vad_threshold = 0.5f;
    int vad_min_speech_duration_ms = 250;
    int vad_min_silence_duration_ms = 100;
    float vad_max_speech_duration_s = FLT_MAX;
    int vad_speech_pad_ms = 30;
    float vad_samples_overlap = 0.1f;
};

HLIE_NAMESPACE_BEGIN

/**
 * 生图请求消息体
 */
struct WhisperRequest {
    struct whisper_full_params params;
    std::string filename;
    std::vector<float> pcmf32; // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM
    int n_processors;
};

struct WhisperResult {
    std::string result;
    bool success = false;
    std::string error_message;
    int64_t predicted_ms;
};

class WhisperModelService final : public IHLIEModelService {
public:
    WhisperModelService() = default;
    ~WhisperModelService() override = default;

    /**
     * 初始化并传递参数给SDCtx
     */
    int init(const std::string& model_dir, const HLIEModelInfo& model_info,
             const std::shared_ptr<HLIEDeviceService>& device_service) override;

    /**
     * 执行模型推理
     */
    void run() override;

    void process() override;

    void stop() override;

    int get_state() override;

    int handle_health(nlohmann::ordered_json& resp);

    void handle_models(nlohmann::ordered_json& resp) const;

    void handle_transcriptions(const httplib::Request& req, httplib::Response& res);

    [[nodiscard]] std::string get_model_key() const {
        return model_key_;
    }

private:
    std::future<WhisperResult> transcription_async(const WhisperRequest& request) const;
    WhisperResult transcription_sync(const WhisperRequest& request) const;

    std::string transcription(const WhisperRequest& request) const;

    whisper_params params_;
    whisper_params default_params_;

    std::string model_key_;

    whisper_context* whisper_ctx_{nullptr};

    struct QueueItem {
        WhisperRequest request;
        std::shared_ptr<std::promise<WhisperResult>> promise;
    };

    std::shared_ptr<HLIEBlockingQueue<QueueItem>> blocking_queue_;

    std::unique_ptr<std::thread> worker_thread_; // 工作线程
    std::atomic<bool> should_stop_{false}; // 线程停止标志
};

HLIE_NAMESPACE_END

#endif

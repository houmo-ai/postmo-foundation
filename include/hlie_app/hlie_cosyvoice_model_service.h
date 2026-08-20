#ifndef HLIE_COSYVOICE_MODEL_SERVICE_H
#define HLIE_COSYVOICE_MODEL_SERVICE_H

#include <future>
#include <memory>
#include <string>

#include "hlie_base/hlie_audio.h"
#include "hlie_base/hlie_blocking_queue.h"
#include "hlie_model_service.h"
#include "hlie_ws_service.h"
#include "httplib.h"
#include "json.hpp"


HLIE_NAMESPACE_BEGIN

class HLIETokenizer;

class HLIECosyVoiceModel;

struct CosyVoiceRequest {
    HLIEQueuedRequestControl control;
    std::string content;
    std::string prompt_text;
    std::string prompt_wav_path;
    int steps = 10;
};

struct CosyVoiceResult {
    bool success = false;
    std::vector<float> wav;
    std::string error_message;
    int64_t predicted_ms = 0;
};

/**
 * Build model inputs for zero-shot voice cloning synthesis.
 *
 * NOTE: This is a placeholder. Full implementation requires:
 * - Text tokenization (CosyVoice3Tokenizer)
 * - Speech token extraction (speech_tokenizer runtime)
 * - Speaker embedding extraction (campplus runtime)
 * - Speech feature extraction (mel spectrogram)
 */
struct FrontendInput {
    std::vector<int> text_tokens;
    int text_len;
    std::vector<int> prompt_tokens;
    int prompt_text_len;
    std::vector<int> llm_prompt_speech_token;
    int llm_prompt_speech_token_len;
    std::vector<int> flow_prompt_speech_token;
    int flow_prompt_speech_token_len;
    std::vector<float> prompt_speech_feat;
    std::vector<float> embedding;
};

/**
 * Speech token extraction result.
 */
struct SpeechToken {
    std::vector<int> data; // Speech tokens [batch, token_len]
    int token_len; // Length of tokens

    // Shape: [1, feat_len / 4]
    int shape() const {
        return token_len;
    }
};

/**
 * Structure to hold LLM performance metrics.
 */
struct LLMStagePerf {
    float llm_stage_wall_s = 0.0f; // Wall time for LLM stage (includes prefill+decode+host overhead)
};


class CosyVoiceModelService final : public IHLIEModelService {
public:
    CosyVoiceModelService() = default;
    ~CosyVoiceModelService() override = default;

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

    std::vector<float> handle_stream(const WhisperStreamRequest& request);

    void handle_speech(const httplib::Request& req, httplib::Response& res);

    [[nodiscard]] std::string get_model_key() const {
        return model_key_;
    }

    /**
     * Run the LLM stage and store generated speech tokens for a request.
     *
     * This is the main entry point for LLM inference, equivalent to the Python
     * llm_job method. It runs the speech token LLM to generate tokens from text.
     *
     * @param text Target text token sequence
     * @param prompt_text Prompt text token sequence
     * @param llm_prompt_speech_token Prompt speech-token sequence
     * @param llm_embedding Speaker embedding used by the LLM
     * @param uuid Unique request identifier for shared state
     */
    void llm_job(const std::vector<int>& text, const std::vector<int>& prompt_text, const std::vector<int>& llm_prompt_speech_token,
                 const std::vector<float>& llm_embedding, const std::string& uuid);

private:
    std::future<CosyVoiceResult> speech_async(const CosyVoiceRequest& request) const;
    CosyVoiceResult speech_sync(const CosyVoiceRequest& request,
                                const std::function<bool()>& should_stop = {}) const;

    std::vector<float> speech(const CosyVoiceRequest& request);

    /**
     * Normalize text and optionally split it into synthesis segments.
     *
     * @param text Input text to normalize
     * @param split Whether to split into multiple segments
     * @param text_frontend Whether to use text frontend normalization
     * @return Vector of normalized text segments, or single text if split=false
     */
    std::vector<std::string> text_normalize(const std::string& text, bool split = true) const;
    int CalcUttLength(const std::string& text, bool is_chinese) const;
    bool ShouldMerge(const std::string& text, bool is_chinese, int merge_len) const;

    std::vector<std::string> SplitParagraph(const std::string& text, int max_tokens, int min_tokens) const;
    /**
     * Process text for zero-shot synthesis.
     *
     * @param tts_text Target text to synthesize
     * @param prompt_text Prompt text for style/content steering
     * @param resample_rate Target sample rate (24000 for CosyVoice2)
     * @return FrontendInput structure ready for model inference
     */
    FrontendInput frontend_zero_shot(const std::string& tts_text, const std::string& prompt_text, const std::string& prompt_wav,
                                     int resample_rate = 24000);

    /**
     * Extract speech tokens from audio file using speech tokenizer.
     *
     * @param prompt_wav Path to prompt audio file
     * @return Pair of (speech_token, token_len)
     */
    std::pair<SpeechToken, int> extract_speech_token(const std::vector<float>& pcm_data_16k) const;

    std::vector<float> extract_spk_embedding(const std::vector<float>& pcm_data_16k) const;

    std::string model_key_;

    std::shared_ptr<HLIECosyVoiceModel> cosyvoice_model_;

    std::shared_ptr<HLIEAudio> audio_processor_;

    // Shared state for multi-threaded inference
    std::mutex lock_;
    std::map<std::string, std::vector<int>> tts_speech_token_dict_;
    std::map<std::string, bool> llm_end_dict_;
    std::map<std::string, float*> hift_cache_dict_;
    std::map<std::string, LLMStagePerf> perf_by_uuid_;

    // Silent tokens that can be skipped
    static constexpr std::array<int, 11> kSilentTokens = {1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323};


    struct QueueItem {
        CosyVoiceRequest request;
        std::shared_ptr<std::promise<CosyVoiceResult>> promise;
    };

    std::shared_ptr<HLIEBlockingQueue<QueueItem>> blocking_queue_;

    std::unique_ptr<HLIEWebSocketService> websocket_service_;

    std::unique_ptr<std::thread> worker_thread_; // 工作线程
    std::atomic<bool> should_stop_{false}; // 线程停止标志

    // TODO: Add tokenizer, text normalizer, and other runtime components
    // CosyVoice3Tokenizer tokenizer_;
    // std::string text_frontend_;
    // ZhNormalizer zh_tn_model_;  // Further
    // EnNormalizer en_tn_model_;  // Further
};

HLIE_NAMESPACE_END

#endif

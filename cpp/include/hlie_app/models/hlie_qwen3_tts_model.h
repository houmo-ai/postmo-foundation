#ifndef HLIE_QWEN3_TTS_MODEL_H
#define HLIE_QWEN3_TTS_MODEL_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "hlie_app/models/hlie_model_base.h"
#include "hlie_base/hlie_gguf_bundle.h"

HLIE_NAMESPACE_BEGIN

enum class Qwen3TTSVariant {
    CustomVoice,
    Base,
};

struct Qwen3TTSModelConfig {
    Qwen3TTSVariant variant = Qwen3TTSVariant::CustomVoice;
    std::string model_size;
    int32_t sample_rate = 0;
    int32_t num_code_groups = 0;
    int32_t talker_hidden_size = 0;
    int32_t text_hidden_size = 0;
    int32_t text_vocab_size = 0;
    int32_t tokenizer_vocab_size = 0;
    int32_t codec_vocab_size = 0;
    int32_t code_predictor_vocab_size = 0;
    int32_t stateful_samples_per_frame = 1920;
    int32_t stateful_initial_output_skip_frames = 4;
    bool code_predictor_sampling = false;
    int32_t tts_bos_token_id = -1;
    int32_t tts_eos_token_id = -1;
    int32_t tts_pad_token_id = -1;
    int32_t codec_bos_id = -1;
    int32_t codec_eos_token_id = -1;
    int32_t codec_pad_id = -1;
    int32_t codec_think_id = -1;
    int32_t codec_nothink_id = -1;
    int32_t codec_think_bos_id = -1;
    int32_t codec_think_eos_id = -1;
    std::unordered_map<std::string, int32_t> codec_language_ids;
    std::unordered_map<std::string, int32_t> speaker_ids;
    std::unordered_map<std::string, std::string> speaker_dialects;
    int32_t reference_sample_rate = 24000;
    int32_t speaker_embedding_dim = 0;
    int32_t reference_max_samples = 0;
    int32_t reference_max_frames = 0;
    int32_t reference_mel_bins = 0;
};

/** Deterministic output used to compare the C++ frontend/prefill with Python. */
struct Qwen3TTSPrefillDebugResult {
    std::vector<int> input_ids;
    int32_t prompt_length = 0;
    int32_t prefill_chunks = 0;
    float prompt_sum = 0.0F;
    float prompt_abs_mean = 0.0F;
    int32_t first_codec_token = -1;
    float first_codec_logit = 0.0F;
    std::vector<int64_t> logits_shape;
    std::vector<float> last_logits_head;
};

struct Qwen3TTSCodecFramesDebugResult {
    Qwen3TTSPrefillDebugResult prefill;
    std::vector<int32_t> talker_tokens;
    std::vector<std::vector<int32_t>> codec_frames;
    bool reached_eos = false;
};

struct Qwen3TTSSynthesisResult {
    std::vector<float> audio;
    std::vector<std::vector<int32_t>> codec_frames;
    bool reached_eos = false;
};

struct Qwen3TTSStreamResult {
    size_t codec_frames = 0;
    size_t audio_samples = 0;
    bool reached_eos = false;
};

using Qwen3TTSAudioChunkCallback =
    std::function<bool(const std::vector<float>&)>;
using Qwen3TTSStopCallback = std::function<bool()>;

struct Qwen3TTSReferenceDebugResult {
    int32_t valid_frames = 0;
    std::vector<int64_t> codec_shape;
    std::vector<int32_t> codec_head;
    std::vector<int64_t> speaker_embedding_shape;
    float speaker_embedding_sum = 0.0F;
    float speaker_embedding_abs_mean = 0.0F;
    std::vector<float> speaker_embedding_head;
};

/** Qwen3-TTS CustomVoice/Base TCIM model bundle. */
class HLIE_API HLIEQwen3TTSModel final : public HLIEModelBase {
public:
    HLIEQwen3TTSModel() = default;
    ~HLIEQwen3TTSModel() = default;

    int load_model(const std::string& model_path);
    int init();
    bool debug_talker_prefill(const std::string& text, const std::string& language,
                              const std::string& speaker, Qwen3TTSPrefillDebugResult& result,
                              std::string& error);
    bool debug_generate_codec_frames(const std::string& text, const std::string& language,
                                     const std::string& speaker, int max_frames,
                                     Qwen3TTSCodecFramesDebugResult& result,
                                     std::string& error);
    bool debug_generate_clone_codec_frames(
        const std::string& text, const std::string& language,
        const std::string& reference_text,
        const std::vector<float>& reference_pcm, int max_frames,
        Qwen3TTSCodecFramesDebugResult& result, std::string& error);
    bool synthesize(const std::string& text, const std::string& language,
                    const std::string& speaker, int max_frames,
                    Qwen3TTSSynthesisResult& result, std::string& error,
                    bool require_eos = false, bool do_sample = false,
                    uint32_t seed = 1024U);
    bool synthesize_clone(const std::string& text, const std::string& language,
                          const std::string& reference_text,
                          const std::vector<float>& reference_pcm,
                          int max_frames, Qwen3TTSSynthesisResult& result,
                          std::string& error, bool require_eos = false,
                          bool do_sample = false, uint32_t seed = 1024U);
    bool synthesize_stream(
        const std::string& text, const std::string& language,
        const std::string& speaker, int max_frames,
        Qwen3TTSStreamResult& result,
        const Qwen3TTSAudioChunkCallback& on_audio_chunk,
        const Qwen3TTSStopCallback& should_stop, std::string& error,
        bool require_eos = false, bool do_sample = false,
        uint32_t seed = 1024U);
    bool synthesize_clone_stream(
        const std::string& text, const std::string& language,
        const std::string& reference_text,
        const std::vector<float>& reference_pcm, int max_frames,
        Qwen3TTSStreamResult& result,
        const Qwen3TTSAudioChunkCallback& on_audio_chunk,
        const Qwen3TTSStopCallback& should_stop, std::string& error,
        bool require_eos = false, bool do_sample = false,
        uint32_t seed = 1024U);
    bool debug_reference_audio(const std::vector<float>& reference_pcm,
                               Qwen3TTSReferenceDebugResult& result,
                               std::string& error);
    bool debug_decode_codec_frames(const std::vector<std::vector<int32_t>>& frames,
                                   std::vector<float>& audio, std::string& error);

    [[nodiscard]] bool ready() const { return ready_; }
    [[nodiscard]] const Qwen3TTSModelConfig& config() const { return config_; }
    [[nodiscard]] bool is_base() const {
        return config_.variant == Qwen3TTSVariant::Base;
    }
    [[nodiscard]] int prefill_length() const { return prefill_length_; }
    [[nodiscard]] int context_length() const { return context_length_; }
    [[nodiscard]] int stateful_chunk_size() const { return stateful_chunk_size_; }

private:
    using CodecFrameCallback =
        std::function<bool(const std::vector<int32_t>&)>;

    bool load_metadata(std::string& error);
    bool load_module(const std::string& tensor_name, std::shared_ptr<tcim::Module>& module,
                     const std::vector<std::string>& dummy_names, std::string& error);
    bool load_embedding(const std::string& tensor_name, const std::vector<int64_t>& shape,
                        std::shared_ptr<tcim::Tensor>& tensor, std::string& error);
    bool load_prefill_decode_pair(const std::string& prefill_name,
                                  const std::string& decode_name,
                                  std::shared_ptr<tcim::Module>& prefill,
                                  std::shared_ptr<tcim::Module>& decode,
                                  std::string& error);
    bool reset_device_caches(const std::shared_ptr<tcim::Module>& module,
                             const std::vector<std::string>& input_names,
                             size_t first_cache_index, std::string& error);
    bool project_text_ids(const std::vector<int>& ids, tcim::Tensor& output, std::string& error);
    bool run_talker_prefill_debug(const std::string& text, const std::string& language,
                                  const std::string& speaker, Qwen3TTSPrefillDebugResult& result,
                                  tcim::Tensor* tts_pad, tcim::Tensor* past_hidden, bool do_sample,
                                  std::string& error);
    bool generate_code_predictor_frame(const tcim::Tensor& past_hidden, int32_t talker_token,
                                       std::vector<int32_t>& predictor_tokens,
                                       std::vector<tcim::Tensor>& predictor_embeds, bool do_sample,
                                       std::string& error);
    bool generate_codec_frames_unlocked(const std::string& text,
                                        const std::string& language,
                                        const std::string& speaker, int max_frames,
                                        Qwen3TTSCodecFramesDebugResult& result,
                                        bool do_sample, std::string& error,
                                        const CodecFrameCallback& on_frame = {},
                                        const Qwen3TTSStopCallback& should_stop = {});
    bool encode_reference_audio(const std::vector<float>& pcm,
                                std::vector<std::vector<int32_t>>& reference_codes,
                                tcim::Tensor& speaker_embedding,
                                std::string& error);
    bool build_base_prompt(const std::string& text, const std::string& language,
                           const std::string& reference_text,
                           const std::vector<std::vector<int32_t>>& reference_codes,
                           const tcim::Tensor& speaker_embedding,
                           tcim::Tensor& talker_input, tcim::Tensor& tts_pad,
                           std::string& error);
    bool run_talker_prefill(const tcim::Tensor& talker_input,
                            Qwen3TTSPrefillDebugResult& result,
                            tcim::Tensor& past_hidden, bool do_sample,
                            std::string& error);
    bool generate_base_codec_frames_unlocked(
        const std::string& text, const std::string& language,
        const std::string& reference_text,
        const std::vector<float>& reference_pcm, int max_frames,
        Qwen3TTSCodecFramesDebugResult& result, bool do_sample,
        std::string& error,
        std::vector<std::vector<int32_t>>* reference_codes_output = nullptr,
        const CodecFrameCallback& on_frame = {},
        const Qwen3TTSStopCallback& should_stop = {});
    bool decode_codec_frames_with_reference_unlocked(
        const std::vector<std::vector<int32_t>>& reference_codes,
        const std::vector<std::vector<int32_t>>& frames,
        std::vector<float>& audio, std::string& error);

    struct StatefulDecoderState {
        std::vector<tcim::Tensor> tensors;
        std::vector<float> latent_audio;
        int32_t kv_valid_length = 0;
        bool device_backed = false;
    };
    bool create_stateful_decoder_state(StatefulDecoderState& state, std::string& error);
    bool prime_stateful_decoder_with_reference(
        const std::vector<std::vector<int32_t>>& reference_codes,
        StatefulDecoderState& state, std::string& error);
    bool decode_stateful_chunk(const std::vector<std::vector<int32_t>>& frames,
                               bool is_final, StatefulDecoderState& state,
                               std::vector<float>& audio, std::string& error);
    bool decode_codec_frames_unlocked(const std::vector<std::vector<int32_t>>& frames,
                                      std::vector<float>& audio, std::string& error);

    // Keep bundle metadata alive until after modules and tensors are destroyed.
    HLIEGgufBundle bundle_;
    Qwen3TTSModelConfig config_;

    std::shared_ptr<tcim::Module> text_projection_model_;
    std::shared_ptr<tcim::Module> talker_prefill_model_;
    std::shared_ptr<tcim::Module> talker_decode_model_;
    std::shared_ptr<tcim::Module> code_predictor_prefill_model_;
    std::shared_ptr<tcim::Module> code_predictor_decode_model_;
    std::shared_ptr<tcim::Module> stateful_decoder_model_;
    std::shared_ptr<tcim::Module> speech_tokenizer_encoder_model_;
    std::shared_ptr<tcim::Module> speaker_encoder_model_;

    std::shared_ptr<tcim::Tensor> talker_token_embedding_;
    std::shared_ptr<tcim::Tensor> talker_text_embedding_;
    std::shared_ptr<tcim::Tensor> code_predictor_embedding_;
    std::shared_ptr<class HLIETokenizer> tokenizer_;

    std::vector<std::string> text_projection_input_names_;
    std::vector<std::string> text_projection_output_names_;
    std::vector<std::string> talker_prefill_input_names_;
    std::vector<std::string> talker_prefill_output_names_;
    std::vector<std::shared_ptr<tcim::Tensor>> talker_prefill_host_inputs_;
    std::vector<std::string> talker_decode_input_names_;
    std::vector<std::string> talker_decode_output_names_;
    std::shared_ptr<tcim::Tensor> talker_decode_valid_length_;
    std::shared_ptr<tcim::Tensor> talker_decode_current_length_;

    std::vector<std::string> code_predictor_prefill_input_names_;
    std::vector<std::string> code_predictor_prefill_output_names_;
    std::vector<std::shared_ptr<tcim::Tensor>> code_predictor_prefill_host_inputs_;
    std::vector<std::string> code_predictor_decode_input_names_;
    std::vector<std::string> code_predictor_decode_output_names_;
    std::shared_ptr<tcim::Tensor> code_predictor_decode_valid_length_;
    std::shared_ptr<tcim::Tensor> code_predictor_decode_current_length_;
    std::shared_ptr<tcim::Tensor> code_predictor_decode_generation_steps_;

    std::vector<std::string> stateful_decoder_input_names_;
    std::vector<std::string> stateful_decoder_output_names_;
    std::vector<std::string> speech_tokenizer_encoder_input_names_;
    std::vector<std::string> speech_tokenizer_encoder_output_names_;
    std::vector<std::string> speaker_encoder_input_names_;
    std::vector<std::string> speaker_encoder_output_names_;

    int text_projection_length_ = 0;
    int prefill_length_ = 0;
    int code_predictor_prefill_length_ = 0;
    int context_length_ = 0;
    int stateful_chunk_size_ = 0;
    int stateful_kv_window_ = 0;
    int reference_encoder_samples_ = 0;
    int reference_encoder_frames_ = 0;
    int speaker_encoder_mel_bins_ = 0;
    bool ready_ = false;
    std::mutex inference_mutex_;
    std::mt19937 sampling_rng_{1024U};
    std::vector<uint8_t> device_zero_scratch_;
};

HLIE_NAMESPACE_END

#endif

#ifndef HLIE_COSYVOICE_MODEL_H
#define HLIE_COSYVOICE_MODEL_H

#include "arg.h"
#include "hlie_base/hlie_base.h"
#include "hlie_model_base.h"
#include "tcim/tcim_runtime.h"
#include "hlie_base/hlie_tcim_util.h"

HLIE_NAMESPACE_BEGIN

class HLIETokenizer;


/**
 * Performance metrics for LLM inference.
 */
struct LLMPerf {
    int prefill_tokens = 0;
    float prefill_s = 0.0f;
    float ttft_s = 0.0f;        // Time to first token
    float decode_s = 0.0f;
    int decode_tokens = 0;
    float llm_total_s = 0.0f;
    float llm_stage_wall_s = 0.0f;
};

/*
 * 整体Modules Params Shapes

╔══════════════════════════════════════════════════════════════╗
║ Model: cosyvoice3_speech_tokenizer_model_                    ║
╠══════════════════════════════════════════════════════════════╣
║ Inputs (3):                                                  ║
║   [0] feats    │ shape: 1 × 128 × 3000 │ float16             ║
║   [1] mask     │ shape: 1 × 20 × 750 × 750 │ float16         ║
║   [2] mask1    │ shape: 1 × 750 × 1280 │ float16             ║
╠══════════════════════════════════════════════════════════════╣
║ Outputs (1):                                                 ║
║   [0] indices  │ shape: 1 × 750 │ int32                      ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║ Model: cosyvoice3_campplus_model_                            ║
╠══════════════════════════════════════════════════════════════╣
║ Inputs (1):                                                  ║
║   [0] input   │ shape: 1 × 1000 × 80 │ float16               ║
╠══════════════════════════════════════════════════════════════╣
║ Outputs (1):                                                 ║
║   [0] output  │ shape: 1 × 192 │ float16                     ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║ Model: cosyvoice3_flow_decoder_model_                        ║
╠══════════════════════════════════════════════════════════════╣
║ Inputs (6):                                                  ║
║   [0] x     │ shape: 2 × 80 × 2048 │ float16                 ║
║   [1] mask  │ shape: 2 × 1 × 2048 │ float16                  ║
║   [2] mu    │ shape: 2 × 80 × 2048 │ float16                 ║
║   [3] t     │ shape: 2 │ float16                             ║
║   [4] spks  │ shape: 2 × 80 │ float16                        ║
║   [5] cond  │ shape: 2 × 80 × 2048 │ float16                 ║
╠══════════════════════════════════════════════════════════════╣
║ Outputs (1):                                                 ║
║   [0] estimator_out │ shape: 2 × 80 × 2048 │ float16         ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║ Model: cosyvoice3_flow_encoder_model_                        ║
╠══════════════════════════════════════════════════════════════╣
║ Inputs (1):                                                  ║
║   [0] token │ shape: 1 × 1024 × 80 │ float16                 ║
╠══════════════════════════════════════════════════════════════╣
║ Outputs (1):                                                 ║
║   [0] output │ shape: 1 × 1024 × 80 │ float16                ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║ Model: cosyvoice3_flow_spk_model_                            ║
╠══════════════════════════════════════════════════════════════╣
║ Inputs (1):                                                  ║
║   [0] embedding │ shape: 1 × 192 │ float16                   ║
╠══════════════════════════════════════════════════════════════╣
║ Outputs (1):                                                 ║
║   [0] output    │ shape: 1 × 80 │ float16                    ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║ Model: cosyvoice3_hift_part1_model_                          ║
╠══════════════════════════════════════════════════════════════╣
║ Inputs (1):                                                  ║
║   [0] speech_feat │ shape: 1 × 80 × 1024 │ float16           ║
╠══════════════════════════════════════════════════════════════╣
║ Outputs (1):                                                 ║
║   [0] reshape_2  │ shape: 1 × 491536 │ float16               ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║ Model: cosyvoice3_hift_part2_model_                          ║
╠══════════════════════════════════════════════════════════════╣
║ Inputs (2):                                                  ║
║   [0] stft        │ shape: 1 × 122881 × 9 × 2 │ float16      ║
║   [1] speech_feat │ shape: 1 × 80 × 1024 │ float16           ║
╠══════════════════════════════════════════════════════════════╣
║ Outputs (1):                                                 ║
║   [0] generated_speech │ shape: 1 × 491520 │ float16         ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║ Model: cosyvoice3_llm_decoder_model_                         ║
╠══════════════════════════════════════════════════════════════╣
║ Inputs (1):                                                  ║
║   [0] input │ shape: 1 × 896 │ float16                       ║
╠══════════════════════════════════════════════════════════════╣
║ Outputs (1):                                                 ║
║   [0] output │ shape: 1 × 6761 │ float16                     ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║ Model: llm_qwen2_decode_model_                               ║
╠══════════════════════════════════════════════════════════════╣
║ Inputs (51):                                                 ║
║   [ 0] input_1        │ shape: 1 × 1 × 896         │ float16 ║
║   [ 1] valid_length   │ shape: 1                   │ int32   ║
║   [ 2] current_length │ shape: 1                   │ int32   ║
║                                                              ║
║   -- KV Cache (int16) --                                     ║
║   [ 3]  layer_0  │ shape: 1 × 2 × 2048 × 64 │ int16          ║
║   [ 4]  layer_1  │ shape: 1 × 2 × 2048 × 64 │ int16          ║
║   [ 5]  layer_2  │ shape: 1 × 2 × 2048 × 64 │ int16          ║
║   ...                                                        ║
║   [24]  layer_21 │ shape: 1 × 2 × 2048 × 64 │ int16          ║
║   [25]  layer_22 │ shape: 1 × 2 × 2048 × 64 │ int16          ║
║   [26]  layer_23 │ shape: 1 × 2 × 2048 × 64 │ int16          ║
║                                                              ║
║   -- KV Cache (float16) --                                   ║
║   [27] layer_0  │ shape: 1 × 2 × 2048 × 64 │ float16         ║
║   [28] layer_1  │ shape: 1 × 2 × 2048 × 64 │ float16         ║
║   [29] layer_2  │ shape: 1 × 2 × 2048 × 64 │ float16         ║
║   ...                                                        ║
║   [48] layer_21 │ shape: 1 × 2 × 2048 × 64 │ float16         ║
║   [49] layer_22 │ shape: 1 × 2 × 2048 × 64 │ float16         ║
║   [50] layer_23 │ shape: 1 × 2 × 2048 × 64 │ float16         ║
╠══════════════════════════════════════════════════════════════╣
║ Outputs (1):                                                 ║
║   [ 0] logits   │ shape: 1 × 1 ×  896   │ float16            ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║ Model: llm_qwen2_prefill_model_                              ║
╠══════════════════════════════════════════════════════════════╣
║ Inputs (51):                                                 ║
║   [ 0] input_1        │ shape: 1 × 256 × 896       │ float16 ║
║   [ 1] valid_length   │ shape: 1                   │ int32   ║
║   [ 2] current_length │ shape: 1                   │ int32   ║
║                                                              ║
║   -- KV Cache (int16) --                                     ║
║   [ 3]  layer_0  │ shape: 1 × 2 × 2048 × 64 │ int16          ║
║   [ 4]  layer_1  │ shape: 1 × 2 × 2048 × 64 │ int16          ║
║   [ 5]  layer_2  │ shape: 1 × 2 × 2048 × 64 │ int16          ║
║   ...                                                        ║
║   [24]  layer_21 │ shape: 1 × 2 × 2048 × 64 │ int16          ║
║   [25]  layer_22 │ shape: 1 × 2 × 2048 × 64 │ int16          ║
║   [26]  layer_23 │ shape: 1 × 2 × 2048 × 64 │ int16          ║
║                                                              ║
║   -- KV Cache (float16) --                                   ║
║   [27] layer_0  │ shape: 1 × 2 × 2048 × 64 │ float16         ║
║   [28] layer_1  │ shape: 1 × 2 × 2048 × 64 │ float16         ║
║   [29] layer_2  │ shape: 1 × 2 × 2048 × 64 │ float16         ║
║   ...                                                        ║
║   [48] layer_21 │ shape: 1 × 2 × 2048 × 64 │ float16         ║
║   [49] layer_22 │ shape: 1 × 2 × 2048 × 64 │ float16         ║
║   [50] layer_23 │ shape: 1 × 2 × 2048 × 64 │ float16         ║
╚══════════════════════════════════════════════════════════════╝
║ Outputs (1):                                                 ║
║   [ 0] logits   │ shape: 1 × 1 ×  896   │ float16            ║
╚══════════════════════════════════════════════════════════════╝
*/

class HLIECosyVoiceModel : public HLIEModelBase {
public:
    int init();
    int load_model(const std::string& model);

    tcim::Tensor text_embedding(const tcim::Tensor& weight, const std::vector<int>& ids);

    int extract_text_token(const std::string& tts_text, std::vector<int>& text_tokens) const;

    std::pair<std::vector<int>,int> speech_tokenizer(const std::vector<TensorFP16>& padded_input, const std::vector<TensorFP16>& mask,
                                                      const std::vector<TensorFP16>& mask1, int feat_len);

    std::vector<float> campplus(const std::vector<TensorFP16>& input) const;

    /**
     * Run LLM inference to generate speech tokens from text.
     *
     * @param text Target text tokens [1, text_len]
     * @param text_len Length of target text
     * @param prompt_text Prompt text tokens [1, prompt_len]
     * @param prompt_text_len Length of prompt text
     * @param prompt_speech_token Prompt speech tokens [1, prompt_speech_len]
     * @param prompt_speech_token_len Length of prompt speech tokens
     * @param llm_embedding Speaker embedding for LLM [embedding_dim]
     * @param sampling Sampling temperature parameter (default: 25)
     * @param max_token_text_ratio Max ratio of generated tokens to text length (default: 20)
     * @param min_token_text_ratio Min ratio before EOS is allowed (default: 2)
     * @param perf Optional performance metrics output
     * @return Generated speech token IDs
     */
    std::vector<int> llm_inference(
        const std::vector<int>& text,
        int text_len,
        const std::vector<int>& prompt_text,
        int prompt_text_len,
        const std::vector<int>& prompt_speech_token,
        int prompt_speech_token_len,
        const std::vector<float>& llm_embedding,
        int sampling = 25,
        float max_token_text_ratio = 20.0f,
        float min_token_text_ratio = 2.0f,
        LLMPerf* perf = nullptr);

    /**
     * Convert speech tokens to waveform using flow decoder and HiFT vocoder.
     *
     * @param token Generated speech token sequence [1, token_len]
     * @param prompt_token Prompt speech token sequence [1, prompt_len]
     * @param prompt_feat Prompt acoustic features (mel spectrogram) [1, 80, prompt_mel_len]
     * @param embedding Speaker embedding for flow decoder [embedding_dim]
     * @param speed Playback speed ratio (default: 1.0)
     * @return Generated waveform as float vector
     */
    std::vector<float> token2wav(const std::vector<int>& token, const std::vector<int>& prompt_token, const std::vector<float>& prompt_feat,
                                 const std::vector<float>& embedding, float speed = 1.0) const;


    /**
     * Get the speech token vocabulary size.
     */
    static int speech_token_size() { return 6561; }

    /**
     * Get the LLM input size (embedding dimension).
     */
    static int llm_input_size() { return 896; }

    /**
     * Get the LLM output size.
     */
    static int llm_output_size() { return 6761; }

    /**
     * Get the pad token ID used in embedding lookup.
     */
    static int pad_token_id() { return 151645; }

private:
    // Sampling function - picks top-k tokens based on weighted scores
    static int sampling_ids(
        const float* weighted_scores,
        const std::vector<int>& decoded_tokens,
        int sampling,
        bool ignore_eos);


    std::shared_ptr<HLIETokenizer> tokenizer_;

    std::shared_ptr<tcim::Module> cosyvoice3_campplus_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_flow_decoder_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_flow_encoder_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_flow_spk_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_hift_part1_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_hift_part2_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_llm_decoder_model_;
    std::shared_ptr<tcim::Module> llm_qwen2_decode_model_;
    std::shared_ptr<tcim::Module> llm_qwen2_prefill_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_speech_tokenizer_model_;

    std::shared_ptr<tcim::Tensor> token_weight_embedding_; // Shape: [151936, 896], dtype=float16
    std::shared_ptr<tcim::Tensor> speech_weight_embedding_; // Shape: [6761, 896], dtype=float16
    std::shared_ptr<tcim::Tensor> flow_input_embedding_; // Shape: [6561, 80], dtype=float16
    std::shared_ptr<tcim::Tensor> llm_sos_embedding_; // Shape: [1,1,896], dtype=float16
    std::shared_ptr<tcim::Tensor> llm_task_id_embedding_; // Shape: [1,1,896], dtype=float16


    std::vector<std::string> speech_tokenizer_input_names_;
    std::vector<std::string> speech_tokenizer_output_names_;


    // LLM model I/O names (cached for performance)
    std::vector<std::string> llm_decoder_input_names_;
    std::vector<std::string> llm_decoder_output_names_;
    std::vector<std::string> prefill_input_names_;
    std::vector<std::string> decode_input_names_;

    std::vector<std::string> flow_input_names_;
    std::vector<std::string> flow_output_names_;
    std::vector<std::string> flow_encoder_input_names_;
    std::vector<std::string> flow_encoder_output_names_;

    std::vector<std::string> flow_spk_input_names_;
    std::vector<std::string> flow_spk_output_names_;
    std::vector<std::string> hift_part1_input_names_;
    std::vector<std::string> hift_part1_output_names_;
    std::vector<std::string> hift_part2_input_names_;
    std::vector<std::string> hift_part2_output_names_;

    std::map<std::string, tcim::Tensor> llm_qwen2_prefill_input_map_;
    std::map<std::string, tcim::Tensor> llm_qwen2_decode_input_map_;

    std::map<std::string, tcim::Tensor> cosyvoice3_speech_tokenizer_input_map_;

    // Model dimension info
    int prefill_length_ = 256;
    int embedding_len_ = 896;
    int context_max_length_ = 0;
    int batch_size_ = 1;

    // Stop token IDs (speech_token_size + 0..199)
    std::vector<int> stop_token_ids_;
};


HLIE_NAMESPACE_END

#endif // HLIE_COSYVOICE_MODEL_H

#ifndef HLIE_COSYVOICE_MODEL_H
#define HLIE_COSYVOICE_MODEL_H

#include "arg.h"
#include "hlie_base/hlie_base.h"
#include "hlie_model_base.h"
#include "tcim/tcim_runtime.h"


HLIE_NAMESPACE_BEGIN

class HLIETokenizer;

class HLIECosyVoiceModel : public HLIEModelBase {
public:
    int init();
    int load_model(const std::string& model);

    tcim::Tensor text_embedding(const tcim::Tensor& weight, const std::vector<int>& ids);

    int extract_text_token(const std::string& tts_text, std::vector<int>& text_tokens) const;

    std::pair<std::vector<int>,int> speech_tokenizer(const std::vector<uint16_t>& padded_input, const std::vector<uint16_t>& mask,
                                                      const std::vector<uint16_t>& mask1);

    std::vector<float> campplus(const std::vector<uint16_t>& input) const;

private:
    std::shared_ptr<HLIETokenizer> tokenizer_;

    std::shared_ptr<tcim::Module> cosyvoice3_campplus_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_flow_decoder_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_flow_encoder_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_flow_spk_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_hift_part1_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_hift_part2_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_llm_decoder_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_llm_qwen2_decode_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_llm_qwen2_prefill_model_;
    std::shared_ptr<tcim::Module> cosyvoice3_speech_tokenizer_model_;

    std::shared_ptr<tcim::Tensor> token_weight_embedding_; // [151936, 896]
    std::shared_ptr<tcim::Tensor> speech_weight_embedding_; // [6761, 896]
    std::shared_ptr<tcim::Tensor> flow_input_embedding_; // [6761, 896]
    std::shared_ptr<tcim::Tensor> llm_sos_embedding_; // [1,1,896]
    std::shared_ptr<tcim::Tensor> llm_task_id_embedding_; // [1,1,896]


    std::vector<std::string> speech_tokenizer_input_names_;
    std::vector<std::string> speech_tokenizer_output_names_;
};


HLIE_NAMESPACE_END

#endif // HLIE_COSYVOICE_MODEL_H

#include "hlie_app/models/hlie_cosyvoice_model.h"

#include <regex>

#include "ggml-cpp.h"
#include "gguf.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_tcim_util.h"
#include "hlie_base/hlie_tokenizer.h"

namespace {
    constexpr std::string_view quant_embedding = "quant_embedding.bin";
    constexpr std::string_view flow_input_embedding = "flow_input_embedding.bin";
    constexpr std::string_view llm_sos_embedding = "llm_sos_embedding.bin";
    constexpr std::string_view llm_speech_embedding = "llm_speech_embedding.bin";
    constexpr std::string_view llm_task_id_embedding = "llm_task_id_embedding.bin";

    constexpr std::string_view cosyvoice3_campplus = "cosyvoice3_campplus.hmm";
    constexpr std::string_view cosyvoice3_flow_decoder = "cosyvoice3_flow_decoder.hmm";
    constexpr std::string_view cosyvoice3_flow_encoder = "cosyvoice3_flow_encoder.hmm";
    constexpr std::string_view cosyvoice3_flow_spk = "cosyvoice3_flow_spk.hmm";
    constexpr std::string_view cosyvoice3_hift_part1 = "cosyvoice3_hift_part1.hmm";
    constexpr std::string_view cosyvoice3_hift_part2 = "cosyvoice3_hift_part2.hmm";
    constexpr std::string_view cosyvoice3_llm_decoder = "cosyvoice3_llm_decoder.hmm";
    constexpr std::string_view cosyvoice3_llm_qwen2_decode = "cosyvoice3_llm_qwen2_decode.hmm";
    constexpr std::string_view cosyvoice3_llm_qwen2_prefill = "cosyvoice3_llm_qwen2_prefill.hmm";
    constexpr std::string_view cosyvoice3_speech_tokenizer = "cosyvoice3_speech_tokenizer.hmm";

    void get_module_input_names() {

    }

} // namespace

HLIE_NAMESPACE_BEGIN



int HLIECosyVoiceModel::init() {
    speech_tokenizer_input_names_.clear();
    speech_tokenizer_output_names_.clear();

    speech_tokenizer_input_names_ = TcimUtil::get_module_input_names(cosyvoice3_speech_tokenizer_model_);
    speech_tokenizer_output_names_ = TcimUtil::get_module_output_names(cosyvoice3_speech_tokenizer_model_);

    return 0;
}

int HLIECosyVoiceModel::load_model(const std::string& model) {
    // Load the main GGUF
    struct ggml_context* ctx_data = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };

    gguf_context* ctx = nullptr;

    try {
        ctx = gguf_init_from_file(model.c_str(), params);
        if (!ctx) {
            LOG_ERROR("failed to load model from {}", model);
            return -1;
        }

        LOG_INFO("load after gguf_init_from_file");

        auto n_kv = gguf_get_n_kv(ctx);

        LOG_INFO("loaded HLIECosyVoiceModel meta data with {} key-value pairs and {} tensors from (version GGUF V3 (latest))", n_kv,
                 model.c_str());

        std::map<std::string_view, std::shared_ptr<tcim::Module>> model_map = {
            {cosyvoice3_campplus, cosyvoice3_campplus_model_},
            {cosyvoice3_flow_decoder, cosyvoice3_flow_decoder_model_},
            {cosyvoice3_flow_encoder, cosyvoice3_flow_encoder_model_},
            {cosyvoice3_flow_spk, cosyvoice3_flow_spk_model_},
            {cosyvoice3_hift_part1, cosyvoice3_hift_part1_model_},
            {cosyvoice3_hift_part2, cosyvoice3_hift_part2_model_},
            {cosyvoice3_llm_decoder, cosyvoice3_llm_decoder_model_},
            {cosyvoice3_llm_qwen2_decode, cosyvoice3_llm_qwen2_decode_model_},
            {cosyvoice3_llm_qwen2_prefill, cosyvoice3_llm_qwen2_prefill_model_},
            {cosyvoice3_speech_tokenizer, cosyvoice3_speech_tokenizer_model_},
        };

        std::map<std::string_view, std::pair<std::shared_ptr<tcim::Tensor>, std::vector<int>>> tensor_map = {
            {quant_embedding,  std::make_pair(token_weight_embedding_, std::vector{151936, 896})},
            {flow_input_embedding,  std::make_pair(flow_input_embedding_, std::vector{6561, 80})},
            {llm_sos_embedding, std::make_pair(llm_sos_embedding_, std::vector{1, 1, 896})},
            {llm_speech_embedding, std::make_pair(speech_weight_embedding_, std::vector{6761, 896})},
            {llm_task_id_embedding, std::make_pair(llm_task_id_embedding_, std::vector{1, 1, 896})},
        };

        const auto n_tensors = gguf_get_n_tensors(ctx);
        for (int i = 0; i < n_tensors; ++i) {
            const char* name = gguf_get_tensor_name(ctx, i);
            const size_t size = gguf_get_tensor_size(ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
            LOG_DEBUG("tensor name is {}, offset: {}, size: {}", name, offset, size);

            if (auto model_pair = model_map.find(name); model_pair != model_map.end()) {
               model_pair->second = create_infer_engine(model, offset, size);
                if (!model_pair->second) {
                    LOG_ERROR("failed to load {}", name);
                    ggml_free(ctx_data);
                    gguf_free(ctx);
                    return false;
                }

                LOG_INFO("succeed to load {}", name);
            } else if (auto tensor_pair = tensor_map.find(name); tensor_pair != tensor_map.end()) {
                LOG_INFO("Start to load {}", name);
                auto token_weight_data = TcimUtil::load_fp16_data(model, offset, size);
                auto tensor = TcimUtil::create_fp16_tensor(token_weight_data.data(), tensor_pair->second.second);
                if (tensor.GetInitStatus() != tcim::OK) {
                    LOG_ERROR("failed to load {}", name);
                    ggml_free(ctx_data);
                    gguf_free(ctx);
                    return false;
                }

                tensor_pair->second.first = std::make_shared<tcim::Tensor>(tensor);
                LOG_INFO("succeed to load {}", name);
            }
        }

        tokenizer_ = std::make_shared<HLIETokenizer>();
        tokenizer_->load_from_gguf_ctx(ctx);

        ggml_free(ctx_data);
        gguf_free(ctx);
    } catch (std::exception& ex) {
        LOG_ERROR("exception occurred when load cosyvoice model", ex.what());
        ggml_free(ctx_data);
        gguf_free(ctx);
    }

    return 0;
}

tcim::Tensor HLIECosyVoiceModel::text_embedding(const tcim::Tensor& weight, const std::vector<int>& ids) {
    return TcimUtil::embedding_lookup(weight, ids);
}

int HLIECosyVoiceModel::extract_text_token(const std::string& tts_text, std::vector<int>& text_tokens) const {
    text_tokens = tokenizer_->encode(tts_text);
    return text_tokens.size();
}

std::pair<std::vector<int>, int> HLIECosyVoiceModel::speech_tokenizer(const std::vector<uint16_t>& padded_input,
                                                                      const std::vector<uint16_t>& mask, const std::vector<uint16_t>& mask1) {

    auto res =TcimUtil::set_module_input(cosyvoice3_speech_tokenizer_model_, speech_tokenizer_input_names_[0], padded_input.data());
    if (res != tcim::OK) {
        LOG_ERROR("Failed to set input tensor {} for the model", speech_tokenizer_input_names_[0]);
        return {{}, -1};
    }

    res =TcimUtil::set_module_input(cosyvoice3_speech_tokenizer_model_, speech_tokenizer_input_names_[1], mask.data());
    if (res != tcim::OK) {
        LOG_ERROR("Failed to set input tensor {} for the model", speech_tokenizer_input_names_[1]);
        return {{}, -1};
    }

    res =TcimUtil::set_module_input(cosyvoice3_speech_tokenizer_model_, speech_tokenizer_input_names_[2], mask1.data());
    if (res != tcim::OK) {
        LOG_ERROR("Failed to set input tensor {} for the model", speech_tokenizer_input_names_[2]);
        return {{}, -1};
    }

    cosyvoice3_speech_tokenizer_model_->Run();
    cosyvoice3_speech_tokenizer_model_->Sync();

    // 输出直接是 token ID 序列，shape: [1, time_frames] 或 [time_frames]
    auto output = cosyvoice3_speech_tokenizer_model_->GetOutput(speech_tokenizer_output_names_[0]);
    auto output_shape = output.Info().Shape();

    LOG_INFO("Speech tokenizer output shape: [{}, {}]", output_shape[0], output_shape[1]);
    LOG_INFO("Output data type: {}", static_cast<int>(output.Info().DataType()));

    // Step 4: 计算实际的 token 长度
    // feat_len = padded_input 实际非零长度，token_len = feat_len / 4
    // padded_input shape: [1, 128, 3000]，需要找到实际的 feat_len
    int64_t padded_len = padded_input.size() / 128;  // 3000
    int64_t feat_len = padded_len;                   // 实际特征长度
    int64_t speech_token_len = feat_len / 4;        // 实际 token 长度

    LOG_INFO("Expected speech token len: {}", speech_token_len);

    // Step 5: 复制输出到 host (直接就是 token ID)
    // 输出可能是 INT32, INT16 或其他类型
    std::vector<int> speech_tokens;

    if (output.Info().DataType() == tcim::DataType::INT32) {
        // 直接复制 INT32 token IDs
        std::vector<int32_t> buffer(output_shape[1]);
        output.Buffer().CopyToHost(buffer.data(), output.MemSize());

        // 转换为 int 并截取实际长度
        speech_tokens.resize(speech_token_len);
        for (int64_t i = 0; i < speech_token_len; ++i) {
            speech_tokens[i] = static_cast<int>(buffer[i]);
        }
    }
    else if (output.Info().DataType() == tcim::DataType::INT16) {
        // 复制 INT16 token IDs
        std::vector<int16_t> buffer(output_shape[1]);
        output.Buffer().CopyToHost(buffer.data(), output.MemSize());

        speech_tokens.resize(speech_token_len);
        for (int64_t i = 0; i < speech_token_len; ++i) {
            speech_tokens[i] = static_cast<int>(buffer[i]);
        }
    }
    else {
        // 其他类型先转 INT32
        auto output_int32 = output.AsType(tcim::DataType::INT32);
        std::vector<int32_t> buffer(output_shape[1]);
        output_int32.Buffer().CopyToHost(buffer.data(), output_int32.MemSize());

        speech_tokens.resize(speech_token_len);
        for (int64_t i = 0; i < speech_token_len; ++i) {
            speech_tokens[i] = static_cast<int>(buffer[i]);
        }
    }

    LOG_INFO("Extracted speech tokens: count={}", speech_tokens.size());

    return {std::move(speech_tokens), static_cast<int>(speech_tokens.size())};
}

std::vector<float> HLIECosyVoiceModel::campplus(const std::vector<uint16_t>& input) const {
    if (!cosyvoice3_campplus_model_) {
        throw std::runtime_error("Campplus model not loaded");
    }


    // Get input/output names (need to cache these like speech_tokenizer)
    static std::vector<std::string> campplus_input_names;
    static std::vector<std::string> campplus_output_names;
    if (campplus_input_names.empty()) {
        campplus_input_names = TcimUtil::get_module_input_names(cosyvoice3_campplus_model_);
        campplus_output_names = TcimUtil::get_module_output_names(cosyvoice3_campplus_model_);
        LOG_INFO("Campplus input names: {}, output names: {}", campplus_input_names[0], campplus_output_names[0]);
    }

    // Step 7: Run inference
    if (campplus_input_names.empty()) {
        throw std::runtime_error("Campplus input names not available");
    }

    // Set input using TcimUtil (handles float32 -> float16 conversion)
    auto status = TcimUtil::set_module_input(cosyvoice3_campplus_model_, campplus_input_names[0], input.data());
    CHECK_TCIM_STATUS(status);

    CHECK_TCIM_STATUS(cosyvoice3_campplus_model_->Run());
    CHECK_TCIM_STATUS(cosyvoice3_campplus_model_->Sync());

    const auto out_info = cosyvoice3_campplus_model_->GetOutputInfo(campplus_output_names[0]).AsContiguous();
    tcim::Tensor out_tensor = tcim::Tensor::CreateHostTensor(out_info);


    // Step 8: Get output and convert to float vector
    cosyvoice3_campplus_model_->GetOutput(campplus_output_names[0]).CastTo(out_tensor);

    const auto* output_tensor_fp16 = static_cast<TensorFP16*>(out_tensor.Buffer().Data());

    const size_t num = out_tensor.MemSize() / sizeof(TensorFP16);

    std::vector<float> embedding(num);
    for (int i = 0; i < num; ++i) {
        embedding[i] = static_cast<float>(output_tensor_fp16[i]);
    }

    return std::move(embedding);
}


HLIE_NAMESPACE_END

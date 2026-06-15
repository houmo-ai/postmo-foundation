#include "hlie_app/models/hlie_cosyvoice_model.h"

#include <cstring>
#include <random>
#include <regex>

#include "ggml-cpp.h"
#include "gguf.h"
#include "hlie_base/hlie_audio.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_sampler.h"
#include "hlie_base/hlie_tcim_util.h"
#include "hlie_base/hlie_tokenizer.h"
#include "hlie_base/librosa.h"

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

    // LLM constants
    constexpr int SPEECH_TOKEN_SIZE = 6561;

    bool IsLlmKCacheInput(const std::string& input_name) {
        static const std::regex pattern(R"(^model_layers_(\d+)_self_attn_kcache_input$)");
        return std::regex_match(input_name, pattern);
    }

    bool IsLlmVCacheInput(const std::string& input_name) {
        static const std::regex pattern(R"(^model_layers_(\d+)_self_attn_vcache_input$)");
        return std::regex_match(input_name, pattern);
    }

    bool IsLlmCacheInput(const std::string& input_name) {
        return IsLlmKCacheInput(input_name) || IsLlmVCacheInput(input_name);
    }

    std::vector<float> TransposeSeqFeatToFeatSeq(const std::vector<float>& input, int seq_len, int feat_dim) {
        std::vector<float> out(static_cast<size_t>(seq_len * feat_dim), 0.0f);
        for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            for (int feat_idx = 0; feat_idx < feat_dim; ++feat_idx) {
                out[static_cast<size_t>(feat_idx * seq_len + seq_idx)] = input[static_cast<size_t>(seq_idx * feat_dim + feat_idx)];
            }
        }
        return out;
    }

    std::vector<float> BuildCosineTimeSpan(int steps) {
        constexpr float kPi = 3.14159265358979323846f;
        std::vector<float> t_span(static_cast<size_t>(steps + 1), 0.0f);
        for (int idx = 0; idx <= steps; ++idx) {
            const float t = static_cast<float>(idx) / static_cast<float>(steps);
            t_span[static_cast<size_t>(idx)] = 1.0f - std::cos(t * 0.5f * kPi);
        }
        return t_span;
    }
    
    
    std::vector<float> ResizeMelLinear(const std::vector<float>& mel, int mel_bins,
                                       int src_frames, int dst_frames) {
        if (src_frames <= 0 || dst_frames <= 0) {
            return {};
        }
        if (src_frames == dst_frames) {
            return mel;
        }

        std::vector<float> resized(static_cast<size_t>(mel_bins * dst_frames), 0.0f);
        for (int mel_idx = 0; mel_idx < mel_bins; ++mel_idx) {
            const float* src = mel.data() + static_cast<size_t>(mel_idx * src_frames);
            float* dst = resized.data() + static_cast<size_t>(mel_idx * dst_frames);
            if (dst_frames == 1) {
                dst[0] = src[0];
                continue;
            }
            for (int frame_idx = 0; frame_idx < dst_frames; ++frame_idx) {
                const float src_pos = static_cast<float>(frame_idx) *
                                      static_cast<float>(src_frames - 1) /
                                      static_cast<float>(dst_frames - 1);
                const int left = static_cast<int>(src_pos);
                const int right = std::min(left + 1, src_frames - 1);
                const float alpha = src_pos - static_cast<float>(left);
                dst[frame_idx] = src[left] * (1.0f - alpha) + src[right] * alpha;
            }
        }
        return resized;
    }

} // namespace

HLIE_NAMESPACE_BEGIN


int HLIECosyVoiceModel::init() {
    hlie_sampler_ = std::make_shared<HLIESampler>();
    speech_tokenizer_input_names_.clear();
    speech_tokenizer_output_names_.clear();

    speech_tokenizer_input_names_ = TcimUtil::get_module_input_names(cosyvoice3_speech_tokenizer_model_);
    speech_tokenizer_output_names_ = TcimUtil::get_module_output_names(cosyvoice3_speech_tokenizer_model_);

    LOG_INFO("HLIECosyVoiceModel::init 1");
    flow_input_names_ = TcimUtil::get_module_input_names(cosyvoice3_flow_decoder_model_);
    flow_output_names_ = TcimUtil::get_module_output_names(cosyvoice3_flow_decoder_model_);
    flow_encoder_input_names_ = TcimUtil::get_module_input_names(cosyvoice3_flow_encoder_model_);
    flow_encoder_output_names_ = TcimUtil::get_module_output_names(cosyvoice3_flow_encoder_model_);
    flow_spk_input_names_ = TcimUtil::get_module_input_names(cosyvoice3_flow_spk_model_);
    flow_spk_output_names_ = TcimUtil::get_module_output_names(cosyvoice3_flow_spk_model_);
    hift_part1_input_names_ = TcimUtil::get_module_input_names(cosyvoice3_hift_part1_model_);
    hift_part1_output_names_ = TcimUtil::get_module_output_names(cosyvoice3_hift_part1_model_);
    hift_part2_input_names_ = TcimUtil::get_module_input_names(cosyvoice3_hift_part2_model_);
    hift_part2_output_names_ = TcimUtil::get_module_output_names(cosyvoice3_hift_part2_model_);

    LOG_INFO("HLIECosyVoiceModel::init 2");
    for (int idx = 0; idx < llm_qwen2_prefill_model_->GetInputNum(); idx++) {
        auto input_name = llm_qwen2_prefill_model_->GetInputName(idx);
        auto input_info = llm_qwen2_prefill_model_->GetInputInfo(input_name).AsContiguous();
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        llm_qwen2_prefill_input_map_.insert(std::pair(input_name, input_tensor));
    }

    LOG_INFO("HLIECosyVoiceModel::init 3");
    for (int idx = 0; idx < llm_qwen2_decode_model_->GetInputNum(); idx++) {
        auto input_name = llm_qwen2_decode_model_->GetInputName(idx);
        auto input_info = llm_qwen2_decode_model_->GetInputInfo(input_name).AsContiguous();
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        llm_qwen2_decode_input_map_.insert(std::pair(input_name, input_tensor));
    }


    for (int idx = 0; idx < cosyvoice3_speech_tokenizer_model_->GetInputNum(); idx++) {
        auto input_name = cosyvoice3_speech_tokenizer_model_->GetInputName(idx);
        auto input_info = cosyvoice3_speech_tokenizer_model_->GetInputInfo(input_name).AsContiguous();
        auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
        cosyvoice3_speech_tokenizer_input_map_.insert(std::pair(input_name, input_tensor));
    }

    std::vector loop_curr_len = {1};
    LOG_WARN("setting cur len {}", decode_input_names_[2]);
    auto cur_len_tensor = llm_qwen2_decode_input_map_.at(decode_input_names_[2]);
    cur_len_tensor.Buffer().CopyFromHost(loop_curr_len.data(), loop_curr_len.size() * sizeof(int));
    llm_qwen2_decode_model_->SetInput(decode_input_names_[2], cur_len_tensor);


    for (int idx = 0; idx < cosyvoice3_flow_decoder_model_->GetInputNum(); idx++) {
        auto name = cosyvoice3_flow_decoder_model_->GetInputName(idx);
        auto info = cosyvoice3_flow_decoder_model_->GetInputInfo(name).AsContiguous();
        flow_decoder_input_maps_[name] = tcim::Tensor::CreateHostTensor(info);
    }

    for (int idx = 0; idx < cosyvoice3_hift_part1_model_->GetInputNum(); idx++) {
        auto name = cosyvoice3_hift_part1_model_->GetInputName(idx);
        auto info = cosyvoice3_hift_part1_model_->GetInputInfo(name).AsContiguous();
        hift_part1_input_maps_[name] = tcim::Tensor::CreateHostTensor(info);
    }

    for (int idx = 0; idx < cosyvoice3_hift_part2_model_->GetInputNum(); idx++) {
        auto name = cosyvoice3_hift_part2_model_->GetInputName(idx);
        auto info = cosyvoice3_hift_part2_model_->GetInputInfo(name).AsContiguous();
        hift_part2_input_maps_[name] = tcim::Tensor::CreateHostTensor(info);
    }

    LOG_INFO("HLIECosyVoiceModel::init 4");

    // Initialize stop token IDs: [6561, 6562, ..., 6760]
    stop_token_ids_.reserve(200);
    for (int i = 0; i < 200; ++i) {
        stop_token_ids_.push_back(SPEECH_TOKEN_SIZE + i);
    }

    // ========== Print all models' input/output information ==========
    auto print_model_info = [](const std::shared_ptr<tcim::Module>& model, const std::string& model_name) {
        LOG_TRACE("╔══════════════════════════════════════════════════════════════╗");
        LOG_TRACE("║ Model: {:<55} ║", model_name);
        LOG_TRACE("╠══════════════════════════════════════════════════════════════╣");

        // Print inputs
        size_t num_inputs = model->GetInputNum();
        LOG_TRACE("║ Inputs ({:2d}):                                                     ║", num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
            auto input_name = model->GetInputName(static_cast<int>(i));
            auto input_info = model->GetInputInfo(input_name).AsContiguous();
            auto shape = input_info.Shape();
            auto dtype = input_info.DataType();

            std::string shape_str;
            for (size_t j = 0; j < shape.size(); ++j) {
                shape_str += std::to_string(shape[j]);
                if (j < shape.size() - 1)
                    shape_str += " × ";
            }

            std::string dtype_str;
            switch (dtype) {
            case tcim::DataType::FLOAT32:
                dtype_str = "float32";
                break;
            case tcim::DataType::FLOAT16:
                dtype_str = "float16";
                break;
            case tcim::DataType::INT32:
                dtype_str = "int32";
                break;
            case tcim::DataType::INT16:
                dtype_str = "int16";
                break;
            case tcim::DataType::INT8:
                dtype_str = "int8";
                break;
            case tcim::DataType::UINT8:
                dtype_str = "uint8";
                break;
            case tcim::DataType::UINT16:
                dtype_str = "uint16";
                break;
            case tcim::DataType::UINT32:
                dtype_str = "uint32";
                break;
            default:
                dtype_str = "unknown";
                break;
            }

            LOG_INFO("║   [{:2d}] {:<20} │ shape: {:<28} │ {:<10} ║", i, input_name.substr(0, 20), shape_str.substr(0, 28), dtype_str);
        }

        // Print outputs
        size_t num_outputs = model->GetOutputNum();
        LOG_TRACE("╠══════════════════════════════════════════════════════════════╣");
        LOG_TRACE("║ Outputs ({:2d}):                                                    ║", num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            auto output_name = model->GetOutputName(static_cast<int>(i));
            auto output_info = model->GetOutputInfo(output_name).AsContiguous();
            auto shape = output_info.Shape();
            auto dtype = output_info.DataType();

            std::string shape_str;
            for (size_t j = 0; j < shape.size(); ++j) {
                shape_str += std::to_string(shape[j]);
                if (j < shape.size() - 1)
                    shape_str += " × ";
            }

            std::string dtype_str;
            switch (dtype) {
            case tcim::DataType::FLOAT32:
                dtype_str = "float32";
                break;
            case tcim::DataType::FLOAT16:
                dtype_str = "float16";
                break;
            case tcim::DataType::INT32:
                dtype_str = "int32";
                break;
            case tcim::DataType::INT16:
                dtype_str = "int16";
                break;
            case tcim::DataType::INT8:
                dtype_str = "int8";
                break;
            case tcim::DataType::UINT8:
                dtype_str = "uint8";
                break;
            case tcim::DataType::UINT16:
                dtype_str = "uint16";
                break;
            case tcim::DataType::UINT32:
                dtype_str = "uint32";
                break;
            default:
                dtype_str = "unknown";
                break;
            }

            LOG_TRACE("║   [{:2d}] {:<20} │ shape: {:<28} │ {:<10} ║", i, output_name.substr(0, 20), shape_str.substr(0, 28), dtype_str);
        }
        LOG_TRACE("╚══════════════════════════════════════════════════════════════╝");
    };

    // Print all model information
    print_model_info(cosyvoice3_speech_tokenizer_model_, "speech_tokenizer");
    print_model_info(cosyvoice3_campplus_model_, "campplus");
    print_model_info(cosyvoice3_flow_decoder_model_, "flow_decoder");
    print_model_info(cosyvoice3_flow_encoder_model_, "flow_encoder");
    print_model_info(cosyvoice3_flow_spk_model_, "flow_spk");
    print_model_info(cosyvoice3_hift_part1_model_, "hift_part1");
    print_model_info(cosyvoice3_hift_part2_model_, "hift_part2");
    print_model_info(cosyvoice3_llm_decoder_model_, "llm_decoder");
    print_model_info(llm_qwen2_decode_model_, "llm_qwen2_decode");
    print_model_info(llm_qwen2_prefill_model_, "llm_qwen2_prefill");

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


        const auto n_tensors = gguf_get_n_tensors(ctx);
        // Helper lambda to load model by name
        auto load_model_by_name = [&](const std::string& tensor_name, auto& model_ptr,
                                      const std::vector<std::string>& dummy_tensor_names = {}) -> bool {
            for (int i = 0; i < n_tensors; ++i) {
                const char* name = gguf_get_tensor_name(ctx, i);
                if (tensor_name == name) {
                    const size_t size = gguf_get_tensor_size(ctx, i);
                    const size_t offset = gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
                    model_ptr = create_infer_engine(model, offset, size, dummy_tensor_names);

                    if (!model_ptr) {
                        LOG_ERROR("failed to load {}", tensor_name);
                        return false;
                    }
                    LOG_INFO("succeed to load {}", tensor_name);
                    return true;
                }
            }
            LOG_ERROR("tensor {} not found in model", tensor_name);
            return false;
        };

        // Helper lambda to load tensor by name
        auto load_tensor_by_name = [&](const std::string& tensor_name, std::shared_ptr<tcim::Tensor>& tensor_ptr,
                                       const std::vector<int64_t>& shape) -> bool {
            for (int i = 0; i < n_tensors; ++i) {
                const char* name = gguf_get_tensor_name(ctx, i);
                if (tensor_name == name) {
                    const size_t size = gguf_get_tensor_size(ctx, i);
                    const size_t offset = gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
                    LOG_INFO("Start to load {}", tensor_name);
                    auto tensor_data = TcimUtil::load_fp16_data(model, offset, size);
                    auto tensor = TcimUtil::create_fp16_tensor(tensor_data.data(), shape);
                    if (tensor.GetInitStatus() != tcim::OK) {
                        LOG_ERROR("failed to load {}", tensor_name);
                        return false;
                    }
                    tensor_ptr = std::make_shared<tcim::Tensor>(tensor);
                    LOG_INFO("succeed to load {}", tensor_name);
                    return true;
                }
            }
            LOG_ERROR("tensor {} not found in model", tensor_name);
            return false;
        };

        // Load all models
        if (!load_model_by_name(std::string(cosyvoice3_campplus), cosyvoice3_campplus_model_))
            return -1;
        if (!load_model_by_name(std::string(cosyvoice3_flow_decoder), cosyvoice3_flow_decoder_model_))
            return -1;
        if (!load_model_by_name(std::string(cosyvoice3_flow_encoder), cosyvoice3_flow_encoder_model_))
            return -1;
        if (!load_model_by_name(std::string(cosyvoice3_flow_spk), cosyvoice3_flow_spk_model_))
            return -1;

        if (!load_model_by_name(std::string(cosyvoice3_llm_decoder), cosyvoice3_llm_decoder_model_))
            return -1;
        if (!load_model_by_name(std::string(cosyvoice3_llm_qwen2_prefill), llm_qwen2_prefill_model_))
            return -1;


        std::vector<std::string> llm_qwen2_docode_dummy_tensor_names;
        std::vector<std::string> llm_qwen2_kcache_input_names;
        std::vector<std::string> llm_qwen2_vcache_input_names;
        prefill_input_names_ = TcimUtil::get_module_input_names(llm_qwen2_prefill_model_);
        for (const auto& input_name : prefill_input_names_) {
            if (IsLlmKCacheInput(input_name)) {
                llm_qwen2_kcache_input_names.push_back(input_name);
            } else if (IsLlmVCacheInput(input_name)) {
                llm_qwen2_vcache_input_names.push_back(input_name);
            }
        }
        llm_qwen2_docode_dummy_tensor_names.reserve(llm_qwen2_kcache_input_names.size() + llm_qwen2_vcache_input_names.size());
        llm_qwen2_docode_dummy_tensor_names.insert(llm_qwen2_docode_dummy_tensor_names.end(), llm_qwen2_kcache_input_names.begin(),
                                                   llm_qwen2_kcache_input_names.end());
        llm_qwen2_docode_dummy_tensor_names.insert(llm_qwen2_docode_dummy_tensor_names.end(), llm_qwen2_vcache_input_names.begin(),
                                                   llm_qwen2_vcache_input_names.end());

        if (!load_model_by_name(std::string(cosyvoice3_llm_qwen2_decode), llm_qwen2_decode_model_, llm_qwen2_docode_dummy_tensor_names))
            return -1;

        for (const auto& input_name : llm_qwen2_docode_dummy_tensor_names) {
            auto cache = llm_qwen2_prefill_model_->GetDevInput(input_name);
            llm_qwen2_decode_model_->SetDevInput(input_name, cache);
        }

        if (!load_model_by_name(std::string(cosyvoice3_speech_tokenizer), cosyvoice3_speech_tokenizer_model_))
            return -1;

        if (!load_model_by_name(std::string(cosyvoice3_hift_part1), cosyvoice3_hift_part1_model_))
            return -1;

        std::string hift2_dummy_input_name = cosyvoice3_hift_part1_model_->GetInputName(0);

        if (!load_model_by_name(std::string(cosyvoice3_hift_part2), cosyvoice3_hift_part2_model_, {hift2_dummy_input_name}))
            return -1;

        auto dummy_input = cosyvoice3_hift_part1_model_->GetDevInput(hift2_dummy_input_name);
        CHECK_TCIM_STATUS(cosyvoice3_hift_part2_model_->SetDevInput(hift2_dummy_input_name, dummy_input));

        // Load all tensors
        if (!load_tensor_by_name(std::string(quant_embedding), token_weight_embedding_, {151936, 896}))
            return -1;
        if (!load_tensor_by_name(std::string(flow_input_embedding), flow_input_embedding_, {6561, 80}))
            return -1;
        if (!load_tensor_by_name(std::string(llm_sos_embedding), llm_sos_embedding_, {1, 1, 896}))
            return -1;
        if (!load_tensor_by_name(std::string(llm_speech_embedding), speech_weight_embedding_, {6761, 896}))
            return -1;
        if (!load_tensor_by_name(std::string(llm_task_id_embedding), llm_task_id_embedding_, {1, 1, 896}))
            return -1;


        // Initialize LLM model I/O names
        decode_input_names_ = TcimUtil::get_module_input_names(llm_qwen2_decode_model_);
        llm_decoder_input_names_ = TcimUtil::get_module_input_names(cosyvoice3_llm_decoder_model_);
        llm_decoder_output_names_ = TcimUtil::get_module_output_names(cosyvoice3_llm_decoder_model_);

        // Extract model dimension info from prefill model
        if (!prefill_input_names_.empty()) {
            const auto input_info = llm_qwen2_prefill_model_->GetInputInfo(prefill_input_names_[0]);
            auto prefill_shape = input_info.AsContiguous().Shape();
            prefill_length_ = prefill_shape[1];
            embedding_len_ = prefill_shape[2];
            LOG_INFO("Prefill model input shape: [{}, {}, {}]", prefill_shape[0], prefill_shape[1], prefill_shape[2]);
        }

        for (const auto& input_name : prefill_input_names_) {
            if (IsLlmCacheInput(input_name)) {
                const auto cache_input_info = llm_qwen2_prefill_model_->GetInputInfo(input_name);
                context_max_length_ = cache_input_info.AsContiguous().Shape()[2];
                break;
            }
        }

        if (!decode_input_names_.empty()) {
            auto decode_batch_info = llm_qwen2_decode_model_->GetInputInfo(decode_input_names_[0]);
            batch_size_ = decode_batch_info.AsContiguous().Shape()[0];
            LOG_INFO("Decode model context_max_length: {}, batch_size: {}", context_max_length_, batch_size_);
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

std::pair<std::vector<int>, int> HLIECosyVoiceModel::speech_tokenizer(const std::vector<TensorFP16>& padded_input,
                                                                      const std::vector<TensorFP16>& mask,
                                                                      const std::vector<TensorFP16>& mask1, int feat_len) {
    // Set inputs to module
    // Input 0: padded_mel [1, 128, 3000]
    // Input 1: mask [1, 20, 750, 750]
    // Input 2: mask1 [1, 750, 1280]

    auto padded_input_tensor = cosyvoice3_speech_tokenizer_input_map_.at(speech_tokenizer_input_names_[0]);
    CHECK_TCIM_STATUS(padded_input_tensor.Buffer().CopyFromHost(padded_input.data(), padded_input.size() * sizeof(TensorFP16)));
    CHECK_TCIM_STATUS(cosyvoice3_speech_tokenizer_model_->SetInput(speech_tokenizer_input_names_[0], padded_input_tensor));

    auto mask_tensor = cosyvoice3_speech_tokenizer_input_map_.at(speech_tokenizer_input_names_[1]);
    CHECK_TCIM_STATUS(mask_tensor.Buffer().CopyFromHost(mask.data(), mask.size() * sizeof(TensorFP16)));
    CHECK_TCIM_STATUS(cosyvoice3_speech_tokenizer_model_->SetInput(speech_tokenizer_input_names_[1], mask_tensor));


    auto mask1_tensor = cosyvoice3_speech_tokenizer_input_map_.at(speech_tokenizer_input_names_[2]);
    CHECK_TCIM_STATUS(mask1_tensor.Buffer().CopyFromHost(mask1.data(), mask1.size() * sizeof(TensorFP16)));
    CHECK_TCIM_STATUS(cosyvoice3_speech_tokenizer_model_->SetInput(speech_tokenizer_input_names_[2], mask1_tensor));

    CHECK_TCIM_STATUS(cosyvoice3_speech_tokenizer_model_->Run());
    CHECK_TCIM_STATUS(cosyvoice3_speech_tokenizer_model_->Sync());

    // 输出直接是 token ID 序列，shape: [1, time_frames] 或 [time_frames]
    auto output = cosyvoice3_speech_tokenizer_model_->GetOutput(speech_tokenizer_output_names_[0]);
    auto output_shape = output.Info().Shape();

    LOG_INFO("Speech tokenizer output shape: [{}, {}]", output_shape[0], output_shape[1]);

    // Step 4: 计算实际的 token 长度
    // feat_len = padded_input 实际非零长度，token_len = feat_len / 4
    // padded_input shape: [1, 128, 3000]，需要找到实际的 feat_len
    int64_t speech_token_len = feat_len / 4; // 实际 token 长度

    LOG_INFO("Expected speech token len: {}", speech_token_len);

    // Step 5: 复制输出到 host (直接就是 token ID)
    std::vector<int32_t> buffer(output_shape[1]);
    output.Buffer().CopyToHost(buffer.data(), output.MemSize());

    // 转换为 int 并截取实际长度
    std::vector<int> speech_tokens;
    speech_tokens.resize(speech_token_len);
    for (int64_t i = 0; i < speech_token_len; ++i) {
        speech_tokens[i] = buffer[i];
        LOG_DEBUG("speech_tokens[{}]: {}", i, speech_tokens[i]);
    }

    LOG_INFO("Extracted speech tokens: count={}", speech_tokens.size());

    return {std::move(speech_tokens), static_cast<int>(speech_tokens.size())};
}

std::vector<float> HLIECosyVoiceModel::campplus(const std::vector<TensorFP16>& input) const {
    // Get input/output names (need to cache these like speech_tokenizer)
    static std::vector<std::string> campplus_input_names;
    static std::vector<std::string> campplus_output_names;
    if (campplus_input_names.empty()) {
        campplus_input_names = TcimUtil::get_module_input_names(cosyvoice3_campplus_model_);
        campplus_output_names = TcimUtil::get_module_output_names(cosyvoice3_campplus_model_);
        LOG_INFO("Campplus input names: {}, output names: {}", campplus_input_names[0], campplus_output_names[0]);
    }

    auto input_tensor_info_fp16 = cosyvoice3_campplus_model_->GetInputInfo(campplus_input_names[0]).AsContiguous();
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_tensor_info_fp16);
    CHECK_TCIM_STATUS(input_tensor.Buffer().CopyFromHost(input.data(), input.size() * sizeof(TensorFP16)));
    CHECK_TCIM_STATUS(cosyvoice3_campplus_model_->SetInput(campplus_input_names[0], input_tensor));

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

std::vector<int> HLIECosyVoiceModel::llm_inference(const std::vector<int>& text, int text_len, const std::vector<int>& prompt_text,
                                                   int prompt_text_len, const std::vector<int>& prompt_speech_token,
                                                   int prompt_speech_token_len, const std::vector<float>& llm_embedding, int sampling,
                                                   float max_token_text_ratio, float min_token_text_ratio, LLMPerf* perf) {

    using namespace std::chrono;

    auto t_start = high_resolution_clock::now();

    // ========== STEP 1: Concatenate prompt_text + text ==========
    std::vector<int> concat_text = prompt_text;
    concat_text.insert(concat_text.end(), text.begin(), text.end());
    int concat_text_len = text_len + prompt_text_len;

    LOG_INFO("LLM inference: concat_text_len={}, prompt_speech_token_len={}", concat_text_len, prompt_speech_token_len);

    // ========== STEP 2: Get text embeddings ==========
    // Equivalent Python: text = self.token_embedding(text)
    auto text_emb = TcimUtil::embedding_lookup(*token_weight_embedding_, concat_text);

    for (auto i : text_emb.Info().Shape()) {
        LOG_DEBUG("LLM inference: text_emb Shape={}, ", i);
    }

    // ========== STEP 3: Get prompt speech token embeddings ==========
    // Direct batch lookup - more efficient than individual lookups
    LOG_DEBUG("LLM inference: STEP 3: Get prompt speech token embeddings, prompt_speech_token_len: {}", prompt_speech_token_len);
    tcim::Tensor prompt_speech_emb;
    if (prompt_speech_token_len > 0) {
        prompt_speech_emb = TcimUtil::embedding_lookup(*speech_weight_embedding_, prompt_speech_token);
    } else {
        // Create empty tensor: [1, 0, 896]
        auto empty_info = tcim::TensorInfo::CreateNDInfo({1, 0, embedding_len_}, tcim::DataType::FLOAT16);
        prompt_speech_emb = tcim::Tensor::CreateHostTensor(empty_info);
    }

    // ========== STEP 4: Build lm_input = [sos_emb, text_emb, task_id_emb, prompt_speech_emb] ==========
    // Prepare individual tensors for concatenation
    LOG_DEBUG("STEP 4: Build lm_input = [sos_emb, text_emb, task_id_emb, prompt_speech_emb]");
    std::vector<tcim::Tensor> lm_input_tensors;
    lm_input_tensors.push_back(*llm_sos_embedding_); // [1, 1, 896]
    lm_input_tensors.push_back(text_emb); // [1, concat_text_len, 896]
    lm_input_tensors.push_back(*llm_task_id_embedding_); // [1, 1, 896]
    lm_input_tensors.push_back(prompt_speech_emb); // [1, prompt_speech_token_len, 896]

    // Concatenate all tensors along sequence dimension
    tcim::Tensor lm_input = TcimUtil::concat_seq(lm_input_tensors);

    // Get lm_input shape
    auto lm_input_shape = lm_input.Info().Shape();
    int seq_length = static_cast<int>(lm_input_shape[1]); // Total sequence length

    if (perf) {
        perf->prefill_tokens = seq_length;
    }

    LOG_INFO("lm_input built: seq_length={}, embedding_len={}", seq_length, embedding_len_);

    // Validate input length
    if (seq_length >= context_max_length_) {
        LOG_ERROR("Input sequence length ({}) exceeds maximum context length ({})", seq_length, context_max_length_);
        throw std::runtime_error("Input sequence too long");
    }

    // Calculate min/max length for generation
    int min_len = static_cast<int>(text_len * min_token_text_ratio);
    int max_len = static_cast<int>(text_len * max_token_text_ratio);

    // ========== STEP 5: PREFILL PHASE ==========
    auto t_prefill0 = high_resolution_clock::now();

    int context_length = 0;
    int prefill_loop_round = (seq_length + prefill_length_ - 1) / prefill_length_;
    LOG_INFO("Prefill loop rounds: {}", prefill_loop_round);

    // Get input names
    const std::string& input_name = prefill_input_names_[0];
    const std::string& valid_length_name = prefill_input_names_[1];
    const std::string& current_length_name = prefill_input_names_[2];

    for (int round_idx = 0; round_idx < prefill_loop_round; ++round_idx) {
        int valid_length = round_idx * prefill_length_ + context_length;
        int current_length;
        std::vector<tcim::Tensor> chunk_tensors;

        if (round_idx == prefill_loop_round - 1) {
            // Last round: use remaining tokens
            current_length = seq_length - round_idx * prefill_length_;
        } else {
            current_length = prefill_length_;
        }
        LOG_DEBUG("prefill round_idx: {} cur length: {}", round_idx, current_length);

        // Extract chunk from lm_input
        // We need to slice lm_input[round_idx * prefill_length : round_idx * prefill_length + current_length]
        // Using concat_seq with a single slice is not directly supported, so we extract manually

        // Get source data pointer
        const auto* lm_input_data = static_cast<const TensorFP16*>(lm_input.Data());
        int start_pos = round_idx * prefill_length_;
        int chunk_len = current_length;

        // Create chunk tensor info: [1, chunk_len, embedding_len]
        auto chunk_info = tcim::TensorInfo::CreateNDInfo({1, chunk_len, embedding_len_}, tcim::DataType::FLOAT16);
        tcim::Tensor chunk_tensor = tcim::Tensor::CreateHostTensor(chunk_info);
        auto* chunk_data = static_cast<TensorFP16*>(chunk_tensor.Data());

        LOG_DEBUG("prefill round_idx: {} cur chunk_data len: {}", round_idx, chunk_len);

        // Copy data for this chunk
        for (int i = 0; i < chunk_len; ++i) {
            const TensorFP16* src_row = lm_input_data + (start_pos + i) * embedding_len_;
            TensorFP16* dst_row = chunk_data + i * embedding_len_;
            std::memcpy(dst_row, src_row, embedding_len_ * sizeof(TensorFP16));
        }

        // ========== Pad using token_embedding with pad_token_id ==========
        std::vector<tcim::Tensor> prefill_tensors;
        int effective_length = chunk_len;
        if (effective_length < prefill_length_) {
            int pad_len = prefill_length_ - effective_length;
            LOG_DEBUG("prefill  round_idx: {} pad_len: {} ", round_idx, pad_len);
            // Create pad_token_ids vector filled with pad_token_id (151645)
            std::vector<int> pad_token_ids(pad_len, pad_token_id()); // pad_token_id = 151645

            // Get padding embeddings using token_embedding
            tcim::Tensor pad_embeds = TcimUtil::embedding_lookup(*token_weight_embedding_, pad_token_ids);

            prefill_tensors.push_back(chunk_tensor);
            prefill_tensors.push_back(pad_embeds);
        } else {
            LOG_DEBUG("prefill  round_idx: {} no pad_len", round_idx);
            prefill_tensors.push_back(chunk_tensor);
        }

        // Concatenate chunk + padding
        tcim::Tensor input_data = TcimUtil::concat_seq(prefill_tensors);

        // Set inputs for prefill model
        llm_qwen2_prefill_model_->SetInput(input_name, input_data);

        // Set valid_length input
        std::vector<int32_t> valid_length_data = {valid_length};
        LOG_DEBUG("prefill valid_length: {} ", valid_length);
        const auto tensor1 = llm_qwen2_prefill_input_map_.at(valid_length_name);
        CHECK_TCIM_STATUS(tensor1.Buffer().CopyFromHost(valid_length_data.data(), valid_length_data.size() * sizeof(int)));
        llm_qwen2_prefill_model_->SetInput(valid_length_name, tensor1);


        // Set current_length input
        LOG_DEBUG("prefill current_length: {} ", current_length);
        std::vector<int32_t> current_length_data = {current_length};
        const auto tensor2 = llm_qwen2_prefill_input_map_.at(current_length_name);
        CHECK_TCIM_STATUS(tensor2.Buffer().CopyFromHost(current_length_data.data(), current_length_data.size() * sizeof(int)));
        llm_qwen2_prefill_model_->SetInput(current_length_name, tensor2);

        llm_qwen2_prefill_model_->Run();
        llm_qwen2_prefill_model_->Sync();
    }

    auto t_prefill1 = high_resolution_clock::now();

    // ========== STEP 6: Get prefill output and run LLM decoder ==========
    auto prefill_dev_opt = llm_qwen2_prefill_model_->GetDevOutput(llm_qwen2_prefill_model_->GetOutputName(0));
    auto llm_decoder_dev_ipt0 = prefill_dev_opt.SelectBatch({0});
    CHECK_TCIM_STATUS(cosyvoice3_llm_decoder_model_->SetInput(llm_decoder_input_names_[0], llm_decoder_dev_ipt0));

    cosyvoice3_llm_decoder_model_->Run();
    cosyvoice3_llm_decoder_model_->Sync();

    // ========== STEP 7: Sample first token ==========
    std::vector<int> out_tokens;
    bool ignore_eos = (min_len > 0);
    auto llm_dev_output_opt = cosyvoice3_llm_decoder_model_->GetDevOutput(llm_decoder_output_names_[0]);
    auto llm_host_output_opt = llm_dev_output_opt.ToHost(true);

    int top_ids =  hlie_sampler_->SamplingIds(llm_host_output_opt, out_tokens, 6761, sampling, ignore_eos);
    auto t_first_token = high_resolution_clock::now();
    /*
    // Check if first token is stop token
    bool is_first_stop = false;
    for (int stop_id : stop_token_ids_) {
        if (top_ids == stop_id) {
            is_first_stop = true;
            break;
        }
    }

    if (is_first_stop) {
        if (perf) {
            perf->prefill_s = duration<float>(t_prefill1 - t_prefill0).count();
            perf->ttft_s = duration<float>(t_first_token - t_start).count();
            perf->decode_s = 0.0f;
            perf->decode_tokens = 0;
            perf->llm_total_s = duration<float>(t_first_token - t_start).count();
        }
        return out_tokens;
    }*/

    out_tokens.push_back(top_ids);
    context_length += seq_length;

    // ========== STEP 8: DECODE LOOP ==========
    auto t_decode0 = high_resolution_clock::now();
    int decode_token_cnt = 0;

    const std::string& decode_input_name = decode_input_names_[0];
    const std::string& decode_valid_length_name = decode_input_names_[1];
    const std::string& decode_current_length_name = decode_input_names_[2];
    const int32_t decode_current_length = 1;

    LOG_DEBUG("CosyVoice3 llm decode start, top_ids: {}, context_length: {}, max_len: {}", top_ids, context_length, max_len);
    for (int i = 1; i < max_len; ++i) {
        auto token_emb = TcimUtil::embedding_lookup(*speech_weight_embedding_, {top_ids});

        // Set decode input
        CHECK_TCIM_STATUS(llm_qwen2_decode_model_->SetInput(decode_input_name, token_emb));

        // Set valid length
        std::vector<int32_t> valid_len_vec = {context_length};
        LOG_DEBUG("CosyVoice3 llm decode [{}], top_ids: {}, context_length: {}", i, top_ids, context_length);
        const auto decode_valid_length_tensor = llm_qwen2_decode_input_map_.at(decode_valid_length_name);
        CHECK_TCIM_STATUS(decode_valid_length_tensor.Buffer().CopyFromHost(valid_len_vec.data(), valid_len_vec.size() * sizeof(int32_t)));
        CHECK_TCIM_STATUS(llm_qwen2_decode_model_->SetInput(decode_valid_length_name, decode_valid_length_tensor));

        const auto decode_current_length_tensor = llm_qwen2_decode_input_map_.at(decode_current_length_name);
        CHECK_TCIM_STATUS(decode_current_length_tensor.Buffer().CopyFromHost(&decode_current_length, sizeof(decode_current_length)));
        CHECK_TCIM_STATUS(llm_qwen2_decode_model_->SetInput(decode_current_length_name, decode_current_length_tensor));

        CHECK_TCIM_STATUS(llm_qwen2_decode_model_->Run());
        CHECK_TCIM_STATUS(llm_qwen2_decode_model_->Sync());

        // ========== STEP 4: Get decode output ==========
        auto decode_dev_opt = llm_qwen2_decode_model_->GetDevOutput(llm_qwen2_decode_model_->GetOutputName(0));
        auto llm_decoder_dev_ipt = decode_dev_opt.SelectBatch({0});

        // ========== STEP 5: Set llm_decoder input ==========
        CHECK_TCIM_STATUS(cosyvoice3_llm_decoder_model_->SetInput(llm_decoder_input_names_[0], llm_decoder_dev_ipt));

        // ========== STEP 6: Run llm_decoder ==========
        // Equivalent Python: self.llm_decoder.run()
        CHECK_TCIM_STATUS(cosyvoice3_llm_decoder_model_->Run());
        CHECK_TCIM_STATUS(cosyvoice3_llm_decoder_model_->Sync());

        // ========== STEP 7: Get logp output and squeeze Sample ==========
        ignore_eos = (i < min_len);
        auto llm_dev_output_opt1 = cosyvoice3_llm_decoder_model_->GetDevOutput(llm_decoder_output_names_[0]);
        auto llm_host_output_opt1 = llm_dev_output_opt1.ToHost(true);
        top_ids = hlie_sampler_->SamplingIds(llm_host_output_opt1, out_tokens, 6761, sampling, ignore_eos);

        context_length += 1;

        // Check stop token
        bool is_stop = false;
        for (int stop_id : stop_token_ids_) {
            if (top_ids == stop_id) {
                is_stop = true;
                break;
            }
        }

        if (is_stop) {
            break;
        }

        out_tokens.push_back(top_ids);
        decode_token_cnt++;
    }

    auto t_decode1 = high_resolution_clock::now();

    if (perf) {
        perf->prefill_s = duration<float>(t_prefill1 - t_prefill0).count();
        perf->ttft_s = duration<float>(t_first_token - t_start).count();
        perf->decode_s = duration<float>(t_decode1 - t_decode0).count();
        perf->decode_tokens = decode_token_cnt;
        perf->llm_total_s = duration<float>(t_decode1 - t_start).count();
    }

    LOG_INFO("LLM inference complete: {} tokens generated", out_tokens.size());

    return out_tokens;
}

std::vector<float> HLIECosyVoiceModel::token2wav(const std::vector<int>& token, const std::vector<int>& prompt_token,
                                                 const std::vector<float>& prompt_feat, const std::vector<float>& embedding,
                                                 float speed) const {

    using Clock = std::chrono::steady_clock;
    const auto elapsed_ms = [](const Clock::time_point begin, const Clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    const auto t_start = Clock::now();

    constexpr int flow_mel_bins_ = 80;
    constexpr int kTokenMelRatio = 2;
    constexpr float kInferenceCfgRate = 0.7f;
    constexpr int kNumSteps = 10; // Number of diffusion steps

    // ========== STEP 1: Concatenate prompt_token + token ==========
    std::vector<int> concat_token = prompt_token;
    concat_token.insert(concat_token.end(), token.begin(), token.end());
    int concat_token_len = static_cast<int>(concat_token.size());

    LOG_INFO("token2wav: concat_token_len={}, prompt_token_len={}, token_len={}", concat_token_len, prompt_token.size(), token.size());

    // ========== STEP 2: Token embedding lookup ==========
    auto token_emb_tensor = TcimUtil::embedding_lookup(*flow_input_embedding_, concat_token);
    auto token_emb_info = token_emb_tensor.Info().AsContiguous();
    auto token_emb_shape = token_emb_info.Shape();

    const auto flow_encoder_input_info =
        cosyvoice3_flow_encoder_model_->GetInputInfo(flow_encoder_input_names_[0]).AsContiguous();
    const auto flow_encoder_input_shape = flow_encoder_input_info.Shape();
    const int encoder_seq_len = static_cast<int>(flow_encoder_input_shape[1]);
    const int encoder_feat_dim = static_cast<int>(flow_encoder_input_shape[2]);
    if (encoder_feat_dim != flow_mel_bins_) {
        throw std::runtime_error("Unexpected flow encoder feature dimension");
    }
    if (concat_token_len > encoder_seq_len) {
        throw std::runtime_error("Flow token length exceeds flow encoder static sequence length");
    }

    int pad_len = encoder_seq_len - static_cast<int>(token_emb_shape[1]);
    if (pad_len < 0) {
        throw std::runtime_error("Embedded flow token length exceeds flow encoder static sequence length");
    }

    // Prepare padded token embedding for flow_encoder
    // Shape: [1, seq_len, 80] -> [1, 1024, 80]
    LOG_DEBUG("Prepare padded token embedding for flow_encoder");
    std::vector<TensorFP16> padded_token_emb;
    padded_token_emb.reserve(static_cast<size_t>(encoder_seq_len * flow_mel_bins_));

    const auto* token_emb_data = static_cast<const TensorFP16*>(token_emb_tensor.Buffer().Data());
    size_t token_emb_size = token_emb_tensor.MemSize() / sizeof(TensorFP16);

    for (size_t i = 0; i < token_emb_size; ++i) {
        padded_token_emb.push_back(token_emb_data[i]);
    }

    // Pad with zeros
    for (int i = 0; i < pad_len * flow_mel_bins_; ++i) {
        padded_token_emb.push_back(static_cast<TensorFP16>(0.0f));
    }
    const auto t_embedding_ready = Clock::now();

    // Set input to flow_encoder and run
    // Input shape: [1, 1024, 80] (float16)
    LOG_DEBUG("Set input to flow_encoder and run");

    auto input_tensor = tcim::Tensor::CreateHostTensor(flow_encoder_input_info);
    CHECK_TCIM_STATUS(input_tensor.Buffer().CopyFromHost(padded_token_emb.data(), padded_token_emb.size() * sizeof(TensorFP16)));
    CHECK_TCIM_STATUS(cosyvoice3_flow_encoder_model_->SetInput(flow_encoder_input_names_[0], input_tensor));

    const auto t_flow_encoder_run_start = Clock::now();
    CHECK_TCIM_STATUS(cosyvoice3_flow_encoder_model_->Run());
    CHECK_TCIM_STATUS(cosyvoice3_flow_encoder_model_->Sync());
    const auto t_flow_encoder_run_end = Clock::now();

    // Get output from flow_encoder
    // Output shape: [1, 1024, 80] (float16)
    LOG_DEBUG("Get output from flow_encoder");
    auto flow_encoder_output = cosyvoice3_flow_encoder_model_->GetOutput(flow_encoder_output_names_[0]);
    auto flow_encoder_output_info = flow_encoder_output.Info().AsContiguous();

    // Convert output to host tensor
    tcim::Tensor flow_encoder_host_tensor = tcim::Tensor::CreateHostTensor(flow_encoder_output_info);
    flow_encoder_output.CastTo(flow_encoder_host_tensor);

    // Copy to std::vector and apply repeat_interleave
    // Equivalent Python: h = h.repeat_interleave(self.token_mel_ratio, dim=1)
    // Shape after repeat: [1, 1024 * 2, 80] = [1, 2048, 80]
    const auto* h_data = static_cast<const TensorFP16*>(flow_encoder_host_tensor.Buffer().Data());
    size_t h_size = flow_encoder_host_tensor.MemSize() / sizeof(TensorFP16);

    std::vector<float> h_float(h_size);
    for (size_t i = 0; i < h_size; ++i) {
        h_float[i] = static_cast<float>(h_data[i]);
    }

    // Reshape to [1, seq, 80] and apply repeat_interleave along dim=1
    LOG_TRACE("Reshape to [1, seq, 80] and apply repeat_interleave along dim=1");
    int h_seq_len = static_cast<int>(h_size) / flow_mel_bins_;
    const int flow_seq_len = h_seq_len * kTokenMelRatio;
    const auto flow_decoder_x_shape =
        cosyvoice3_flow_decoder_model_->GetInputInfo(cosyvoice3_flow_decoder_model_->GetInputName(0)).Shape();
    if (flow_decoder_x_shape.size() < 3 || flow_decoder_x_shape[1] != flow_mel_bins_ ||
        flow_decoder_x_shape[2] != flow_seq_len) {
        throw std::runtime_error("Flow encoder output length does not match flow decoder input shape");
    }
    std::vector<float> h_repeated;
    h_repeated.reserve(static_cast<size_t>(flow_seq_len * flow_mel_bins_));
    for (int i = 0; i < h_seq_len; ++i) {
        for (int r = 0; r < kTokenMelRatio; ++r) {
            for (int j = 0; j < flow_mel_bins_; ++j) {
                h_repeated.push_back(h_float[i * flow_mel_bins_ + j]);
            }
        }
    }
    const auto t_flow_encoder_done = Clock::now();
    LOG_INFO("token2wav perf: token_embedding_and_pad={:.3f}ms, flow_encoder_run_sync={:.3f}ms, flow_encoder_output_postprocess={:.3f}ms",
             elapsed_ms(t_start, t_embedding_ready), elapsed_ms(t_flow_encoder_run_start, t_flow_encoder_run_end),
             elapsed_ms(t_flow_encoder_run_end, t_flow_encoder_done));

    // ========== STEP 3: Process speaker embedding ==========
    // Equivalent Python: embedding = F.normalize(embedding, dim=1)
    // First normalize the embedding
    LOG_DEBUG("========== STEP 3: Process speaker embedding ==========");
    float emb_norm = 0.0f;
    for (float v : embedding) {
        emb_norm += v * v;
    }
    emb_norm = std::sqrt(std::max(emb_norm, 1e-12f));

    std::vector<TensorFP16> normalized_emb;
    normalized_emb.reserve(embedding.size());
    for (float v : embedding) {
        normalized_emb.push_back(static_cast<TensorFP16>(v * 1.0f / emb_norm));
    }

    // Run flow_spk (spk_embed_affine_layer) with embedding
    // Input shape: [1, 192] (float16), Output shape: [1, 80] (float16)
    LOG_DEBUG("Run flow_spk (spk_embed_affine_layer) with embedding");
    auto spk_input_tensor_info_fp16 = cosyvoice3_flow_spk_model_->GetInputInfo(flow_spk_input_names_[0]).AsContiguous();
    auto spk_input_input_tensor = tcim::Tensor::CreateHostTensor(spk_input_tensor_info_fp16);
    CHECK_TCIM_STATUS(spk_input_input_tensor.Buffer().CopyFromHost(normalized_emb.data(), normalized_emb.size() * sizeof(TensorFP16)));
    CHECK_TCIM_STATUS(cosyvoice3_flow_spk_model_->SetInput(flow_spk_input_names_[0], spk_input_input_tensor));

    const auto t_flow_spk_run_start = Clock::now();
    CHECK_TCIM_STATUS(cosyvoice3_flow_spk_model_->Run());
    CHECK_TCIM_STATUS(cosyvoice3_flow_spk_model_->Sync());
    const auto t_flow_spk_run_end = Clock::now();


    // Get output from flow_spk
    auto flow_spk_output = cosyvoice3_flow_spk_model_->GetOutput(flow_spk_output_names_[0]);
    auto flow_spk_output_info = flow_spk_output.Info().AsContiguous();

    // Convert to host tensor
    tcim::Tensor flow_spk_host_tensor = tcim::Tensor::CreateHostTensor(flow_spk_output_info);
    flow_spk_output.CastTo(flow_spk_host_tensor);

    // Copy spk embedding to vector
    const auto* spk_data = static_cast<const TensorFP16*>(flow_spk_host_tensor.Buffer().Data());
    size_t spk_size = flow_spk_host_tensor.MemSize() / sizeof(TensorFP16);
    std::vector<float> speaker_embedding(spk_size);
    for (size_t i = 0; i < spk_size; ++i) {
        speaker_embedding[i] = static_cast<float>(spk_data[i]);
    }
    const auto t_flow_spk_done = Clock::now();

    LOG_TRACE("cosyvoice3_flow_spk_model_ out spk_size: {}", spk_size);
    LOG_INFO("token2wav perf: speaker_prepare={:.3f}ms, flow_spk_run_sync={:.3f}ms, flow_spk_output={:.3f}ms",
             elapsed_ms(t_flow_encoder_done, t_flow_spk_run_start), elapsed_ms(t_flow_spk_run_start, t_flow_spk_run_end),
             elapsed_ms(t_flow_spk_run_end, t_flow_spk_done));

    // ========== STEP 4: Prepare flow decoder inputs ==========
    // Calculate mel lengths
    // Equivalent Python: mel_len1, mel_len2 = prompt_feat.shape[1], token_len * 2 - prompt_feat.shape[1]
    // ========== STEP 4: Prepare flow decoder inputs ==========
    LOG_DEBUG("========== STEP 4: Prepare flow decoder inputs ==========");

    const int mel_len1 = static_cast<int>(prompt_feat.size()) / flow_mel_bins_;
    const int mel_len2 = concat_token_len * kTokenMelRatio - mel_len1;
    if (mel_len2 <= 0) {
        LOG_WARN("token2wav: invalid mel_len2={}, concat_token_len={}, prompt_mel_len={}", mel_len2, concat_token_len, mel_len1);
        return {};
    }
    const int total_mel_len = mel_len1 + mel_len2;
    if (total_mel_len > flow_seq_len) {
        throw std::runtime_error("Total mel length exceeds flow decoder static sequence length");
    }

    LOG_INFO("token2wav: prompt_mel_len={}, mel_len2={}, total_mel_len={}", mel_len1, mel_len2, total_mel_len);

    std::vector<float> conds(static_cast<size_t>(flow_seq_len * flow_mel_bins_), 0.0f);
    for (int idx = 0; idx < mel_len1 * flow_mel_bins_ && idx < static_cast<int>(prompt_feat.size()); ++idx) {
        conds[static_cast<size_t>(idx)] = static_cast<float>(prompt_feat[static_cast<size_t>(idx)]);
    }
    std::vector<float> conds_t = TransposeSeqFeatToFeatSeq(conds, flow_seq_len, flow_mel_bins_);
    std::vector<float> mask(static_cast<size_t>(flow_seq_len), 0.0f);
    for (int idx = 0; idx < total_mel_len; ++idx) {
        mask[static_cast<size_t>(idx)] = 1.0f;
    }
    std::vector<float> mu = TransposeSeqFeatToFeatSeq(h_repeated, flow_seq_len, flow_mel_bins_);
    const auto t_flow_inputs_ready = Clock::now();
    LOG_INFO("token2wav perf: flow_decoder_static_input_prepare={:.3f}ms",
             elapsed_ms(t_flow_spk_done, t_flow_inputs_ready));


    // ========== STEP 5: Diffusion sampling loop ==========

    // ---------- noise ----------
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(flow_mel_bins_ * flow_seq_len), 0.0f);
    for (auto& value : x) {
        value = normal_dist(rng);
    }

    LOG_DEBUG("x 随机噪声生成完毕, shape : {}x{}, size{}", flow_mel_bins_, flow_seq_len, x.size());

    const std::vector<float> t_span = BuildCosineTimeSpan(kNumSteps);
    float t = t_span[0];
    float dt = t_span[1] - t_span[0];

    std::vector<float> x_in(static_cast<size_t>(2 * flow_mel_bins_ * flow_seq_len), 0.0f);
    std::vector<float> mask_in(static_cast<size_t>(2 * flow_seq_len), 0.0f);
    std::vector<float> mu_in(static_cast<size_t>(2 * flow_mel_bins_ * flow_seq_len), 0.0f);
    std::vector<float> t_in(2, 0.0f);
    std::vector<float> cond_in(static_cast<size_t>(2 * flow_mel_bins_ * flow_seq_len), 0.0f);
    std::vector<float> spks_in(2 * flow_mel_bins_, 0.0f);

    double decoder_host_prepare_ms = 0.0;
    double decoder_tensor_convert_ms = 0.0;
    double decoder_h2d_ms = 0.0;
    double decoder_set_input_ms = 0.0;
    double decoder_run_sync_ms = 0.0;
    double decoder_d2h_ms = 0.0;
    double decoder_output_update_ms = 0.0;
    const auto t_diffusion_start = Clock::now();

    for (int step = 1; step < static_cast<int>(t_span.size()); ++step) {
        const auto t_step_start = Clock::now();
        std::copy(x.begin(), x.end(), x_in.begin());
        std::copy(x.begin(), x.end(), x_in.begin() + x.size());
        std::copy(mask.begin(), mask.end(), mask_in.begin());
        std::copy(mask.begin(), mask.end(), mask_in.begin() + mask.size());
        std::fill(mu_in.begin(), mu_in.end(), 0.0f);
        std::copy(mu.begin(), mu.end(), mu_in.begin());
        std::fill(cond_in.begin(), cond_in.end(), 0.0f);
        std::copy(conds_t.begin(), conds_t.end(), cond_in.begin());
        std::fill(spks_in.begin(), spks_in.end(), 0.0f);
        std::copy(speaker_embedding.begin(), speaker_embedding.begin() + std::min(speaker_embedding.size(), spks_in.size()),
                  spks_in.begin());
        std::fill(t_in.begin(), t_in.end(), t);
        const auto t_host_prepare_done = Clock::now();

        const auto x_tensor = TcimUtil::float_to_tensor_vec(x_in);
        const auto mask_tensor = TcimUtil::float_to_tensor_vec(mask_in);
        const auto mu_tensor = TcimUtil::float_to_tensor_vec(mu_in);
        const auto t_tensor = TcimUtil::float_to_tensor_vec(t_in);
        const auto spks_tensor = TcimUtil::float_to_tensor_vec(spks_in);
        const auto cond_tensor = TcimUtil::float_to_tensor_vec(cond_in);
        const auto t_tensor_convert_done = Clock::now();

        const auto x_name = cosyvoice3_flow_decoder_model_->GetInputName(0);
        const auto mask_name = cosyvoice3_flow_decoder_model_->GetInputName(1);
        const auto mu_name = cosyvoice3_flow_decoder_model_->GetInputName(2);
        const auto t_name = cosyvoice3_flow_decoder_model_->GetInputName(3);
        const auto spks_name = cosyvoice3_flow_decoder_model_->GetInputName(4);
        const auto cond_name = cosyvoice3_flow_decoder_model_->GetInputName(5);

        CHECK_TCIM_STATUS(
            flow_decoder_input_maps_.at(x_name).Buffer().CopyFromHost(x_tensor.data(), flow_decoder_input_maps_.at(x_name).MemSize()));
        CHECK_TCIM_STATUS(flow_decoder_input_maps_.at(mask_name).Buffer().CopyFromHost(mask_tensor.data(),
                                                                                       flow_decoder_input_maps_.at(mask_name).MemSize()));
        CHECK_TCIM_STATUS(
            flow_decoder_input_maps_.at(mu_name).Buffer().CopyFromHost(mu_tensor.data(), flow_decoder_input_maps_.at(mu_name).MemSize()));
        CHECK_TCIM_STATUS(
            flow_decoder_input_maps_.at(t_name).Buffer().CopyFromHost(t_tensor.data(), flow_decoder_input_maps_.at(t_name).MemSize()));
        CHECK_TCIM_STATUS(flow_decoder_input_maps_.at(spks_name).Buffer().CopyFromHost(spks_tensor.data(),
                                                                                       flow_decoder_input_maps_.at(spks_name).MemSize()));
        CHECK_TCIM_STATUS(flow_decoder_input_maps_.at(cond_name).Buffer().CopyFromHost(cond_tensor.data(),
                                                                                       flow_decoder_input_maps_.at(cond_name).MemSize()));
        const auto t_h2d_done = Clock::now();

        CHECK_TCIM_STATUS(cosyvoice3_flow_decoder_model_->SetInput(x_name, flow_decoder_input_maps_.at(x_name)));
        CHECK_TCIM_STATUS(cosyvoice3_flow_decoder_model_->SetInput(mask_name, flow_decoder_input_maps_.at(mask_name)));
        CHECK_TCIM_STATUS(cosyvoice3_flow_decoder_model_->SetInput(mu_name, flow_decoder_input_maps_.at(mu_name)));
        CHECK_TCIM_STATUS(cosyvoice3_flow_decoder_model_->SetInput(t_name, flow_decoder_input_maps_.at(t_name)));
        CHECK_TCIM_STATUS(cosyvoice3_flow_decoder_model_->SetInput(spks_name, flow_decoder_input_maps_.at(spks_name)));
        CHECK_TCIM_STATUS(cosyvoice3_flow_decoder_model_->SetInput(cond_name, flow_decoder_input_maps_.at(cond_name)));
        const auto t_set_input_done = Clock::now();
        CHECK_TCIM_STATUS(cosyvoice3_flow_decoder_model_->Run());
        CHECK_TCIM_STATUS(cosyvoice3_flow_decoder_model_->Sync());
        const auto t_run_sync_done = Clock::now();

        auto dphi_dt_output = cosyvoice3_flow_decoder_model_->GetDevOutput(cosyvoice3_flow_decoder_model_->GetOutputName(0)).ToHost(true);
        const auto t_d2h_done = Clock::now();
        const auto dphi_shape = cosyvoice3_flow_decoder_model_->GetOutputInfo(cosyvoice3_flow_decoder_model_->GetOutputName(0)).Shape();
        std::vector<float> dphi_dt_all = TcimUtil::tensor_to_float_vec(dphi_dt_output.Buffer().Data(), TcimUtil::numel(dphi_shape));
        const size_t batch_stride = static_cast<size_t>(flow_mel_bins_ * flow_seq_len);
        for (size_t idx = 0; idx < batch_stride; ++idx) {
            const float dphi = dphi_dt_all[idx];
            const float cfg_dphi = dphi_dt_all[idx + batch_stride];
            x[idx] += dt * ((1.0f + kInferenceCfgRate) * dphi - kInferenceCfgRate * cfg_dphi);
        }
        t += dt;
        if (step < static_cast<int>(t_span.size()) - 1) {
            dt = t_span[static_cast<size_t>(step + 1)] - t;
        }
        const auto t_step_done = Clock::now();

        const double host_prepare_ms = elapsed_ms(t_step_start, t_host_prepare_done);
        const double tensor_convert_ms = elapsed_ms(t_host_prepare_done, t_tensor_convert_done);
        const double h2d_ms = elapsed_ms(t_tensor_convert_done, t_h2d_done);
        const double set_input_ms = elapsed_ms(t_h2d_done, t_set_input_done);
        const double run_sync_ms = elapsed_ms(t_set_input_done, t_run_sync_done);
        const double d2h_ms = elapsed_ms(t_run_sync_done, t_d2h_done);
        const double output_update_ms = elapsed_ms(t_d2h_done, t_step_done);
        decoder_host_prepare_ms += host_prepare_ms;
        decoder_tensor_convert_ms += tensor_convert_ms;
        decoder_h2d_ms += h2d_ms;
        decoder_set_input_ms += set_input_ms;
        decoder_run_sync_ms += run_sync_ms;
        decoder_d2h_ms += d2h_ms;
        decoder_output_update_ms += output_update_ms;

        LOG_INFO("token2wav flow_decoder step {}/{}: total={:.3f}ms, host_prepare={:.3f}ms, tensor_convert={:.3f}ms, h2d={:.3f}ms, set_input={:.3f}ms, run_sync={:.3f}ms, d2h={:.3f}ms, output_update={:.3f}ms",
                 step, t_span.size() - 1, elapsed_ms(t_step_start, t_step_done), host_prepare_ms, tensor_convert_ms, h2d_ms,
                 set_input_ms, run_sync_ms, d2h_ms, output_update_ms);
    }
    const auto t_diffusion_end = Clock::now();
    LOG_INFO("token2wav flow_decoder summary: steps={}, total={:.3f}ms, host_prepare={:.3f}ms, tensor_convert={:.3f}ms, h2d={:.3f}ms, set_input={:.3f}ms, run_sync={:.3f}ms, d2h={:.3f}ms, output_update={:.3f}ms",
             t_span.size() - 1, elapsed_ms(t_diffusion_start, t_diffusion_end), decoder_host_prepare_ms,
             decoder_tensor_convert_ms, decoder_h2d_ms, decoder_set_input_ms, decoder_run_sync_ms, decoder_d2h_ms,
             decoder_output_update_ms);

    std::vector<float> mel_spectrogram(static_cast<size_t>(flow_mel_bins_ * mel_len2), 0.0f);
    for (int mel_idx = 0; mel_idx < flow_mel_bins_; ++mel_idx) {
        const size_t src_offset = static_cast<size_t>(mel_idx * flow_seq_len + mel_len1);
        const size_t dst_offset = static_cast<size_t>(mel_idx * mel_len2);
        std::copy(x.begin() + src_offset, x.begin() + src_offset + mel_len2, mel_spectrogram.begin() + dst_offset);
    }

    LOG_INFO("token2wav: mel_spectrogram feat size = {}", mel_spectrogram.size());
    const auto t_mel_extract_done = Clock::now();
    LOG_INFO("token2wav perf: mel_extract={:.3f}ms", elapsed_ms(t_diffusion_end, t_mel_extract_done));

    // ========== STEP 6: HiFT vocoder ==========
    int mel_bins_ = 80;
    int padded_mel_frames_ = 1024;
    int waveform_hop_ = 480;

    const int mel_len2_voc = static_cast<int>(mel_spectrogram.size() / mel_bins_);
    std::vector<float> speech_feat = mel_spectrogram;
    int mel_frames_for_model = mel_len2_voc;

    if (speed != 1.0f && mel_len2_voc > 0) {
        mel_frames_for_model = std::max(1, static_cast<int>(mel_len2_voc / speed));
        speech_feat = ResizeMelLinear(speech_feat, mel_bins_, mel_len2_voc, mel_frames_for_model);
    }

    std::vector<float> padded_speech_feat(static_cast<size_t>(mel_bins_ * padded_mel_frames_), 0.0f);
    const int copied_frames = std::min(mel_frames_for_model, padded_mel_frames_);
    for (int mel_idx = 0; mel_idx < mel_bins_; ++mel_idx) {
        const size_t src_offset = static_cast<size_t>(mel_idx * mel_frames_for_model);
        const size_t dst_offset = static_cast<size_t>(mel_idx * padded_mel_frames_);
        std::copy(speech_feat.begin() + src_offset, speech_feat.begin() + src_offset + copied_frames,
                  padded_speech_feat.begin() + dst_offset);
    }
    const auto t_hift_prepare_done = Clock::now();

    const auto hift_part1_input_name = cosyvoice3_hift_part1_model_->GetInputName(0);
    auto speech_feat_tensor = TcimUtil::float_to_tensor_vec(padded_speech_feat);
    CHECK_TCIM_STATUS(hift_part1_input_maps_.at(hift_part1_input_name).Buffer().CopyFromHost(
        speech_feat_tensor.data(), hift_part1_input_maps_.at(hift_part1_input_name).MemSize()));
    CHECK_TCIM_STATUS(cosyvoice3_hift_part1_model_->SetInput(hift_part1_input_name, hift_part1_input_maps_.at(hift_part1_input_name)));
    const auto t_hift_part1_run_start = Clock::now();
    CHECK_TCIM_STATUS(cosyvoice3_hift_part1_model_->Run());
    CHECK_TCIM_STATUS(cosyvoice3_hift_part1_model_->Sync());
    const auto t_hift_part1_run_end = Clock::now();

    auto hift_part1_output = cosyvoice3_hift_part1_model_->GetDevOutput(cosyvoice3_hift_part1_model_->GetOutputName(0)).ToHost(true);
    const auto hift_part1_output_shape = cosyvoice3_hift_part1_model_->GetOutputInfo(cosyvoice3_hift_part1_model_->GetOutputName(0)).Shape();
    std::vector<float> stft_input = TcimUtil::tensor_to_float_vec(hift_part1_output.Buffer().Data(), TcimUtil::numel(hift_part1_output_shape));
    const auto t_hift_part1_output_done = Clock::now();

    auto stft_frames = librosa::Feature::stft(stft_input, 16, 4, "hann", false, "reflect");
    const auto t_stft_done = Clock::now();
    const auto hift_part2_input_name = cosyvoice3_hift_part2_model_->GetInputName(0);
    const auto hift_part2_input_shape = cosyvoice3_hift_part2_model_->GetInputInfo(hift_part2_input_name).Shape();
    const int target_frames = static_cast<int>(hift_part2_input_shape[1]);
    const int target_bins = static_cast<int>(hift_part2_input_shape[2]);
    std::vector<float> stft_flat(static_cast<size_t>(target_frames * target_bins * 2), 0.0f);
    const int copied_stft_frames = std::min(target_frames, static_cast<int>(stft_frames.size()));
    for (int frame_idx = 0; frame_idx < copied_stft_frames; ++frame_idx) {
        const int copied_bins = std::min(target_bins, static_cast<int>(stft_frames[frame_idx].size()));
        for (int bin_idx = 0; bin_idx < copied_bins; ++bin_idx) {
            const size_t offset = static_cast<size_t>((frame_idx * target_bins + bin_idx) * 2);
            stft_flat[offset] = stft_frames[frame_idx][bin_idx].real();
            stft_flat[offset + 1] = stft_frames[frame_idx][bin_idx].imag();
        }
    }

    auto stft_tensor = TcimUtil::float_to_tensor_vec(stft_flat);
    CHECK_TCIM_STATUS(hift_part2_input_maps_.at(hift_part2_input_name).Buffer().CopyFromHost(
        stft_tensor.data(), hift_part2_input_maps_.at(hift_part2_input_name).MemSize()));
    CHECK_TCIM_STATUS(cosyvoice3_hift_part2_model_->SetInput(hift_part2_input_name, hift_part2_input_maps_.at(hift_part2_input_name)));
    const auto t_hift_part2_run_start = Clock::now();
    CHECK_TCIM_STATUS(cosyvoice3_hift_part2_model_->Run());
    CHECK_TCIM_STATUS(cosyvoice3_hift_part2_model_->Sync());
    const auto t_hift_part2_run_end = Clock::now();

    auto hift_part2_output = cosyvoice3_hift_part2_model_->GetDevOutput(cosyvoice3_hift_part2_model_->GetOutputName(0)).ToHost(true);
    const auto hift_part2_output_shape = cosyvoice3_hift_part2_model_->GetOutputInfo(cosyvoice3_hift_part2_model_->GetOutputName(0)).Shape();
    std::vector<float> utterance_audio = TcimUtil::tensor_to_float_vec(hift_part2_output.Buffer().Data(), TcimUtil::numel(hift_part2_output_shape));

    const size_t expected_samples = static_cast<size_t>(std::max(mel_len2_voc, 0) * waveform_hop_);
    if (utterance_audio.size() > expected_samples) {
        utterance_audio.resize(expected_samples);
    }
    const auto t_end = Clock::now();

    LOG_INFO("token2wav HiFT summary: prepare={:.3f}ms, part1_input={:.3f}ms, part1_run_sync={:.3f}ms, part1_output={:.3f}ms, stft={:.3f}ms, part2_input={:.3f}ms, part2_run_sync={:.3f}ms, part2_output_trim={:.3f}ms",
             elapsed_ms(t_mel_extract_done, t_hift_prepare_done), elapsed_ms(t_hift_prepare_done, t_hift_part1_run_start),
             elapsed_ms(t_hift_part1_run_start, t_hift_part1_run_end),
             elapsed_ms(t_hift_part1_run_end, t_hift_part1_output_done), elapsed_ms(t_hift_part1_output_done, t_stft_done),
             elapsed_ms(t_stft_done, t_hift_part2_run_start), elapsed_ms(t_hift_part2_run_start, t_hift_part2_run_end),
             elapsed_ms(t_hift_part2_run_end, t_end));
    LOG_INFO("token2wav completed in {:.3f} ms", elapsed_ms(t_start, t_end));
    
    return utterance_audio;
}

HLIE_NAMESPACE_END

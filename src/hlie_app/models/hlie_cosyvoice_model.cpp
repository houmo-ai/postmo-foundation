#include "hlie_app/models/hlie_cosyvoice_model.h"

#include <random>
#include <regex>

#include "ggml-cpp.h"
#include "gguf.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_audio.h"
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

    // LLM constants
    constexpr int SPEECH_TOKEN_SIZE = 6561;
    constexpr int MAX_SILENT_TOKEN_NUM = 5;
    constexpr int INPUT_SEQUENCE_LENGTH = 256;

 // RAS sampling constants
    constexpr float DEFAULT_TOP_P = 0.8f;
    constexpr int DEFAULT_TOP_K = 25;
    constexpr int DEFAULT_WIN_SIZE = 10;
    constexpr float DEFAULT_TAU_R = 0.1f;

    // Default random engine
    std::mt19937 g(std::random_device{}());

    /**
     * Compute softmax of weighted_scores.
     */
    std::vector<float> softmax(const float* weighted_scores, int size) {
        std::vector<float> exp_scores(size);
        float max_score = weighted_scores[0];
        for (int i = 1; i < size; ++i) {
            if (weighted_scores[i] > max_score) {
                max_score = weighted_scores[i];
            }
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < size; ++i) {
            exp_scores[i] = std::exp(weighted_scores[i] - max_score);
            sum_exp += exp_scores[i];
        }

        for (int i = 0; i < size; ++i) {
            exp_scores[i] /= sum_exp;
        }
        return exp_scores;
    }

    /**
     * Random sampling - sample one token from the full probability distribution.
     * Equivalent to Python: weighted_scores.softmax(dim=0).multinomial(1, replacement=True).item()
     */
    int random_sampling(const float* weighted_scores, int token_size) {
        std::vector<float> probs = softmax(weighted_scores, token_size);

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(g);

        float cum_prob = 0.0f;
        for (int i = 0; i < token_size; ++i) {
            cum_prob += probs[i];
            if (r <= cum_prob) {
                return i;
            }
        }
        return token_size - 1;  // fallback
    }

    /**
     * Nucleus sampling - sample one token from the truncated top-p and top-k candidate set.
     * Equivalent to Python's nucleus_sampling function.
     */
    int nucleus_sampling(const float* weighted_scores, float top_p = DEFAULT_TOP_P, int top_k = DEFAULT_TOP_K) {
        std::vector<float> probs = softmax(weighted_scores, SPEECH_TOKEN_SIZE);

        // Sort by probability descending, keeping track of indices
        std::vector<std::pair<float, int>> sorted;
        sorted.reserve(SPEECH_TOKEN_SIZE);
        for (int i = 0; i < SPEECH_TOKEN_SIZE; ++i) {
            sorted.emplace_back(probs[i], i);
        }

        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Select candidates based on top_p and top_k
        std::vector<float> cand_probs;
        std::vector<int> cand_indices;
        float cum_prob = 0.0f;

        for (int i = 0; i < static_cast<int>(sorted.size()); ++i) {
            if (cum_prob < top_p && static_cast<int>(cand_probs.size()) < top_k) {
                cum_prob += sorted[i].first;
                cand_probs.push_back(sorted[i].first);
                cand_indices.push_back(sorted[i].second);
            } else {
                break;
            }
        }

        // Sample from candidates using multinomial
        if (cand_probs.empty()) {
            return sorted[0].second;
        }

        // Normalize probabilities for multinomial sampling
        float total = 0.0f;
        for (float p : cand_probs) {
            total += p;
        }
        for (float& p : cand_probs) {
            p /= total;
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(g);

        float cum = 0.0f;
        for (size_t i = 0; i < cand_probs.size(); ++i) {
            cum += cand_probs[i];
            if (r <= cum) {
                return cand_indices[i];
            }
        }
        return cand_indices.back();
    }

    /**
     * RAS (Repetition-Aware Sampling) - applies repetition-aware logic on top of nucleus sampling.
     * Equivalent to Python's ras_sampling function.
     */
    int ras_sampling(const float* weighted_scores, const std::vector<int>& decoded_tokens, int sampling,
                     float top_p = DEFAULT_TOP_P, int top_k = DEFAULT_TOP_K,
                     int win_size = DEFAULT_WIN_SIZE, float tau_r = DEFAULT_TAU_R) {
        // Step 1: Get top candidate using nucleus sampling
        int top_ids = nucleus_sampling(weighted_scores, top_p, top_k);

        // Step 2: Count repetitions in the last win_size tokens
        int rep_num = 0;
        int start_idx = static_cast<int>(decoded_tokens.size()) - win_size;
        if (start_idx < 0) {
            start_idx = 0;
        }
        for (int i = start_idx; i < static_cast<int>(decoded_tokens.size()); ++i) {
            if (decoded_tokens[i] == top_ids) {
                ++rep_num;
            }
        }

        // Step 3: If repetition rate exceeds threshold, fall back to random sampling
        if (rep_num >= win_size * tau_r) {
            top_ids = random_sampling(weighted_scores, SPEECH_TOKEN_SIZE);
        }

        return top_ids;
    }


    /**
     * Simple argmax sampling - select the token with highest score.
     * For testing purposes.
     */
    int argmax_sampling(const float* weighted_scores, int token_size = SPEECH_TOKEN_SIZE) {
        int max_idx = 0;
        float max_score = weighted_scores[0];
        for (int i = 1; i < token_size; ++i) {
            if (weighted_scores[i] > max_score) {
                max_score = weighted_scores[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

} // namespace

HLIE_NAMESPACE_BEGIN


int HLIECosyVoiceModel::init() {
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

    LOG_INFO("HLIECosyVoiceModel::init 4");

    // Initialize stop token IDs: [6561, 6562, ..., 6760]
    stop_token_ids_.reserve(200);
    for (int i = 0; i < 200; ++i) {
        stop_token_ids_.push_back(SPEECH_TOKEN_SIZE + i);
    }

    // ========== Print all models' input/output information ==========
    auto print_model_info = [](const std::shared_ptr<tcim::Module>& model, const std::string& model_name) {
        LOG_INFO("╔══════════════════════════════════════════════════════════════╗");
        LOG_INFO("║ Model: {:<55} ║", model_name);
        LOG_INFO("╠══════════════════════════════════════════════════════════════╣");

        // Print inputs
        size_t num_inputs = model->GetInputNum();
        LOG_INFO("║ Inputs ({:2d}):                                                     ║", num_inputs);
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
        LOG_INFO("╠══════════════════════════════════════════════════════════════╣");
        LOG_INFO("║ Outputs ({:2d}):                                                    ║", num_outputs);
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

            LOG_INFO("║   [{:2d}] {:<20} │ shape: {:<28} │ {:<10} ║", i, output_name.substr(0, 20), shape_str.substr(0, 28), dtype_str);
        }
        LOG_INFO("╚══════════════════════════════════════════════════════════════╝");
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

        std::vector<std::string> dummy_tensor_names;

        const auto n_tensors = gguf_get_n_tensors(ctx);
        // Helper lambda to load model by name
        auto load_model_by_name = [&](const std::string& tensor_name, auto& model_ptr) -> bool {
            for (int i = 0; i < n_tensors; ++i) {
                const char* name = gguf_get_tensor_name(ctx, i);
                if (tensor_name == name) {
                    const size_t size = gguf_get_tensor_size(ctx, i);
                    const size_t offset = gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
                    if (tensor_name == cosyvoice3_llm_qwen2_decode) {
                        for (auto i : dummy_tensor_names) {
                            LOG_INFO(" dummy_tensor_names {} from {}", i, tensor_name);
                        }
                        model_ptr = create_infer_engine(model, offset, size, dummy_tensor_names);
                    } else {
                        model_ptr = create_infer_engine(model, offset, size);
                    }

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
        if (!load_model_by_name(std::string(cosyvoice3_hift_part1), cosyvoice3_hift_part1_model_))
            return -1;
        if (!load_model_by_name(std::string(cosyvoice3_hift_part2), cosyvoice3_hift_part2_model_))
            return -1;
        if (!load_model_by_name(std::string(cosyvoice3_llm_decoder), cosyvoice3_llm_decoder_model_))
            return -1;
        if (!load_model_by_name(std::string(cosyvoice3_llm_qwen2_prefill), llm_qwen2_prefill_model_))
            return -1;


        prefill_input_names_ = TcimUtil::get_module_input_names(llm_qwen2_prefill_model_);
        for (const auto& input_name : prefill_input_names_) {
            if (input_name.find("model_layers") != std::string::npos) {
                dummy_tensor_names.push_back(input_name);
            }
        }

        // Todo: Decode need init dummy and cache
        if (!load_model_by_name(std::string(cosyvoice3_llm_qwen2_decode), llm_qwen2_decode_model_))
            return -1;

        for (const auto& input_name : dummy_tensor_names) {
            auto cache = llm_qwen2_prefill_model_->GetDevInput(input_name);
            llm_qwen2_decode_model_->SetInput(input_name, cache);
        }

        if (!load_model_by_name(std::string(cosyvoice3_speech_tokenizer), cosyvoice3_speech_tokenizer_model_))
            return -1;

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

        if (!decode_input_names_.empty()) {
            // decode_input_names[3] should be the context max length
            const auto decode_input_info = llm_qwen2_decode_model_->GetInputInfo(decode_input_names_[3]);
            context_max_length_ = decode_input_info.AsContiguous().Shape()[2];

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
    LOG_INFO("Output data type: {}", static_cast<int>(output.Info().DataType()));

    // Step 4: 计算实际的 token 长度
    // feat_len = padded_input 实际非零长度，token_len = feat_len / 4
    // padded_input shape: [1, 128, 3000]，需要找到实际的 feat_len
    int64_t speech_token_len = feat_len / 4; // 实际 token 长度

    LOG_INFO("Expected speech token len: {}", speech_token_len);

    // Step 5: 复制输出到 host (直接就是 token ID)
    // 输出可能是 INT32, INT16 或其他类型
    std::vector<int> speech_tokens{490, 499, 495, 243, 273, 273, 30, 27, 273, 27, 27, 27, 27, 30, 246, 492, 501, 5092, 2904, 4856, 5956, 4948,
        2760, 3652, 3808, 6316, 6040, 4816, 5060, 3976, 6236, 5992, 3320, 2104, 2096, 4236, 2160, 1928, 1685, 1823, 3950, 5424, 1532, 3720, 1588,
        2043, 5660, 4928, 5864, 5864, 987, 492, 4228, 5656, 3712, 3692, 1477, 19, 1, 17, 404, 2180, 6044, 2368, 831, 2730, 3410, 5816, 6276, 1624,
        1385, 5032, 4600, 2168, 1086, 4600, 4432, 3400, 105, 1311, 6008, 5848, 6036, 579, 3430, 2025, 1648, 1648, 4352, 4028, 4432, 4432, 2244,
        1271, 1271, 6104, 5128, 5308, 5080, 5052, 4052, 3810, 3828, 128, 73, 73, 316, 45, 27, 27, 27, 243, 244, 252, 243, 243, 243, 243, 243, 243, 243, 244, 244, 244};
/*
    // 直接复制 INT32 token IDs
    std::vector<int32_t> buffer(output_shape[1]);
    output.Buffer().CopyToHost(buffer.data(), output.MemSize());

    // 转换为 int 并截取实际长度
    speech_tokens.resize(speech_token_len);
    for (int64_t i = 0; i < speech_token_len; ++i) {
        speech_tokens[i] = static_cast<int>(buffer[i]);
        LOG_DEBUG("speech_tokens[{}]: {}", i, speech_tokens[i]);
    }
*/
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

int HLIECosyVoiceModel::sampling_ids(const float* weighted_scores, const std::vector<int>& decoded_tokens, const int sampling,
                                     const bool ignore_eos) {
    int num_trials = 0;
    constexpr int max_trials = 100;
    int top_ids = 0;
    while (true) {
        top_ids = ras_sampling(weighted_scores, decoded_tokens, sampling);
        if (!ignore_eos || top_ids != SPEECH_TOKEN_SIZE) {
            break;
        }

        num_trials++;
        if (num_trials > max_trials) {
            LOG_ERROR("sampling reaches max_trials {} and still get eos", max_trials);
            throw std::runtime_error("sampling reaches max_trials and still get eos");
        }
    }

    return top_ids;
}

std::vector<int> HLIECosyVoiceModel::llm_inference(const std::vector<int>& text, int text_len, const std::vector<int>& prompt_text,
                                                   int prompt_text_len, const std::vector<int>& prompt_speech_token,
                                                   int prompt_speech_token_len, const std::vector<float>& llm_embedding, int sampling,
                                                   float max_token_text_ratio, float min_token_text_ratio, LLMPerf* perf) {

    using namespace std::chrono;

    auto t_start = high_resolution_clock::now();

    // ========== STEP 1: Concatenate prompt_text + text ==========
    // Equivalent Python: text = torch.concat([prompt_text, text], dim=1)
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
    // Equivalent Python: prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
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
    // Equivalent Python:
    // lm_input = torch.concat([self.sos_emb, text, self.task_id_emb, prompt_speech_token_emb], dim=1)

    // Prepare individual tensors for concatenation
    // - sos_emb: [1, 1, 896]
    // - text_emb: [1, concat_text_len, 896]
    // - task_id_emb: [1, 1, 896]
    // - prompt_speech_emb: [1, prompt_speech_token_len, 896]
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
    int min_len = static_cast<int>((text_len - prompt_text_len)*min_token_text_ratio);
    int max_len = static_cast<int>((text_len - prompt_text_len)*max_token_text_ratio);

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
            // Equivalent Python: inputs_embeds = lm_input[:, round_idx * self.prefill_length : seq_length, :]
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
        // Equivalent Python:
        // pad_input_ids = torch.zeros((self.prefill_length - effective_length), dtype=torch.long)
        // pad_input_ids.fill_(self.pad_token_id)  # 151645
        // pad_input_ids = pad_input_ids.unsqueeze(0)
        // pad_embeds = self.token_embedding(pad_input_ids)

        std::vector<tcim::Tensor> prefill_tensors;
        int effective_length = chunk_len;
        if (effective_length < prefill_length_) {
            int pad_len = prefill_length_ - effective_length;
            LOG_DEBUG("prefill  round_idx: {} pad_len: {} ", round_idx,pad_len);
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
    // auto prefill_out = llm_qwen2_prefill_model_->GetDevOutput(llm_qwen2_prefill_model_->GetOutputName(0));
    auto prefill_dev_opt = llm_qwen2_prefill_model_->GetDevOutput(llm_qwen2_prefill_model_->GetOutputName(0));
    auto llm_decoder_dev_ipt0 = prefill_dev_opt.SelectBatch({0});
    CHECK_TCIM_STATUS(cosyvoice3_llm_decoder_model_->SetInput(llm_decoder_input_names_[0], llm_decoder_dev_ipt0));

    cosyvoice3_llm_decoder_model_->Run();
    cosyvoice3_llm_decoder_model_->Sync();

    // Get log probabilities
    auto decoder_out = cosyvoice3_llm_decoder_model_->GetOutput(llm_decoder_output_names_[0]);
    tcim::Tensor decoder_out_tensor = tcim::Tensor::CreateHostTensor(decoder_out.Info().AsContiguous());
    decoder_out.CastTo(decoder_out_tensor);

    const auto* logp_fp16 = static_cast<const TensorFP16*>(decoder_out_tensor.Buffer().Data());
    std::vector<float> logp_scores(llm_output_size());
    for (int i = 0; i < llm_output_size(); ++i) {
        logp_scores[i] = static_cast<float>(logp_fp16[i]);
    }

    // ========== STEP 7: Sample first token ==========
    std::vector<int> out_tokens;
    bool ignore_eos = (min_len > 0);

    int top_ids = sampling_ids(logp_scores.data(), out_tokens, sampling, ignore_eos);
    auto t_first_token = high_resolution_clock::now();

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
    }

    out_tokens.push_back(top_ids);
    context_length += seq_length;

    // ========== STEP 8: DECODE LOOP ==========
    auto t_decode0 = high_resolution_clock::now();
    int decode_token_cnt = 0;

    const std::string& decode_input_name = decode_input_names_[0];
    const std::string& decode_valid_length_name = decode_input_names_[1];

    LOG_DEBUG("CosyVoice3 llm infer decode loop start, top_ids: {}, context_length: {}, max_len: {}", top_ids, context_length, max_len);
    for (int i = 1; i < max_len; ++i) {
        // Get embedding for current token: lm_input = self.speech_embedding.weight[top_ids]
        auto token_emb = TcimUtil::embedding_lookup(*speech_weight_embedding_, {top_ids});

        // Set decode input
        CHECK_TCIM_STATUS(llm_qwen2_decode_model_->SetInput(decode_input_name, token_emb));

        // Set valid length
        std::vector<int32_t> valid_len_vec = {context_length};
        LOG_DEBUG("CosyVoice3 llm infer decode loop[{}], top_ids: {}, context_length: {}", i, top_ids, context_length);
        const auto decode_valid_length_tensor = llm_qwen2_decode_input_map_.at(decode_valid_length_name);
        CHECK_TCIM_STATUS(decode_valid_length_tensor.Buffer().CopyFromHost(valid_len_vec.data(), valid_len_vec.size() * sizeof(int32_t)));
        CHECK_TCIM_STATUS( llm_qwen2_decode_model_->SetInput(decode_valid_length_name, decode_valid_length_tensor));

        CHECK_TCIM_STATUS(llm_qwen2_decode_model_->Run());
        CHECK_TCIM_STATUS(llm_qwen2_decode_model_->Sync());

        // ========== STEP 4: Get decode output ==========
        // Equivalent Python: decode_dev_opt = self.decode.get_dev_output(...).select_batch([0])
        auto decode_dev_opt = llm_qwen2_decode_model_->GetDevOutput(llm_qwen2_decode_model_->GetOutputName(0));
        auto llm_decoder_dev_ipt = decode_dev_opt.SelectBatch({0});

        // ========== STEP 5: Set llm_decoder input ==========
        // Equivalent Python: self.llm_decoder.set_input(..., llm_decoder_dev_ipt)
        CHECK_TCIM_STATUS(cosyvoice3_llm_decoder_model_->SetInput(llm_decoder_input_names_[0], llm_decoder_dev_ipt));

        // ========== STEP 6: Run llm_decoder ==========
        // Equivalent Python: self.llm_decoder.run()
        CHECK_TCIM_STATUS(cosyvoice3_llm_decoder_model_->Run());
        CHECK_TCIM_STATUS( cosyvoice3_llm_decoder_model_->Sync());

        // ========== STEP 7: Get logp output and squeeze ==========
        // Equivalent Python: logp = self.llm_decoder.get_dev_output(...).to_host().numpy()
        //                   logp = torch.from_numpy(logp).squeeze(dim=0)
        // auto logp_out = cosyvoice3_llm_decoder_model_->GetOutput(llm_decoder_output_names_[0]); //  logits   │ shape: 1 × 1 × 896  │ float16

        auto logits_info  = cosyvoice3_llm_decoder_model_->GetOutputInfo(llm_decoder_output_names_[0]).AsContiguous();

        tcim::Tensor logits_tensor = tcim::Tensor::CreateHostTensor(logits_info);
        cosyvoice3_llm_decoder_model_->GetOutput(llm_decoder_output_names_[0]).CastTo(logits_tensor);
        auto* p_logits = static_cast<TensorFP16*>(logits_tensor.Buffer().Data());

        // Convert to float scores
        std::vector<float> logp_scores2(llm_output_size());
        for (int j = 0; j < llm_output_size(); ++j) {
            logp_scores2[j] = static_cast<float>(p_logits[j]);
        }

        // ========== STEP 8: Sample next token ==========
        // Equivalent Python: top_ids = self.sampling_ids(logp.squeeze(dim=0), ...)
        ignore_eos = (i < min_len);
        top_ids = sampling_ids(logp_scores2.data(), out_tokens, sampling, ignore_eos);
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

    using namespace std::chrono;

    auto t_start = high_resolution_clock::now();

    // Flow decoder constants (matching Python implementation)
    constexpr int kFlowInputSize = 80;
    constexpr int kFlowOutputSize = 80;
    constexpr int kVocabSize = 6561;
    constexpr int kMaxSeqLen = 2048;
    constexpr int kTokenMelRatio = 2;
    constexpr float kInferenceCfgRate = 0.7f;
    constexpr int kNumSteps = 10;  // Number of diffusion steps

    // ========== STEP 1: Concatenate prompt_token + token ==========
    // Equivalent Python: token = torch.concat([prompt_token, token], dim=1)
    std::vector<int> concat_token = prompt_token;
    concat_token.insert(concat_token.end(), token.begin(), token.end());
    int concat_token_len = static_cast<int>(concat_token.size());

    LOG_INFO("token2wav: concat_token_len={}, prompt_token_len={}, token_len={}", concat_token_len, prompt_token.size(), token.size());

    // ========== STEP 2: Token embedding lookup ==========
    // Equivalent Python: token = self.input_embedding(token)
    auto token_emb_tensor = TcimUtil::embedding_lookup(*flow_input_embedding_, concat_token);
    auto token_emb_info = token_emb_tensor.Info().AsContiguous();
    auto token_emb_shape = token_emb_info.Shape();

    // Pad to max sequence length (1024)
    // Equivalent Python: token = F.pad(token, (0, 0, 0, 1024 - token.shape[1]), value=0)
    constexpr int kPadLen = 1024;
    int pad_len = kPadLen - static_cast<int>(token_emb_shape[1]);
    if (pad_len < 0) {
        LOG_WARN("token sequence length {} exceeds padding length {}, truncating", token_emb_shape[1], kPadLen);
        pad_len = 0;
    }

    // Prepare padded token embedding for flow_encoder
    // Shape: [1, seq_len, 80] -> [1, 1024, 80]
    LOG_DEBUG("Prepare padded token embedding for flow_encoder");
    std::vector<float> padded_token_emb;
    padded_token_emb.reserve(kPadLen * kFlowInputSize);

    const auto* token_emb_data = static_cast<const TensorFP16*>(token_emb_tensor.Buffer().Data());
    size_t token_emb_size = token_emb_tensor.MemSize() / sizeof(TensorFP16);

    for (size_t i = 0; i < token_emb_size; ++i) {
        padded_token_emb.push_back(static_cast<float>(token_emb_data[i]));
    }

    // Pad with zeros
    for (int i = 0; i < pad_len * kFlowInputSize; ++i) {
        padded_token_emb.push_back(0.0f);
    }

    // Set input to flow_encoder and run
    // Input shape: [1, 1024, 80] (float16)
    LOG_DEBUG("Set input to flow_encoder and run");
    CHECK_TCIM_STATUS(TcimUtil::set_module_input(cosyvoice3_flow_encoder_model_, flow_encoder_input_names_[0], padded_token_emb.data()));

    CHECK_TCIM_STATUS(cosyvoice3_flow_encoder_model_->Run());
    CHECK_TCIM_STATUS(cosyvoice3_flow_encoder_model_->Sync());

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
    LOG_DEBUG("Reshape to [1, seq, 80] and apply repeat_interleave along dim=1");
    int h_seq_len = static_cast<int>(h_size) / kFlowInputSize;
    std::vector<float> h_repeated;
    h_repeated.reserve(h_seq_len * kTokenMelRatio * kFlowInputSize);
    for (int i = 0; i < h_seq_len; ++i) {
        for (int r = 0; r < kTokenMelRatio; ++r) {
            for (int j = 0; j < kFlowInputSize; ++j) {
                h_repeated.push_back(h_float[i * kFlowInputSize + j]);
            }
        }
    }

    // ========== STEP 3: Process speaker embedding ==========
    // Equivalent Python: embedding = F.normalize(embedding, dim=1)
    // First normalize the embedding
    LOG_DEBUG("========== STEP 3: Process speaker embedding ==========");
    float emb_norm = 0.0f;
    for (float v : embedding) {
        emb_norm += v * v;
    }
    emb_norm = std::sqrt(emb_norm);
    std::vector<float> normalized_emb;
    normalized_emb.reserve(embedding.size());
    for (float v : embedding) {
        normalized_emb.push_back(v / emb_norm);
    }

    // Run flow_spk (spk_embed_affine_layer) with embedding
    // Input shape: [1, 192] (float16), Output shape: [1, 80] (float16)
    LOG_DEBUG("Run flow_spk (spk_embed_affine_layer) with embedding");
    CHECK_TCIM_STATUS(TcimUtil::set_module_input(cosyvoice3_flow_spk_model_, flow_spk_input_names_[0], normalized_emb.data()));

    CHECK_TCIM_STATUS(cosyvoice3_flow_spk_model_->Run());
    CHECK_TCIM_STATUS(cosyvoice3_flow_spk_model_->Sync());

    // Get output from flow_spk
    auto flow_spk_output = cosyvoice3_flow_spk_model_->GetOutput(flow_spk_output_names_[0]);
    auto flow_spk_output_info = flow_spk_output.Info().AsContiguous();

    // Convert to host tensor
    tcim::Tensor flow_spk_host_tensor = tcim::Tensor::CreateHostTensor(flow_spk_output_info);
    flow_spk_output.CastTo(flow_spk_host_tensor);

    // Copy spk embedding to vector
    const auto* spk_data = static_cast<const TensorFP16*>(flow_spk_host_tensor.Buffer().Data());
    size_t spk_size = flow_spk_host_tensor.MemSize() / sizeof(TensorFP16);
    std::vector<float> spk_emb_float(spk_size);
    for (size_t i = 0; i < spk_size; ++i) {
        spk_emb_float[i] = static_cast<float>(spk_data[i]);
    }

    // ========== STEP 4: Prepare flow decoder inputs ==========
    // Calculate mel lengths
    // Equivalent Python: mel_len1, mel_len2 = prompt_feat.shape[1], token_len * 2 - prompt_feat.shape[1]
    LOG_DEBUG("========== STEP 4: Prepare flow decoder inputs ==========");
    int prompt_mel_len = static_cast<int>(prompt_feat.size()) / kFlowInputSize;
    int mel_len2 = concat_token_len * kTokenMelRatio - prompt_mel_len;
    int total_mel_len = prompt_mel_len + mel_len2;

    LOG_INFO("token2wav: prompt_mel_len={}, mel_len2={}, total_mel_len={}", prompt_mel_len, mel_len2, total_mel_len);

    // Prepare conditioning tensor (conds)
    // Shape: [1, 2048, 80], filled with zeros except prompt part
    // Equivalent Python: conds = torch.zeros([1, 2048, 80])
    //                   conds[:, :mel_len1] = prompt_feat
    std::vector<float> conds(kMaxSeqLen * kFlowInputSize, 0.0f);
    for (int i = 0; i < prompt_mel_len && i < kMaxSeqLen; ++i) {
        for (int j = 0; j < kFlowInputSize; ++j) {
            int prompt_idx = i * kFlowInputSize + j;
            int cond_idx = i * kFlowInputSize + j;
            if (prompt_idx < static_cast<int>(prompt_feat.size())) {
                conds[cond_idx] = prompt_feat[prompt_idx];
            }
        }
    }

    // Prepare mask
    // Equivalent Python: mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]), 2048)).to(h)
    // Shape: [1, 1, 2048]
    std::vector<float> mask(kMaxSeqLen, 1.0f);
    for (int i = total_mel_len; i < kMaxSeqLen; ++i) {
        mask[i] = 0.0f;
    }

    // Prepare mu (mean) tensor from h_repeated
    // Shape: [1, 80, 2048]
    std::vector<float> mu(kMaxSeqLen * kFlowInputSize, 0.0f);
    int mu_seq_len = std::min(static_cast<int>(h_repeated.size()) / kFlowInputSize, kMaxSeqLen);
    for (int i = 0; i < mu_seq_len; ++i) {
        for (int j = 0; j < kFlowInputSize; ++j) {
            mu[i * kFlowInputSize + j] = h_repeated[i * kFlowInputSize + j];
        }
    }

    // Prepare random noise for diffusion
    // Equivalent Python: rand_noise = torch.randn([1, 80, 50 * 300])
    //                    x = rand_noise[:, :, : mu.size(2)] * 1.0
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x(kMaxSeqLen * kFlowInputSize, 0.0f);
    for (int i = 0; i < mu_seq_len; ++i) {
        for (int j = 0; j < kFlowInputSize; ++j) {
            x[i * kFlowInputSize + j] = dist(rng);
        }
    }

    // Prepare time span for diffusion schedule
    // Equivalent Python: t_span = torch.linspace(0, 1, 10 + 1)
    //                    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    std::vector<float> t_span(kNumSteps + 1);
    for (int i = 0; i <= kNumSteps; ++i) {
        float t = static_cast<float>(i) / kNumSteps;
        t_span[i] = 1.0f - std::cos(t * M_PI * 0.5f);
    }
    float dt = t_span[1] - t_span[0];
    float t = t_span[0];

    // ========== STEP 5: Diffusion sampling loop ==========
    // Equivalent Python:
    //   for step in range(1, len(t_span)):
    //       # Classifier-Free Guidance
    //       x_in[:] = x
    //       mask_in[:] = mask
    //       mu_in[0] = mu
    //       t_in[:] = t
    //       spks_in[0] = embedding
    //       cond_in[0] = conds
    //       # Run flow_decoder and apply CFG
    //       x = x + dt * dphi_dt

    // Prepare CFG input tensors (batch size 2)
    // Shape: [2, 80, 2048]
    std::vector<float> x_in(kMaxSeqLen * kFlowInputSize * 2, 0.0f);
    std::vector<float> mask_in(kMaxSeqLen * 2, 0.0f);
    std::vector<float> mu_in(kMaxSeqLen * kFlowInputSize * 2, 0.0f);
    std::vector<float> t_in(2, 0.0f);
    std::vector<float> spks_in(2 * kFlowInputSize, 0.0f);
    std::vector<float> cond_in(kMaxSeqLen * kFlowInputSize * 2, 0.0f);

    std::vector<float> sol;  // Store intermediate results

    for (int step = 1; step < kNumSteps; ++step) {
        // Prepare CFG inputs
        // Copy x to both slots of x_in
        for (int i = 0; i < mu_seq_len * kFlowInputSize; ++i) {
            x_in[i] = x[i];
            x_in[mu_seq_len * kFlowInputSize + i] = x[i];
        }

        // Copy mask to both slots
        for (int i = 0; i < kMaxSeqLen; ++i) {
            mask_in[i] = mask[i];
            mask_in[kMaxSeqLen + i] = mask[i];
        }

        // Copy mu to first slot only (CFG: unconditional)
        for (int i = 0; i < mu_seq_len * kFlowInputSize; ++i) {
            mu_in[i] = mu[i];
        }

        // Set time step for both
        t_in[0] = t;
        t_in[1] = t;

        // Set speaker embedding for first slot (CFG: unconditional)
        for (size_t i = 0; i < spk_emb_float.size(); ++i) {
            spks_in[i] = spk_emb_float[i];
        }

        // Copy conds to first slot only
        for (int i = 0; i < kMaxSeqLen * kFlowInputSize; ++i) {
            cond_in[i] = conds[i];
        }

        // Set inputs to flow_decoder
        CHECK_TCIM_STATUS(TcimUtil::set_module_input(cosyvoice3_flow_decoder_model_, flow_input_names_[0], x_in.data()));
        CHECK_TCIM_STATUS(TcimUtil::set_module_input(cosyvoice3_flow_decoder_model_, flow_input_names_[1], mask_in.data()));
        CHECK_TCIM_STATUS(TcimUtil::set_module_input(cosyvoice3_flow_decoder_model_, flow_input_names_[2], mu_in.data()));
        CHECK_TCIM_STATUS(TcimUtil::set_module_input(cosyvoice3_flow_decoder_model_, flow_input_names_[3], t_in.data()));
        CHECK_TCIM_STATUS(TcimUtil::set_module_input(cosyvoice3_flow_decoder_model_, flow_input_names_[4], spks_in.data()));
        CHECK_TCIM_STATUS(TcimUtil::set_module_input(cosyvoice3_flow_decoder_model_, flow_input_names_[5], cond_in.data()));

        // Run flow_decoder
        CHECK_TCIM_STATUS(cosyvoice3_flow_decoder_model_->Run());
        CHECK_TCIM_STATUS(cosyvoice3_flow_decoder_model_->Sync());

        // Get output
        auto flow_decoder_output = cosyvoice3_flow_decoder_model_->GetOutput(flow_output_names_[0]);
        auto flow_decoder_output_info = flow_decoder_output.Info().AsContiguous();

        // Convert to host tensor
        tcim::Tensor flow_decoder_host_tensor = tcim::Tensor::CreateHostTensor(flow_decoder_output_info);
        flow_decoder_output.CastTo(flow_decoder_host_tensor);

        // Copy output and apply CFG
        const auto* dphi_dt_data = static_cast<const TensorFP16*>(flow_decoder_host_tensor.Buffer().Data());
        size_t dphi_dt_size = flow_decoder_host_tensor.MemSize() / sizeof(TensorFP16);

        // Split into unconditional and conditional predictions
        // Output shape: [2, 80, 2048]
        // First half is unconditional, second half is conditional
        int out_size = static_cast<int>(dphi_dt_size) / 2;

        // Apply CFG: (1.0 + cfg_rate) * unconditional - cfg_rate * conditional
        for (int i = 0; i < out_size; ++i) {
            float unconditional = static_cast<float>(dphi_dt_data[i]);
            float conditional = static_cast<float>(dphi_dt_data[out_size + i]);
            float dphi = (1.0f + kInferenceCfgRate) * unconditional - kInferenceCfgRate * conditional;
            x[i] += dt * dphi;
        }

        // Update time
        t = t_span[step];
        if (step < kNumSteps - 1) {
            dt = t_span[step + 1] - t;
        }

        // Store intermediate result
        if (step == kNumSteps - 1) {
            sol = x;
        }
    }

    // Extract the final feature
    // Equivalent Python: feat = sol[-1].float()
    //                    feat = feat[:, :, mel_len1 : mel_len1 + mel_len2]
    std::vector<float> feat;
    feat.reserve(mel_len2 * kFlowInputSize);
    for (int i = prompt_mel_len; i < prompt_mel_len + mel_len2; ++i) {
        for (int j = 0; j < kFlowInputSize; ++j) {
            int idx = i * kFlowInputSize + j;
            if (idx < static_cast<int>(sol.size())) {
                feat.push_back(sol[idx]);
            }
        }
    }

    LOG_INFO("token2wav: feat size = {}", feat.size());

    // ========== STEP 6: HiFT vocoder ==========
    // Prepare mel spectrogram for hift_part1
    // Shape: [1, 80, 1024]
    constexpr int kHiftSeqLen = 1024;
    std::vector<float> speech_feat(kHiftSeqLen * kFlowInputSize, 0.0f);
    int feat_len = static_cast<int>(feat.size()) / kFlowInputSize;
    for (int i = 0; i < feat_len && i < kHiftSeqLen; ++i) {
        for (int j = 0; j < kFlowInputSize; ++j) {
            speech_feat[i * kFlowInputSize + j] = feat[i * kFlowInputSize + j];
        }
    }

    // Run hift_part1
    // Input shape: [1, 80, 1024], Output shape: [1, 491536]
    LOG_DEBUG("Run hift_part1");
    CHECK_TCIM_STATUS(TcimUtil::set_module_input(cosyvoice3_hift_part1_model_, hift_part1_input_names_[0], speech_feat.data()));

    CHECK_TCIM_STATUS(cosyvoice3_hift_part1_model_->Run());
    CHECK_TCIM_STATUS(cosyvoice3_hift_part1_model_->Sync());

    // Get output from hift_part1
    auto hift_part1_output = cosyvoice3_hift_part1_model_->GetOutput(hift_part1_output_names_[0]);
    auto hift_part1_output_info = hift_part1_output.Info().AsContiguous();

    // Convert to host tensor
    tcim::Tensor hift_part1_host_tensor = tcim::Tensor::CreateHostTensor(hift_part1_output_info);
    hift_part1_output.CastTo(hift_part1_host_tensor);

    // Prepare STFT input for hift_part2
    // Shape: [1, 122881, 9, 2]
    // This is a complex STFT representation
    const auto* hift_part1_data = static_cast<const TensorFP16*>(hift_part1_host_tensor.Buffer().Data());
    size_t hift_part1_size = hift_part1_host_tensor.MemSize() / sizeof(TensorFP16);

    std::vector<float> hift_part1_float(hift_part1_size);
    for (size_t i = 0; i < hift_part1_size; ++i) {
        hift_part1_float[i] = static_cast<float>(hift_part1_data[i]);
    }

    // ========== Compute STFT for hift_part2 input ==========
    // Python equivalent:
    //   stft_opt = torch.stft(stft_ipt, 16, 4, 16, window=hann_window,
    //                         onesided=True, center=False, return_complex=False)
    //   # Output shape: [1, 122881, 9, 2]

    constexpr int kNfft = 16;
    constexpr int kHopLength = 4;

    LOG_DEBUG("Compute STFT for hift_part2 input");

    // Create STFT cache (sin/cos tables for 16-point FFT)
    hlie_audio_cache stft_cache;
    stft_cache.fill_sin_cos_table(kNfft);

    // Create 16-point Hann window
    std::vector<float> hann_window(kNfft);
    for (int i = 0; i < kNfft; i++) {
        hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / kNfft));
    }

    // Calculate output dimensions
    int input_len = static_cast<int>(hift_part1_size);  // 491536
    int n_frames = (input_len - kNfft) / kHopLength + 1;  // 122881
    int n_freq_bins = kNfft / 2 + 1;  // 9
    size_t stft_output_size = (size_t)n_frames * n_freq_bins * 2;  // 2211858

    // Prepare output buffer
    std::vector<float> stft_output(stft_output_size);

    // Call STFT
    size_t written = STFTUtil::hlie_audio_stft(
        stft_cache,                    // cache with sin/cos tables
        hift_part1_float.data(),       // input signal [491536]
        input_len,                     // 491536
        kNfft,                         // 16
        kHopLength,                    // 4
        hann_window.data(),            // 16-point Hann window
        kNfft,                         // window_len = 16
        stft_output.data(),            // output buffer
        stft_output.size()             // buffer capacity
    );

    if (written == 0) {
        LOG_ERROR("STFT computation failed");
        return {};
    }


    // Run hift_part2
    // Inputs: [1, 122881, 9, 2] and [1, 80, 1024]
    // Output: [1, 491520]
    LOG_DEBUG("Run hift_part2, stft_output.size: {}", stft_output.size());
    //  Todo: input shape 不对，hift_part1_float 的shape还是 419536
    CHECK_TCIM_STATUS(TcimUtil::set_module_input(cosyvoice3_hift_part2_model_, hift_part2_input_names_[0], stft_output.data()));
    CHECK_TCIM_STATUS(TcimUtil::set_module_input(cosyvoice3_hift_part2_model_, hift_part2_input_names_[1], speech_feat.data()));

    CHECK_TCIM_STATUS(cosyvoice3_hift_part2_model_->Run());
    CHECK_TCIM_STATUS(cosyvoice3_hift_part2_model_->Sync());

    // Get output from hift_part2
    auto hift_part2_output = cosyvoice3_hift_part2_model_->GetOutput(hift_part2_output_names_[0]);
    auto hift_part2_output_info = hift_part2_output.Info().AsContiguous();

    // Convert to host tensor
    tcim::Tensor hift_part2_host_tensor = tcim::Tensor::CreateHostTensor(hift_part2_output_info);
    hift_part2_output.CastTo(hift_part2_host_tensor);

    // Get final waveform
    LOG_DEBUG("Get final waveform");
    const auto* waveform_data = static_cast<const TensorFP16*>(hift_part2_host_tensor.Buffer().Data());
    size_t waveform_size = hift_part2_host_tensor.MemSize() / sizeof(TensorFP16);

    std::vector<float> waveform(waveform_size);
    for (size_t i = 0; i < waveform_size; ++i) {
        waveform[i] = static_cast<float>(waveform_data[i]);
    }

    // Truncate to expected length
    // Equivalent Python: tts_speech = tts_speech[:, : 480 * mel_len2]
    int expected_len = 480 * mel_len2;
    if (waveform.size() > static_cast<size_t>(expected_len)) {
        waveform.resize(expected_len);
    }

    auto t_end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t_end - t_start);
    LOG_INFO("token2wav completed in {} ms", duration.count());

    return waveform;
}

HLIE_NAMESPACE_END

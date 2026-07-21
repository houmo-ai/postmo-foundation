#include "hlie_app/models/hlie_qwen3_tts_model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string_view>
#include <unordered_set>

#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_tcim_util.h"
#include "hlie_base/hlie_tokenizer.h"
#include "nlohmann/json.hpp"

namespace {
constexpr std::string_view kTextProjection = "qwen3_tts.text_projection.hmm";
constexpr std::string_view kTalkerPrefill = "qwen3_tts.talker.prefill.hmm";
constexpr std::string_view kTalkerDecode = "qwen3_tts.talker.decode.hmm";
constexpr std::string_view kCodePredictorPrefill = "qwen3_tts.code_predictor.prefill.hmm";
constexpr std::string_view kCodePredictorDecode = "qwen3_tts.code_predictor.decode.hmm";
constexpr std::string_view kStatefulDecoder = "qwen3_tts.stateful_decoder.hmm";
constexpr std::string_view kTalkerTokenEmbedding = "quant_embedding.bin";
constexpr std::string_view kTalkerTextEmbedding = "text_embedding.bin";
constexpr std::string_view kCodePredictorEmbedding = "quant_embedding_code_predictor.bin";

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](const unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

tcim::Tensor slice_seq(const tcim::Tensor& input, const int start, const int length) {
    const auto& shape = input.Info().Shape();
    if (shape.size() != 3 || start < 0 || length < 0 || start + length > shape[1]) {
        throw std::invalid_argument("invalid sequence slice");
    }
    const int64_t hidden = shape[2];
    const auto info = tcim::TensorInfo::CreateNDInfo(
        {shape[0], length, hidden}, input.Info().DataType());
    auto output = tcim::Tensor::CreateHostTensor(info);
    const size_t row_bytes = static_cast<size_t>(hidden) * input.Info().DataTypeSize();
    std::memcpy(output.Data(), static_cast<const uint8_t*>(input.Data()) +
                               static_cast<size_t>(start) * row_bytes,
                static_cast<size_t>(length) * row_bytes);
    return output;
}

tcim::Tensor add_fp16(const tcim::Tensor& lhs, const tcim::Tensor& rhs) {
    if (lhs.Info().Shape() != rhs.Info().Shape() ||
        lhs.Info().DataType() != tcim::DataType::FLOAT16 ||
        rhs.Info().DataType() != tcim::DataType::FLOAT16) {
        throw std::invalid_argument("Qwen3-TTS embedding add shape/type mismatch");
    }
    auto output = tcim::Tensor::CreateHostTensor(lhs.Info().AsContiguous());
    size_t count = 1;
    for (const int64_t dimension : lhs.Info().Shape()) {
        count *= static_cast<size_t>(dimension);
    }
    const auto* left = static_cast<const TensorFP16*>(lhs.Data());
    const auto* right = static_cast<const TensorFP16*>(rhs.Data());
    auto* out = static_cast<TensorFP16*>(output.Data());
    for (size_t i = 0; i < count; ++i) {
        out[i] = left[i] + right[i];
    }
    return output;
}

int64_t tensor_numel(const std::vector<int64_t>& shape) {
    int64_t total = 1;
    for (const int64_t dimension : shape) {
        total *= dimension;
    }
    return total;
}

std::vector<float> fp16_values(const tcim::Tensor& tensor) {
    const size_t count = static_cast<size_t>(tensor_numel(tensor.Info().Shape()));
    const auto* source = static_cast<const TensorFP16*>(tensor.Data());
    std::vector<float> values(count);
    std::transform(source, source + count, values.begin(),
                   [](const TensorFP16 value) { return static_cast<float>(value); });
    return values;
}

int32_t select_top_k_token(std::vector<float>& values, const size_t offset,
                           const int64_t vocab, const bool do_sample,
                           std::mt19937& generator) {
    if (vocab <= 0 || offset > values.size() ||
        static_cast<uint64_t>(vocab) > values.size() - offset) {
        throw std::invalid_argument("invalid logits vocabulary range");
    }
    const auto begin = values.begin() + static_cast<std::ptrdiff_t>(offset);
    if (!do_sample) {
        return static_cast<int32_t>(
            std::distance(begin, std::max_element(begin, begin + vocab)));
    }

    constexpr size_t kTopK = 50;
    constexpr float kTemperature = 0.9F;
    std::vector<int32_t> candidates(static_cast<size_t>(vocab));
    std::iota(candidates.begin(), candidates.end(), 0);
    const size_t candidate_count = std::min(kTopK, candidates.size());
    std::partial_sort(
        candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(candidate_count),
        candidates.end(), [&](const int32_t lhs, const int32_t rhs) {
            return values[offset + static_cast<size_t>(lhs)] >
                   values[offset + static_cast<size_t>(rhs)];
        });
    candidates.resize(candidate_count);
    const float max_score = values[offset + static_cast<size_t>(candidates.front())];
    std::vector<double> weights;
    weights.reserve(candidate_count);
    for (const int32_t token : candidates) {
        const float score = values[offset + static_cast<size_t>(token)];
        weights.push_back(std::isfinite(score)
                              ? std::exp(static_cast<double>((score - max_score) / kTemperature))
                              : 0.0);
    }
    if (std::none_of(weights.begin(), weights.end(),
                     [](const double value) { return value > 0.0; })) {
        return candidates.front();
    }
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return candidates[distribution(generator)];
}

int32_t select_talker_logits(const tcim::Tensor& logits,
                             const std::vector<int32_t>& generated_tokens,
                             const int32_t eos_token_id, const bool do_sample,
                             std::mt19937& generator) {
    const auto& shape = logits.Info().Shape();
    if (shape.empty() || logits.Info().DataType() != tcim::DataType::FLOAT16) {
        throw std::invalid_argument("expected FLOAT16 Talker logits");
    }
    const int64_t vocab = shape.back();
    const int64_t total = tensor_numel(shape);
    auto values = fp16_values(logits);
    const size_t offset = static_cast<size_t>(total - vocab);

    // Match demo.py: repetition penalty, minimum EOS length, reserved-token
    // suppression, temperature=0.9 and top-k=50.
    const std::unordered_set<int32_t> unique_generated(
        generated_tokens.begin(), generated_tokens.end());
    for (const int32_t token : unique_generated) {
        if (token >= 0 && token < vocab) {
            float& score = values[offset + static_cast<size_t>(token)];
            score = score < 0.0F ? score * 1.05F : score / 1.05F;
        }
    }
    if (generated_tokens.size() < 2 && eos_token_id >= 0 && eos_token_id < vocab) {
        values[offset + static_cast<size_t>(eos_token_id)] = -std::numeric_limits<float>::infinity();
    }
    for (int64_t token = std::max<int64_t>(0, vocab - 1024); token < vocab; ++token) {
        if (token != eos_token_id) {
            values[offset + static_cast<size_t>(token)] = -std::numeric_limits<float>::infinity();
        }
    }
    return select_top_k_token(values, offset, vocab, do_sample, generator);
}

int32_t select_subtalker_logits(const tcim::Tensor& logits, const bool do_sample,
                                std::mt19937& generator) {
    const auto& shape = logits.Info().Shape();
    if (shape.empty() || logits.Info().DataType() != tcim::DataType::FLOAT16) {
        throw std::invalid_argument("expected FLOAT16 CodePredictor logits");
    }
    const int64_t vocab = shape.back();
    const int64_t total = tensor_numel(shape);
    auto values = fp16_values(logits);
    const size_t offset = static_cast<size_t>(total - vocab);
    return select_top_k_token(values, offset, vocab, do_sample, generator);
}

tcim::Tensor sum_fp16_vectors(const std::vector<tcim::Tensor>& inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument("cannot sum an empty tensor list");
    }
    const auto& shape = inputs.front().Info().Shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != 1 ||
        inputs.front().Info().DataType() != tcim::DataType::FLOAT16) {
        throw std::invalid_argument("expected [1,1,H] FLOAT16 embeddings");
    }
    auto output = tcim::Tensor::CreateHostTensor(inputs.front().Info().AsContiguous());
    auto* out = static_cast<TensorFP16*>(output.Data());
    for (int64_t hidden = 0; hidden < shape[2]; ++hidden) {
        float sum = 0.0F;
        for (const auto& input : inputs) {
            if (input.Info().Shape() != shape) {
                throw std::invalid_argument("codec embedding shape mismatch");
            }
            sum += static_cast<float>(static_cast<const TensorFP16*>(input.Data())[hidden]);
        }
        out[hidden] = TensorFP16(sum);
    }
    return output;
}
}

HLIE_NAMESPACE_BEGIN

bool HLIEQwen3TTSModel::load_metadata(std::string& error) {
    std::string model_type;
    if (!bundle_.get_string("model_type", model_type, true, &error) || model_type != "qwen3_tts") {
        if (error.empty()) {
            error = "GGUF model_type is not qwen3_tts";
        }
        return false;
    }

    std::string language_json;
    std::string speaker_json;
    std::string dialect_json;
    const bool loaded = bundle_.get_string("qwen3_tts.model_size", config_.model_size, true, &error) &&
           bundle_.get_int32("qwen3_tts.sample_rate", config_.sample_rate, true, &error) &&
           bundle_.get_int32("qwen3_tts.num_code_groups", config_.num_code_groups, true, &error) &&
           bundle_.get_int32("qwen3_tts.talker.hidden_size", config_.talker_hidden_size, true, &error) &&
           bundle_.get_int32("qwen3_tts.text.hidden_size", config_.text_hidden_size, true, &error) &&
           bundle_.get_int32("qwen3_tts.text.vocab_size", config_.text_vocab_size, true, &error) &&
           bundle_.get_int32("qwen3_tts.tokenizer.vocab_size", config_.tokenizer_vocab_size, true, &error) &&
           bundle_.get_int32("qwen3_tts.codec.vocab_size", config_.codec_vocab_size, true, &error) &&
           bundle_.get_int32("qwen3_tts.code_predictor.vocab_size", config_.code_predictor_vocab_size, true, &error) &&
           bundle_.get_int32("qwen3_tts.tts_bos_token_id", config_.tts_bos_token_id, true, &error) &&
           bundle_.get_int32("qwen3_tts.tts_eos_token_id", config_.tts_eos_token_id, true, &error) &&
           bundle_.get_int32("qwen3_tts.tts_pad_token_id", config_.tts_pad_token_id, true, &error) &&
           bundle_.get_int32("qwen3_tts.codec_bos_id", config_.codec_bos_id, true, &error) &&
           bundle_.get_int32("qwen3_tts.codec_eos_token_id", config_.codec_eos_token_id, true, &error) &&
           bundle_.get_int32("qwen3_tts.codec_pad_id", config_.codec_pad_id, true, &error) &&
           bundle_.get_int32("qwen3_tts.codec_think_id", config_.codec_think_id, true, &error) &&
           bundle_.get_int32("qwen3_tts.codec_nothink_id", config_.codec_nothink_id, true, &error) &&
           bundle_.get_int32("qwen3_tts.codec_think_bos_id", config_.codec_think_bos_id, true, &error) &&
           bundle_.get_int32("qwen3_tts.codec_think_eos_id", config_.codec_think_eos_id, true, &error) &&
           bundle_.get_string("qwen3_tts.codec_language_id", language_json, true, &error) &&
           bundle_.get_string("qwen3_tts.spk_id", speaker_json, true, &error) &&
           bundle_.get_string("qwen3_tts.spk_is_dialect", dialect_json, true, &error);
    if (!loaded) {
        return false;
    }

    int32_t optional_value = 0;
    if (bundle_.get_int32("qwen3_tts.stateful.samples_per_frame", optional_value,
                          false, nullptr)) {
        config_.stateful_samples_per_frame = optional_value;
    }
    if (bundle_.get_int32("qwen3_tts.stateful.initial_output_skip_frames",
                          optional_value, false, nullptr)) {
        config_.stateful_initial_output_skip_frames = optional_value;
    }
    if (bundle_.get_int32("qwen3_tts.sampling.code_predictor", optional_value,
                          false, nullptr)) {
        config_.code_predictor_sampling = optional_value != 0;
    }

    try {
        // Keep the parsed JSON values alive while iterating over items(). The
        // items proxy contains a reference to its JSON object.
        const auto language_map = nlohmann::json::parse(language_json);
        const auto speaker_map = nlohmann::json::parse(speaker_json);
        const auto dialect_map = nlohmann::json::parse(dialect_json);
        if (!language_map.is_object() || !speaker_map.is_object() || !dialect_map.is_object()) {
            error = "Qwen3-TTS language/speaker metadata must be JSON objects";
            return false;
        }
        for (const auto& [name, value] : language_map.items()) {
            if (value.is_number_integer()) {
                config_.codec_language_ids[lower_ascii(name)] = value.get<int32_t>();
            } else {
                LOG_WARN("Ignoring Qwen3-TTS language without an integer ID: {}", name);
            }
        }
        for (const auto& [name, value] : speaker_map.items()) {
            if (value.is_number_integer()) {
                config_.speaker_ids[lower_ascii(name)] = value.get<int32_t>();
            } else {
                LOG_WARN("Ignoring Qwen3-TTS speaker without an integer ID: {}", name);
            }
        }
        for (const auto& [name, value] : dialect_map.items()) {
            if (value.is_string() && !value.get<std::string>().empty()) {
                config_.speaker_dialects[lower_ascii(name)] = lower_ascii(value.get<std::string>());
            }
        }
    } catch (const nlohmann::json::exception& ex) {
        error = std::string("invalid Qwen3-TTS language/speaker metadata: ") + ex.what();
        return false;
    }
    return !config_.codec_language_ids.empty() && !config_.speaker_ids.empty();
}

bool HLIEQwen3TTSModel::load_module(const std::string& tensor_name,
                                    std::shared_ptr<tcim::Module>& module,
                                    const std::vector<std::string>& dummy_names,
                                    std::string& error) {
    const auto span = bundle_.find_tensor(tensor_name);
    if (!span) {
        error = "missing Qwen3-TTS component: " + tensor_name;
        return false;
    }
    module = create_infer_engine(bundle_.path(), span->offset, span->size, dummy_names);
    if (!module) {
        error = "failed to load Qwen3-TTS component: " + tensor_name;
        return false;
    }
    return true;
}

bool HLIEQwen3TTSModel::load_embedding(const std::string& tensor_name,
                                       const std::vector<int64_t>& shape,
                                       std::shared_ptr<tcim::Tensor>& tensor,
                                       std::string& error) {
    tcim::Tensor value;
    if (!bundle_.load_fp16_tensor(tensor_name, shape, value, &error)) {
        return false;
    }
    tensor = std::make_shared<tcim::Tensor>(std::move(value));
    return true;
}

bool HLIEQwen3TTSModel::load_prefill_decode_pair(
    const std::string& prefill_name, const std::string& decode_name,
    std::shared_ptr<tcim::Module>& prefill, std::shared_ptr<tcim::Module>& decode,
    std::string& error) {
    if (!load_module(prefill_name, prefill, {}, error)) {
        return false;
    }

    std::vector<std::string> cache_names;
    for (const auto& name : TcimUtil::get_module_input_names(prefill)) {
        if (name.find("model_layers") != std::string::npos) {
            cache_names.push_back(name);
        }
    }
    if (cache_names.empty()) {
        error = "prefill component exposes no model_layers cache inputs: " + prefill_name;
        return false;
    }
    if (!load_module(decode_name, decode, cache_names, error)) {
        return false;
    }
    for (const auto& name : cache_names) {
        const auto cache = prefill->GetDevInput(name);
        if (decode->SetDevInput(name, cache) != tcim::Status::OK) {
            error = "failed to bind prefill/decode cache input: " + name;
            return false;
        }
    }
    return true;
}

bool HLIEQwen3TTSModel::reset_device_caches(
    const std::shared_ptr<tcim::Module>& module,
    const std::vector<std::string>& input_names, const size_t first_cache_index,
    std::string& error) {
    if (!module || first_cache_index > input_names.size()) {
        error = "invalid Qwen3-TTS cache reset request";
        return false;
    }
    size_t largest_cache = 0;
    for (size_t index = first_cache_index; index < input_names.size(); ++index) {
        largest_cache = std::max(largest_cache,
                                 module->GetDevInput(input_names[index]).Info().MemSize());
    }
    if (device_zero_scratch_.size() < largest_cache) {
        device_zero_scratch_.resize(largest_cache, uint8_t{0});
    }
    for (size_t index = first_cache_index; index < input_names.size(); ++index) {
        auto device_cache = module->GetDevInput(input_names[index]);
        const size_t cache_size = device_cache.Info().MemSize();
        const auto status = device_cache.Buffer().CopyFromHost(
            device_zero_scratch_.data(), cache_size);
        if (status != tcim::Status::OK) {
            error = "failed to clear Qwen3-TTS device cache: " + input_names[index];
            return false;
        }
    }
    return true;
}

int HLIEQwen3TTSModel::load_model(const std::string& model_path) {
    ready_ = false;
    std::string error;
    if (device_ids_.empty()) {
        LOG_ERROR("Qwen3-TTS devices must be configured before loading");
        return -1;
    }
    if (!bundle_.open(model_path, &error) || !load_metadata(error)) {
        LOG_ERROR("Failed to load Qwen3-TTS GGUF metadata: {}", error);
        return -1;
    }
    if (config_.model_size != "0b6" && config_.model_size != "1b7") {
        LOG_ERROR("Unsupported Qwen3-TTS model size: {}", config_.model_size);
        return -1;
    }
    if (config_.num_code_groups != 16 || config_.sample_rate != 24000 ||
        config_.stateful_samples_per_frame <= 0 ||
        config_.stateful_initial_output_skip_frames < 0 ||
        config_.tokenizer_vocab_size > config_.text_vocab_size) {
        LOG_ERROR("Invalid Qwen3-TTS bundle dimensions or sample rate");
        return -1;
    }

    if (!load_embedding(std::string(kTalkerTokenEmbedding),
                        {config_.codec_vocab_size, config_.talker_hidden_size},
                        talker_token_embedding_, error) ||
        !load_embedding(std::string(kTalkerTextEmbedding),
                        {config_.text_vocab_size, config_.text_hidden_size},
                        talker_text_embedding_, error)) {
        LOG_ERROR("Failed to load Qwen3-TTS embedding: {}", error);
        return -1;
    }

    if (!load_embedding(std::string(kCodePredictorEmbedding),
                        {config_.num_code_groups - 1,
                         config_.code_predictor_vocab_size,
                         config_.talker_hidden_size},
                        code_predictor_embedding_, error)) {
        LOG_ERROR("Failed to load Qwen3-TTS Code Predictor embedding: {}", error);
        return -1;
    }

    if (!load_module(std::string(kTextProjection), text_projection_model_, {}, error) ||
        !load_prefill_decode_pair(std::string(kTalkerPrefill), std::string(kTalkerDecode),
                                  talker_prefill_model_, talker_decode_model_, error) ||
        !load_prefill_decode_pair(std::string(kCodePredictorPrefill), std::string(kCodePredictorDecode),
                                  code_predictor_prefill_model_, code_predictor_decode_model_, error) ||
        !load_module(std::string(kStatefulDecoder), stateful_decoder_model_, {}, error)) {
        LOG_ERROR("Failed to load Qwen3-TTS TCIM component: {}", error);
        return -1;
    }

    return init();
}

int HLIEQwen3TTSModel::init() {
    if (!text_projection_model_ || !talker_prefill_model_ || !talker_decode_model_ ||
        !code_predictor_prefill_model_ || !code_predictor_decode_model_ ||
        !stateful_decoder_model_) {
        LOG_ERROR("Qwen3-TTS init called with missing components");
        return -1;
    }

    text_projection_input_names_ = TcimUtil::get_module_input_names(text_projection_model_);
    text_projection_output_names_ = TcimUtil::get_module_output_names(text_projection_model_);
    talker_prefill_input_names_ = TcimUtil::get_module_input_names(talker_prefill_model_);
    talker_prefill_output_names_ = TcimUtil::get_module_output_names(talker_prefill_model_);
    talker_decode_input_names_ = TcimUtil::get_module_input_names(talker_decode_model_);
    talker_decode_output_names_ = TcimUtil::get_module_output_names(talker_decode_model_);
    code_predictor_prefill_input_names_ = TcimUtil::get_module_input_names(code_predictor_prefill_model_);
    code_predictor_prefill_output_names_ = TcimUtil::get_module_output_names(code_predictor_prefill_model_);
    code_predictor_decode_input_names_ = TcimUtil::get_module_input_names(code_predictor_decode_model_);
    code_predictor_decode_output_names_ = TcimUtil::get_module_output_names(code_predictor_decode_model_);
    stateful_decoder_input_names_ = TcimUtil::get_module_input_names(stateful_decoder_model_);
    stateful_decoder_output_names_ = TcimUtil::get_module_output_names(stateful_decoder_model_);
    const auto& talker_inputs = talker_prefill_input_names_;
    const auto& talker_decode_inputs = talker_decode_input_names_;
    const auto& stateful_inputs = stateful_decoder_input_names_;
    if (text_projection_input_names_.empty() || text_projection_output_names_.empty() ||
        talker_inputs.size() < 4 || talker_prefill_output_names_.size() < 2 ||
        talker_decode_inputs.size() < 4 || talker_decode_output_names_.size() < 2 ||
        code_predictor_prefill_input_names_.size() < 5 || code_predictor_prefill_output_names_.empty() ||
        code_predictor_decode_input_names_.size() < 5 || code_predictor_decode_output_names_.empty() ||
        stateful_inputs.size() < 8 ||
        stateful_decoder_output_names_.size() != stateful_inputs.size() - 2) {
        LOG_ERROR("Qwen3-TTS component I/O signature is incomplete");
        return -1;
    }

    const auto prefill_shape = talker_prefill_model_->GetInputInfo(talker_inputs[0]).AsContiguous().Shape();
    const auto projection_shape = text_projection_model_->GetInputInfo(text_projection_input_names_[0]).AsContiguous().Shape();
    const auto code_predictor_prefill_shape = code_predictor_prefill_model_->GetInputInfo(
        code_predictor_prefill_input_names_[0]).AsContiguous().Shape();
    const auto context_shape = talker_decode_model_->GetInputInfo(talker_decode_inputs[3]).AsContiguous().Shape();
    const auto stateful_shape = stateful_decoder_model_->GetInputInfo(stateful_inputs[0]).AsContiguous().Shape();
    if (projection_shape.size() < 3 || prefill_shape.size() < 3 ||
        code_predictor_prefill_shape.size() < 3 || context_shape.size() < 3 ||
        stateful_shape.size() < 3 || projection_shape[2] != config_.text_hidden_size ||
        prefill_shape[2] != config_.talker_hidden_size ||
        code_predictor_prefill_shape[2] != config_.talker_hidden_size ||
        stateful_shape[2] != config_.num_code_groups) {
        LOG_ERROR("Qwen3-TTS component shapes do not match GGUF metadata");
        return -1;
    }
    text_projection_length_ = static_cast<int>(projection_shape[1]);
    prefill_length_ = static_cast<int>(prefill_shape[1]);
    code_predictor_prefill_length_ = static_cast<int>(code_predictor_prefill_shape[1]);
    context_length_ = static_cast<int>(context_shape[2]);
    stateful_chunk_size_ = static_cast<int>(stateful_shape[1]);
    const auto kv_shape = stateful_decoder_model_->GetInputInfo(
        stateful_inputs[7]).AsContiguous().Shape();
    if (kv_shape.size() < 3 || kv_shape[2] <= 0) {
        LOG_ERROR("Qwen3-TTS Stateful Decoder has an invalid KV cache shape");
        return -1;
    }
    stateful_kv_window_ = static_cast<int>(kv_shape[2]);

    tokenizer_ = std::make_shared<HLIETokenizer>();
    if (!tokenizer_->load_from_gguf_ctx(bundle_.context()) ||
        tokenizer_->vocab_size() != static_cast<size_t>(config_.tokenizer_vocab_size)) {
        LOG_ERROR("Qwen3-TTS tokenizer failed to load or vocabulary size mismatched");
        return -1;
    }

    talker_prefill_host_inputs_.clear();
    talker_prefill_host_inputs_.reserve(talker_inputs.size());
    for (const auto& name : talker_inputs) {
        const auto info = talker_prefill_model_->GetInputInfo(name).AsContiguous();
        auto tensor = std::make_shared<tcim::Tensor>(tcim::Tensor::CreateHostTensor(info));
        if (tensor->GetInitStatus() != tcim::Status::OK) {
            LOG_ERROR("Failed to allocate Qwen3-TTS prefill host input: {}", name);
            return -1;
        }
        std::memset(tensor->Data(), 0, info.MemSize());
        talker_prefill_host_inputs_.push_back(std::move(tensor));
    }

    const auto make_host_input = [](const std::shared_ptr<tcim::Module>& module,
                                    const std::string& name) {
        const auto info = module->GetInputInfo(name).AsContiguous();
        auto tensor = std::make_shared<tcim::Tensor>(tcim::Tensor::CreateHostTensor(info));
        if (tensor->GetInitStatus() == tcim::Status::OK) {
            std::memset(tensor->Data(), 0, info.MemSize());
        }
        return tensor;
    };
    talker_decode_valid_length_ = make_host_input(talker_decode_model_, talker_decode_input_names_[1]);
    talker_decode_current_length_ = make_host_input(talker_decode_model_, talker_decode_input_names_[2]);

    code_predictor_prefill_host_inputs_.clear();
    code_predictor_prefill_host_inputs_.reserve(code_predictor_prefill_input_names_.size());
    for (const auto& name : code_predictor_prefill_input_names_) {
        code_predictor_prefill_host_inputs_.push_back(make_host_input(code_predictor_prefill_model_, name));
    }
    code_predictor_decode_valid_length_ = make_host_input(
        code_predictor_decode_model_, code_predictor_decode_input_names_[1]);
    code_predictor_decode_current_length_ = make_host_input(
        code_predictor_decode_model_, code_predictor_decode_input_names_[2]);
    code_predictor_decode_generation_steps_ = make_host_input(
        code_predictor_decode_model_, code_predictor_decode_input_names_.back());
    if (talker_decode_valid_length_->GetInitStatus() != tcim::Status::OK ||
        talker_decode_current_length_->GetInitStatus() != tcim::Status::OK ||
        code_predictor_decode_valid_length_->GetInitStatus() != tcim::Status::OK ||
        code_predictor_decode_current_length_->GetInitStatus() != tcim::Status::OK ||
        code_predictor_decode_generation_steps_->GetInitStatus() != tcim::Status::OK) {
        LOG_ERROR("Failed to allocate Qwen3-TTS decode host inputs");
        return -1;
    }
    const int32_t one = 1;
    CHECK_TCIM_STATUS(talker_decode_current_length_->Buffer().CopyFromHost(&one, sizeof(one)));
    CHECK_TCIM_STATUS(talker_decode_model_->SetInput(
        talker_decode_input_names_[2], *talker_decode_current_length_));
    CHECK_TCIM_STATUS(code_predictor_decode_current_length_->Buffer().CopyFromHost(&one, sizeof(one)));
    CHECK_TCIM_STATUS(code_predictor_decode_model_->SetInput(
        code_predictor_decode_input_names_[2], *code_predictor_decode_current_length_));
    ready_ = true;
    LOG_INFO("Qwen3-TTS {} loaded: hidden={}, prefill={}, context={}, stream_chunk={}, "
             "sample_rate={}, samples_per_frame={}, initial_skip_frames={}, code_predictor_sampling={}",
             config_.model_size, config_.talker_hidden_size, prefill_length_, context_length_,
             stateful_chunk_size_, config_.sample_rate,
             config_.stateful_samples_per_frame,
             config_.stateful_initial_output_skip_frames,
             config_.code_predictor_sampling);
    return 0;
}

bool HLIEQwen3TTSModel::project_text_ids(const std::vector<int>& ids, tcim::Tensor& output,
                                         std::string& error) {
    if (ids.empty()) {
        error = "text projection received no token IDs";
        return false;
    }
    try {
        const auto embeddings = TcimUtil::embedding_lookup(*talker_text_embedding_, ids);
        std::vector<tcim::Tensor> chunks;
        for (int start = 0; start < static_cast<int>(ids.size()); start += text_projection_length_) {
            const int length = std::min(text_projection_length_, static_cast<int>(ids.size()) - start);
            auto input = slice_seq(embeddings, start, length);
            CHECK_TCIM_STATUS(text_projection_model_->SetInput(text_projection_input_names_[0], input));
            CHECK_TCIM_STATUS(text_projection_model_->Run());
            CHECK_TCIM_STATUS(text_projection_model_->Sync());
            auto host = text_projection_model_->GetDevOutput(text_projection_output_names_[0]).ToHost(true);
            if (host.Info().Shape().size() != 3 || host.Info().Shape()[1] < length) {
                error = "unexpected text projection output shape";
                return false;
            }
            chunks.push_back(host.Info().Shape()[1] == length ? host : slice_seq(host, 0, length));
        }
        output = chunks.size() == 1 ? chunks.front() : TcimUtil::concat_seq(chunks);
        return true;
    } catch (const std::exception& ex) {
        error = std::string("text projection failed: ") + ex.what();
        return false;
    }
}

bool HLIEQwen3TTSModel::debug_talker_prefill(
    const std::string& text, const std::string& language, const std::string& speaker,
    Qwen3TTSPrefillDebugResult& result, std::string& error) {
    std::lock_guard lock(inference_mutex_);
    return run_talker_prefill_debug(text, language, speaker, result, nullptr, nullptr,
                                    false, error);
}

bool HLIEQwen3TTSModel::run_talker_prefill_debug(
    const std::string& text, const std::string& language, const std::string& speaker,
    Qwen3TTSPrefillDebugResult& result, tcim::Tensor* tts_pad_output,
    tcim::Tensor* past_hidden_output, const bool do_sample, std::string& error) {
    result = {};
    if (!ready_ || text.empty()) {
        error = ready_ ? "Qwen3-TTS text must not be empty" : "Qwen3-TTS model is not ready";
        return false;
    }

    try {
        const std::string language_key = lower_ascii(language.empty() ? "chinese" : language);
        const std::string speaker_key = lower_ascii(speaker.empty() ? "vivian" : speaker);
        const auto speaker_it = config_.speaker_ids.find(speaker_key);
        if (speaker_it == config_.speaker_ids.end()) {
            error = "unsupported Qwen3-TTS speaker: " + speaker_key;
            return false;
        }

        std::optional<int32_t> language_id;
        if (language_key != "auto") {
            const auto language_it = config_.codec_language_ids.find(language_key);
            if (language_it == config_.codec_language_ids.end()) {
                error = "unsupported Qwen3-TTS language: " + language_key;
                return false;
            }
            language_id = language_it->second;
        }
        if ((language_key == "chinese" || language_key == "auto")) {
            if (const auto dialect = config_.speaker_dialects.find(speaker_key);
                dialect != config_.speaker_dialects.end()) {
                language_id = config_.codec_language_ids.at(dialect->second);
            }
        }

        const std::string assistant_text =
            "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
        result.input_ids = tokenizer_->tokenize(assistant_text);
        if (result.input_ids.size() < 9) {
            error = "Qwen3-TTS tokenizer produced an invalid assistant prompt";
            return false;
        }

        tcim::Tensor tts_special;
        if (!project_text_ids({config_.tts_bos_token_id, config_.tts_eos_token_id,
                               config_.tts_pad_token_id}, tts_special, error)) {
            return false;
        }
        const auto tts_bos = slice_seq(tts_special, 0, 1);
        const auto tts_eos = slice_seq(tts_special, 1, 1);
        const auto tts_pad = slice_seq(tts_special, 2, 1);

        std::vector<int> codec_ids;
        if (language_id) {
            codec_ids = {config_.codec_think_id, config_.codec_think_bos_id,
                         *language_id, config_.codec_think_eos_id};
        } else {
            codec_ids = {config_.codec_nothink_id, config_.codec_think_bos_id,
                         config_.codec_think_eos_id};
        }
        codec_ids.push_back(speaker_it->second);
        codec_ids.push_back(config_.codec_pad_id);
        codec_ids.push_back(config_.codec_bos_id);
        const auto codec_embedding = TcimUtil::embedding_lookup(*talker_token_embedding_, codec_ids);

        tcim::Tensor role;
        const std::vector<int> role_ids(result.input_ids.begin(), result.input_ids.begin() + 3);
        if (!project_text_ids(role_ids, role, error)) {
            return false;
        }
        std::vector<int> prefix_text_ids(codec_ids.size() - 2, config_.tts_pad_token_id);
        prefix_text_ids.push_back(config_.tts_bos_token_id);
        tcim::Tensor prefix_text;
        if (!project_text_ids(prefix_text_ids, prefix_text, error)) {
            return false;
        }
        auto prefix = add_fp16(prefix_text, slice_seq(codec_embedding, 0,
                                                      static_cast<int>(codec_ids.size()) - 1));
        auto talker_input = TcimUtil::concat_seq({role, prefix});
        talker_input = slice_seq(talker_input, 0, static_cast<int>(talker_input.Info().Shape()[1]) - 1);

        const std::vector<int> body_ids(result.input_ids.begin() + 3, result.input_ids.end() - 5);
        tcim::Tensor body;
        if (!project_text_ids(body_ids, body, error)) {
            return false;
        }
        auto text_tail = TcimUtil::concat_seq({body, tts_eos});
        const auto codec_pad = TcimUtil::embedding_lookup(
            *talker_token_embedding_, std::vector<int>(body_ids.size() + 1, config_.codec_pad_id));
        text_tail = add_fp16(text_tail, codec_pad);
        const auto final_position = add_fp16(
            tts_pad, TcimUtil::embedding_lookup(*talker_token_embedding_, {config_.codec_bos_id}));
        talker_input = TcimUtil::concat_seq({talker_input, text_tail, final_position});
        result.prompt_length = static_cast<int32_t>(talker_input.Info().Shape()[1]);
        const size_t prompt_elements = static_cast<size_t>(TcimUtil::numel(talker_input.Info().Shape()));
        const auto prompt_values = TcimUtil::tensor_to_float_vec(talker_input.Data(), prompt_elements);
        double prompt_sum = 0.0;
        double prompt_abs_sum = 0.0;
        for (const float value : prompt_values) {
            prompt_sum += value;
            prompt_abs_sum += std::abs(value);
        }
        result.prompt_sum = static_cast<float>(prompt_sum);
        result.prompt_abs_mean = static_cast<float>(prompt_abs_sum / prompt_values.size());
        if (result.prompt_length >= context_length_) {
            error = "Qwen3-TTS prompt exceeds Talker context length";
            return false;
        }

        const int padded_length = ((result.prompt_length + prefill_length_ - 1) / prefill_length_) * prefill_length_;
        if (padded_length > result.prompt_length) {
            const auto padding = TcimUtil::embedding_lookup(
                *talker_token_embedding_, std::vector<int>(padded_length - result.prompt_length, 0));
            talker_input = TcimUtil::concat_seq({talker_input, padding});
        }
        if (!reset_device_caches(talker_prefill_model_, talker_prefill_input_names_,
                                 3, error)) {
            return false;
        }

        result.prefill_chunks = padded_length / prefill_length_;
        for (int round = 0; round < result.prefill_chunks; ++round) {
            const int start = round * prefill_length_;
            const int current_length = std::min(prefill_length_, result.prompt_length - start);
            auto chunk = slice_seq(talker_input, start, prefill_length_);
            auto& valid = *talker_prefill_host_inputs_[1];
            auto& current = *talker_prefill_host_inputs_[2];
            const int32_t valid_value = start;
            const int32_t current_value = current_length;
            CHECK_TCIM_STATUS(valid.Buffer().CopyFromHost(&valid_value, sizeof(valid_value)));
            CHECK_TCIM_STATUS(current.Buffer().CopyFromHost(&current_value, sizeof(current_value)));
            CHECK_TCIM_STATUS(talker_prefill_model_->SetInput(talker_prefill_input_names_[0], chunk));
            CHECK_TCIM_STATUS(talker_prefill_model_->SetInput(talker_prefill_input_names_[1], valid));
            CHECK_TCIM_STATUS(talker_prefill_model_->SetInput(talker_prefill_input_names_[2], current));
            CHECK_TCIM_STATUS(talker_prefill_model_->Run());
            CHECK_TCIM_STATUS(talker_prefill_model_->Sync());
        }

        const auto logits = talker_prefill_model_->GetDevOutput(talker_prefill_output_names_[0]).ToHost(true);
        result.logits_shape = logits.Info().Shape();
        if (result.logits_shape.empty() || logits.Info().DataType() != tcim::DataType::FLOAT16) {
            error = "unexpected Qwen3-TTS Talker logits output";
            return false;
        }
        const int64_t vocab = result.logits_shape.back();
        const int64_t total = TcimUtil::numel(result.logits_shape);
        const auto values = TcimUtil::tensor_to_float_vec(logits.Data(), static_cast<size_t>(total));
        const size_t offset = static_cast<size_t>(total - vocab);
        const auto begin = values.begin() + static_cast<std::ptrdiff_t>(offset);
        result.first_codec_token = select_talker_logits(
            logits, {}, config_.codec_eos_token_id, do_sample, sampling_rng_);
        result.first_codec_logit = values[
            offset + static_cast<size_t>(result.first_codec_token)];
        const size_t head_size = std::min<size_t>(16, static_cast<size_t>(vocab));
        result.last_logits_head.assign(begin, begin + static_cast<std::ptrdiff_t>(head_size));
        if (tts_pad_output) {
            *tts_pad_output = tts_pad;
        }
        if (past_hidden_output) {
            *past_hidden_output = talker_prefill_model_->GetDevOutput(
                talker_prefill_output_names_[1]).ToHost(true);
        }
        LOG_INFO("Qwen3-TTS debug prefill: tokens={}, prompt={}, chunks={}, first_codec_token={}",
                 result.input_ids.size(), result.prompt_length, result.prefill_chunks,
                 result.first_codec_token);
        return true;
    } catch (const std::exception& ex) {
        error = std::string("Qwen3-TTS Talker prefill failed: ") + ex.what();
        return false;
    }
}

bool HLIEQwen3TTSModel::generate_code_predictor_frame(
    const tcim::Tensor& past_hidden, const int32_t talker_token,
    std::vector<int32_t>& predictor_tokens, std::vector<tcim::Tensor>& predictor_embeds,
    const bool do_sample, std::string& error) {
    predictor_tokens.clear();
    predictor_embeds.clear();
    try {
        const auto code_embedding = [this](const int group, const int32_t token) {
            const auto& shape = code_predictor_embedding_->Info().Shape();
            if (shape.size() != 3 || group < 0 || group >= shape[0] ||
                token < 0 || token >= shape[1]) {
                throw std::out_of_range("CodePredictor embedding index is out of range");
            }
            const int64_t hidden = shape[2];
            auto output = tcim::Tensor::CreateHostTensor(tcim::TensorInfo::CreateNDInfo(
                {1, 1, hidden}, tcim::DataType::FLOAT16));
            const auto* source = static_cast<const TensorFP16*>(code_predictor_embedding_->Data()) +
                                 (static_cast<int64_t>(group) * shape[1] + token) * hidden;
            std::memcpy(output.Data(), source, static_cast<size_t>(hidden) * sizeof(TensorFP16));
            return output;
        };

        const auto talker_embedding = TcimUtil::embedding_lookup(
            *talker_token_embedding_, {talker_token});
        auto predictor_input = TcimUtil::concat_seq({past_hidden, talker_embedding});
        if (predictor_input.Info().Shape()[1] != code_predictor_prefill_length_) {
            error = "CodePredictor prefill input length mismatch";
            return false;
        }

        if (!reset_device_caches(code_predictor_prefill_model_,
                                 code_predictor_prefill_input_names_, 3, error)) {
            return false;
        }
        const int32_t zero = 0;
        const int32_t current_length = code_predictor_prefill_length_;
        auto& valid = *code_predictor_prefill_host_inputs_[1];
        auto& current = *code_predictor_prefill_host_inputs_[2];
        auto& generation = *code_predictor_prefill_host_inputs_.back();
        CHECK_TCIM_STATUS(valid.Buffer().CopyFromHost(&zero, sizeof(zero)));
        CHECK_TCIM_STATUS(current.Buffer().CopyFromHost(&current_length, sizeof(current_length)));
        CHECK_TCIM_STATUS(generation.Buffer().CopyFromHost(&zero, sizeof(zero)));
        CHECK_TCIM_STATUS(code_predictor_prefill_model_->SetInput(
            code_predictor_prefill_input_names_[0], predictor_input));
        CHECK_TCIM_STATUS(code_predictor_prefill_model_->SetInput(
            code_predictor_prefill_input_names_[1], valid));
        CHECK_TCIM_STATUS(code_predictor_prefill_model_->SetInput(
            code_predictor_prefill_input_names_[2], current));
        CHECK_TCIM_STATUS(code_predictor_prefill_model_->SetInput(
            code_predictor_prefill_input_names_.back(), generation));
        CHECK_TCIM_STATUS(code_predictor_prefill_model_->Run());
        CHECK_TCIM_STATUS(code_predictor_prefill_model_->Sync());

        auto logits = code_predictor_prefill_model_->GetDevOutput(
            code_predictor_prefill_output_names_[0]).ToHost(true);
        int32_t next_token = select_subtalker_logits(logits, do_sample, sampling_rng_);
        predictor_tokens.push_back(next_token);

        int32_t context_length = code_predictor_prefill_length_;
        for (int step = 0; step < config_.num_code_groups - 2; ++step) {
            auto next_embedding = code_embedding(step, next_token);
            predictor_embeds.push_back(next_embedding);

            const int32_t generation_step = step + 1;
            CHECK_TCIM_STATUS(code_predictor_decode_valid_length_->Buffer().CopyFromHost(
                &context_length, sizeof(context_length)));
            CHECK_TCIM_STATUS(code_predictor_decode_generation_steps_->Buffer().CopyFromHost(
                &generation_step, sizeof(generation_step)));
            CHECK_TCIM_STATUS(code_predictor_decode_model_->SetInput(
                code_predictor_decode_input_names_[0], next_embedding));
            CHECK_TCIM_STATUS(code_predictor_decode_model_->SetInput(
                code_predictor_decode_input_names_[1], *code_predictor_decode_valid_length_));
            CHECK_TCIM_STATUS(code_predictor_decode_model_->SetInput(
                code_predictor_decode_input_names_.back(), *code_predictor_decode_generation_steps_));
            CHECK_TCIM_STATUS(code_predictor_decode_model_->Run());
            CHECK_TCIM_STATUS(code_predictor_decode_model_->Sync());

            logits = code_predictor_decode_model_->GetDevOutput(
                code_predictor_decode_output_names_[0]).ToHost(true);
            next_token = select_subtalker_logits(logits, do_sample, sampling_rng_);
            predictor_tokens.push_back(next_token);
            ++context_length;
        }
        if (predictor_tokens.size() != static_cast<size_t>(config_.num_code_groups - 1) ||
            predictor_embeds.size() != static_cast<size_t>(config_.num_code_groups - 2)) {
            error = "CodePredictor produced an incomplete codec frame";
            return false;
        }
        return true;
    } catch (const std::exception& ex) {
        error = std::string("CodePredictor frame generation failed: ") + ex.what();
        return false;
    }
}

bool HLIEQwen3TTSModel::debug_generate_codec_frames(
    const std::string& text, const std::string& language, const std::string& speaker,
    const int max_frames, Qwen3TTSCodecFramesDebugResult& result, std::string& error) {
    std::lock_guard lock(inference_mutex_);
    if (max_frames <= 0 || max_frames > 32) {
        error = "max_frames must be between 1 and 32";
        return false;
    }
    return generate_codec_frames_unlocked(
        text, language, speaker, max_frames, result, false, error);
}

bool HLIEQwen3TTSModel::generate_codec_frames_unlocked(
    const std::string& text, const std::string& language, const std::string& speaker,
    const int max_frames, Qwen3TTSCodecFramesDebugResult& result, const bool do_sample,
    std::string& error) {
    result = {};
    if (max_frames <= 0 || max_frames > context_length_) {
        error = "max_frames must be between 1 and the model context length";
        return false;
    }

    tcim::Tensor tts_pad;
    tcim::Tensor past_hidden;
    if (!run_talker_prefill_debug(text, language, speaker, result.prefill,
                                  &tts_pad, &past_hidden, do_sample, error)) {
        return false;
    }
    if (result.prefill.prompt_length + max_frames >= context_length_) {
        error = "Qwen3-TTS prompt plus max_frames exceeds the model context length";
        return false;
    }

    try {
        int32_t talker_token = result.prefill.first_codec_token;
        int32_t past_seq_length = result.prefill.prompt_length;
        for (int frame_index = 0; frame_index < max_frames; ++frame_index) {
            result.talker_tokens.push_back(talker_token);
            std::vector<int32_t> predictor_tokens;
            std::vector<tcim::Tensor> predictor_embeds;
            if (!generate_code_predictor_frame(past_hidden, talker_token,
                                               predictor_tokens, predictor_embeds,
                                               do_sample && config_.code_predictor_sampling,
                                               error)) {
                return false;
            }

            std::vector<int32_t> frame;
            frame.reserve(config_.num_code_groups);
            frame.push_back(talker_token);
            frame.insert(frame.end(), predictor_tokens.begin(), predictor_tokens.end());
            result.codec_frames.push_back(std::move(frame));
            if (frame_index + 1 >= max_frames) {
                break;
            }

            std::vector<tcim::Tensor> codec_hiddens;
            codec_hiddens.reserve(1 + predictor_embeds.size());
            codec_hiddens.push_back(TcimUtil::embedding_lookup(
                *talker_token_embedding_, {talker_token}));
            codec_hiddens.insert(codec_hiddens.end(), predictor_embeds.begin(), predictor_embeds.end());
            auto decode_input = add_fp16(sum_fp16_vectors(codec_hiddens), tts_pad);

            CHECK_TCIM_STATUS(talker_decode_valid_length_->Buffer().CopyFromHost(
                &past_seq_length, sizeof(past_seq_length)));
            CHECK_TCIM_STATUS(talker_decode_model_->SetInput(
                talker_decode_input_names_[0], decode_input));
            CHECK_TCIM_STATUS(talker_decode_model_->SetInput(
                talker_decode_input_names_[1], *talker_decode_valid_length_));
            CHECK_TCIM_STATUS(talker_decode_model_->Run());
            CHECK_TCIM_STATUS(talker_decode_model_->Sync());

            const auto logits = talker_decode_model_->GetDevOutput(
                talker_decode_output_names_[0]).ToHost(true);
            past_hidden = talker_decode_model_->GetDevOutput(
                talker_decode_output_names_[1]).ToHost(true);
            talker_token = select_talker_logits(
                logits, result.talker_tokens, config_.codec_eos_token_id,
                do_sample, sampling_rng_);
            ++past_seq_length;
            if (talker_token == config_.codec_eos_token_id) {
                result.reached_eos = true;
                break;
            }
        }
        LOG_INFO("Qwen3-TTS {} codec generation produced {} frames",
                 do_sample ? "sampled" : "greedy", result.codec_frames.size());
        return true;
    } catch (const std::exception& ex) {
        error = std::string("Qwen3-TTS codec frame generation failed: ") + ex.what();
        return false;
    }
}

bool HLIEQwen3TTSModel::create_stateful_decoder_state(
    StatefulDecoderState& state, std::string& error) {
    state = {};
    try {
        // Inputs 1..3 are convolution/latent history and 7.. are KV caches.
        // Scalar controls and audio_codes are supplied per invocation.
        state.tensors.reserve(stateful_decoder_input_names_.size() - 4);
        for (size_t index = 1; index < stateful_decoder_input_names_.size(); ++index) {
            if (index >= 4 && index <= 6) {
                continue;
            }
            const auto info = stateful_decoder_model_->GetInputInfo(
                stateful_decoder_input_names_[index]).AsContiguous();
            auto tensor = tcim::Tensor::CreateHostTensor(info);
            if (tensor.GetInitStatus() != tcim::Status::OK) {
                error = "failed to allocate Stateful Decoder state input: " +
                        stateful_decoder_input_names_[index];
                return false;
            }
            std::memset(tensor.Data(), 0, info.MemSize());
            state.tensors.push_back(std::move(tensor));
        }
        return true;
    } catch (const std::exception& ex) {
        error = std::string("failed to create Stateful Decoder state: ") + ex.what();
        return false;
    }
}

bool HLIEQwen3TTSModel::decode_stateful_chunk(
    const std::vector<std::vector<int32_t>>& frames, const bool is_final,
    StatefulDecoderState& state, std::vector<float>& audio, std::string& error) {
    audio.clear();
    if (frames.empty() && is_final && !state.latent_audio.empty()) {
        audio = std::move(state.latent_audio);
        return true;
    }
    if (frames.size() > static_cast<size_t>(stateful_chunk_size_) ||
        (!is_final && frames.size() != static_cast<size_t>(stateful_chunk_size_))) {
        error = "Stateful Decoder requires a full non-final chunk and at most one chunk when final";
        return false;
    }

    try {
        const auto codes_info = stateful_decoder_model_->GetInputInfo(
            stateful_decoder_input_names_[0]).AsContiguous();
        if (codes_info.DataType() != tcim::DataType::INT32) {
            error = "Stateful Decoder audio_codes input must be INT32";
            return false;
        }
        auto codes = tcim::Tensor::CreateHostTensor(codes_info);
        std::memset(codes.Data(), 0, codes_info.MemSize());
        auto* code_data = static_cast<int32_t*>(codes.Data());
        for (size_t frame = 0; frame < frames.size(); ++frame) {
            if (frames[frame].size() != static_cast<size_t>(config_.num_code_groups)) {
                error = "Stateful Decoder received an invalid codec frame width";
                return false;
            }
            for (size_t group = 0; group < frames[frame].size(); ++group) {
                const int32_t token = frames[frame][group];
                const int32_t vocabulary = group == 0
                    ? config_.codec_vocab_size : config_.code_predictor_vocab_size;
                if (token < 0 || token >= vocabulary) {
                    error = "Stateful Decoder received an out-of-range codec token";
                    return false;
                }
            }
            std::copy(frames[frame].begin(), frames[frame].end(),
                      code_data + frame * static_cast<size_t>(config_.num_code_groups));
        }
        CHECK_TCIM_STATUS(stateful_decoder_model_->SetInput(
            stateful_decoder_input_names_[0], codes));

        size_t state_index = 0;
        for (size_t input_index = 1; input_index < stateful_decoder_input_names_.size(); ++input_index) {
            if (input_index >= 4 && input_index <= 6) {
                continue;
            }
            const auto status = state.device_backed
                ? stateful_decoder_model_->SetDevInput(
                      stateful_decoder_input_names_[input_index], state.tensors[state_index])
                : stateful_decoder_model_->SetInput(
                      stateful_decoder_input_names_[input_index], state.tensors[state_index]);
            CHECK_TCIM_STATUS(status);
            ++state_index;
        }

        auto set_scalar = [&](const size_t input_index, const auto value) {
            const auto info = stateful_decoder_model_->GetInputInfo(
                stateful_decoder_input_names_[input_index]).AsContiguous();
            auto tensor = tcim::Tensor::CreateHostTensor(info);
            CHECK_TCIM_STATUS(tensor.Buffer().CopyFromHost(&value, sizeof(value)));
            CHECK_TCIM_STATUS(stateful_decoder_model_->SetInput(
                stateful_decoder_input_names_[input_index], tensor));
        };
        set_scalar(4, TensorFP16(is_final ? 1.0F : 0.0F));
        set_scalar(5, state.kv_valid_length);
        set_scalar(6, static_cast<int32_t>(frames.size()));

        CHECK_TCIM_STATUS(stateful_decoder_model_->Run());
        CHECK_TCIM_STATUS(stateful_decoder_model_->Sync());

        const auto wav = stateful_decoder_model_->GetDevOutput(
            stateful_decoder_output_names_[0]).ToHost(true);
        const auto valid_tensor = stateful_decoder_model_->GetDevOutput(
            stateful_decoder_output_names_[1]).ToHost(true);
        if (wav.Info().DataType() != tcim::DataType::FLOAT16 ||
            valid_tensor.Info().DataType() != tcim::DataType::INT32) {
            error = "Stateful Decoder returned unexpected output types";
            return false;
        }
        const int32_t valid_samples = *static_cast<const int32_t*>(valid_tensor.Data());
        const auto wav_values = fp16_values(wav);
        const size_t audio_start = state.kv_valid_length == 0
            ? static_cast<size_t>(config_.stateful_samples_per_frame) *
                  static_cast<size_t>(config_.stateful_initial_output_skip_frames)
            : 0U;
        const size_t audio_end = audio_start + static_cast<size_t>(std::max(0, valid_samples));
        if (audio_end > wav_values.size()) {
            error = "Stateful Decoder valid_samples exceeds final_wav capacity";
            return false;
        }
        audio.assign(wav_values.begin() + static_cast<std::ptrdiff_t>(audio_start),
                     wav_values.begin() + static_cast<std::ptrdiff_t>(audio_end));
        if (is_final) {
            state.latent_audio.clear();
        } else {
            state.latent_audio.assign(
                wav_values.begin() + static_cast<std::ptrdiff_t>(audio_end), wav_values.end());
        }

        state.tensors.clear();
        state.tensors.reserve(stateful_decoder_output_names_.size() - 2);
        for (size_t output_index = 2; output_index < stateful_decoder_output_names_.size(); ++output_index) {
            state.tensors.push_back(stateful_decoder_model_->GetDevOutput(
                stateful_decoder_output_names_[output_index]));
        }
        state.device_backed = true;
        state.kv_valid_length = std::min<int32_t>(
            stateful_kv_window_,
            state.kv_valid_length + static_cast<int32_t>(frames.size()));
        return true;
    } catch (const std::exception& ex) {
        error = std::string("Stateful Decoder inference failed: ") + ex.what();
        return false;
    }
}

bool HLIEQwen3TTSModel::synthesize(
    const std::string& text, const std::string& language, const std::string& speaker,
    const int max_frames, Qwen3TTSSynthesisResult& result, std::string& error,
    const bool require_eos, const bool do_sample, const uint32_t seed) {
    std::lock_guard lock(inference_mutex_);
    result = {};
    if (do_sample) {
        // Isolate the random sequence per request so earlier requests cannot
        // change the voice, pace, or token path of an identical later request.
        sampling_rng_.seed(seed);
    }
    Qwen3TTSCodecFramesDebugResult generated;
    if (!generate_codec_frames_unlocked(
            text, language, speaker, max_frames, generated, do_sample, error)) {
        return false;
    }

    result.codec_frames = std::move(generated.codec_frames);
    result.reached_eos = generated.reached_eos;
    if (require_eos && !result.reached_eos) {
        error = "Qwen3-TTS generation reached max_frames before EOS";
        return false;
    }
    if (!decode_codec_frames_unlocked(result.codec_frames, result.audio, error)) {
        return false;
    }
    LOG_INFO("Qwen3-TTS synthesized {} codec frames into {} samples (eos={})",
             result.codec_frames.size(), result.audio.size(), result.reached_eos);
    return true;
}

bool HLIEQwen3TTSModel::decode_codec_frames_unlocked(
    const std::vector<std::vector<int32_t>>& frames, std::vector<float>& audio,
    std::string& error) {
    audio.clear();
    if (frames.empty()) {
        error = "Stateful Decoder requires at least one codec frame";
        return false;
    }
    StatefulDecoderState state;
    if (!create_stateful_decoder_state(state, error)) {
        return false;
    }
    for (size_t begin = 0; begin < frames.size();
         begin += static_cast<size_t>(stateful_chunk_size_)) {
        const size_t end = std::min(
            frames.size(), begin + static_cast<size_t>(stateful_chunk_size_));
        const bool final = end == frames.size() &&
                           end - begin < static_cast<size_t>(stateful_chunk_size_);
        std::vector<std::vector<int32_t>> chunk(frames.begin() + begin,
                                                frames.begin() + end);
        std::vector<float> chunk_audio;
        if (!decode_stateful_chunk(chunk, final, state, chunk_audio, error)) {
            return false;
        }
        audio.insert(audio.end(), chunk_audio.begin(), chunk_audio.end());
    }
    if (frames.size() % static_cast<size_t>(stateful_chunk_size_) == 0) {
        std::vector<float> tail;
        if (!decode_stateful_chunk({}, true, state, tail, error)) {
            return false;
        }
        audio.insert(audio.end(), tail.begin(), tail.end());
    }
    return true;
}

bool HLIEQwen3TTSModel::debug_decode_codec_frames(
    const std::vector<std::vector<int32_t>>& frames, std::vector<float>& audio,
    std::string& error) {
    std::lock_guard lock(inference_mutex_);
    if (!ready_) {
        error = "Qwen3-TTS model is not ready";
        return false;
    }
    if (frames.size() > static_cast<size_t>(context_length_)) {
        error = "codec frame count exceeds the model context length";
        return false;
    }
    return decode_codec_frames_unlocked(frames, audio, error);
}

HLIE_NAMESPACE_END

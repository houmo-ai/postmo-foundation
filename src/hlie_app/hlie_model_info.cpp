#include "hlie_app/hlie_model_info.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <regex>
#include <unordered_map>

#include "gguf.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_gguf_util.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"
#include "hlie_app/hlie_model_family_registry.h"
#include "json.hpp"
#include "log.h"


HLIE_NAMESPACE_BEGIN

namespace {
    std::string to_lower(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) { return std::tolower(c); });
        return result;
    }

    const std::regex kParamRegex(R"((\d+(?:\.\d+)?)\s*[bBmM])", std::regex::icase);
    const std::regex kSizeLabelRegex(R"((\d+(?:\.\d+)?)\s*([bBmM]))", std::regex::icase);

    uint64_t parse_param_count(const std::string& size_label) {
        std::smatch m;
        if (!std::regex_search(size_label, m, kSizeLabelRegex)) {
            return 0;
        }

        const double value = std::stod(m[1].str());
        const std::string unit = to_lower(m[2].str());
        const double multiplier = unit == "b" ? 1000.0 * 1000.0 * 1000.0 : 1000.0 * 1000.0;
        return static_cast<uint64_t>(value * multiplier);
    }

    double weight_bits_from_file_type(const uint32_t file_type) {
        switch (file_type) {
        case 0:  return 32.0;
        case 1:  return 16.0;
        case 2:  return 4.0;
        case 3:  return 4.0;
        case 7:  return 8.0;
        case 8:  return 5.0;
        case 9:  return 5.0;
        case 10: return 2.0;
        case 12: return 3.0;
        case 14: return 6.0;
        case 15: return 4.0;
        case 16: return 4.0;
        default: return 4.0;
        }
    }

    int bytes_to_mb_ceil(const double bytes) {
        if (bytes <= 0) {
            return 0;
        }

        return static_cast<int>(std::ceil(bytes / (1000.0 * 1000.0)));
    }

    bool get_u32_key(gguf_context* ctx, const std::string& key, uint32_t& value) {
        return HLIEGgufUtil::get_key<uint32_t>(ctx, key, value, false);
    }

    std::string get_arch_prefix(gguf_context* ctx) {
        std::string arch;
        if (!HLIEGgufUtil::get_key<std::string>(ctx, "general.architecture", arch, false) || arch.empty()) {
            return {};
        }

        return arch + ".";
    }

    std::filesystem::path memory_estimates_path() {
        const auto& config_service = Singleton<HLIEConfigService>::instance();
        const auto bundle_path = config_service->get_config_bundle().memory_estimates;
        if (!bundle_path.empty()) {
            return bundle_path;
        }
        const auto config_path = config_service->get_config_path();
        if (config_path.empty()) {
            return "memory_estimates.json";
        }
        return std::filesystem::path(config_path).parent_path() / "memory_estimates.json";
    }

    struct GgufMetadata {
        std::string architecture;
        std::string tokenizer_model;
        std::string projector_type;
        std::string vision_projector_type;
        std::string audio_projector_type;
        bool has_vision_encoder = false;
        bool has_audio_encoder = false;
        bool valid = false;
    };

    GgufMetadata read_gguf_metadata(const std::string& path) {
        GgufMetadata metadata;
        if (path.empty()) {
            return metadata;
        }

        ggml_context* ctx_data = nullptr;
        gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx_data,
        };
        gguf_context* ctx = gguf_init_from_file(path.c_str(), params);
        if (!ctx) {
            return metadata;
        }

        try {
            metadata.valid = true;
            HLIEGgufUtil::get_key<std::string>(ctx, "general.architecture", metadata.architecture, false);
            HLIEGgufUtil::get_key<std::string>(ctx, "tokenizer.ggml.model", metadata.tokenizer_model, false);
            HLIEGgufUtil::get_key<std::string>(ctx, "clip.projector_type", metadata.projector_type, false);
            HLIEGgufUtil::get_key<std::string>(ctx, "clip.vision.projector_type", metadata.vision_projector_type, false);
            HLIEGgufUtil::get_key<std::string>(ctx, "clip.audio.projector_type", metadata.audio_projector_type, false);
            HLIEGgufUtil::get_key<bool>(ctx, "clip.has_vision_encoder", metadata.has_vision_encoder, false);
            HLIEGgufUtil::get_key<bool>(ctx, "clip.has_audio_encoder", metadata.has_audio_encoder, false);
        } catch (const std::exception& ex) {
            LOG_WARN("failed to read GGUF metadata from {}, ex: {}", path, ex.what());
        }
        gguf_free(ctx);
        return metadata;
    }

    bool is_known_service_kind(const std::string& value) {
        return value == "llm" || value == "sd" || value == "whisper" || value == "cosyvoice" || value == "tts" ||
            value == "qwen3_tts" || value == "document";
    }

} // namespace

int HLIEModelInfo::parse_from_file(const std::string& file_path) {
    try {
        capabilities_ = {};
        role_ = HLIEModelRole::Unknown;
        effective_llama_params_ = {};
        matched_llama_presets_.clear();
        llama_param_sources_.clear();
        effective_llama_params_hash_.clear();
        has_effective_llama_params_ = false;
        model_family_.clear();
        architecture_.clear();
        service_kind_.clear();
        std::ifstream file(file_path);
        if (!file.is_open()) {
            LOG_ERROR("Error: Could not open file: {}", file_path);
            return -3;
        }

        nlohmann::json config;
        file >> config;
        if (const auto schema = config.find("Schema"); schema != config.end()) {
            if (!schema->is_string() || schema->get<std::string>() != "hlie.model-launch" ||
                config.value("SchemaVersion", 0) != 1) {
                LOG_ERROR("Unsupported model launch manifest schema: {}", file_path);
                return -2;
            }
        }

        model_path_ = config.value("ModelPath", "");
        model_type_ = to_lower(config.value("ModelType", ""));
        service_kind_ = to_lower(config.value("ServiceKind", ""));
        model_family_ = to_lower(config.value("ModelFamily", ""));
        architecture_ = to_lower(config.value("Architecture", ""));
        role_ = role_from_string(config.value("Role", ""));
        if (service_kind_.empty() && is_known_service_kind(model_type_)) {
            service_kind_ = model_type_;
        }
        if (model_family_.empty() && !model_type_.empty() && !is_known_service_kind(model_type_)) {
            model_family_ = model_type_;
        }
        context_size_ = config.value("ContextSize", 0);
        cores_ = config.value("Cores", 0);
        batch_num_ = config.value("BatchNum", 1);
        xh_version_ = config.value("XHVersion", 2);
        is_multi_model_ = config.value("IsMM", false);
        if (const auto family = HLIEModelFamilyRegistry::detect_projector_family(model_family_); !family.empty()) {
            model_family_ = family;
            is_multi_model_ = true;
        }
        if (const auto capabilities = config.find("Capabilities"); capabilities != config.end() && capabilities->is_object()) {
            capabilities_.text = capabilities->value("text", true);
            capabilities_.vision = capabilities->value("vision", false);
            capabilities_.audio = capabilities->value("audio", false);
        }
        params_ = config.value("Params", "");
        mm_proj_path_ = config.value("MMProj", "");
        model_version_ = config.value("Version", "");
        estimated_mem_mb_ = config.value("EstimatedMemMB", 0);
        clip_model_ = config.value("ClipModel", "");
        clip_l_model_ = config.value("ClipLModel", "");
        t5_model_ = config.value("T5Model", "");
        devices_ = config.value("Devices", "");
        if (const auto launch = config.find("LlamaLaunch"); launch != config.end() && launch->is_object()) {
            const auto effective = launch->find("EffectiveParams");
            std::string params_error;
            if (effective != launch->end() &&
                HLIELlmParamSet::from_json(*effective, effective_llama_params_, &params_error, false)) {
                matched_llama_presets_ = launch->value("MatchedPresets", std::vector<std::string>{});
                llama_param_sources_ = launch->value("Sources", std::vector<std::string>{});
                effective_llama_params_hash_ = launch->value("ParamsHash", effective_llama_params_.stable_hash());
                has_effective_llama_params_ = true;
            } else if (effective != launch->end()) {
                LOG_ERROR("Invalid LlamaLaunch.EffectiveParams in {}: {}", file_path, params_error);
                return -2;
            }
        }
        mmdit_model_ = config.value("MMDitModel", "");
        vae_model_ = config.value("VaeModel", "");
        if (!model_path_.empty()) {
            const std::filesystem::path p(model_path_);
            model_name_ = p.stem().string(); // 去掉扩展名
            // ===== device =====
            device_num_ = estimate_device_num(model_name_);
        } else if (!clip_model_.empty()) {
            model_name_ = "sd3";
            is_llm_ = false;
        } else {
            model_name_.clear();
        }
        inspect_main_gguf_metadata();
        infer_service_kind_from_name();
        if (!mm_proj_path_.empty()) {
            set_mm_proj_path(mm_proj_path_, true);
        }
        is_llm_ = service_kind_ != "sd";
        if (model_type_.empty()) {
            model_type_ = !model_family_.empty() ? model_family_ : service_kind_;
        }
        model_key_ = build_model_key(*this);
        if (estimated_mem_mb_ <= 0 && !model_path_.empty()) {
            estimated_mem_mb_ = estimate_mem_from_gguf(model_path_, context_size_, batch_num_);
        }
    } catch (const nlohmann::json::exception& ex) {
        LOG_ERROR("JSON Error: {}", ex.what());
        return -2;
    } catch (const std::exception& ex) {
        LOG_ERROR("General Error: {}", ex.what());
        return -1;
    }

    return 0;
}

int HLIEModelInfo::parse_from_gguf_filename(const std::string& file_path) {
    const int result = parse_identity_from_gguf_filename(file_path);
    if (result != 0) {
        return result;
    }
    inspect_main_gguf_metadata();
    infer_service_kind_from_name();
    model_type_ = !model_family_.empty() ? model_family_ : service_kind_;
    is_llm_ = service_kind_ != "sd";
    model_key_ = build_model_key(*this);
    estimated_mem_mb_ = estimate_mem_from_gguf(model_path_, context_size_, batch_num_);
    return 0;
}

void HLIEModelInfo::inspect_main_gguf_metadata() {
    const auto filename_family = HLIEModelFamilyRegistry::detect_projector_family(model_name_);
    const auto metadata = read_gguf_metadata(model_path_);
    if (!metadata.valid) {
        if (!filename_family.empty()) {
            model_family_ = filename_family;
            is_multi_model_ = true;
        }
        return;
    }

    architecture_ = to_lower(metadata.architecture);
    auto metadata_family = HLIEModelFamilyRegistry::detect_metadata_family(
        metadata.architecture, metadata.projector_type, metadata.vision_projector_type, metadata.audio_projector_type);
    if (metadata_family.empty()) {
        metadata_family = HLIEModelFamilyRegistry::detect_metadata_family(metadata.tokenizer_model);
    }

    if (!filename_family.empty()) {
        const bool gemma_conflict = (!metadata_family.empty() && metadata_family != filename_family) &&
            (metadata_family == HLIEModelFamilyRegistry::GEMMA4 || filename_family == HLIEModelFamilyRegistry::GEMMA4);
        if (gemma_conflict) {
            LOG_WARN("GGUF architecture {} conflicts with filename family {} for {}; using metadata family",
                     metadata_family, filename_family, model_path_);
            model_family_ = metadata_family;
        } else {
            // Filename families can be a more specific product name than the
            // underlying architecture (for example qwen3.6 built on qwen35).
            model_family_ = filename_family;
        }
    } else if (!metadata_family.empty()) {
        model_family_ = metadata_family;
    }

    if (!HLIEModelFamilyRegistry::detect_projector_family(model_family_).empty()) {
        is_multi_model_ = true;
    }
}

void HLIEModelInfo::inspect_mmproj_gguf_metadata() {
    const auto metadata = read_gguf_metadata(mm_proj_path_);
    if (!metadata.valid) {
        if (vl_spec_ != "unknown") {
            capabilities_.vision = true;
        }
        if (model_family_.find("asr") != std::string::npos || model_family_.find("aligner") != std::string::npos) {
            capabilities_.audio = true;
        }
        return;
    }

    capabilities_.vision = metadata.has_vision_encoder || !metadata.vision_projector_type.empty();
    capabilities_.audio = metadata.has_audio_encoder || !metadata.audio_projector_type.empty();
    if (!capabilities_.vision && !capabilities_.audio && !metadata.projector_type.empty()) {
        if (model_family_.find("asr") != std::string::npos || model_family_.find("aligner") != std::string::npos) {
            capabilities_.audio = true;
        } else {
            capabilities_.vision = true;
        }
    }

    if (model_family_.empty()) {
        model_family_ = HLIEModelFamilyRegistry::detect_metadata_family(
            metadata.architecture, metadata.projector_type, metadata.vision_projector_type, metadata.audio_projector_type);
    }
}

void HLIEModelInfo::set_mm_proj_path(const std::string& mm_proj_path, const bool inspect_metadata) {
    mm_proj_path_ = mm_proj_path;
    vl_spec_ = mm_proj_path_.empty() ? std::string{} : parse_mmproj_vl_spec(mm_proj_path_);
    if (mm_proj_path_.empty()) {
        capabilities_.vision = false;
        capabilities_.audio = false;
        infer_role_from_name();
        return;
    }

    is_multi_model_ = true;
    if (inspect_metadata) {
        inspect_mmproj_gguf_metadata();
    }
    infer_role_from_name();
    model_key_ = build_model_key(*this);
}

void HLIEModelInfo::ScanLocalModelsFromDir(std::vector<HLIEModelInfo>& result, const std::string& dir) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || ec) {
        return;
    }

    std::vector<std::filesystem::path> llm_files;
    std::vector<std::filesystem::path> mmproj_main_files;
    std::vector<std::filesystem::path> mmproj_files;

    // 分类
    for (std::filesystem::directory_iterator it(dir, ec), end; it != end && !ec; it.increment(ec)) {
        const auto& entry = *it;
        if (!entry.is_regular_file(ec) || ec)
            continue;

        const auto& path = entry.path();
        if (to_lower(path.extension().string()) != ".gguf")
            continue;

        if (const std::string name = to_lower(path.filename().string()); name.find("mmproj") != std::string::npos) {
            mmproj_files.push_back(path);
        } else if (HLIEModelFamilyRegistry::is_projector_main_model_name(name)) {
            mmproj_main_files.push_back(path);
        } else {
            llm_files.push_back(path);
        }
    }
    if (ec) {
        LOG_WARN("Model directory scan stopped early for {}, ex: {}", dir, ec.message());
    }

    const auto by_filename = [](const std::filesystem::path& lhs, const std::filesystem::path& rhs) {
        return to_lower(lhs.filename().string()) < to_lower(rhs.filename().string());
    };
    std::sort(llm_files.begin(), llm_files.end(), by_filename);
    std::sort(mmproj_main_files.begin(), mmproj_main_files.end(), by_filename);
    std::sort(mmproj_files.begin(), mmproj_files.end(), by_filename);

    // mmproj 索引
    std::unordered_map<std::string, std::vector<std::filesystem::path>> mmproj_index;

    for (auto& p : mmproj_files) {
        const std::string key = HLIEModelFamilyRegistry::build_projector_match_key(p.filename().string());
        if (!key.empty())
            mmproj_index[key].push_back(p);
    }

    // VL 主模型 × mmproj
    for (auto& p : mmproj_main_files) {
        const std::string filename = p.filename().string();
        const std::string key = HLIEModelFamilyRegistry::build_projector_match_key(filename);

        if (key.empty()) { // 名称不规范，直接用最简陋的方式构建info
            LOG_WARN("{} parse info failed, use basic path only ",filename);
            if (HLIEModelFamilyRegistry::supports_standalone(filename)) {
                HLIEModelInfo base;
                if (base.parse_from_gguf_filename(p.string()) == 0) {
                    base.is_multi_model_ = true;
                    base.set_mm_proj_path({}, false);
                    result.push_back(base);
                }
            }
            continue;
        }

        if (HLIEModelFamilyRegistry::supports_standalone(filename) && mmproj_index.find(key) == mmproj_index.end()) {
            HLIEModelInfo base;
            if (base.parse_from_gguf_filename(p.string()) == 0) {
                base.is_multi_model_ = true;
                base.set_mm_proj_path({}, false);
                result.push_back(base);
            }
            continue;
        }

        auto it = mmproj_index.find(key);
        if (it == mmproj_index.end()) {
            LOG_WARN("No matching mmproj found for {}", filename);
            continue; // 没 mmproj，直接跳过（对齐你 C#）

        }

        HLIEModelInfo base;
        if (base.parse_from_gguf_filename(p.string()) != 0)
            continue;

        base.is_multi_model_ = true;

        for (auto& mmproj : it->second) {
            HLIEModelInfo model = base;
            model.set_mm_proj_path(mmproj.string(), true);
            result.push_back(model);
        }
    }

    // 普通 LLM
    for (auto& p : llm_files) {
        if (HLIEModelInfo model; model.parse_from_gguf_filename(p.string()) == 0) {
            model.is_multi_model_ = false;
            model.set_mm_proj_path({}, false);
            result.push_back(model);
        } else {
            LOG_WARN("{} parse info failed, use basic path only ",p.string());
            HLIEModelInfo base;
            base.is_multi_model_ = false;
            base.model_path_ = p.string();
            base.model_name_ = p.stem().string();
            base.set_mm_proj_path({}, false);
            result.push_back(base);
        }
    }
}

int HLIEModelInfo::estimate_mem(const std::string& model_name) {
    if (model_name.empty()) {
        return 0;
    }

    if (const int empirical_mem = estimate_empirical_mem(model_name); empirical_mem > 0) {
        return empirical_mem;
    }

    const std::string name = to_lower(model_name);

    if (name.find("qwen2.5") != std::string::npos) {
        return 8192;
    }

    if (name.find("qwen3") != std::string::npos) {
        return 10240;
    }

    if (name.find("gte") != std::string::npos || name.find("bge") != std::string::npos) {
        return 1536;
    }

    if (name.find("whisper") != std::string::npos) {
        return 2048;
    }

    return 4096;
}

int HLIEModelInfo::estimate_empirical_mem(const std::string& model_name) {
    if (model_name.empty()) {
        return 0;
    }

    const auto path = memory_estimates_path();
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_WARN("memory estimate rules not found: {}", path.string());
        return 0;
    }

    try {
        nlohmann::json config;
        file >> config;
        const std::string name = to_lower(model_name);
        const int chips = estimate_device_num(name);
        std::smatch param_match;
        std::string model_params;
        if (std::regex_search(name, param_match, kParamRegex)) {
            model_params = to_lower(param_match.str());
            model_params.erase(std::remove_if(model_params.begin(), model_params.end(),
                                              [](const unsigned char c) { return std::isspace(c); }),
                               model_params.end());
        }

        for (const auto& rule : config.value("rules", nlohmann::json::array())) {
            const std::string family = to_lower(rule.value("model_family", ""));
            std::string params = to_lower(rule.value("params", ""));
            params.erase(std::remove_if(params.begin(), params.end(), [](const unsigned char c) { return std::isspace(c); }),
                         params.end());
            const int rule_chips = rule.value("chips", 1);
            const double per_device_mb = rule.value("memory_per_device_mb", 0.0);
            if (family.empty() || params.empty() || per_device_mb <= 0 || rule_chips != chips) {
                continue;
            }
            if (name.find(family) == std::string::npos || model_params != params) {
                continue;
            }

            const int total_mb = static_cast<int>(std::ceil(per_device_mb * rule_chips));
            LOG_INFO("empirical memory estimate for {} matched {} {} {}chips: {}MB per device, {}MB total",
                     model_name, family, params, rule_chips, per_device_mb, total_mb);
            return total_mb;
        }
    } catch (const std::exception& ex) {
        LOG_WARN("failed to parse memory estimate rules {}, ex: {}", path.string(), ex.what());
    }

    return 0;
}

int HLIEModelInfo::estimate_mem_from_gguf(const std::string& model_path, const int context_size, const int batch_num) {
    if (model_path.empty()) {
        return 0;
    }

    ggml_context* ctx_data = nullptr;
    gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };

    gguf_context* ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        LOG_WARN("failed to init gguf for memory estimate: {}", model_path);
        return 0;
    }

    try {
        const std::string prefix = get_arch_prefix(ctx);

        std::string size_label;
        HLIEGgufUtil::get_key<std::string>(ctx, "general.size_label", size_label, false);

        uint32_t file_type = 0;
        HLIEGgufUtil::get_key<uint32_t>(ctx, "general.file_type", file_type, false);

        uint64_t tensor_bytes = 0;
        const int64_t n_tensors = gguf_get_n_tensors(ctx);
        for (int64_t i = 0; i < n_tensors; ++i) {
            tensor_bytes += gguf_get_tensor_size(ctx, i);
        }

        const uint64_t param_count = parse_param_count(size_label);
        const double weight_bits = weight_bits_from_file_type(file_type);
        const double param_weight_bytes = static_cast<double>(param_count) * weight_bits / 8.0;
        const double base_weight_bytes = param_weight_bytes > 0 ? param_weight_bytes : static_cast<double>(tensor_bytes);

        uint32_t block_count = 0;
        uint32_t ctx_train = 0;
        uint32_t head_count_kv = 0;
        uint32_t key_length = 0;
        uint32_t value_length = 0;
        if (!prefix.empty()) {
            get_u32_key(ctx, prefix + "block_count", block_count);
            get_u32_key(ctx, prefix + "context_length", ctx_train);
            get_u32_key(ctx, prefix + "attention.head_count_kv", head_count_kv);
            get_u32_key(ctx, prefix + "attention.key_length", key_length);
            get_u32_key(ctx, prefix + "attention.value_length", value_length);
        }

        const uint32_t runtime_ctx = context_size > 0 ? static_cast<uint32_t>(context_size) : ctx_train;
        const uint32_t runtime_batch = batch_num > 0 ? static_cast<uint32_t>(batch_num) : 1;
        const double kv_cache_bytes = static_cast<double>(runtime_batch) * runtime_ctx * block_count * head_count_kv *
            (key_length + value_length) * 2.0;

        constexpr double kLoadUpperFactor = 1.9;
        const double estimate_bytes = base_weight_bytes * kLoadUpperFactor;
        const int estimate_mb = bytes_to_mb_ceil(estimate_bytes);

        LOG_INFO("GGUF memory estimate for {}: size_label={}, file_type={}, weight_bits={}, param_weight={}MB, tensor={}MB, kv_cache={}MB, load_upper={}MB",
                 model_path, size_label, file_type, weight_bits, bytes_to_mb_ceil(param_weight_bytes), bytes_to_mb_ceil(tensor_bytes),
                 bytes_to_mb_ceil(kv_cache_bytes), estimate_mb);

        gguf_free(ctx);
        return estimate_mb;
    } catch (const std::exception& ex) {
        LOG_WARN("failed to estimate memory from gguf {}, ex: {}", model_path, ex.what());
        gguf_free(ctx);
        return 0;
    }
}

int HLIEModelInfo::get_estimated_mem() const {
    if (const int empirical_mem = estimate_empirical_mem(model_name_); empirical_mem > 0) {
        return empirical_mem;
    }
    return estimated_mem_mb_ > 0 ? estimated_mem_mb_ : estimate_mem(model_name_);
}

void HLIEModelInfo::build_llm(const std::string& model_path, const std::string& model_type, const int context_size, const int cores,
                              const int batch_num, const int xh_version, const bool is_multi_model, const std::string& params,
                              const std::string& mm_proj_path, const std::string& devices) {
    model_path_ = model_path;
    model_type_ = to_lower(model_type);
    service_kind_ = is_known_service_kind(model_type_) ? model_type_ : "llm";
    if (!model_type_.empty() && !is_known_service_kind(model_type_)) {
        model_family_ = model_type_;
    }
    context_size_ = context_size;
    cores_ = cores;
    batch_num_ = batch_num;
    xh_version_ = xh_version;
    is_multi_model_ = is_multi_model;
    params_ = params;
    devices_ = devices;
    if (!model_path_.empty()) {
        model_name_ = std::filesystem::path(model_path_).stem().string();
        inspect_main_gguf_metadata();
    }
    set_mm_proj_path(mm_proj_path, true);
    infer_service_kind_from_name();
    is_llm_ = true;
    estimated_mem_mb_ = estimate_mem_from_gguf(model_path_, context_size_, batch_num_);
}

int HLIEModelInfo::to_json_file(const std::string& file_path, const bool launch_manifest) const {
    nlohmann::json config = {
        {"ModelPath", model_path_}, {"ModelType", model_type_},  {"ContextSize", context_size_}, {"Cores", cores_},
        {"BatchNum", batch_num_},   {"XHVersion", xh_version_},  {"IsMM", is_multi_model_},      {"Params", params_},
        {"MMProj", mm_proj_path_},  {"Version", model_version_}, {"ClipModel", clip_model_},     {"ClipLModel", clip_l_model_},
        {"T5Model", t5_model_},     {"Devices", devices_},       {"MMDitModel", mmdit_model_},   {"VaeModel", vae_model_},
        {"EstimatedMemMB", estimated_mem_mb_}, {"ServiceKind", service_kind_}, {"ModelFamily", model_family_},
        {"Architecture", architecture_}, {"Role", role_to_string(role_)},
        {"Capabilities", {{"text", capabilities_.text}, {"vision", capabilities_.vision}, {"audio", capabilities_.audio}}}};
    if (has_effective_llama_params_) {
        config["LlamaLaunch"] = {
            {"EffectiveParams", effective_llama_params_.to_json()},
            {"MatchedPresets", matched_llama_presets_},
            {"Sources", llama_param_sources_},
            {"ParamsHash", effective_llama_params_hash_},
        };
    }
    if (launch_manifest) {
        config["Schema"] = "hlie.model-launch";
        config["SchemaVersion"] = 1;
    }

    try {
        std::ofstream file(file_path);
        if (!file.is_open()) {
            LOG_ERROR("Could not open file for writing: {} ", file_path);
            return -1;
        }
        file << config.dump(4); // 缩进 4 格，方便阅读
    } catch (const std::exception& ex) {
        LOG_ERROR("Error writing JSON file {}: {}", file_path, ex.what());
        return -1;
    }

    return 0;
}

HLIE_NAMESPACE_END

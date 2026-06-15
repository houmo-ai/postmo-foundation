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
#include "json.hpp"
#include "log.h"


HLIE_NAMESPACE_BEGIN

namespace {
    std::string to_lower(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) { return std::tolower(c); });
        return result;
    }

    const std::regex kVersionRegex(R"(v\d+\.\d+\.\d+)", std::regex::icase);
    const std::regex kXhRegex(R"(xh\s*(\d+))", std::regex::icase);
    const std::regex kParamRegex(R"((\d+(?:\.\d+)?)\s*[bBmM])", std::regex::icase);
    const std::regex kContextRegex(R"((\d+)\s*[kK])");
    const std::regex kBatchRegex(R"(b(\d+))", std::regex::icase);
    const std::regex kCoreRegex(R"((\d+)\s*core)", std::regex::icase);
    const std::regex kChipRegex(R"((\d+)\s*chips?)", std::regex::icase);
    const std::regex kDeviceRegex(R"((\d+)d)", std::regex::icase);
    const std::regex kSizeLabelRegex(R"((\d+(?:\.\d+)?)\s*([bBmM]))", std::regex::icase);
    const std::regex kVlFamilyRegex(R"(qwen2\.5-vl|qwen3-vl|qwen3\.5|qwen3\.6)", std::regex::icase);
    const std::regex kMmprojFamilyRegex(
        R"(qwen2\.5-vl|qwen3-vl|qwen3\.5|qwen3\.6|qwen3[-_]asr|qwen3[-_]forcedaligner|glm-ocr)",
        std::regex::icase);
    const std::regex kOptionalMmprojVlFamilyRegex(R"(qwen3\.5|qwen3\.6)", std::regex::icase);
    const std::regex kVlResRegex(R"(vit_(\d+)(?:[_x](\d+))?)", std::regex::icase);
    const std::regex kBuildTimeRegex(R"((\d{8}|\d{14}))");

    std::string build_mmproj_match_key(const std::string& name) {
        std::smatch m;

        if (!std::regex_search(name, m, kMmprojFamilyRegex))
            return {};
        std::string model_family = to_lower(m.str());
        std::replace(model_family.begin(), model_family.end(), '_', '-');

        if (!std::regex_search(name, m, kXhRegex))
            return {};
        const std::string xh = to_lower(m.str());

        if (!std::regex_search(name, m, kParamRegex))
            return {};
        const std::string param = to_lower(m.str());

        if (!std::regex_search(name, m, kVersionRegex))
            return {};
        const std::string version = to_lower(m.str());

        std::string build_time;
        if (std::regex_search(name, m, kBuildTimeRegex)) {
            build_time = m.str(); // 20251228 or 20251228103045
        }

        // qwen3.5/qwen3.6 mmproj files can be built at a different time from
        // their base model, so build time is not part of their match key.
        if (!std::regex_search(name, kOptionalMmprojVlFamilyRegex) && !build_time.empty()) {
            return xh + "_" + model_family + "_" + param + "_" + version + "_" + build_time;
        }

        // 兼容老模型（没 build time）
        return xh + "_" + model_family + "_" + param + "_" + version;
    }

    bool is_optional_mmproj_vl_model(const std::string& name) {
        return std::regex_search(name, kOptionalMmprojVlFamilyRegex);
    }

    bool is_mmproj_main_model_name(const std::string& name) {
        return name.find("mmproj") == std::string::npos && std::regex_search(name, kMmprojFamilyRegex);
    }

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
        const auto config_path = Singleton<HLIEConfigService>::instance()->get_config_path();
        if (config_path.empty()) {
            return "memory_estimates.json";
        }
        return std::filesystem::path(config_path).parent_path() / "memory_estimates.json";
    }

} // namespace

int HLIEModelInfo::parse_from_file(const std::string& file_path) {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            LOG_ERROR("Error: Could not open file: {}", file_path);
            return -3;
        }

        nlohmann::json config;
        file >> config;

        model_path_ = config.value("ModelPath", "");
        model_type_ = config.value("ModelType", "");
        model_type_ = to_lower(model_type_);
        context_size_ = config.value("ContextSize", 0);
        cores_ = config.value("Cores", 0);
        batch_num_ = config.value("BatchNum", 1);
        xh_version_ = config.value("XHVersion", 2);
        is_multi_model_ = config.value("IsMM", false);
        if (std::regex_search(model_type_, kMmprojFamilyRegex)) {
            is_multi_model_ = true;
        }
        params_ = config.value("Params", "");
        mm_proj_path_ = config.value("MMProj", "");
        if (!mm_proj_path_.empty()) {
            vl_spec_ = parse_mmproj_vl_spec(mm_proj_path_);
        }
        model_version_ = config.value("Version", "");
        estimated_mem_mb_ = config.value("EstimatedMemMB", 0);
        clip_model_ = config.value("ClipModel", "");
        clip_l_model_ = config.value("ClipLModel", "");
        t5_model_ = config.value("T5Model", "");
        devices_ = config.value("Devices", "");
        mmdit_model_ = config.value("MMDitModel", "");
        vae_model_ = config.value("VaeModel", "");
        is_llm_ = true;
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
    std::filesystem::path p(file_path);
    const std::string name = to_lower(p.filename().string());

    model_path_ = file_path;

    auto find = [&](const std::regex& r, std::smatch& m) { return std::regex_search(name, m, r); };

    std::smatch m;

    // ===== xh（必需）=====
    // if (!std::regex_search(name, m, kXhRegex)) {
    //     return -1;
    // }

    // xh_version_ = std::stoi(m[1].str());

    // ===== version=====
    if (!find(kVersionRegex, m)) {
        return -2;
    }

    model_version_ = m.str();

    // ===== build time=====
    if (find(kBuildTimeRegex, m))
        build_time_ = m.str();
    else
        build_time_.clear();

    // ===== param=====
    if (find(kParamRegex, m))
        params_ = m.str();
    else
        params_.clear();

    // ===== context=====
    if (find(kContextRegex, m))
        context_size_ = std::stoi(m[1]) * 1024;
    else
        context_size_ = 6400000;

    // ===== batch=====
    if (find(kBatchRegex, m))
        batch_num_ = std::stoi(m[1]);
    else
        batch_num_ = 1;

    // ===== core=====
    if (find(kCoreRegex, m))
        cores_ = std::stoi(m[1]);
    else
        cores_ = 2;

    // ===== device =====
    // 1chip / 2chips
    device_num_ = estimate_device_num(name);

    // ===== VL family 判断 =====
    if (find(kMmprojFamilyRegex, m)) {
        model_type_ = m.str();
        is_multi_model_ = true;
    } else {
        // 非 VL：尽量提 family
        if (const auto pos = name.find("qwen"); pos != std::string::npos) {
            const auto end = name.find_first_of("-_", pos);
            model_type_ = name.substr(pos, end - pos);
        } else {
            model_type_.clear();
        }
        is_multi_model_ = false;
    }

    // ===== mmproj 的 vl_spec（如果是 mmproj 文件）=====
    if (name.find("mmproj") != std::string::npos) {
        if (find(kVlResRegex, m)) {
            if (m[2].matched)
                vl_spec_ = "vit_" + m[1].str() + "_" + m[2].str();
            else
                vl_spec_ = "vit_" + m[1].str();
        } else {
            vl_spec_ = "unknown";
        }
    } else {
        vl_spec_.clear();
    }

    if (!model_path_.empty()) {
        const std::filesystem::path model_path_tmp(model_path_);
        model_name_ = model_path_tmp.filename().stem().string();
    } else {
        model_name_.clear();
    }

    is_llm_ = true;
    estimated_mem_mb_ = estimate_mem_from_gguf(model_path_, context_size_, batch_num_);
    return 0;
}

std::string HLIEModelInfo::parse_mmproj_vl_spec(const std::string& name) {
    std::smatch m;
    if (std::regex_search(name, m, kVlResRegex)) {
        if (m[2].matched)
            return "vit_" + m[1].str() + "_" + m[2].str();
        return "vit_" + m[1].str();
    }
    return "unknown";
}

std::string HLIEModelInfo::build_model_key(const HLIEModelInfo& model_info) {
    if (model_info.get_mm_proj_path().empty()) {
        return model_info.get_model_name();
    }

    if (!model_info.vl_spec_.empty() && model_info.vl_spec_ != "unknown") {
        return model_info.get_model_name() + "_" + model_info.vl_spec_;
    }

    const std::filesystem::path mmproj_path(model_info.get_mm_proj_path());
    return model_info.get_model_name() + "_mmproj_" + mmproj_path.stem().string();
}

bool HLIEModelInfo::is_embedding_model() const {
    return HlieUtil::contains_ignore_case(model_name_, "bge") || HlieUtil::contains_ignore_case(model_name_, "gte") ||
        HlieUtil::contains_ignore_case(model_name_, "embedding");
}

bool HLIEModelInfo::is_rerank_model() const {
    return HlieUtil::contains_ignore_case(model_name_, "rerank");
}

void HLIEModelInfo::ScanLocalModelsFromDir(std::vector<HLIEModelInfo>& result, const std::string& dir) {
    if (!std::filesystem::exists(dir)) {
        return;
    }

    std::vector<std::filesystem::path> llm_files;
    std::vector<std::filesystem::path> mmproj_main_files;
    std::vector<std::filesystem::path> mmproj_files;

    // 分类
    for (auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file())
            continue;

        const auto& path = entry.path();
        if (path.extension() != ".gguf")
            continue;

        if (const std::string name = to_lower(path.filename().string()); name.find("mmproj") != std::string::npos) {
            mmproj_files.push_back(path);
        } else if (is_mmproj_main_model_name(name)) {
            mmproj_main_files.push_back(path);
        } else {
            llm_files.push_back(path);
        }
    }

    // mmproj 索引
    std::unordered_map<std::string, std::vector<std::filesystem::path>> mmproj_index;

    for (auto& p : mmproj_files) {
        const std::string key = build_mmproj_match_key(p.filename().string());
        if (!key.empty())
            mmproj_index[key].push_back(p);
    }

    // VL 主模型 × mmproj
    for (auto& p : mmproj_main_files) {
        const std::string filename = p.filename().string();
        const std::string key = build_mmproj_match_key(filename);

        if (key.empty()) { // 名称不规范，直接用最简陋的方式构建info
            LOG_WARN("{} parse info failed, use basic path only ",filename);
            if (is_optional_mmproj_vl_model(filename)) {
                HLIEModelInfo base;
                if (base.parse_from_gguf_filename(p.string()) == 0) {
                    base.is_multi_model_ = true;
                    base.mm_proj_path_.clear();
                    base.vl_spec_.clear();
                    result.push_back(base);
                }
            }
            continue;
        }

        if (is_optional_mmproj_vl_model(filename) && mmproj_index.find(key) == mmproj_index.end()) {
            HLIEModelInfo base;
            if (base.parse_from_gguf_filename(p.string()) == 0) {
                base.is_multi_model_ = true;
                base.mm_proj_path_.clear();
                base.vl_spec_.clear();
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
            model.mm_proj_path_ = mmproj.string();
            model.vl_spec_ = parse_mmproj_vl_spec(mmproj.filename().string());
            model.model_key_ = build_model_key(model);
            result.push_back(model);
        }
    }

    // 普通 LLM
    for (auto& p : llm_files) {
        if (HLIEModelInfo model; model.parse_from_gguf_filename(p.string()) == 0) {
            model.is_multi_model_ = false;
            model.mm_proj_path_.clear();
            model.vl_spec_.clear();
            result.push_back(model);
        } else {
            LOG_WARN("{} parse info failed, use basic path only ",p.string());
            HLIEModelInfo base;
            base.is_multi_model_ = false;
            base.model_path_ = p.string();
            base.model_name_ = p.stem().string();
            base.mm_proj_path_.clear();
            base.vl_spec_.clear();
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

int HLIEModelInfo::estimate_device_num(const std::string& model_name) {
    std::smatch match;
    if (std::regex_search(model_name, match, kChipRegex)) {
        return std::max(1, std::stoi(match[1].str()));
    }
    if (std::regex_search(model_name, match, kDeviceRegex)) {
        return std::max(1, std::stoi(match[1].str()));
    }
    return 1;
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
    context_size_ = context_size;
    cores_ = cores;
    batch_num_ = batch_num;
    xh_version_ = xh_version;
    is_multi_model_ = is_multi_model;
    params_ = params;
    mm_proj_path_ = mm_proj_path;
    devices_ = devices;
    estimated_mem_mb_ = estimate_mem_from_gguf(model_path_, context_size_, batch_num_);
}

void HLIEModelInfo::build_sd(const std::string& model_type, const std::string& vae_model, const std::string& t5_model,
                             const std::string& clip_model, const std::string& clip_l_model, const std::string& mmdit_model,
                             const std::string& devices) {
    model_type_ = to_lower(model_type);
    vae_model_ = vae_model;
    t5_model_ = t5_model;
    clip_model_ = clip_model;
    clip_l_model_ = clip_l_model;
    mmdit_model_ = mmdit_model;
    devices_ = devices;
}

int HLIEModelInfo::to_json_file(const std::string& file_path) const {
    const nlohmann::json config = {
        {"ModelPath", model_path_}, {"ModelType", model_type_},  {"ContextSize", context_size_}, {"Cores", cores_},
        {"BatchNum", batch_num_},   {"XHVersion", xh_version_},  {"IsMM", is_multi_model_},      {"Params", params_},
        {"MMProj", mm_proj_path_},  {"Version", model_version_}, {"ClipModel", clip_model_},     {"ClipLModel", clip_l_model_},
        {"T5Model", t5_model_},     {"Devices", devices_},       {"MMDitModel", mmdit_model_},   {"VaeModel", vae_model_},
        {"EstimatedMemMB", estimated_mem_mb_}};

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

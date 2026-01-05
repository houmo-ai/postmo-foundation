#include "hlie_app/hlie_model_info.h"

#include <fstream>
#include <regex>

#include "hlie_base/hlie_logger.h"
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
    const std::regex kParamRegex(R"((\d+)\s*[bBmM])", std::regex::icase);
    const std::regex kContextRegex(R"((\d+)\s*[kK])");
    const std::regex kBatchRegex(R"(b(\d+))", std::regex::icase);
    const std::regex kCoreRegex(R"((\d+)\s*core)", std::regex::icase);
    const std::regex kVlFamilyRegex(R"(qwen2\.5-vl|qwen3-vl)", std::regex::icase);
    const std::regex kVlResRegex(R"(vit_(\d+)(?:[_x](\d+))?)", std::regex::icase);
    const std::regex kBuildTimeRegex(R"((\d{8}|\d{14}))");

    std::string build_vl_match_key(const std::string& name) {
        std::smatch m;

        if (!std::regex_search(name, m, kVlFamilyRegex))
            return {};
        const std::string vl_family = to_lower(m.str());

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

        // xh2_qwen3-vl_4b_v0.7.0_20251228
        if (!build_time.empty()) {
            return xh + "_" + vl_family + "_" + param + "_" + version + "_" + build_time;
        }

        // 兼容老模型（没 build time）
        return xh + "_" + vl_family + "_" + param + "_" + version;
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
        xh_version_ = config.value("XHVersion", 1);
        is_multi_model_ = config.value("IsMM", false);
        params_ = config.value("Params", "");
        mm_proj_path_ = config.value("MMProj", "");
        model_version_ = config.value("Version", "");
        clip_model_ = config.value("ClipModel", "");
        clip_l_model_ = config.value("ClipLModel", "");
        t5_model_ = config.value("T5Model", "");
        devices_ = config.value("Devices", "");
        mmdit_model_ = config.value("MMDitModel", "");
        vae_model_ = config.value("VaeModel", "");
        is_llm_ = true;
        if (!model_path_.empty()) {
            const std::filesystem::path p(model_path_);
            model_name_ = p.stem().string();   // 去掉扩展名
        } else if (!clip_model_.empty()) {
            model_name_ = "sd3";
            is_llm_ = false;
        } else {
            model_name_.clear();
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
    if (!std::regex_search(name, m, kXhRegex)) {
        return -1;
    }

    xh_version_ = std::stoi(m[1].str());

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

    // ===== device=====
    device_num_ = 1;
    if (const std::regex device_regex(R"((\d+)d)", std::regex::icase); std::regex_search(name, m, device_regex)) {
        device_num_ = std::stoi(m[1].str());
    }

    // ===== VL family 判断 =====
    if (find(kVlFamilyRegex, m)) {
        model_type_ = m.str(); // qwen2.5-vl / qwen3-vl
        is_multi_model_ = true;
    } else {
        // 非 VL：尽量提 family
        auto pos = name.find("qwen");
        if (pos != std::string::npos) {
            auto end = name.find_first_of("-_", pos);
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

void HLIEModelInfo::ScanLocalModelsFromDir(std::vector<HLIEModelInfo>& result, const std::string& dir) {
    if (!std::filesystem::exists(dir)) {
        return;
    }

    std::vector<std::filesystem::path> llm_files;
    std::vector<std::filesystem::path> vl_main_files;
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
        } else if (name.find("vl") != std::string::npos) {
            vl_main_files.push_back(path);
        } else {
            llm_files.push_back(path);
        }
    }

    // mmproj 索引
    std::unordered_map<std::string, std::vector<std::filesystem::path>> mmproj_index;

    for (auto& p : mmproj_files) {
        const std::string key = build_vl_match_key(p.filename().string());
        if (!key.empty())
            mmproj_index[key].push_back(p);
    }

    // VL 主模型 × mmproj
    for (auto& p : vl_main_files) {
        const std::string filename = p.filename().string();
        const std::string key = build_vl_match_key(filename);

        if (key.empty())
            continue;

        auto it = mmproj_index.find(key);
        if (it == mmproj_index.end())
            continue; // 没 mmproj，直接跳过（对齐你 C#）

        HLIEModelInfo base;
        if (base.parse_from_gguf_filename(p.string()) != 0)
            continue;

        base.is_multi_model_ = true;

        for (auto& mmproj : it->second) {
            HLIEModelInfo model = base;
            model.mm_proj_path_ = mmproj.string();
            model.vl_spec_ = parse_mmproj_vl_spec(mmproj.filename().string());
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
        }
    }
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
        {"T5Model", t5_model_},     {"Devices", devices_},       {"MMDitModel", mmdit_model_},   {"VaeModel", vae_model_}};

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

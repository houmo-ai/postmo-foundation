#include "hlie_app/hlie_model_info.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <regex>

#include "hlie_app/hlie_model_family_registry.h"

HLIE_NAMESPACE_BEGIN

namespace {

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool contains_ignore_case(const std::string& value, const std::string& needle) {
    return to_lower(value).find(to_lower(needle)) != std::string::npos;
}

bool is_known_service_kind(const std::string& value) {
    return value == "llm" || value == "sd" || value == "whisper" || value == "cosyvoice" || value == "tts" ||
        value == "qwen3_tts" || value == "document";
}

const std::regex kChipRegex(R"((\d+)\s*chips?)", std::regex::icase);
const std::regex kDeviceRegex(R"((\d+)d)", std::regex::icase);
const std::regex kVlResRegex(R"(vit_(\d+)(?:[_x](\d+))?)", std::regex::icase);
const std::regex kVersionRegex(R"(v\d+\.\d+\.\d+)", std::regex::icase);
const std::regex kParamRegex(R"((\d+(?:\.\d+)?)\s*[bBmM])", std::regex::icase);
const std::regex kContextRegex(R"((\d+)\s*[kK])");
const std::regex kBatchRegex(R"(b(\d+))", std::regex::icase);
const std::regex kCoreRegex(R"((\d+)\s*core)", std::regex::icase);
const std::regex kBuildTimeRegex(R"((\d{8}|\d{14}))");

} // namespace

int HLIEModelInfo::parse_identity_from_gguf_filename(const std::string& file_path) {
    const std::filesystem::path path(file_path);
    const std::string name = to_lower(path.filename().string());

    model_path_ = file_path;
    service_kind_ = "llm";
    model_type_.clear();
    model_family_.clear();
    architecture_.clear();
    capabilities_ = {};
    is_multi_model_ = false;
    mm_proj_path_.clear();
    estimated_mem_mb_ = 0;

    std::smatch match;
    model_version_ = std::regex_search(name, match, kVersionRegex) ? match.str() : std::string{};
    build_time_ = std::regex_search(name, match, kBuildTimeRegex) ? match.str() : std::string{};
    params_ = std::regex_search(name, match, kParamRegex) ? match.str() : std::string{};
    context_size_ = std::regex_search(name, match, kContextRegex) ? std::stoi(match[1].str()) * 1024 : 6400000;
    batch_num_ = std::regex_search(name, match, kBatchRegex) ? std::stoi(match[1].str()) : 1;
    cores_ = std::regex_search(name, match, kCoreRegex) ? std::stoi(match[1].str()) : 2;
    device_num_ = estimate_device_num(name);

    if (const auto family = HLIEModelFamilyRegistry::detect_projector_family(name); !family.empty()) {
        model_family_ = family;
        is_multi_model_ = true;
    } else if (const auto pos = name.find("qwen"); pos != std::string::npos) {
        const auto end = name.find_first_of("-_", pos);
        model_family_ = name.substr(pos, end - pos);
    }

    if (name.find("mmproj") != std::string::npos) {
        vl_spec_ = parse_mmproj_vl_spec(name);
    } else {
        vl_spec_.clear();
    }

    model_name_ = path.filename().stem().string();
    infer_service_kind_from_name();
    model_type_ = !model_family_.empty() ? model_family_ : service_kind_;
    is_llm_ = service_kind_ != "sd";
    model_key_ = build_model_key(*this);
    return 0;
}

void HLIEModelInfo::infer_service_kind_from_name() {
    const std::string name = to_lower(model_name_);
    if (name.find("qwen3-tts") != std::string::npos || name.find("qwen3_tts") != std::string::npos) {
        service_kind_ = "qwen3_tts";
    } else if (name.find("whisper") != std::string::npos) {
        service_kind_ = "whisper";
    } else if (name.find("cosyvoice") != std::string::npos) {
        service_kind_ = "cosyvoice";
    } else if (!clip_model_.empty() || name == "sd3") {
        service_kind_ = "sd";
    } else if (service_kind_.empty() || !is_known_service_kind(service_kind_)) {
        service_kind_ = "llm";
    }
}

void HLIEModelInfo::set_service_kind(const std::string& service_kind) {
    service_kind_ = to_lower(service_kind);
    if (service_kind_.empty()) {
        service_kind_ = "llm";
    }
    is_llm_ = service_kind_ != "sd";
}

std::string HLIEModelInfo::parse_mmproj_vl_spec(const std::string& name) {
    std::smatch match;
    if (!std::regex_search(name, match, kVlResRegex)) {
        return "unknown";
    }
    if (match[2].matched) {
        return "vit_" + match[1].str() + "_" + match[2].str();
    }
    return "vit_" + match[1].str();
}

std::string HLIEModelInfo::build_model_key(const HLIEModelInfo& model_info) {
    if (model_info.get_mm_proj_path().empty() || model_info.vl_spec_.empty() || model_info.vl_spec_ == "unknown") {
        return model_info.get_model_name();
    }
    return model_info.get_model_name() + "_" + model_info.vl_spec_;
}

bool HLIEModelInfo::is_embedding_model() const {
    return contains_ignore_case(model_name_, "bge") || contains_ignore_case(model_name_, "gte") ||
        contains_ignore_case(model_name_, "embedding");
}

bool HLIEModelInfo::is_rerank_model() const {
    return contains_ignore_case(model_name_, "rerank");
}

bool HLIEModelInfo::is_mtp() const {
    return contains_ignore_case(model_name_, "mtp");
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

void HLIEModelInfo::build_sd(const std::string& model_type, const std::string& vae_model, const std::string& t5_model,
                             const std::string& clip_model, const std::string& clip_l_model, const std::string& mmdit_model,
                             const std::string& devices) {
    model_type_ = to_lower(model_type);
    service_kind_ = "sd";
    model_family_ = "sd3";
    model_name_ = "sd3";
    model_key_ = model_name_;
    is_llm_ = false;
    vae_model_ = vae_model;
    t5_model_ = t5_model;
    clip_model_ = clip_model;
    clip_l_model_ = clip_l_model;
    mmdit_model_ = mmdit_model;
    devices_ = devices;
}

HLIE_NAMESPACE_END

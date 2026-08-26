#include "hlie_core/backend/hlie_model_family_registry.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <vector>

HLIE_NAMESPACE_BEGIN

namespace {

struct ModelFamilyRule {
    const char* canonical_name;
    std::regex filename_pattern;
    bool supports_standalone;
    bool match_build_time;
};

const std::regex kVersionRegex(R"(v\d+\.\d+\.\d+)", std::regex::icase);
const std::regex kXhRegex(R"(xh\s*(\d+))", std::regex::icase);
const std::regex kParamRegex(R"((\d+(?:\.\d+)?)\s*[bBmM])", std::regex::icase);
const std::regex kBuildTimeRegex(R"((\d{8}|\d{14}))");
const std::regex kActiveParamsRegex(R"((?:^|[-_])(a\d+(?:\.\d+)?b)(?:[-_.]|$))", std::regex::icase);
const std::regex kExpertSizeRegex(R"((?:^|[-_])(e\d+(?:\.\d+)?b)(?:[-_.]|$))", std::regex::icase);
const std::regex kMtpRegex(R"((?:^|[-_])(mtp)(?:[-_.]|$))", std::regex::icase);
const std::regex kQuantizationRegex(R"((?:^|[-_])(w\d+a\d+(?:h\d+)?)(?:[-_.]|$))", std::regex::icase);

std::string to_lower(const std::string& value) {
    std::string result = value;
    std::transform(result.begin(), result.end(), result.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return result;
}

std::string to_compact_lower(const std::string& value) {
    std::string result = to_lower(value);
    result.erase(std::remove_if(result.begin(), result.end(), [](const unsigned char c) {
        return std::isspace(c);
    }), result.end());
    return result;
}

std::string pairing_variant_signature(const std::string& name) {
    std::string signature;
    const auto append_match = [&](const std::regex& pattern) {
        std::smatch match;
        if (std::regex_search(name, match, pattern) && match.size() > 1) {
            signature += "_" + to_compact_lower(match[1].str());
        }
    };

    // Fixed order makes the signature independent of token order in a filename.
    append_match(kActiveParamsRegex);
    append_match(kExpertSizeRegex);
    append_match(kMtpRegex);
    append_match(kQuantizationRegex);
    return signature;
}

const std::vector<ModelFamilyRule>& model_family_rules() {
    // Order more specific names first. supports_standalone means the base GGUF
    // remains registrable when no companion projector is present.
    static const std::vector<ModelFamilyRule> rules{
        {"qwen2.5-vl", std::regex(R"(qwen2\.5[-_]vl)", std::regex::icase), false, true},
        {"qwen3-vl", std::regex(R"(qwen3[-_]vl)", std::regex::icase), false, true},
        {"qwen3.5", std::regex(R"(qwen3\.5)", std::regex::icase), true, false},
        {"qwen3.6", std::regex(R"(qwen3\.6)", std::regex::icase), true, false},
        {HLIEModelFamilyRegistry::GEMMA4,
         std::regex(R"(gemma(?:4|[-_ ]4))", std::regex::icase), true, false},
        {"copaw-flash", std::regex(R"(copaw[-_]flash)", std::regex::icase), false, true},
        {"glm-asr", std::regex(R"(glm[-_]asr)", std::regex::icase), false, true},
        {"qwen3-asr", std::regex(R"(qwen3[-_]asr)", std::regex::icase), false, true},
        {"qwen3-forcealigner", std::regex(R"(qwen3[-_](?:force|forced)aligner)", std::regex::icase), false, true},
        {"glm-ocr", std::regex(R"(glm[-_]ocr)", std::regex::icase), false, true},
    };
    return rules;
}

const ModelFamilyRule* find_rule(const std::string& name) {
    const auto& rules = model_family_rules();
    const auto it = std::find_if(rules.begin(), rules.end(), [&](const ModelFamilyRule& rule) {
        return std::regex_search(name, rule.filename_pattern);
    });
    return it == rules.end() ? nullptr : &*it;
}

std::string normalize_architecture(std::string architecture) {
    architecture = to_compact_lower(architecture);
    std::replace(architecture.begin(), architecture.end(), '_', '-');

    if (architecture == "qwen35" || architecture == "qwen35moe") {
        return "qwen3.5";
    }
    if (architecture == "qwen3vl" || architecture == "qwen3vlmoe") {
        return "qwen3-vl";
    }
    if (architecture == "qwen3moe" || architecture == "qwen3next") {
        return "qwen3";
    }
    if (architecture == "gemma-4") {
        return HLIEModelFamilyRegistry::GEMMA4;
    }
    return architecture;
}

std::string family_from_projector_type(const std::string& projector_type) {
    const std::string type = to_compact_lower(projector_type);
    if (type.find("gemma4") != std::string::npos) {
        return HLIEModelFamilyRegistry::GEMMA4;
    }
    if (type.find("qwen3") != std::string::npos) {
        return "qwen3-vl";
    }
    if (type.find("glma") != std::string::npos) {
        return "glm-asr";
    }
    if (type.find("glm") != std::string::npos) {
        return "glm-ocr";
    }
    return {};
}

} // namespace

std::string HLIEModelFamilyRegistry::detect_projector_family(const std::string& name) {
    const auto* rule = find_rule(name);
    return rule ? rule->canonical_name : std::string{};
}

std::string HLIEModelFamilyRegistry::detect_metadata_family(const std::string& architecture,
                                                            const std::string& projector_type,
                                                            const std::string& vision_projector_type,
                                                            const std::string& audio_projector_type) {
    const std::string normalized_arch = normalize_architecture(architecture);
    if (!normalized_arch.empty() && normalized_arch != "clip") {
        return normalized_arch;
    }

    for (const auto* type : {&vision_projector_type, &audio_projector_type, &projector_type}) {
        if (const auto family = family_from_projector_type(*type); !family.empty()) {
            return family;
        }
    }
    return {};
}

bool HLIEModelFamilyRegistry::supports_standalone(const std::string& name) {
    const auto* rule = find_rule(name);
    return rule && rule->supports_standalone;
}

bool HLIEModelFamilyRegistry::is_projector_main_model_name(const std::string& name) {
    return to_lower(name).find("mmproj") == std::string::npos && find_rule(name) != nullptr;
}

std::string HLIEModelFamilyRegistry::build_projector_match_key(const std::string& name) {
    const auto* rule = find_rule(name);
    if (!rule) {
        return {};
    }

    std::smatch match;
    if (!std::regex_search(name, match, kXhRegex)) {
        return {};
    }
    const std::string xh = to_compact_lower(match.str());

    if (!std::regex_search(name, match, kParamRegex)) {
        return {};
    }
    const std::string param = to_compact_lower(match.str());

    if (!std::regex_search(name, match, kVersionRegex)) {
        return {};
    }
    const std::string version = to_lower(match.str());

    std::string build_time;
    if (std::regex_search(name, match, kBuildTimeRegex)) {
        build_time = match.str();
    }

    const std::string prefix = xh + "_" + rule->canonical_name + "_" + param +
        pairing_variant_signature(name) + "_" + version;
    if (rule->match_build_time && !build_time.empty()) {
        return prefix + "_" + build_time;
    }
    return prefix;
}

HLIE_NAMESPACE_END

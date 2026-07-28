#ifndef HLIE_LLM_PRESET_RESOLVER_H
#define HLIE_LLM_PRESET_RESOLVER_H

#include <string>
#include <vector>

#include "hlie_app/hlie_llm_param_set.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

struct HLIELlmPresetRule {
    std::string id;
    std::string source;
    int priority = 0;
    nlohmann::json match = nlohmann::json::object();
    HLIELlmParamSet params;
};

class HLIE_API HLIELlmPresetCatalog {
public:
    bool load_file(const std::string& path, const std::string& source, std::string* error = nullptr);
    bool load_json(const nlohmann::json& document, const std::string& source, std::string* error = nullptr);

    [[nodiscard]] const std::vector<HLIELlmPresetRule>& rules() const { return rules_; }
    [[nodiscard]] const std::string& catalog_version() const { return catalog_version_; }
    [[nodiscard]] bool empty() const { return rules_.empty(); }

private:
    std::string catalog_version_;
    std::vector<HLIELlmPresetRule> rules_;
};

struct HLIELlmPresetResolution {
    HLIELlmParamSet params;
    std::vector<std::string> matched_presets;
    std::vector<std::string> sources;
    std::string params_hash;
};

class HLIE_API HLIELlmPresetResolver {
public:
    static bool rule_matches(const HLIELlmPresetRule& rule, const HLIEModelInfo& model_info,
                             std::string* mismatch_reason = nullptr);
    static HLIELlmPresetResolution resolve(const HLIEModelInfo& model_info,
                                           const HLIELlmParamSet& base,
                                           const std::vector<const HLIELlmPresetCatalog*>& catalogs);
};

HLIE_NAMESPACE_END

#endif // HLIE_LLM_PRESET_RESOLVER_H

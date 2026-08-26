#include "hlie_app/hlie_llm_preset_resolver.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <tuple>

#include "hlie_base/hlie_version.h"

HLIE_NAMESPACE_BEGIN

namespace {

void set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

std::string normalize(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool glob_match(const std::string& pattern_value, const std::string& candidate_value) {
    const std::string pattern = normalize(pattern_value);
    const std::string candidate = normalize(candidate_value);
    size_t p = 0;
    size_t c = 0;
    size_t star = std::string::npos;
    size_t retry = 0;
    while (c < candidate.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == candidate[c])) {
            ++p;
            ++c;
        } else if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            retry = c;
        } else if (star != std::string::npos) {
            p = star + 1;
            c = ++retry;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') {
        ++p;
    }
    return p == pattern.size();
}

std::vector<int> parse_version(std::string version) {
    version = normalize(std::move(version));
    if (!version.empty() && version.front() == 'v') {
        version.erase(version.begin());
    }
    std::vector<int> parts;
    size_t begin = 0;
    while (begin <= version.size()) {
        const size_t end = version.find('.', begin);
        const std::string token = version.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (token == "x" || token == "*") {
            parts.push_back(-1);
        } else {
            try {
                parts.push_back(token.empty() ? 0 : std::stoi(token));
            } catch (...) {
                return {};
            }
        }
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return parts;
}

int compare_version(const std::string& lhs, const std::string& rhs) {
    auto a = parse_version(lhs);
    auto b = parse_version(rhs);
    const size_t count = std::max(a.size(), b.size());
    a.resize(count, 0);
    b.resize(count, 0);
    for (size_t i = 0; i < count; ++i) {
        if (b[i] < 0) return 0;
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

bool match_string(const nlohmann::json& match, const char* field, const std::string& actual,
                  const bool allow_glob = false) {
    const auto it = match.find(field);
    if (it == match.end()) return true;
    if (!it->is_string()) return false;
    return allow_glob ? glob_match(it->get<std::string>(), actual)
                      : normalize(it->get<std::string>()) == normalize(actual);
}

bool match_int(const nlohmann::json& match, const char* field, const int actual) {
    const auto it = match.find(field);
    return it == match.end() || (it->is_number_integer() && it->get<int>() == actual);
}

bool match_bool(const nlohmann::json& match, const char* field, const bool actual) {
    const auto it = match.find(field);
    return it == match.end() || (it->is_boolean() && it->get<bool>() == actual);
}

bool parse_rule(const nlohmann::json& value, const std::string& source,
                HLIELlmPresetRule& rule, std::string* error) {
    if (!value.is_object()) {
        set_error(error, "preset rule must be an object");
        return false;
    }
    rule.id = value.value("id", "");
    if (rule.id.empty()) {
        set_error(error, "preset rule id is required");
        return false;
    }
    rule.source = source;
    rule.priority = value.value("priority", 0);
    rule.match = value.value("match", nlohmann::json::object());
    if (!rule.match.is_object()) {
        set_error(error, "preset rule match must be an object: " + rule.id);
        return false;
    }
    static const std::set<std::string> string_fields{
        "model_key", "model_name", "service_kind", "role", "model_family", "architecture",
        "params", "vl_spec", "model_version_min", "model_version_max",
    };
    static const std::set<std::string> int_fields{
        "context_size", "batch_num", "xh_version", "device_count",
    };
    for (auto it = rule.match.begin(); it != rule.match.end(); ++it) {
        if (string_fields.find(it.key()) != string_fields.end()) {
            if (!it->is_string()) {
                set_error(error, "preset match field must be a string: " + rule.id + "." + it.key());
                return false;
            }
        } else if (int_fields.find(it.key()) != int_fields.end()) {
            if (!it->is_number_integer()) {
                set_error(error, "preset match field must be an integer: " + rule.id + "." + it.key());
                return false;
            }
        } else if (it.key() == "multimodal") {
            if (!it->is_boolean()) {
                set_error(error, "preset match field must be a boolean: " + rule.id + ".multimodal");
                return false;
            }
        } else {
            set_error(error, "unknown preset match field: " + rule.id + "." + it.key());
            return false;
        }
    }
    std::string param_error;
    if (!HLIELlmParamSet::from_json(value.value("params", nlohmann::json::object()),
                                    rule.params, &param_error, false)) {
        set_error(error, "invalid params for preset " + rule.id + ": " + param_error);
        return false;
    }
    return true;
}

} // namespace

bool HLIELlmPresetCatalog::load_file(const std::string& path, const std::string& source, std::string* error) {
    std::ifstream input(path);
    if (!input) {
        set_error(error, "cannot open preset catalog: " + path);
        return false;
    }
    try {
        nlohmann::json document;
        input >> document;
        return load_json(document, source, error);
    } catch (const std::exception& ex) {
        set_error(error, "failed to parse preset catalog " + path + ": " + ex.what());
        return false;
    }
}

bool HLIELlmPresetCatalog::load_json(const nlohmann::json& document, const std::string& source,
                                     std::string* error) {
    if (!document.is_object() || document.value("schema_version", 0) != 1) {
        set_error(error, "preset catalog schema_version must be 1");
        return false;
    }

    if (const auto runtime = document.find("runtime"); runtime != document.end()) {
        if (!runtime->is_object()) {
            set_error(error, "preset catalog runtime must be an object");
            return false;
        }
        const std::string min_version = runtime->value("hlie_min_version", "");
        const std::string max_version = runtime->value("hlie_max_version", "");
        if ((!min_version.empty() && compare_version(HLIE_VERSION, min_version) < 0) ||
            (!max_version.empty() && compare_version(HLIE_VERSION, max_version) > 0)) {
            set_error(error, "preset catalog is not compatible with HLIECpp " + std::string(HLIE_VERSION));
            return false;
        }
    }

    std::vector<HLIELlmPresetRule> parsed;
    const auto append_rules = [&](const char* field) -> bool {
        const auto it = document.find(field);
        if (it == document.end()) return true;
        if (!it->is_array()) {
            set_error(error, std::string(field) + " must be an array");
            return false;
        }
        for (const auto& item : *it) {
            HLIELlmPresetRule rule;
            if (!parse_rule(item, source, rule, error)) return false;
            parsed.push_back(std::move(rule));
        }
        return true;
    };
    if (!append_rules("presets") || !append_rules("rules")) {
        return false;
    }

    if (const auto models = document.find("models"); models != document.end()) {
        if (!models->is_object()) {
            set_error(error, "models must be an object");
            return false;
        }
        for (auto it = models->begin(); it != models->end(); ++it) {
            if (!it.value().is_object()) {
                set_error(error, "exact model override must be an object: " + it.key());
                return false;
            }
            nlohmann::json rule_json = it.value();
            rule_json["id"] = rule_json.value("id", "model:" + it.key());
            rule_json["priority"] = rule_json.value("priority", 100000);
            rule_json["match"] = {{"model_key", it.key()}};
            HLIELlmPresetRule rule;
            if (!parse_rule(rule_json, source, rule, error)) return false;
            parsed.push_back(std::move(rule));
        }
    }

    std::set<std::string> ids;
    std::map<std::tuple<int, std::string, std::string>, std::string> conflict_index;
    for (const auto& rule : parsed) {
        if (!ids.insert(rule.id).second) {
            set_error(error, "duplicate preset rule id: " + rule.id);
            return false;
        }
        const auto params = rule.params.to_json(true);
        for (auto it = params.begin(); it != params.end(); ++it) {
            const auto conflict_key = std::make_tuple(rule.priority, rule.match.dump(), it.key());
            const std::string value = it.value().dump();
            if (const auto existing = conflict_index.find(conflict_key);
                existing != conflict_index.end() && existing->second != value) {
                set_error(error, "conflicting preset rules for param " + it.key() + " at priority " +
                                 std::to_string(rule.priority));
                return false;
            }
            conflict_index[conflict_key] = value;
        }
    }

    catalog_version_ = document.value("catalog_version", "");
    rules_ = std::move(parsed);
    return true;
}

bool HLIELlmPresetResolver::rule_matches(const HLIELlmPresetRule& rule, const HLIEModelInfo& model_info,
                                         std::string* mismatch_reason) {
    const auto& match = rule.match;
    const bool matched =
        match_string(match, "model_key", model_info.model_key_) &&
        match_string(match, "model_name", model_info.get_model_name(), true) &&
        match_string(match, "service_kind", model_info.get_service_kind()) &&
        match_string(match, "role", model_info.get_role_name()) &&
        match_string(match, "model_family", model_info.get_model_family()) &&
        match_string(match, "architecture", model_info.get_architecture()) &&
        match_string(match, "params", model_info.get_params()) &&
        match_string(match, "vl_spec", model_info.vl_spec_) &&
        match_int(match, "context_size", model_info.get_context_size()) &&
        match_int(match, "batch_num", model_info.get_batch_num()) &&
        match_int(match, "xh_version", model_info.get_xh_version()) &&
        match_int(match, "device_count", model_info.get_device_num()) &&
        match_bool(match, "multimodal", model_info.is_multi_model_);
    if (!matched) {
        set_error(mismatch_reason, "identity fields did not match");
        return false;
    }

    if (const auto it = match.find("model_version_min"); it != match.end()) {
        if (!it->is_string() || compare_version(model_info.get_model_version(), it->get<std::string>()) < 0) {
            set_error(mismatch_reason, "model version is below minimum");
            return false;
        }
    }
    if (const auto it = match.find("model_version_max"); it != match.end()) {
        if (!it->is_string() || compare_version(model_info.get_model_version(), it->get<std::string>()) > 0) {
            set_error(mismatch_reason, "model version is above maximum");
            return false;
        }
    }
    return true;
}

HLIELlmPresetResolution HLIELlmPresetResolver::resolve(
    const HLIEModelInfo& model_info, const HLIELlmParamSet& base,
    const std::vector<const HLIELlmPresetCatalog*>& catalogs) {
    HLIELlmPresetResolution result;
    result.params = base;

    for (const auto* catalog : catalogs) {
        if (!catalog) continue;
        std::vector<const HLIELlmPresetRule*> matched;
        for (const auto& rule : catalog->rules()) {
            if (rule_matches(rule, model_info)) {
                matched.push_back(&rule);
            }
        }
        std::sort(matched.begin(), matched.end(), [](const auto* lhs, const auto* rhs) {
            return std::make_tuple(lhs->priority, lhs->match.size(), lhs->id) <
                   std::make_tuple(rhs->priority, rhs->match.size(), rhs->id);
        });
        for (const auto* rule : matched) {
            result.params.merge_layer(rule->params);
            result.matched_presets.push_back(rule->id);
            if (result.sources.empty() || result.sources.back() != rule->source) {
                result.sources.push_back(rule->source);
            }
        }
    }
    result.params_hash = result.params.stable_hash();
    return result;
}

HLIE_NAMESPACE_END

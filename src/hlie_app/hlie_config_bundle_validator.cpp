#include "hlie_app/hlie_config_bundle_validator.h"

#include <filesystem>
#include <fstream>

#include "hlie_app/hlie_llm_preset_resolver.h"
#include "hlie_base/hlie_config_service.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

namespace {

bool require_regular_file(const std::filesystem::path& path, const char* label, std::string& error) {
    std::error_code filesystem_error;
    if (std::filesystem::is_regular_file(path, filesystem_error) && !filesystem_error) {
        return true;
    }
    error = std::string(label) + " is missing or is not a regular file: " + path.string();
    return false;
}

bool read_json_file(const std::filesystem::path& path, nlohmann::json& document, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        error = "cannot open JSON file: " + path.string();
        return false;
    }
    try {
        input >> document;
        return true;
    } catch (const nlohmann::json::exception& exception) {
        error = "invalid JSON file " + path.string() + ": " + exception.what();
        return false;
    }
}

bool validate_schema(const std::filesystem::path& path, std::string& error) {
    nlohmann::json schema;
    if (!require_regular_file(path, "config_schema.json", error) || !read_json_file(path, schema, error)) {
        return false;
    }
    if (!schema.is_object() || !schema.contains("version") || !schema["version"].is_number_integer() ||
        schema["version"].get<double>() <= 0.0 || !schema.contains("sections") || !schema["sections"].is_object()) {
        error = "config_schema.json must contain a positive integer version and an object sections field: " + path.string();
        return false;
    }
    return true;
}

bool validate_memory_estimates(const std::filesystem::path& path, std::string& error) {
    nlohmann::json document;
    if (!require_regular_file(path, "memory_estimates.json", error) || !read_json_file(path, document, error)) {
        return false;
    }
    if (!document.is_object() || !document.contains("rules") || !document["rules"].is_array()) {
        error = "memory_estimates.json must contain a rules array: " + path.string();
        return false;
    }
    std::size_t index = 0;
    for (const auto& rule : document["rules"]) {
        const bool valid = rule.is_object() && rule.contains("model_family") && rule["model_family"].is_string() &&
            !rule["model_family"].get_ref<const std::string&>().empty() && rule.contains("params") &&
            rule["params"].is_string() && !rule["params"].get_ref<const std::string&>().empty() &&
            rule.contains("chips") && rule["chips"].is_number_integer() && rule["chips"].get<double>() > 0.0 &&
            rule.contains("memory_per_device_mb") && rule["memory_per_device_mb"].is_number() &&
            rule["memory_per_device_mb"].get<double>() > 0.0;
        if (!valid) {
            error = "memory_estimates.json contains an invalid rule at index " + std::to_string(index) + ": " + path.string();
            return false;
        }
        ++index;
    }
    return true;
}

bool validate_catalog(const std::filesystem::path& path, const char* source, std::string& error) {
    HLIELlmPresetCatalog catalog;
    return catalog.load_file(path.string(), source, &error);
}

} // namespace

HLIEConfigBundleValidationResult HLIEConfigBundleValidator::validate(const HLIEConfigService& config_service) {
    HLIEConfigBundleValidationResult result;
    const auto bundle = config_service.get_config_bundle();
    if (!bundle.explicit_override) {
        result.notes.emplace_back("Using platform-default configuration bundle compatibility mode");
        return result;
    }

    if (!validate_schema(bundle.config_schema, result.error) ||
        !validate_memory_estimates(bundle.memory_estimates, result.error)) {
        result.valid = false;
        return result;
    }

    const bool presets_enabled = config_service.get_bool("model", "llm_presets_enabled", true, true);
    const bool preset_strict = config_service.get_bool("model", "llm_preset_strict", true, true);
    if (presets_enabled) {
        if (!require_regular_file(bundle.model_presets, "model_presets.json", result.error)) {
            result.valid = false;
            return result;
        }
        std::string catalog_error;
        if (!validate_catalog(bundle.model_presets, "official", catalog_error)) {
            if (preset_strict) {
                result.valid = false;
                result.error = "invalid model preset catalog " + bundle.model_presets.string() + ": " + catalog_error;
                return result;
            }
            result.warnings.push_back("Ignoring invalid model preset catalog in non-strict mode: " + catalog_error);
        }
    } else {
        result.notes.emplace_back("LLM preset catalogs are disabled; model_presets.json validation was skipped");
    }

    std::error_code filesystem_error;
    if (std::filesystem::exists(bundle.model_overrides, filesystem_error) && !filesystem_error) {
        if (!std::filesystem::is_regular_file(bundle.model_overrides, filesystem_error) || filesystem_error) {
            result.valid = false;
            result.error = "model_overrides.json exists but is not a regular file: " + bundle.model_overrides.string();
            return result;
        }
        std::string catalog_error;
        if (!validate_catalog(bundle.model_overrides, "local", catalog_error)) {
            if (preset_strict) {
                result.valid = false;
                result.error = "invalid model override catalog " + bundle.model_overrides.string() + ": " + catalog_error;
                return result;
            }
            result.warnings.push_back("Ignoring invalid model override catalog in non-strict mode: " + catalog_error);
        }
    } else if (filesystem_error) {
        result.valid = false;
        result.error = "cannot inspect model override catalog " + bundle.model_overrides.string() + ": " + filesystem_error.message();
        return result;
    } else {
        result.notes.emplace_back("Optional model override catalog is not present: " + bundle.model_overrides.string());
    }

    result.notes.emplace_back("Explicit configuration bundle validation succeeded");
    return result;
}

HLIE_NAMESPACE_END

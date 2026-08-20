#include "hlie_app/hlie_llm_param_set.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>
#include <string_view>

HLIE_NAMESPACE_BEGIN

namespace {

void set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

std::string scalar_to_string(const HLIELlmParamValue& value) {
    if (const auto* v = std::get_if<int64_t>(&value)) {
        return std::to_string(*v);
    }
    if (const auto* v = std::get_if<double>(&value)) {
        std::ostringstream out;
        out << std::setprecision(15) << *v;
        return out.str();
    }
    if (const auto* v = std::get_if<std::string>(&value)) {
        return *v;
    }
    return {};
}

} // namespace

bool HLIELlmParamSet::is_framework_owned(const std::string& key) {
    static constexpr std::array<std::string_view, 14> owned{
        "--model", "--mmproj", "--host", "--port", "--device_id", "--instance_id", "--config",
        "--json_info", "--launch_manifest", "--launch_model_key", "--launch_generation",
        "--launch_token", "--api-key", "--log-file",
    };
    return std::find(owned.begin(), owned.end(), key) != owned.end();
}

bool HLIELlmParamSet::from_json(const nlohmann::json& value, HLIELlmParamSet& result,
                                std::string* error, const bool allow_framework_keys) {
    if (!value.is_object()) {
        set_error(error, "llama params must be a JSON object");
        return false;
    }

    HLIELlmParamSet parsed;
    for (auto it = value.begin(); it != value.end(); ++it) {
        const std::string key = it.key();
        if (key.rfind("--", 0) != 0) {
            set_error(error, "llama param key must start with --: " + key);
            return false;
        }
        if (!allow_framework_keys && is_framework_owned(key)) {
            set_error(error, "framework-owned llama param cannot be overridden: " + key);
            return false;
        }

        const auto& raw = it.value();
        if (raw.is_null()) {
            parsed.set(key, std::monostate{});
        } else if (raw.is_boolean()) {
            parsed.set(key, raw.get<bool>());
        } else if (raw.is_number_integer()) {
            parsed.set(key, raw.get<int64_t>());
        } else if (raw.is_number_float()) {
            parsed.set(key, raw.get<double>());
        } else if (raw.is_string()) {
            parsed.set(key, raw.get<std::string>());
        } else if (raw.is_array()) {
            std::vector<std::string> values;
            for (const auto& item : raw) {
                if (!item.is_string()) {
                    set_error(error, "repeated llama param values must be strings: " + key);
                    return false;
                }
                values.push_back(item.get<std::string>());
            }
            parsed.set(key, std::move(values));
        } else {
            set_error(error, "unsupported llama param value type: " + key);
            return false;
        }
    }

    result = std::move(parsed);
    return true;
}

void HLIELlmParamSet::set(std::string key, HLIELlmParamValue value) {
    values_[std::move(key)] = std::move(value);
}

void HLIELlmParamSet::erase(const std::string& key) {
    values_.erase(key);
}

void HLIELlmParamSet::merge_layer(const HLIELlmParamSet& layer) {
    for (const auto& [key, value] : layer.values_) {
        if (std::holds_alternative<std::monostate>(value) ||
            (std::holds_alternative<bool>(value) && !std::get<bool>(value))) {
            values_.erase(key);
            continue;
        }
        values_[key] = value;
    }
}

bool HLIELlmParamSet::contains(const std::string& key) const {
    return values_.find(key) != values_.end();
}

const HLIELlmParamValue* HLIELlmParamSet::get(const std::string& key) const {
    const auto it = values_.find(key);
    return it == values_.end() ? nullptr : &it->second;
}

nlohmann::json HLIELlmParamSet::to_json(const bool include_tombstones) const {
    nlohmann::json result = nlohmann::json::object();
    for (const auto& [key, value] : values_) {
        if (std::holds_alternative<std::monostate>(value)) {
            if (include_tombstones) {
                result[key] = nullptr;
            }
        } else if (const auto* v = std::get_if<bool>(&value)) {
            if (*v || include_tombstones) {
                result[key] = *v;
            }
        } else if (const auto* v = std::get_if<int64_t>(&value)) {
            result[key] = *v;
        } else if (const auto* v = std::get_if<double>(&value)) {
            result[key] = *v;
        } else if (const auto* v = std::get_if<std::string>(&value)) {
            result[key] = *v;
        } else if (const auto* v = std::get_if<std::vector<std::string>>(&value)) {
            result[key] = *v;
        }
    }
    return result;
}

std::vector<std::string> HLIELlmParamSet::to_argv(const std::string& program) const {
    std::vector<std::string> result{program};
    for (const auto& [key, value] : values_) {
        if (std::holds_alternative<std::monostate>(value)) {
            continue;
        }
        if (const auto* enabled = std::get_if<bool>(&value)) {
            if (*enabled) {
                result.push_back(key);
            }
            continue;
        }
        if (const auto* repeated = std::get_if<std::vector<std::string>>(&value)) {
            for (const auto& item : *repeated) {
                result.push_back(key);
                result.push_back(item);
            }
            continue;
        }
        result.push_back(key);
        result.push_back(scalar_to_string(value));
    }
    return result;
}

std::string HLIELlmParamSet::stable_hash() const {
    const std::string canonical = to_json(false).dump();
    uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char c : canonical) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

HLIE_NAMESPACE_END

#include <algorithm>

#include "hlie_base/hlie_config_service.h"

#include <fstream>

#include "INIReader.h"
#include "hlie_base/hlie_logger.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

namespace {
    std::string to_upper(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(),
                       [](unsigned char c){ return std::toupper(c); });
        return result;
    }

    std::string to_lower(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return result;
    }

    std::string replace_dot_with_underscore(const std::string& str) {
        std::string result = str;
        std::replace(result.begin(), result.end(), '.', '_');
        return result;
    }

    std::string env_var_name(const std::string& section, const std::string& key) {
        const std::string upper_section = to_upper(replace_dot_with_underscore(section));
        const std::string upper_key = to_upper(key);
        return ENV_VAR_PREFIX + upper_section + "__" + upper_key;
    }
}

int HLIEConfigService::parse_from_file(const std::string& file_path) {
    config_reader_ = std::make_shared<INIReader>(file_path);
    if (config_reader_->ParseError() < 0) {
        LOG_ERROR("parse config file failed from {}", file_path);
        return -1;
    }

    return 0;
}

std::shared_ptr<INIReader> HLIEConfigService::get_reader() const {
    return config_reader_;
}

int HLIEConfigService::get_int(const std::string & section, const std::string & key, const int default_value, const bool env_first) const {
    if (env_first) {
        const std::string env_var = env_var_name(section, key);

        if (const char* env_value = std::getenv(env_var.c_str()); env_value != nullptr) {
            try {
                return std::stoi(env_value);
            } catch (const std::exception& ex) {
                LOG_ERROR("Cannot parse env {}, ex: {}", env_var, ex.what());
            }
        }
    }

    const auto result = config_reader_->GetInteger(section, key, default_value);
    return result;
}

std::string HLIEConfigService::get_string(const std::string & section, const std::string & key, const std::string& default_value, const bool env_first) const {
    if (env_first) {
        const std::string env_var = env_var_name(section, key);
        if (const char* env_value = std::getenv(env_var.c_str()); env_value != nullptr) {
            LOG_INFO("Config {} : {} from env", env_var, env_value);
            return {env_value};
        }
    }

    const auto result = config_reader_->GetString(section, key, default_value);
    return result;
}

bool HLIEConfigService::get_bool(const std::string& section, const std::string& key, const bool default_value, const bool env_first) const {
    if (env_first) {
        const std::string env_var = env_var_name(section, key);
        if (const char* env_value = std::getenv(env_var.c_str()); env_value != nullptr) {
            try {
                std::string val(env_value);
                std::transform(val.begin(), val.end(), val.begin(), tolower);
                LOG_INFO("Config {} : {} from env", env_var, env_value);

                if (val == "true" || val == "1" || val == "yes" || val == "on") {
                    return true;
                }

                if (val == "false" || val == "0" || val == "no" || val == "off") {
                    return false;
                }
            } catch (const std::exception& ex) {
                LOG_ERROR("Cannot parse env {}, ex: {}", env_var, ex.what());
            }
        }
    }

    const auto result = config_reader_->GetBoolean(section, key, default_value);
    return result;
}

int HLIEModelInfo::parse_from_file(const std::string& file_path) {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            LOG_ERROR("Error: Could not open file: {}", file_path);
            return -3;
        }

        nlohmann::json config;
        file >> config;

        model_name_ = config.value("ModelName", "");
        model_type_ = config.value("ModelType", "");
        model_type_ = to_lower(model_type_);
        context_size_ = config.value("ContextSize", 0);
        cores_ = config.value("Cores", 0);
        xh_version_ = config.value("XHVersion", 1);
        is_multi_model_ = config.value("IsMM", false);
        params_ = config.value("Params", "");
        mm_proj_path_ = config.value("MMProj", "");
        model_version_ = config.value("Version", "");
    } catch (const nlohmann::json::exception& ex) {
        LOG_ERROR("JSON Error: {}", ex.what());
        return -2;
    } catch (const std::exception& ex) {
        LOG_ERROR("General Error: {}", ex.what());
        return -1;
    }

    return 0;
}

HLIE_NAMESPACE_END

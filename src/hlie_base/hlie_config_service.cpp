#include <algorithm>

#include "hlie_base/hlie_config_service.h"

#include "INIReader.h"
#include "hlie_base/hlie_logger.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

namespace {
    std::string to_upper(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) { return std::toupper(c); });
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

    bool is_option(const char* arg) {
        return arg != nullptr && arg[0] == '-';
    }

    // 判断下一个参数是否为选项值（非选项即为值）
    bool next_is_value(char* argv[], int current_index, int max_length) {
        // 检查是否有下一个参数，且下一个参数不是选项
        return (current_index + 1 < max_length) && (argv[current_index + 1] != nullptr) && !is_option(argv[current_index + 1]);
    }
} // namespace

int HLIEConfigService::parse(const std::string& config_path) {
    file_path_ = config_path;
    if (file_path_.empty()) {
        HLIELogger::get_logger()->info("file path is empty");
        return -1;
    }

    config_reader_ = std::make_shared<INIReader>(file_path_);
    if (config_reader_->ParseError() < 0) {
        HLIELogger::get_logger()->error("parse config file failed from {}", file_path_);
        config_reader_.reset();
        return -1;
    }

    return 0;
}

std::shared_ptr<INIReader> HLIEConfigService::get_reader() const {
    return config_reader_;
}

int HLIEConfigService::get_int(const std::string& section, const std::string& key, const int default_value, const bool env_first) const {
    if (env_first || !config_reader_) {
        const std::string env_var = env_var_name(section, key);
        HLIELogger::get_logger()->info("Try to get {} from env", env_var);
        if (const char* env_value = std::getenv(env_var.c_str()); env_value != nullptr) {
            HLIELogger::get_logger()->info("Got config {} : {} from env", env_var, env_value);
            try {
                return std::stoi(env_value);
            } catch (const std::exception& ex) {
                HLIELogger::get_logger()->error("Cannot parse env {}, ex: {}", env_var, ex.what());
            }
        }
    }

    if (!config_reader_) {
        HLIELogger::get_logger()->warn("config reader is null, just return the [{}][{}] with default_value[{}]", section, key, default_value);
        return default_value;
    }

    const auto result = config_reader_->GetInteger(section, key, default_value);
    return result;
}

std::string HLIEConfigService::get_string(const std::string& section, const std::string& key, const std::string& default_value,
                                          const bool env_first) const {
    if (env_first || !config_reader_) {
        const std::string env_var = env_var_name(section, key);
        if (HLIELogger::get_logger()) {
            HLIELogger::get_logger()->info("Try to get {} from env", env_var);
        }
        if (const char* env_value = std::getenv(env_var.c_str()); env_value != nullptr) {
            if (HLIELogger::get_logger()) {
                HLIELogger::get_logger()->info("Got config {} : {} from env", env_var, env_value);
            }

            return {env_value};
        }
    }

    if (!config_reader_) {
        if (HLIELogger::get_logger()) {
            HLIELogger::get_logger()->warn("config reader is null, just return the [{}][{}] with default_value[{}]", section, key, default_value);
        }

        return default_value;
    }

    const auto result = config_reader_->GetString(section, key, default_value);
    return result;
}

bool HLIEConfigService::get_bool(const std::string& section, const std::string& key, const bool default_value, const bool env_first) const {
    if (env_first || !config_reader_) {
        const std::string env_var = env_var_name(section, key);
        HLIELogger::get_logger()->info("Try to get {} from env", env_var);
        if (const char* env_value = std::getenv(env_var.c_str()); env_value != nullptr) {
            try {
                std::string val(env_value);
                std::transform(val.begin(), val.end(), val.begin(), tolower);
                HLIELogger::get_logger()->info("Got config {} : {} from env", env_var, env_value);

                if (val == "true" || val == "1" || val == "yes" || val == "on") {
                    return true;
                }

                if (val == "false" || val == "0" || val == "no" || val == "off") {
                    return false;
                }
            } catch (const std::exception& ex) {
                HLIELogger::get_logger()->error("Cannot parse env {}, ex: {}", env_var, ex.what());
            }
        }
    }

    if (!config_reader_) {
        HLIELogger::get_logger()->warn("config reader is null, just return the [{}][{}] with default_value[{}]", section, key, default_value);
        return default_value;
    }

    const auto result = config_reader_->GetBoolean(section, key, default_value);
    return result;
}

void HLIEConfigService::set_args(const int argc, char** argv) {
    args_.clear();
    for (int i = 0; i < argc; ++i) {
        if (const auto opt = argv[i]; is_option(opt)) {
            if (next_is_value(argv, i, argc)) {
                args_[opt] = argv[i + 1];
                i++;
            } else {
                args_[opt] = "";
            }
            std::cout << "program arg:" << opt << " " << args_[opt] << std::endl;
        }
    }
}

const std::unordered_map<std::string, std::string>& HLIEConfigService::get_args() const {
    return args_;
}

HLIE_NAMESPACE_END

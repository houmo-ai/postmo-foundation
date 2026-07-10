#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>

#include "hlie_base/hlie_config_service.h"

#include "INIReader.h"
#include "hlie_base/hlie_logger.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

namespace {
    std::string trim(const std::string& str) {
        const auto first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return "";
        }

        const auto last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, last - first + 1);
    }

    bool is_comment_or_empty(const std::string& line) {
        const std::string value = trim(line);
        return value.empty() || value[0] == '#' || value[0] == ';';
    }

    bool parse_section_line(const std::string& line, std::string& section) {
        const std::string value = trim(line);
        if (value.size() < 3 || value.front() != '[') {
            return false;
        }

        const auto end_pos = value.find(']');
        if (end_pos == std::string::npos || end_pos == 1) {
            return false;
        }

        section = trim(value.substr(1, end_pos - 1));
        return !section.empty();
    }

    bool parse_key_value_line(const std::string& line, std::string& key, std::string& value) {
        if (is_comment_or_empty(line)) {
            return false;
        }

        const auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            return false;
        }

        key = trim(line.substr(0, eq_pos));
        value = trim(line.substr(eq_pos + 1));
        return !key.empty();
    }

    std::string format_config_line(const std::string& key, const std::string& value) {
        return key + " = " + value;
    }

    std::string read_text_file(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return "";
        }

        std::ostringstream oss;
        oss << in.rdbuf();
        return oss.str();
    }

    std::vector<std::string> split_lines(const std::string& content) {
        std::vector<std::string> lines;
        std::istringstream iss(content);
        std::string line;
        while (std::getline(iss, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            lines.push_back(line);
        }
        return lines;
    }

    bool write_lines(const std::string& path, const std::vector<std::string>& lines) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            return false;
        }

        for (size_t i = 0; i < lines.size(); ++i) {
            out << lines[i];
            if (i + 1 < lines.size()) {
                out << '\n';
            }
        }
        return true;
    }

    bool is_valid_section_name(const std::string& section) {
        return !section.empty() && section.find_first_of("[]\r\n") == std::string::npos;
    }

    bool is_valid_key_name(const std::string& key) {
        return !key.empty() && key.find_first_of("=\r\n") == std::string::npos;
    }

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
    std::lock_guard lock(mutex_);
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
    std::lock_guard lock(mutex_);
    return config_reader_;
}

std::string HLIEConfigService::get_config_path() const {
    std::lock_guard lock(mutex_);
    return file_path_;
}

std::unordered_map<std::string, std::unordered_map<std::string, std::string>> HLIEConfigService::get_config(
    const std::string& section) const {
    std::lock_guard lock(mutex_);
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> config;
    if (file_path_.empty()) {
        return config;
    }

    const std::string content = read_text_file(file_path_);
    if (content.empty()) {
        return config;
    }

    std::string current_section;
    for (const auto& line : split_lines(content)) {
        std::string parsed_section;
        if (parse_section_line(line, parsed_section)) {
            current_section = parsed_section;
            continue;
        }

        if (current_section.empty() || (!section.empty() && current_section != section)) {
            continue;
        }

        std::string key;
        std::string value;
        if (parse_key_value_line(line, key, value)) {
            config[current_section][key] = value;
        }
    }

    if (!section.empty() && config.find(section) == config.end()) {
        config[section] = {};
    }
    return config;
}

int HLIEConfigService::update_config(
    const std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& updates) {
    std::lock_guard lock(mutex_);
    if (file_path_.empty()) {
        return -1;
    }

    for (const auto& [section, values] : updates) {
        if (!is_valid_section_name(section)) {
            return -2;
        }

        for (const auto& [key, value] : values) {
            if (!is_valid_key_name(key) || value.find_first_of("\r\n") != std::string::npos) {
                return -2;
            }
        }
    }

    const std::string content = read_text_file(file_path_);
    std::vector<std::string> lines = split_lines(content);
    std::unordered_map<std::string, std::set<std::string>> applied;
    std::string current_section;

    for (auto& line : lines) {
        std::string parsed_section;
        if (parse_section_line(line, parsed_section)) {
            current_section = parsed_section;
            continue;
        }

        const auto section_it = updates.find(current_section);
        if (section_it == updates.end()) {
            continue;
        }

        std::string key;
        std::string old_value;
        if (!parse_key_value_line(line, key, old_value)) {
            continue;
        }

        const auto value_it = section_it->second.find(key);
        if (value_it == section_it->second.end()) {
            continue;
        }

        line = format_config_line(key, value_it->second);
        applied[current_section].insert(key);
    }

    for (const auto& [section, values] : updates) {
        if (values.empty()) {
            continue;
        }

        size_t section_start = lines.size();
        size_t section_end = lines.size();
        current_section.clear();
        for (size_t i = 0; i < lines.size(); ++i) {
            std::string parsed_section;
            if (!parse_section_line(lines[i], parsed_section)) {
                continue;
            }

            if (parsed_section == section) {
                section_start = i;
                section_end = lines.size();
                for (size_t j = i + 1; j < lines.size(); ++j) {
                    std::string next_section;
                    if (parse_section_line(lines[j], next_section)) {
                        section_end = j;
                        break;
                    }
                }
                break;
            }
        }

        std::vector<std::string> missing_lines;
        for (const auto& [key, value] : values) {
            if (applied[section].find(key) == applied[section].end()) {
                missing_lines.push_back(format_config_line(key, value));
            }
        }

        if (missing_lines.empty()) {
            continue;
        }

        if (section_start == lines.size()) {
            if (!lines.empty() && !lines.back().empty()) {
                lines.emplace_back("");
            }
            lines.push_back("[" + section + "]");
            lines.insert(lines.end(), missing_lines.begin(), missing_lines.end());
        } else {
            lines.insert(lines.begin() + static_cast<std::ptrdiff_t>(section_end), missing_lines.begin(), missing_lines.end());
        }
    }

    if (!write_lines(file_path_, lines)) {
        return -3;
    }

    config_reader_ = std::make_shared<INIReader>(file_path_);
    if (config_reader_->ParseError() < 0) {
        config_reader_.reset();
        return -4;
    }

    return 0;
}

int HLIEConfigService::delete_config_keys(const std::string& section, const std::vector<std::string>& keys) {
    std::lock_guard lock(mutex_);
    if (file_path_.empty()) {
        return -1;
    }

    if (!is_valid_section_name(section) || keys.empty()) {
        return -2;
    }

    std::set<std::string> key_set;
    for (const auto& key : keys) {
        if (!is_valid_key_name(key)) {
            return -2;
        }
        key_set.insert(key);
    }

    const std::string content = read_text_file(file_path_);
    std::vector<std::string> lines = split_lines(content);
    std::vector<std::string> kept_lines;
    kept_lines.reserve(lines.size());

    std::string current_section;
    int deleted_count = 0;
    for (const auto& line : lines) {
        std::string parsed_section;
        if (parse_section_line(line, parsed_section)) {
            current_section = parsed_section;
            kept_lines.push_back(line);
            continue;
        }

        if (current_section == section) {
            std::string key;
            std::string value;
            if (parse_key_value_line(line, key, value) && key_set.find(key) != key_set.end()) {
                ++deleted_count;
                continue;
            }
        }

        kept_lines.push_back(line);
    }

    if (deleted_count == 0) {
        return -5;
    }

    if (!write_lines(file_path_, kept_lines)) {
        return -3;
    }

    config_reader_ = std::make_shared<INIReader>(file_path_);
    if (config_reader_->ParseError() < 0) {
        config_reader_.reset();
        return -4;
    }

    return deleted_count;
}

int HLIEConfigService::get_int(const std::string& section, const std::string& key, const int default_value, const bool env_first) const {
    std::lock_guard lock(mutex_);
    if (env_first || !config_reader_) {
        const std::string env_var = env_var_name(section, key);
        HLIELogger::get_logger()->debug("Try to get {} from env", env_var);
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
    std::lock_guard lock(mutex_);
    if (env_first || !config_reader_) {
        const std::string env_var = env_var_name(section, key);
        if (HLIELogger::get_logger()) {
            HLIELogger::get_logger()->debug("Try to get {} from env", env_var);
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
    std::lock_guard lock(mutex_);
    if (env_first || !config_reader_) {
        const std::string env_var = env_var_name(section, key);
        HLIELogger::get_logger()->debug("Try to get {} from env", env_var);
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

#ifndef HLIE_CONFIG_SERVICE_H
#define HLIE_CONFIG_SERVICE_H

#include <memory>
#include <string>

#include "hlie_base.h"

class INIReader;

const std::string ENV_VAR_PREFIX = "HLIE__";

HLIE_NAMESPACE_BEGIN

class HLIE_API HLIEConfigService {
public:
    HLIEConfigService() = default;
    ~HLIEConfigService() = default;

    int parse_from_file(const std::string& file_path);

    [[nodiscard]] std::shared_ptr<INIReader> get_reader() const;

    [[nodiscard]] int get_int(const std::string& section, const std::string& key, int default_value, bool env_first = false) const;
    [[nodiscard]] std::string get_string(const std::string& section, const std::string& key, const std::string& default_value = "", bool env_first = false) const;
    [[nodiscard]] bool get_bool(const std::string & section, const std::string & key, bool default_value, bool env_first = false) const;

private:
    std::shared_ptr<INIReader>  config_reader_;
};

class HLIE_API HLIEModelInfo {
public:
    int parse_from_file(const std::string& file_path = "desc.json");

    [[nodiscard]] const std::string& get_model_name() const { return model_name_; }
    [[nodiscard]] const std::string& get_model_type() const { return model_type_; }
    [[nodiscard]] const std::string& get_model_version() const { return model_version_; }
    [[nodiscard]] const std::string& get_mm_proj_path() const { return mm_proj_path_; }
    [[nodiscard]] const std::string& get_params() const { return params_; }
    [[nodiscard]] int get_xh_version() const { return xh_version_; }
    [[nodiscard]] int get_context_size() const { return context_size_; }
    [[nodiscard]] int get_cores() const { return cores_; }
    [[nodiscard]] bool get_is_multi_model() const { return is_multi_model_; }

private:
    std::string model_name_;
    std::string model_type_;
    std::string model_version_;
    std::string mm_proj_path_;
    std::string params_;
    int xh_version_{1};
    int context_size_{2048};
    int cores_{0};
    bool is_multi_model_{false};
};

HLIE_NAMESPACE_END

#endif

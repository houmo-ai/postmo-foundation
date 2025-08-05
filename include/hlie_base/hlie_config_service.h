#ifndef HLIE_CONFIG_SERVICE_H
#define HLIE_CONFIG_SERVICE_H

#include <memory>
#include <string>
#include <utility>

#include "hlie_base.h"

class INIReader;

const std::string ENV_VAR_PREFIX = "HLIE__";

HLIE_NAMESPACE_BEGIN

class HLIE_API HLIEConfigService {
public:
    explicit HLIEConfigService(std::string config_path): file_path_(std::move(config_path)) {}

    ~HLIEConfigService() = default;

    int parse();

    [[nodiscard]] std::shared_ptr<INIReader> get_reader() const;

    [[nodiscard]] int get_int(const std::string& section, const std::string& key, int default_value, bool env_first = false) const;
    [[nodiscard]] std::string get_string(const std::string& section, const std::string& key, const std::string& default_value = "", bool env_first = false) const;
    [[nodiscard]] bool get_bool(const std::string & section, const std::string & key, bool default_value, bool env_first = false) const;

private:
    std::shared_ptr<INIReader> config_reader_;
    std::string file_path_;
};

HLIE_NAMESPACE_END

#endif

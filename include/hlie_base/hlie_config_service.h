#ifndef HLIE_CONFIG_SERVICE_H
#define HLIE_CONFIG_SERVICE_H

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "hlie_base.h"
#include "hlie_singleton.h"

class INIReader;

const std::string ENV_VAR_PREFIX = "HLIE__";

HLIE_NAMESPACE_BEGIN

class HLIE_API HLIEConfigService {
public:
    HLIEConfigService() = default;

    ~HLIEConfigService() = default;

    int parse(const std::string& config_path);

    [[nodiscard]] std::shared_ptr<INIReader> get_reader() const;
    [[nodiscard]] std::string get_config_path() const;
    [[nodiscard]] std::unordered_map<std::string, std::unordered_map<std::string, std::string>> get_config(
        const std::string& section = "") const;
    int update_config(const std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& updates);
    int delete_config_keys(const std::string& section, const std::vector<std::string>& keys);

    [[nodiscard]] int get_int(const std::string& section, const std::string& key, int default_value, bool env_first = false) const;
    [[nodiscard]] std::string get_string(const std::string& section, const std::string& key, const std::string& default_value = "", bool env_first = false) const;
    [[nodiscard]] bool get_bool(const std::string & section, const std::string & key, bool default_value, bool env_first = false) const;

    void set_args(int argc, char** argv);
    [[nodiscard]] const std::unordered_map<std::string, std::string>& get_args() const;

private:

    std::shared_ptr<INIReader> config_reader_;
    std::string file_path_;
    mutable std::mutex mutex_;

    std::unordered_map<std::string, std::string> args_;
};

HLIE_NAMESPACE_END

#endif

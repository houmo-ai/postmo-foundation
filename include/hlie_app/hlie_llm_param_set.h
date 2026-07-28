#ifndef HLIE_LLM_PARAM_SET_H
#define HLIE_LLM_PARAM_SET_H

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "hlie_base/hlie_base.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

using HLIELlmParamValue = std::variant<std::monostate, bool, int64_t, double, std::string,
                                       std::vector<std::string>>;

class HLIE_API HLIELlmParamSet {
public:
    using Storage = std::map<std::string, HLIELlmParamValue>;

    static bool from_json(const nlohmann::json& value, HLIELlmParamSet& result,
                          std::string* error = nullptr, bool allow_framework_keys = false);
    static bool is_framework_owned(const std::string& key);

    void set(std::string key, HLIELlmParamValue value);
    void erase(const std::string& key);
    void merge_layer(const HLIELlmParamSet& layer);

    [[nodiscard]] bool contains(const std::string& key) const;
    [[nodiscard]] const HLIELlmParamValue* get(const std::string& key) const;
    [[nodiscard]] const Storage& values() const { return values_; }
    [[nodiscard]] bool empty() const { return values_.empty(); }
    [[nodiscard]] size_t size() const { return values_.size(); }

    [[nodiscard]] nlohmann::json to_json(bool include_tombstones = false) const;
    [[nodiscard]] std::vector<std::string> to_argv(const std::string& program = "hlie-llm") const;
    [[nodiscard]] std::string stable_hash() const;

private:
    Storage values_;
};

HLIE_NAMESPACE_END

#endif // HLIE_LLM_PARAM_SET_H

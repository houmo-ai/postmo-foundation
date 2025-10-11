#ifndef HLIE_UTIL_H
#define HLIE_UTIL_H

#include <filesystem>
#include <string>
#include <utility>


#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

namespace fs = std::filesystem;

class HLIE_API HlieUtil {
public:
    template <typename... Args>
    static std::string join_paths(Args&&... args) {
        fs::path result;
        (result /= ... /= fs::path(std::forward<Args>(args)));
        return result.lexically_normal().string();
    }


    static int set_env(const std::string& key, const std::string& value);

    static std::string uuid();

    static bool is_valid_devices(const std::string& input);

    static int parse_natural_num(const std::string& input, bool include_zero = true);

    static bool is_relative_path(const std::string& path);
    static std::string extract_filename(const std::string& path);
    static bool has_model_params(const std::string& str);
};

HLIE_NAMESPACE_END

#endif //HLIE_UTIL_H

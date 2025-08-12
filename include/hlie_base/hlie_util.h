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

};

HLIE_NAMESPACE_END

#endif //HLIE_UTIL_H

//
// Created by tuo.wang on 2025/8/5.
//

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


};

HLIE_NAMESPACE_END

#endif //HLIE_UTIL_H

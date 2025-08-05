//
// Created by tuo.wang on 2025/8/5.
//

#ifndef HLIE_UTIL_H
#define HLIE_UTIL_H

#include <string>


#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class HLIE_API HlieUtil {
public:
    static std::string join_paths(const std::string& base, const std::string& part);
};

HLIE_NAMESPACE_END

#endif //HLIE_UTIL_H

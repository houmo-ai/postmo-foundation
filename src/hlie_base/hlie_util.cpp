#include "hlie_base/hlie_util.h"
#include <filesystem>

namespace fs = std::filesystem;

HLIE_NAMESPACE_BEGIN

std::string HlieUtil::join_paths(const std::string& base, const std::string& part) {
    const fs::path path = fs::path(base) / part;
    return path.lexically_normal().string();
}


HLIE_NAMESPACE_END

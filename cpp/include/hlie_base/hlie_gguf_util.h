#ifndef HLIE_GGUF_UTIL_H
#define HLIE_GGUF_UTIL_H

#include <filesystem>
#include <string>
#include <utility>
#include <vector>


#include "gguf.h"
#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

namespace fs = std::filesystem;

class HLIE_API HLIEGgufUtil {
public:
    /**
     *从 GGUF 上下文获取指定 key 的值
     *
     * @param meta GGUF 上下文
     * @param key 要获取的键名
     * @param result 输出结果
     * @param required 是否必需（如果为 true 且找不到则抛异常）
     * @return 是否成功找到
     */
    template <typename T>
    static bool get_key(gguf_context* meta, const std::string& key, T& result, bool required = true);

    static void replace_all(std::string& s, const std::string& search, const std::string& replace);
    static std::string gguf_data_to_str(enum gguf_type type, const void* data, int i);
    static std::string gguf_kv_to_str(const struct gguf_context* ctx_gguf, int i);
};

HLIE_NAMESPACE_END

#endif // HLIE_GGUF_UTIL_H

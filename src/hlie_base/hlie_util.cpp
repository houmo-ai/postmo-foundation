#include "hlie_base/hlie_util.h"

#include <regex>

#include "hlie_base/hlie_logger.h"


#if defined(_WIN32) || defined(_WIN64)  // Windows 平台
    #include <objbase.h>
    #pragma comment(lib, "Rpcrt4.lib")
#elif defined(__unix__) // Linux
    #include <uuid/uuid.h>
#else
    #include <random>
    #include <sstream>
    #include <iomanip>
#endif

HLIE_NAMESPACE_BEGIN


int HlieUtil::set_env(const std::string& key, const std::string& value) {
    if (key.empty()) {
        LOG_ERROR("key cannot be empty");
        return -1;
    }

#ifdef _WIN32
    const std::string env_var = key + "=" + value;
    _putenv(env_var.c_str());
#else
    setenv(key.c_str(), value.c_str(), 1); // 1 表示覆盖已存在的变量
#endif

    LOG_INFO("Update ENV {}={}", key, value);
    return 0;
}

std::string HlieUtil::uuid() {
#if defined(_WIN32) || defined(_WIN64)
    UUID uuid;
    UuidCreate(&uuid);
    RPC_CSTR str;
    UuidToStringA(&uuid, &str);
    std::string result(reinterpret_cast<char*>(str));
    RpcStringFreeA(&str);
    return result;

#elif defined(__unix__)
    uuid_t uuid;
    uuid_generate_random(uuid);
    char str[37];
    uuid_unparse_lower(uuid, str);
    return std::string(str);

#else
    // 纯 C++17 随机数版本
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);

    uint32_t data[4];
    for (auto& d : data) d = dis(gen);

    // UUID v4 规则
    data[1] = (data[1] & 0xFFFF0FFF) | 0x00004000;
    data[2] = (data[2] & 0x3FFFFFFF) | 0x80000000;

    std::stringstream ss;
    ss << std::hex << std::setfill('0')
       << std::setw(8) << data[0] << "-"
       << std::setw(4) << ((data[1] >> 16) & 0xFFFF) << "-"
       << std::setw(4) << (data[1] & 0xFFFF) << "-"
       << std::setw(4) << ((data[2] >> 16) & 0xFFFF) << "-"
       << std::setw(4) << (data[2] & 0xFFFF)
       << std::setw(8) << data[3];
    return ss.str();
#endif
}

bool HlieUtil::is_valid_devices(const std::string& input) {
    const std::regex pattern("^\\d+(,\\d+)*$");
    return std::regex_match(input, pattern);
}

int HlieUtil::parse_natural_num(const std::string& input, const bool include_zero) {
    if (input.empty()) {
        return -1; // 空字符串无效
    }

    try {
        size_t pos;
        const long num = std::stol(input, &pos); // 用 stol 避免 stoi 的溢出问题

        // 检查是否所有字符都被转换，且数值在 int 范围内
        if (pos != input.size() || num < 0 || num > INT_MAX) {
            return -1;
        }

        // 检查前导零（不允许 "01"，但允许 "0"）
        if (input.size() > 1 && input[0] == '0') {
            return -1;
        }

        if (!include_zero && num == 0) {
            return -1;
        }

        return static_cast<int>(num);
    } catch (...) {
        return -1; // 捕获所有异常（无效字符串、超出范围等）
    }
}

bool HlieUtil::is_relative_path(const std::string& path) {
    return path[0] == '.' || path[0] == 'H';
}

HLIE_NAMESPACE_END

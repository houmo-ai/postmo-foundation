#include "hlie_base/hlie_util.h"

#include <regex>
#include <sstream>

#include "hlie_base/hlie_logger.h"


#if defined(_WIN32) || defined(_WIN64) // Windows 平台
#include <objbase.h>
#pragma comment(lib, "Rpcrt4.lib")
#else
#include <iomanip>
#include <random>
#include <sstream>
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

#else
    // 纯 C++17 随机数版本
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);

    uint32_t data[4];
    for (auto& d : data)
        d = dis(gen);

    // UUID v4 规则
    data[1] = (data[1] & 0xFFFF0FFF) | 0x00004000;
    data[2] = (data[2] & 0x3FFFFFFF) | 0x80000000;

    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) << data[0] << "-" << std::setw(4) << ((data[1] >> 16) & 0xFFFF) << "-"
       << std::setw(4) << (data[1] & 0xFFFF) << "-" << std::setw(4) << ((data[2] >> 16) & 0xFFFF) << "-" << std::setw(4)
       << (data[2] & 0xFFFF) << std::setw(8) << data[3];
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
#ifdef _WIN32
    // UNC 路径 \\server\...
    if (path.size() >= 2 && path[0] == '\\' && path[1] == '\\') {
        return false;
    }
    // 驱动器开头 C:\ 或者 C:/
    if (path.size() >= 3 && std::isalpha(path[0]) && path[1] == ':' && (path[2] == '\\' || path[2] == '/')) {
        return false;
    }
#else
    // Unix/Linux 绝对路径以 / 开头
    if (path[0] == '/') {
        return false;
    }
#endif

    // 其他情况都是相对路径
    return true;
}

/**
 * @brief 从文件路径中提取文件名（不含扩展名）
 * @param path 文件路径，可以是绝对路径或相对路径
 * @return 纯文件名，不包含路径和扩展名
 */
std::string HlieUtil::extract_filename(const std::string& path) {
    // 处理Windows和Linux的路径分隔符
    size_t last_slash = path.find_last_of("/\\");
    std::string filename = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);

    // 移除文件扩展名
    size_t last_dot = filename.find_last_of('.');
    if (last_dot != std::string::npos) {
        filename = filename.substr(0, last_dot);
    }

    return filename;
}

/**
 * @brief 检查字符串是否已包含模型参数格式（如 -7B、-8B）
 * @param str 输入字符串
 * @return 如果包含模型参数格式则返回true
 */
bool HlieUtil::has_model_params(const std::string& str) {
    static const std::regex params_regex(".*-\\d+B.*");
    return std::regex_match(str, params_regex);
}

std::string HlieUtil::get_datetime_str() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);

    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d-%H-%M"); // 年-月-日-时-分
    return oss.str();
}

std::tuple<int, int, int> HlieUtil::parse_version(const std::string& v) {
    int major = 0, minor = 0, patch = 0;

    std::string s = v;
    if (!s.empty() && s[0] == 'v')
        s.erase(0, 1); // 去掉 'v'

    char dot;
    std::istringstream iss(s);
    iss >> major >> dot >> minor >> dot >> patch;

    return {major, minor, patch};
}

bool HlieUtil::version_ge(const std::string& v, const std::string& target) {
    return parse_version(v) >= parse_version(target);
}

bool HlieUtil::contains_ignore_case(std::string s, const std::string& key) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    std::string k = key;
    std::transform(k.begin(), k.end(), k.begin(), ::tolower);

    return s.find(k) != std::string::npos;
}

HLIE_NAMESPACE_END

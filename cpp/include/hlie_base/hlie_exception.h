#ifndef HLIE_EXCEPTION_H
#define HLIE_EXCEPTION_H

#include <exception>
#include <string>
#include <utility>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

/**
 * @brief 基础异常类，所有Web服务器相关异常的基类
 * 继承自std::exception，支持错误码和错误消息
 */
class WebServerException : public std::exception {
public:
    /**
     * @brief 构造函数
     * @param ret_code 错误码
     * @param err_msg 错误消息
     */
    WebServerException(int32_t ret_code, const std::string& err_msg) : ret_code_(ret_code), err_msg_(err_msg) {}

    /**
     * @brief 析构函数，确保多态析构
     */
    ~WebServerException() noexcept override = default;

    /**
     * @brief 获取错误消息
     * @return 错误消息的C字符串表示
     */
    [[nodiscard]] const char* what() const noexcept override { return err_msg_.c_str(); }

    /**
     * @brief 获取错误码
     * @return 错误码
     */
    [[nodiscard]] int32_t get_ret_code() const noexcept { return ret_code_; }

    /**
     * @brief 获取错误消息
     * @return 错误消息
     */
    [[nodiscard]] const std::string& get_err_msg() const noexcept { return err_msg_; }

protected:
    int32_t ret_code_; // 错误码
    std::string err_msg_; // 错误消息
};

/**
 * @brief HTTP相关异常
 */
class HttpException final : public WebServerException {
public:
    using WebServerException::WebServerException;
};

/**
 * @brief 通用异常
 */
class CommonException final : public WebServerException {
public:
    using WebServerException::WebServerException;
};


/**
 * @brief 资源相关异常
 */
class ResourceException final : public WebServerException {
public:
    using WebServerException::WebServerException;
};

class InferenceQueueFullException final : public std::exception {
public:
    explicit InferenceQueueFullException(std::string message) : message_(std::move(message)) {}
    [[nodiscard]] const char* what() const noexcept override { return message_.c_str(); }

private:
    std::string message_;
};

/**
 * @brief 权限相关异常
 */
class AuthException final : public WebServerException {
public:
    using WebServerException::WebServerException;
};

HLIE_NAMESPACE_END

#endif // HLIE_EXCEPTION_H

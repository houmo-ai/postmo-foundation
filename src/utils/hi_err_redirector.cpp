#include "hi_err_redirector.h"

#include <iostream>
#include "spdlog/sinks/daily_file_sink.h"

void HiErrRedirector::start(const std::string& log_filename) {
    auto logger = spdlog::daily_logger_mt("HI.CPP", log_filename, 0, 0);
    logger->set_level(spdlog::level::debug);

    // 重定向 stderr (cerr / clog)
    stderr_buf_ = std::make_unique<SpdlogRedirectBuffer>(spdlog::level::err, logger);
    old_stderr_buf_ = std::cerr.rdbuf(stderr_buf_.get());
    setvbuf(stderr, nullptr, _IONBF, 0); // 禁用stderr缓冲
}

void HiErrRedirector::stop() const {
    // 恢复原始缓冲区
    std::cerr.rdbuf(old_stderr_buf_);
}

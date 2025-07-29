#include "hlie_base/hlie_std_err_redirector.h"

#include <iostream>
#include "spdlog/sinks/daily_file_sink.h"

void HLIEStdErrRedirector::start() {
    // 重定向 stderr (cerr/clog)
    stderr_buf_ = std::make_unique<SpdlogRedirectBuffer>();
    old_stderr_buf_ = std::cerr.rdbuf(stderr_buf_.get());
    setvbuf(stderr, nullptr, _IONBF, 0); // 禁用stderr缓冲
}

void HLIEStdErrRedirector::stop() const {
    // 恢复原始缓冲区
    std::cerr.rdbuf(old_stderr_buf_);
}

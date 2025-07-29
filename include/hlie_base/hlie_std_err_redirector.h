#ifndef HLIE_STD_ERR_REDIRECTOR
#define HLIE_STD_ERR_REDIRECTOR

#include <cstdio>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

#include <fcntl.h>

#include "hlie_base/hlie_base.h"

class SpdlogRedirectBuffer : public std::streambuf {
protected:
    int overflow(int c) override {
        if (c != EOF) {
            buffer_.push_back(static_cast<char>(c));
            if (c == '\n')
                flush();
        }
        return c;
    }

    int sync() override {
        if (!buffer_.empty())
            flush();
        return 0;
    }

private:
    void flush() {
        if (!buffer_.empty()) {
            spdlog::default_logger()->error(buffer_);
            buffer_.clear();
        }
    }

    std::string buffer_;
};


class HLIE_API HLIEStdErrRedirector {
public:
    HLIEStdErrRedirector() = default;
    ~HLIEStdErrRedirector() = default;

    void start();
    void stop() const;

private:
    std::unique_ptr<SpdlogRedirectBuffer> stderr_buf_;
    std::streambuf* old_stderr_buf_;
};

#endif

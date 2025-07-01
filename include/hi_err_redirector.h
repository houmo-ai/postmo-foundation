#ifndef HI_ERR_REDIRECTOR
#define HI_ERR_REDIRECTOR

#include <cstdio>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

#include <fcntl.h>

class SpdlogRedirectBuffer : public std::streambuf {
public:
    SpdlogRedirectBuffer(const spdlog::level::level_enum level, const std::shared_ptr<spdlog::logger>& logger) :
        level_(level), logger_(logger) {}

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
            logger_->log(level_, buffer_);
            buffer_.clear();
        }
    }

    std::string buffer_;
    spdlog::level::level_enum level_;
    std::shared_ptr<spdlog::logger> logger_;
};


class HiErrRedirector {
public:
    HiErrRedirector() = default;
    ~HiErrRedirector() = default;

    void start(const std::string& log_filename);
    void stop() const;

private:
    std::unique_ptr<SpdlogRedirectBuffer> stderr_buf_;
    std::streambuf* old_stderr_buf_;
};

#endif

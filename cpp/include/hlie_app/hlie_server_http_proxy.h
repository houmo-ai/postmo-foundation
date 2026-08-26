#ifndef HLIE_REQUEST_PROXY_H
#define HLIE_REQUEST_PROXY_H

#include "hlie_base/hlie_base.h"
#include "hlie_app/util/hlie_proxy_timeout.h"
#include "server/server-http.h"

#include <memory>
#include <mutex>


HLIE_NAMESPACE_BEGIN

struct HLIE_API ServerHttpProxy : server_http_res {
    std::function<void()> cleanup = nullptr;
    ServerHttpProxy(const std::string & method,
                      const std::string & host,
                      int port,
                      const std::string & path,
                      const std::map<std::string, std::string> & headers,
                       const std::string & body,
                       const std::map<std::string, uploaded_file> & files,
                       const std::function<bool()>& should_stop,
                       const ProxyTimeoutOptions& timeout_options = {});
    ~ServerHttpProxy() override;

    // Stop the upstream request. Safe to call repeatedly from both the stream
    // provider and its completion callback.
    void cancel();

    // Available after next() reaches EOF. HTTP response trailers are received
    // only after the upstream chunked body has completed.
    [[nodiscard]] std::map<std::string, std::string> response_trailers() const;

private:
    std::thread thread;
    mutable std::mutex cleanup_mutex_;
    bool cleaned_up_ = false;

    struct trailer_state_t;
    std::shared_ptr<trailer_state_t> trailer_state_;

    struct msg_t {
        std::map<std::string, std::string> headers;
        int status = 0;
        std::string data;
        std::string content_type;
    };
};

HLIE_NAMESPACE_END

#endif

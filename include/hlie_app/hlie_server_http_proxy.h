#ifndef HLIE_REQUEST_PROXY_H
#define HLIE_REQUEST_PROXY_H

#include "hlie_base/hlie_base.h"
#include "server/server-http.h"


HLIE_NAMESPACE_BEGIN

struct HLIE_API ServerHttpProxy : server_http_res {
    std::function<void()> cleanup = nullptr;
    ServerHttpProxy(const std::string & method,
                      const std::string & host,
                      int port,
                      const std::string & path,
                      const std::map<std::string, std::string> & headers,
                      const std::string & body,
                      const std::function<bool()>& should_stop);
    ~ServerHttpProxy() override {
        if (cleanup) {
            cleanup();
        }
    }
private:
    std::thread thread;
    struct msg_t {
        std::map<std::string, std::string> headers;
        int status = 0;
        std::string data;
        std::string content_type;
    };
};

HLIE_NAMESPACE_END

#endif

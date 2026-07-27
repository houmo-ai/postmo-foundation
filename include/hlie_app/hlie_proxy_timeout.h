#ifndef HLIE_PROXY_TIMEOUT_H
#define HLIE_PROXY_TIMEOUT_H

#include <string>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

struct HLIE_API ProxyTimeoutOptions {
    int connect_timeout_ms = 1000;
    int non_stream_read_timeout_sec = 1800;
    int stream_idle_timeout_sec = 120;
    int write_timeout_sec = 30;
    int max_request_timeout_sec = 3600;

    [[nodiscard]] ProxyTimeoutOptions normalized() const;
    [[nodiscard]] int read_timeout_for_body(const std::string& body) const;
};

HLIE_NAMESPACE_END

#endif // HLIE_PROXY_TIMEOUT_H

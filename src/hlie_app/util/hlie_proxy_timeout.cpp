#include "hlie_app/util/hlie_proxy_timeout.h"

#include <algorithm>

#include "json.hpp"

HLIE_NAMESPACE_BEGIN

ProxyTimeoutOptions ProxyTimeoutOptions::normalized() const {
    ProxyTimeoutOptions result = *this;
    result.connect_timeout_ms = std::clamp(result.connect_timeout_ms, 100, 10000);
    result.non_stream_read_timeout_sec = std::clamp(result.non_stream_read_timeout_sec, 60, 3600);
    result.stream_idle_timeout_sec = std::clamp(result.stream_idle_timeout_sec, 10, 600);
    result.write_timeout_sec = std::clamp(result.write_timeout_sec, 5, 300);
    result.max_request_timeout_sec = std::clamp(result.max_request_timeout_sec, 300, 7200);
    return result;
}

int ProxyTimeoutOptions::read_timeout_for_body(const std::string& body) const {
    const auto options = normalized();
    const auto request = nlohmann::json::parse(body, nullptr, false);
    const bool streaming = request.is_object() && request.contains("stream") && request["stream"].is_boolean() &&
        request["stream"].get<bool>();
    return streaming ? options.stream_idle_timeout_sec : options.non_stream_read_timeout_sec;
}

HLIE_NAMESPACE_END

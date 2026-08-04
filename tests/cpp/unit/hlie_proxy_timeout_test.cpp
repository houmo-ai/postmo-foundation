#include "hlie_app/util/hlie_proxy_timeout.h"

#include <catch_amalgamated.hpp>

using namespace hm_HLIE;

TEST_CASE("Proxy timeout defaults distinguish streaming idle from non-stream waits", "[proxy][timeout]") {
    const ProxyTimeoutOptions options;
    CHECK(options.read_timeout_for_body(R"({"stream":true})") == 120);
    CHECK(options.read_timeout_for_body(R"({"stream":false})") == 1800);
    CHECK(options.read_timeout_for_body(R"({"model":"demo"})") == 1800);
    CHECK(options.read_timeout_for_body("not-json") == 1800);
}

TEST_CASE("Proxy timeout configuration is clamped to operational limits", "[proxy][timeout]") {
    ProxyTimeoutOptions input;
    input.connect_timeout_ms = 1;
    input.non_stream_read_timeout_sec = 99999;
    input.stream_idle_timeout_sec = 1;
    input.write_timeout_sec = 99999;
    input.max_request_timeout_sec = 1;

    const auto options = input.normalized();
    CHECK(options.connect_timeout_ms == 100);
    CHECK(options.non_stream_read_timeout_sec == 3600);
    CHECK(options.stream_idle_timeout_sec == 10);
    CHECK(options.write_timeout_sec == 300);
    CHECK(options.max_request_timeout_sec == 300);
}

TEST_CASE("Proxy timeout request classification only accepts boolean stream true", "[proxy][timeout]") {
    ProxyTimeoutOptions options;
    options.non_stream_read_timeout_sec = 900;
    options.stream_idle_timeout_sec = 45;

    CHECK(options.read_timeout_for_body(R"({"stream":true})") == 45);
    CHECK(options.read_timeout_for_body(R"({"stream":"true"})") == 900);
    CHECK(options.read_timeout_for_body(R"({"stream":1})") == 900);
}

#include <catch_amalgamated.hpp>

#include <string>

#include "hlie_app/util/hlie_http_log_sanitizer.h"

using namespace hm_HLIE;

TEST_CASE("HTTP log sanitizer recognizes only known static UI paths") {
    CHECK(http_log::is_static_asset_path("/"));
    CHECK(http_log::is_static_asset_path("/bundle.js"));
    CHECK(http_log::is_static_asset_path("/bundle.css"));
    CHECK_FALSE(http_log::is_static_asset_path("/v1/models"));
    CHECK_FALSE(http_log::is_static_asset_path("/api/bundle.js"));
}

TEST_CASE("HTTP log sanitizer handles JSON content types case-insensitively") {
    CHECK(http_log::is_json_content_type("application/json; charset=utf-8"));
    CHECK(http_log::is_json_content_type("Application/Problem+JSON"));
    CHECK_FALSE(http_log::is_json_content_type("application/javascript"));
    CHECK_FALSE(http_log::is_json_content_type(""));
}

TEST_CASE("HTTP log sanitizer redacts request and response base64 fields") {
    const std::string request =
        R"({"messages":[{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,SECRET"}}]}]})";
    const std::string response = R"({"data":[{"b64_json":"SECRET"}]})";

    const auto request_summary =
        http_log::summarize_body(request, "application/json", http_log::BodyDirection::request);
    const auto response_summary =
        http_log::summarize_body(response, "application/json", http_log::BodyDirection::response);

    CHECK(request_summary.find("SECRET") == std::string::npos);
    CHECK(request_summary.find("redacted image_url") != std::string::npos);
    CHECK(response_summary.find("SECRET") == std::string::npos);
    CHECK(response_summary.find("redacted b64_json") != std::string::npos);
}

TEST_CASE("HTTP log sanitizer recursively redacts common credential fields") {
    const std::string body =
        R"({"authorization":"Bearer SECRET","nested":{"api_key":"KEY","password":"PASS","launch_token":"WORKER","token":"GENERIC"},"max_tokens":32})";
    const auto summary =
        http_log::summarize_body(body, "application/json", http_log::BodyDirection::request);

    CHECK(summary.find("SECRET") == std::string::npos);
    CHECK(summary.find("KEY") == std::string::npos);
    CHECK(summary.find("PASS") == std::string::npos);
    CHECK(summary.find("WORKER") == std::string::npos);
    CHECK(summary.find("GENERIC") == std::string::npos);
    CHECK(summary.find("\"max_tokens\":32") != std::string::npos);
}

TEST_CASE("HTTP log sanitizer never emits non-JSON body contents") {
    const std::string javascript = "var secret = 'must not be logged';";
    const auto summary =
        http_log::summarize_body(javascript, "application/javascript", http_log::BodyDirection::response);

    CHECK(summary == "<non-json response: content_type=application/javascript, size=34 bytes>");
    CHECK(summary.find("secret") == std::string::npos);
}

TEST_CASE("HTTP log sanitizer summarizes invalid and oversized JSON without parsing errors") {
    const auto invalid =
        http_log::summarize_body("var broken", "application/json", http_log::BodyDirection::request);
    CHECK(invalid == "<invalid-json request: content_type=application/json, size=10 bytes>");

    const std::string oversized(http_log::MAX_JSON_PARSE_BYTES + 1, 'x');
    const auto oversized_summary =
        http_log::summarize_body(oversized, "application/json", http_log::BodyDirection::request);
    CHECK(oversized_summary.find("<oversized-json request:") == 0);
}

TEST_CASE("HTTP log sanitizer applies a hard output limit") {
    const std::string large_json = "{\"value\":\"" + std::string(8000, 'x') + "\"}";
    const auto summary =
        http_log::summarize_body(large_json, "application/json", http_log::BodyDirection::response);

    CHECK(summary.size() <= http_log::MAX_LOG_BODY_BYTES);
    CHECK(summary.find("<truncated, original_size=") != std::string::npos);
}

TEST_CASE("HTTP log sanitizer protects log structure from content-type control characters") {
    const auto summary =
        http_log::summarize_body("abc", "text/plain\nforged", http_log::BodyDirection::request);
    CHECK(summary == "<non-json request: content_type=invalid, size=3 bytes>");
}

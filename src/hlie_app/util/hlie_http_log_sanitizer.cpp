#include "hlie_app/util/hlie_http_log_sanitizer.h"

#include <cctype>
#include <string>

#include "json.hpp"

HLIE_NAMESPACE_BEGIN

namespace http_log {
namespace {

std::string normalized_media_type(const std::string_view content_type) {
    const auto parameter = content_type.find(';');
    auto media_type = content_type.substr(0, parameter);
    while (!media_type.empty() && std::isspace(static_cast<unsigned char>(media_type.front()))) {
        media_type.remove_prefix(1);
    }
    while (!media_type.empty() && std::isspace(static_cast<unsigned char>(media_type.back()))) {
        media_type.remove_suffix(1);
    }
    if (media_type.empty() || media_type.size() > 96) {
        return "unknown";
    }

    std::string result;
    result.reserve(media_type.size());
    for (const unsigned char character : media_type) {
        if (!(std::isalnum(character) || character == '/' || character == '+' || character == '-' || character == '.')) {
            return "invalid";
        }
        result.push_back(static_cast<char>(std::tolower(character)));
    }
    return result;
}

const char* direction_name(const BodyDirection direction) {
    return direction == BodyDirection::request ? "request" : "response";
}

bool is_sensitive_key(std::string key) {
    for (char& character : key) {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    return key == "api_key" || key == "apikey" || key == "authorization" ||
        key == "access_token" || key == "refresh_token" || key == "password" ||
        key == "secret" || key == "client_secret";
}

void redact_sensitive_fields(nlohmann::json& value) {
    if (value.is_object()) {
        for (auto& [key, child] : value.items()) {
            if (is_sensitive_key(key)) {
                child = "<redacted>";
            } else {
                redact_sensitive_fields(child);
            }
        }
    } else if (value.is_array()) {
        for (auto& child : value) {
            redact_sensitive_fields(child);
        }
    }
}

std::string metadata_summary(const char* kind,
                             const std::string_view content_type,
                             const std::size_t size,
                             const BodyDirection direction) {
    return "<" + std::string(kind) + " " + direction_name(direction) + ": content_type=" +
        normalized_media_type(content_type) + ", size=" + std::to_string(size) + " bytes>";
}

void redact_base64_fields(nlohmann::json& body_json) {
    if (body_json.contains("messages") && body_json["messages"].is_array()) {
        for (auto& message : body_json["messages"]) {
            if (!message.is_object() || !message.contains("content") || !message["content"].is_array()) {
                continue;
            }
            for (auto& content_item : message["content"]) {
                if (!content_item.is_object() || !content_item.contains("type") ||
                    !content_item["type"].is_string() || content_item["type"] != "image_url" ||
                    !content_item.contains("image_url") || !content_item["image_url"].is_object()) {
                    continue;
                }
                auto& image_url = content_item["image_url"];
                if (image_url.contains("url") && image_url["url"].is_string()) {
                    const auto original_size = image_url["url"].get_ref<const std::string&>().size();
                    image_url["url"] = "<redacted image_url, size=" + std::to_string(original_size) + " bytes>";
                }
            }
        }
    }

    if (body_json.contains("data") && body_json["data"].is_array()) {
        for (auto& item : body_json["data"]) {
            if (item.is_object() && item.contains("b64_json") && item["b64_json"].is_string()) {
                const auto original_size = item["b64_json"].get_ref<const std::string&>().size();
                item["b64_json"] = "<redacted b64_json, size=" + std::to_string(original_size) + " bytes>";
            }
        }
    }
}

std::string bounded_utf8(std::string value) {
    if (value.size() <= MAX_LOG_BODY_BYTES) {
        return value;
    }

    const std::string suffix = "...<truncated, original_size=" + std::to_string(value.size()) + " bytes>";
    std::size_t prefix_size = MAX_LOG_BODY_BYTES > suffix.size() ? MAX_LOG_BODY_BYTES - suffix.size() : 0;
    while (prefix_size > 0 &&
           (static_cast<unsigned char>(value[prefix_size]) & 0xC0U) == 0x80U) {
        --prefix_size;
    }
    value.resize(prefix_size);
    value += suffix;
    return value;
}

} // namespace

bool is_static_asset_path(const std::string_view path) {
    return path == "/" || path == "/index.html" || path == "/loading.html" ||
        path == "/bundle.js" || path == "/bundle.css" || path == "/favicon.ico";
}

bool is_json_content_type(const std::string_view content_type) {
    const std::string media_type = normalized_media_type(content_type);
    return media_type == "application/json" ||
        (media_type.size() > 5 && media_type.compare(media_type.size() - 5, 5, "+json") == 0);
}

std::string summarize_body(const std::string_view body,
                           const std::string_view content_type,
                           const BodyDirection direction) {
    if (body.empty()) {
        return {};
    }
    if (!is_json_content_type(content_type)) {
        return metadata_summary("non-json", content_type, body.size(), direction);
    }
    if (body.size() > MAX_JSON_PARSE_BYTES) {
        return metadata_summary("oversized-json", content_type, body.size(), direction);
    }

    auto body_json = nlohmann::json::parse(body, nullptr, false);
    if (body_json.is_discarded()) {
        return metadata_summary("invalid-json", content_type, body.size(), direction);
    }

    redact_sensitive_fields(body_json);
    redact_base64_fields(body_json);
    return bounded_utf8(body_json.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace));
}

} // namespace http_log

HLIE_NAMESPACE_END

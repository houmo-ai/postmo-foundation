#ifndef HLIE_HTTP_LOG_SANITIZER_H
#define HLIE_HTTP_LOG_SANITIZER_H

#include <cstddef>
#include <string>
#include <string_view>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

namespace http_log {

constexpr std::size_t MAX_JSON_PARSE_BYTES = 64U * 1024U;
constexpr std::size_t MAX_LOG_BODY_BYTES = 4U * 1024U;

enum class BodyDirection {
    request,
    response,
};

[[nodiscard]] HLIE_API bool is_static_asset_path(std::string_view path);
[[nodiscard]] HLIE_API bool is_json_content_type(std::string_view content_type);

/**
 * Produce a bounded, single-line representation suitable for debug logging.
 * JSON is parsed only below MAX_JSON_PARSE_BYTES and known base64 fields are
 * redacted. Non-JSON and oversized bodies are represented by metadata only.
 */
[[nodiscard]] HLIE_API std::string summarize_body(std::string_view body,
                                                  std::string_view content_type,
                                                  BodyDirection direction);

} // namespace http_log

HLIE_NAMESPACE_END

#endif // HLIE_HTTP_LOG_SANITIZER_H

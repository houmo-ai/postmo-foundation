#ifndef HLIE_DOCUMENT_CORE_H
#define HLIE_DOCUMENT_CORE_H

#include <string>

#include "hlie_app/document/hlie_document_types.h"
#include "hlie_base/hlie_base.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

// Hardware-independent document post-processing shared by the runtime and
// focused unit tests.
namespace DocumentCore {

void apply_page_margin_policy(DocumentPreprocessResult& result, bool include_page_margins);
[[nodiscard]] nlohmann::json build_page_layout_metrics(const DocumentPage& page, bool include_page_margins);
[[nodiscard]] nlohmann::json build_layout_metrics(const DocumentPreprocessResult& result, bool include_page_margins);
[[nodiscard]] std::string merge_document_text(const DocumentPreprocessResult& result);
[[nodiscard]] bool is_low_confidence_fallback_region(const DocumentLayoutRegion& region);
[[nodiscard]] bool looks_like_non_transcription_response(std::string text);

} // namespace DocumentCore

HLIE_NAMESPACE_END

#endif // HLIE_DOCUMENT_CORE_H

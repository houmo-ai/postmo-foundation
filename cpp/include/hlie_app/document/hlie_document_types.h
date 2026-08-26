#ifndef HLIE_DOCUMENT_TYPES_H
#define HLIE_DOCUMENT_TYPES_H

#include <string>
#include <vector>

#include "hlie_app/document/hlie_pdf_renderer.h"
#include "hlie_base/hlie_base.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

enum class DocumentRegionKind {
    Unknown,
    Text,
    Title,
    Table,
    Figure,
    Formula,
    Header,
    Footer,
};

struct DocumentBoundingBox {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
};

struct DocumentPoint {
    float x = 0.0f;
    float y = 0.0f;
};

struct DocumentRegionCrop {
    int width = 0;
    int height = 0;
    int channels = 3;
    int stride = 0;
    DocumentBoundingBox source_box;
    std::string mode;
    raw_buffer pixels;
};

struct DocumentLayoutRegion {
    int id = 0;
    int class_id = -1;
    int source_query = -1;
    int order = -1;
    DocumentRegionKind kind = DocumentRegionKind::Unknown;
    std::string label;
    std::string task_type;
    float score = 0.0f;
    DocumentBoundingBox box;
    std::vector<DocumentPoint> polygon;
    DocumentRegionCrop crop;
    std::string text;
};

struct DocumentPage {
    PdfPageImage image;
    std::vector<DocumentLayoutRegion> regions;
};

struct DocumentPreprocessResult {
    std::string source;
    std::string filename;
    std::string content_type;
    std::vector<DocumentPage> pages;
};

const char* document_region_kind_to_string(DocumentRegionKind kind);
nlohmann::json document_page_to_json(const DocumentPage& page);
nlohmann::json document_preprocess_result_to_json(const DocumentPreprocessResult& result);

HLIE_NAMESPACE_END

#endif // HLIE_DOCUMENT_TYPES_H

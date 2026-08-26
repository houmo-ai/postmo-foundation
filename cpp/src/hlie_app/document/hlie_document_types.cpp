#include "hlie_app/document/hlie_document_types.h"

#include <algorithm>

HLIE_NAMESPACE_BEGIN

const char* document_region_kind_to_string(const DocumentRegionKind kind) {
    switch (kind) {
    case DocumentRegionKind::Text:
        return "text";
    case DocumentRegionKind::Title:
        return "title";
    case DocumentRegionKind::Table:
        return "table";
    case DocumentRegionKind::Figure:
        return "figure";
    case DocumentRegionKind::Formula:
        return "formula";
    case DocumentRegionKind::Header:
        return "header";
    case DocumentRegionKind::Footer:
        return "footer";
    case DocumentRegionKind::Unknown:
    default:
        return "unknown";
    }
}

nlohmann::json document_page_to_json(const DocumentPage& page) {
    nlohmann::json regions = nlohmann::json::array();
    const float width = static_cast<float>(std::max(1, page.image.width));
    const float height = static_cast<float>(std::max(1, page.image.height));
    for (const auto& region : page.regions) {
        regions.push_back({
            {"id", region.id},
            {"class_id", region.class_id},
            {"order", region.order >= 0 ? nlohmann::json(region.order) : nlohmann::json(nullptr)},
            {"label", region.label},
            {"task_type", region.task_type},
            {"kind", document_region_kind_to_string(region.kind)},
            {"score", region.score},
            {"box", {
                {"x0", region.box.x0},
                {"y0", region.box.y0},
                {"x1", region.box.x1},
                {"y1", region.box.y1},
            }},
            {"bbox_2d", {
                static_cast<int>(region.box.x0 / width * 1000.0f),
                static_cast<int>(region.box.y0 / height * 1000.0f),
                static_cast<int>(region.box.x1 / width * 1000.0f),
                static_cast<int>(region.box.y1 / height * 1000.0f),
            }},
            {"polygon", nlohmann::json::array()},
            {"crop", region.crop.pixels.empty()
                ? nlohmann::json(nullptr)
                : nlohmann::json({
                    {"width", region.crop.width},
                    {"height", region.crop.height},
                    {"channels", region.crop.channels},
                    {"stride", region.crop.stride},
                    {"bytes", region.crop.pixels.size()},
                    {"mode", region.crop.mode},
                    {"source_box", {
                        {"x0", region.crop.source_box.x0},
                        {"y0", region.crop.source_box.y0},
                        {"x1", region.crop.source_box.x1},
                        {"y1", region.crop.source_box.y1},
                    }},
                    {"source_bbox_2d", {
                        static_cast<int>(region.crop.source_box.x0 / width * 1000.0f),
                        static_cast<int>(region.crop.source_box.y0 / height * 1000.0f),
                        static_cast<int>(region.crop.source_box.x1 / width * 1000.0f),
                        static_cast<int>(region.crop.source_box.y1 / height * 1000.0f),
                    }},
                })},
            {"text", region.text},
        });
        auto& polygon_json = regions.back()["polygon"];
        for (const auto& point : region.polygon) {
            polygon_json.push_back({
                static_cast<int>(point.x / width * 1000.0f),
                static_cast<int>(point.y / height * 1000.0f),
            });
        }
    }

    return {
        {"index", page.image.page_index},
        {"width", page.image.width},
        {"height", page.image.height},
        {"channels", page.image.channels},
        {"stride", page.image.stride},
        {"bytes", page.image.pixels.size()},
        {"regions", regions},
    };
}

nlohmann::json document_preprocess_result_to_json(const DocumentPreprocessResult& result) {
    nlohmann::json pages = nlohmann::json::array();
    for (const auto& page : result.pages) {
        pages.push_back(document_page_to_json(page));
    }

    return {
        {"source", result.source},
        {"filename", result.filename},
        {"content_type", result.content_type},
        {"page_count", result.pages.size()},
        {"pages", pages},
    };
}

HLIE_NAMESPACE_END

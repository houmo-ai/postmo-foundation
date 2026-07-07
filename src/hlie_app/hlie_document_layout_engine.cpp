#include "hlie_app/hlie_document_layout_engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iterator>
#include <tuple>

#include "gguf.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_tcim_util.h"

HLIE_NAMESPACE_BEGIN

namespace {
    constexpr int kLayoutInputSize = 800;
    constexpr float kScoreThreshold = 0.30f;
    constexpr float kNmsSameClassThreshold = 0.60f;
    constexpr float kNmsDiffClassThreshold = 0.98f;
    constexpr float kMinBoxSidePixels = 2.0f;
    constexpr int kMaxRegions = 300;

    const std::array<const char*, 25> kLayoutLabels = {
        "abstract", "algorithm", "aside_text", "chart", "content", "display_formula", "doc_title",
        "figure_title", "footer", "footer_image", "footnote", "formula_number", "header", "header_image",
        "image", "inline_formula", "number", "paragraph_title", "reference", "reference_content",
        "seal", "table", "text", "vertical_text", "vision_footnote",
    };

    bool visual_reading_order_less(const DocumentLayoutRegion& lhs, const DocumentLayoutRegion& rhs);
    std::vector<DocumentPoint> box_to_polygon(const DocumentBoundingBox& box);

    std::string shape_to_string(const std::vector<int64_t>& shape) {
        std::string result;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) {
                result += " x ";
            }
            result += std::to_string(shape[i]);
        }
        return result;
    }

    const char* tcim_dtype_to_string(const tcim::DataType dtype) {
        switch (dtype) {
        case tcim::DataType::INT8:
            return "int8";
        case tcim::DataType::UINT8:
            return "uint8";
        case tcim::DataType::INT16:
            return "int16";
        case tcim::DataType::UINT16:
            return "uint16";
        case tcim::DataType::INT32:
            return "int32";
        case tcim::DataType::UINT32:
            return "uint32";
        case tcim::DataType::FLOAT16:
            return "float16";
        case tcim::DataType::FLOAT32:
            return "float32";
        default:
            return "unknown";
        }
    }

    nlohmann::json tensor_info_to_json(const std::string& name, const tcim::TensorInfo& info) {
        const auto contiguous = info.AsContiguous();
        return {
            {"name", name},
            {"shape", contiguous.Shape()},
            {"shape_text", shape_to_string(contiguous.Shape())},
            {"dtype", tcim_dtype_to_string(contiguous.DataType())},
            {"mem_size", contiguous.MemSize()},
        };
    }

    float sigmoid(const float value) {
        if (value >= 0.0f) {
            const float z = std::exp(-value);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(value);
        return z / (1.0f + z);
    }

    float clamp_float(const float value, const float lo, const float hi) {
        return std::max(lo, std::min(value, hi));
    }

    float box_area(const DocumentBoundingBox& box) {
        return std::max(0.0f, box.x1 - box.x0) * std::max(0.0f, box.y1 - box.y0);
    }

    float box_iou(const DocumentBoundingBox& a, const DocumentBoundingBox& b) {
        const float x0 = std::max(a.x0, b.x0);
        const float y0 = std::max(a.y0, b.y0);
        const float x1 = std::min(a.x1, b.x1);
        const float y1 = std::min(a.y1, b.y1);
        const float intersection = std::max(0.0f, x1 - x0 + 1.0f) * std::max(0.0f, y1 - y0 + 1.0f);
        const float area_a = std::max(0.0f, a.x1 - a.x0 + 1.0f) * std::max(0.0f, a.y1 - a.y0 + 1.0f);
        const float area_b = std::max(0.0f, b.x1 - b.x0 + 1.0f) * std::max(0.0f, b.y1 - b.y0 + 1.0f);
        const float union_area = area_a + area_b - intersection;
        return union_area <= 0.0f ? 0.0f : intersection / union_area;
    }

    float box_intersection_area(const DocumentBoundingBox& a, const DocumentBoundingBox& b) {
        const float x0 = std::max(a.x0, b.x0);
        const float y0 = std::max(a.y0, b.y0);
        const float x1 = std::min(a.x1, b.x1);
        const float y1 = std::min(a.y1, b.y1);
        return std::max(0.0f, x1 - x0) * std::max(0.0f, y1 - y0);
    }

    bool is_textual_task(const DocumentLayoutRegion& region) {
        return region.task_type == "text" || region.task_type == "table" || region.task_type == "formula";
    }

    bool is_dark_pixel(const uint8_t* pixel) {
        const int luminance = (static_cast<int>(pixel[0]) * 30 + static_cast<int>(pixel[1]) * 59 +
                               static_cast<int>(pixel[2]) * 11) / 100;
        return luminance < 245;
    }

    bool should_suppress_region(const DocumentLayoutRegion& candidate, const DocumentLayoutRegion& existing) {
        if (candidate.task_type == "abandon" || existing.task_type == "abandon") {
            return box_iou(candidate.box, existing.box) >= kNmsSameClassThreshold;
        }

        const float intersection = box_intersection_area(candidate.box, existing.box);
        const float candidate_area = box_area(candidate.box);
        const float existing_area = box_area(existing.box);
        if (candidate_area <= 1.0f || existing_area <= 1.0f || intersection <= 0.0f) {
            return false;
        }

        const float overlap_min = intersection / std::max(1.0f, std::min(candidate_area, existing_area));
        if (candidate.class_id == existing.class_id && overlap_min >= 0.78f) {
            return true;
        }
        const float candidate_coverage = intersection / std::max(1.0f, candidate_area);
        const float existing_coverage = intersection / std::max(1.0f, existing_area);
        if (is_textual_task(candidate) && is_textual_task(existing) && overlap_min >= 0.86f &&
            candidate_coverage >= 0.82f && existing_coverage >= 0.82f) {
            return true;
        }
        return box_iou(candidate.box, existing.box) >=
               (candidate.class_id == existing.class_id ? kNmsSameClassThreshold : kNmsDiffClassThreshold);
    }

    PdfPageImage crop_page_image(const PdfPageImage& image, const int y0, const int height) {
        PdfPageImage crop;
        crop.page_index = image.page_index;
        crop.width = image.width;
        crop.height = std::max(0, std::min(height, image.height - y0));
        crop.channels = image.channels;
        crop.stride = crop.width * crop.channels;
        if (crop.width <= 0 || crop.height <= 0 || image.pixels.empty()) {
            return crop;
        }
        crop.pixels.resize(static_cast<size_t>(crop.stride) * static_cast<size_t>(crop.height));
        for (int y = 0; y < crop.height; ++y) {
            const auto* src = image.pixels.data() + static_cast<size_t>(y0 + y) * static_cast<size_t>(image.stride);
            auto* dst = crop.pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(crop.stride);
            std::copy(src, src + static_cast<size_t>(crop.stride), dst);
        }
        return crop;
    }

    void offset_region(DocumentLayoutRegion& region, const float dx, const float dy,
                       const int image_width, const int image_height) {
        region.box.x0 = clamp_float(region.box.x0 + dx, 0.0f, static_cast<float>(image_width));
        region.box.x1 = clamp_float(region.box.x1 + dx, 0.0f, static_cast<float>(image_width));
        region.box.y0 = clamp_float(region.box.y0 + dy, 0.0f, static_cast<float>(image_height));
        region.box.y1 = clamp_float(region.box.y1 + dy, 0.0f, static_cast<float>(image_height));
        for (auto& point : region.polygon) {
            point.x = clamp_float(point.x + dx, 0.0f, static_cast<float>(image_width));
            point.y = clamp_float(point.y + dy, 0.0f, static_cast<float>(image_height));
        }
    }

    void keep_regions_in_vertical_band(std::vector<DocumentLayoutRegion>& regions, const float y0, const float y1) {
        regions.erase(std::remove_if(regions.begin(), regions.end(), [&](const DocumentLayoutRegion& region) {
            if (region.task_type == "abandon") {
                return true;
            }
            const float center_y = (region.box.y0 + region.box.y1) * 0.5f;
            return center_y < y0 || center_y >= y1;
        }), regions.end());
    }

    void merge_regions(std::vector<DocumentLayoutRegion>& base, std::vector<DocumentLayoutRegion> extra) {
        if (extra.empty()) {
            return;
        }
        base.insert(base.end(), std::make_move_iterator(extra.begin()), std::make_move_iterator(extra.end()));
        std::sort(base.begin(), base.end(), [](const DocumentLayoutRegion& lhs, const DocumentLayoutRegion& rhs) {
            return lhs.score > rhs.score;
        });

        std::vector<DocumentLayoutRegion> kept;
        kept.reserve(base.size());
        for (const auto& candidate : base) {
            bool suppressed = false;
            for (const auto& existing : kept) {
                if (should_suppress_region(candidate, existing)) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                kept.push_back(candidate);
            }
        }
        std::sort(kept.begin(), kept.end(), visual_reading_order_less);
        for (size_t i = 0; i < kept.size(); ++i) {
            kept[i].id = static_cast<int>(i);
        }
        base = std::move(kept);
    }

    void add_projection_fallback_regions(const PdfPageImage& image, std::vector<DocumentLayoutRegion>& regions) {
        if (image.width <= 0 || image.height <= 0 || image.channels < 3 || image.pixels.empty()) {
            return;
        }

        const int min_row_ink = std::max(6, image.width / 260);
        const int merge_gap = std::max(14, image.height / 100);
        const int pad_x = std::max(6, image.width / 180);
        const int pad_y = std::max(4, image.height / 360);
        const int min_band_height = std::max(8, image.height / 260);
        const int min_fallback_segment_height = std::max(32, image.height / 80);
        const int min_band_width = std::max(40, image.width / 24);

        struct Band {
            int y0 = 0;
            int y1 = 0;
            int x0 = 0;
            int x1 = 0;
        };
        std::vector<Band> bands;
        Band current;
        int gap = 0;
        bool in_band = false;

        for (int y = 0; y < image.height; ++y) {
            int ink = 0;
            int x_min = image.width;
            int x_max = 0;
            const auto* row = image.pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(image.stride);
            for (int x = 0; x < image.width; ++x) {
                const auto* pixel = row + static_cast<size_t>(x) * static_cast<size_t>(image.channels);
                if (!is_dark_pixel(pixel)) {
                    continue;
                }
                ++ink;
                x_min = std::min(x_min, x);
                x_max = std::max(x_max, x);
            }

            if (ink >= min_row_ink) {
                if (!in_band) {
                    current = {y, y + 1, x_min, x_max + 1};
                    in_band = true;
                } else {
                    current.y1 = y + 1;
                    current.x0 = std::min(current.x0, x_min);
                    current.x1 = std::max(current.x1, x_max + 1);
                }
                gap = 0;
                continue;
            }

            if (in_band) {
                ++gap;
                if (gap > merge_gap) {
                    current.y1 -= gap;
                    bands.push_back(current);
                    in_band = false;
                    gap = 0;
                }
            }
        }
        if (in_band) {
            current.y1 -= gap;
            bands.push_back(current);
        }

        std::vector<DocumentLayoutRegion> fallback_regions;
        for (const auto& band : bands) {
            DocumentBoundingBox box{
                static_cast<float>(std::max(0, band.x0 - pad_x)),
                static_cast<float>(std::max(0, band.y0 - pad_y)),
                static_cast<float>(std::min(image.width, band.x1 + pad_x)),
                static_cast<float>(std::min(image.height, band.y1 + pad_y)),
            };
            if (box.x1 - box.x0 < static_cast<float>(min_band_width) ||
                box.y1 - box.y0 < static_cast<float>(min_band_height)) {
                continue;
            }

            const int box_y0 = std::max(0, static_cast<int>(std::floor(box.y0)));
            const int box_y1 = std::min(image.height, static_cast<int>(std::ceil(box.y1)));
            if (box_y1 <= box_y0) {
                continue;
            }
            std::vector<uint8_t> covered_rows(static_cast<size_t>(box_y1 - box_y0), 0);
            const float box_width = std::max(1.0f, box.x1 - box.x0);
            for (const auto& region : regions) {
                if (!is_textual_task(region)) {
                    continue;
                }
                const float x0 = std::max(box.x0, region.box.x0);
                const float x1 = std::min(box.x1, region.box.x1);
                if (x1 <= x0 || (x1 - x0) / box_width < 0.25f) {
                    continue;
                }
                const int y0 = std::max(box_y0, static_cast<int>(std::floor(region.box.y0)));
                const int y1 = std::min(box_y1, static_cast<int>(std::ceil(region.box.y1)));
                for (int y = y0; y < y1; ++y) {
                    covered_rows[static_cast<size_t>(y - box_y0)] = 1;
                }
            }

            const int split_merge_gap = std::max(6, image.height / 220);
            int segment_y0 = -1;
            int segment_last = -1;
            int gap = 0;
            auto emit_segment = [&](const int y0, const int y1) {
                DocumentBoundingBox segment_box{
                    box.x0,
                    static_cast<float>(std::max(0, y0 - pad_y)),
                    box.x1,
                    static_cast<float>(std::min(image.height, y1 + pad_y)),
                };
                if (segment_box.y1 - segment_box.y0 < static_cast<float>(min_fallback_segment_height)) {
                    return;
                }
                DocumentLayoutRegion region;
                region.source_query = -1;
                region.class_id = -1;
                region.order = 100000 + static_cast<int>(fallback_regions.size());
                region.label = "text";
                region.task_type = "text";
                region.kind = DocumentRegionKind::Text;
                region.score = 0.20f;
                region.box = segment_box;
                region.polygon = box_to_polygon(segment_box);
                fallback_regions.push_back(std::move(region));
            };

            for (int y = box_y0; y < box_y1; ++y) {
                const bool covered = covered_rows[static_cast<size_t>(y - box_y0)] != 0;
                if (!covered) {
                    if (segment_y0 < 0) {
                        segment_y0 = y;
                    }
                    segment_last = y;
                    gap = 0;
                    continue;
                }
                if (segment_y0 >= 0) {
                    ++gap;
                    if (gap > split_merge_gap) {
                        emit_segment(segment_y0, segment_last + 1);
                        segment_y0 = -1;
                        segment_last = -1;
                        gap = 0;
                    }
                }
            }
            if (segment_y0 >= 0) {
                emit_segment(segment_y0, segment_last + 1);
            }
        }

        if (!fallback_regions.empty()) {
            const auto before_count = regions.size();
            merge_regions(regions, std::move(fallback_regions));
            LOG_INFO("Document layout projection fallback page={}, regions_before={}, regions_after={}",
                     image.page_index, before_count, regions.size());
        }
    }

    const char* label_for_class(const int class_id) {
        if (class_id >= 0 && class_id < static_cast<int>(kLayoutLabels.size())) {
            return kLayoutLabels[static_cast<size_t>(class_id)];
        }
        return "unknown";
    }

    DocumentRegionKind kind_for_label(const std::string& label) {
        if (label == "table") {
            return DocumentRegionKind::Table;
        }
        if (label == "chart" || label == "image" || label == "seal") {
            return DocumentRegionKind::Figure;
        }
        if (label == "display_formula" || label == "inline_formula" || label == "formula_number") {
            return DocumentRegionKind::Formula;
        }
        if (label == "doc_title" || label == "paragraph_title" || label == "figure_title") {
            return DocumentRegionKind::Title;
        }
        if (label == "header" || label == "header_image") {
            return DocumentRegionKind::Header;
        }
        if (label == "footer" || label == "footer_image") {
            return DocumentRegionKind::Footer;
        }
        return DocumentRegionKind::Text;
    }

    std::vector<DocumentPoint> box_to_polygon(const DocumentBoundingBox& box) {
        return {
            {box.x0, box.y0},
            {box.x1, box.y0},
            {box.x1, box.y1},
            {box.x0, box.y1},
        };
    }

    std::string task_type_for_label(const std::string& label) {
        if (label == "table") {
            return "table";
        }
        if (label == "display_formula" || label == "inline_formula") {
            return "formula";
        }
        if (label == "chart" || label == "image") {
            return "skip";
        }
        if (label == "aside_text" || label == "reference" || label == "footer_image" || label == "header_image") {
            return "abandon";
        }
        return "text";
    }

    bool is_preserved_when_contained(const int class_id) {
        const std::string label = label_for_class(class_id);
        return label == "image" || label == "seal" || label == "chart";
    }

    bool is_contained(const DocumentBoundingBox& inner, const DocumentBoundingBox& outer) {
        const float inner_area = box_area(inner);
        if (inner_area <= 0.0f) {
            return false;
        }
        const float x0 = std::max(inner.x0, outer.x0);
        const float y0 = std::max(inner.y0, outer.y0);
        const float x1 = std::min(inner.x1, outer.x1);
        const float y1 = std::min(inner.y1, outer.y1);
        const float intersection = std::max(0.0f, x1 - x0) * std::max(0.0f, y1 - y0);
        return intersection / inner_area >= 0.8f;
    }

    float cubic_weight(const float x) {
        constexpr float a = -0.75f;
        const float ax = std::abs(x);
        if (ax <= 1.0f) {
            return (a + 2.0f) * ax * ax * ax - (a + 3.0f) * ax * ax + 1.0f;
        }
        if (ax < 2.0f) {
            return a * ax * ax * ax - 5.0f * a * ax * ax + 8.0f * a * ax - 4.0f * a;
        }
        return 0.0f;
    }

    float cross_product(const DocumentPoint& origin, const DocumentPoint& a, const DocumentPoint& b) {
        return (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
    }

    std::vector<DocumentPoint> convex_hull(std::vector<DocumentPoint> points) {
        if (points.size() <= 3) {
            return points;
        }
        std::sort(points.begin(), points.end(), [](const DocumentPoint& lhs, const DocumentPoint& rhs) {
            if (lhs.x == rhs.x) {
                return lhs.y < rhs.y;
            }
            return lhs.x < rhs.x;
        });
        points.erase(std::unique(points.begin(), points.end(), [](const DocumentPoint& lhs, const DocumentPoint& rhs) {
                         return std::abs(lhs.x - rhs.x) < 1e-3f && std::abs(lhs.y - rhs.y) < 1e-3f;
                     }),
                     points.end());
        if (points.size() <= 3) {
            return points;
        }

        std::vector<DocumentPoint> hull;
        hull.reserve(points.size() * 2);
        for (const auto& point : points) {
            while (hull.size() >= 2 &&
                   cross_product(hull[hull.size() - 2], hull[hull.size() - 1], point) <= 0.0f) {
                hull.pop_back();
            }
            hull.push_back(point);
        }
        const size_t lower_size = hull.size();
        for (auto it = points.rbegin() + 1; it != points.rend(); ++it) {
            while (hull.size() > lower_size &&
                   cross_product(hull[hull.size() - 2], hull[hull.size() - 1], *it) <= 0.0f) {
                hull.pop_back();
            }
            hull.push_back(*it);
        }
        if (!hull.empty()) {
            hull.pop_back();
        }
        return hull;
    }

    float point_line_distance(const DocumentPoint& point, const DocumentPoint& start, const DocumentPoint& end) {
        const float dx = end.x - start.x;
        const float dy = end.y - start.y;
        const float denom = std::sqrt(dx * dx + dy * dy);
        if (denom <= 1e-6f) {
            const float px = point.x - start.x;
            const float py = point.y - start.y;
            return std::sqrt(px * px + py * py);
        }
        return std::abs(dy * point.x - dx * point.y + end.x * start.y - end.y * start.x) / denom;
    }

    std::vector<DocumentPoint> simplify_polygon(const std::vector<DocumentPoint>& points, const float epsilon) {
        if (points.size() <= 8) {
            return points;
        }

        std::vector<DocumentPoint> closed = points;
        closed.push_back(points.front());
        std::vector<bool> keep(closed.size(), false);
        keep.front() = true;
        keep.back() = true;

        std::function<void(size_t, size_t)> simplify_range = [&](const size_t first, const size_t last) {
            if (last <= first + 1) {
                return;
            }
            float max_distance = 0.0f;
            size_t split = first;
            for (size_t i = first + 1; i < last; ++i) {
                const float distance = point_line_distance(closed[i], closed[first], closed[last]);
                if (distance > max_distance) {
                    max_distance = distance;
                    split = i;
                }
            }
            if (max_distance > epsilon) {
                keep[split] = true;
                simplify_range(first, split);
                simplify_range(split, last);
            }
        };
        simplify_range(0, closed.size() - 1);

        std::vector<DocumentPoint> simplified;
        for (size_t i = 0; i + 1 < closed.size(); ++i) {
            if (keep[i]) {
                simplified.push_back(closed[i]);
            }
        }
        return simplified.size() >= 3 ? simplified : points;
    }

    std::vector<DocumentPoint> polygon_from_mask(const std::vector<float>& masks, const int query, const int query_count,
                                                 const int mask_height, const int mask_width,
                                                 const DocumentBoundingBox& box, const int image_width, const int image_height) {
        if (masks.empty() || query < 0 || query >= query_count || mask_height <= 0 || mask_width <= 0 ||
            image_width <= 0 || image_height <= 0) {
            return {};
        }

        const int x_min = static_cast<int>(std::round(box.x0));
        const int y_min = static_cast<int>(std::round(box.y0));
        const int x_max = static_cast<int>(std::round(box.x1));
        const int y_max = static_cast<int>(std::round(box.y1));
        const int box_width = x_max - x_min;
        const int box_height = y_max - y_min;
        if (box_width <= 0 || box_height <= 0) {
            return {};
        }

        const float scale_x = static_cast<float>(mask_width) / static_cast<float>(image_width);
        const float scale_y = static_cast<float>(mask_height) / static_cast<float>(image_height);
        int x_start = static_cast<int>(std::round(static_cast<float>(x_min) * scale_x));
        int x_end = static_cast<int>(std::round(static_cast<float>(x_max) * scale_x));
        int y_start = static_cast<int>(std::round(static_cast<float>(y_min) * scale_y));
        int y_end = static_cast<int>(std::round(static_cast<float>(y_max) * scale_y));
        x_start = std::max(0, std::min(x_start, mask_width));
        x_end = std::max(0, std::min(x_end, mask_width));
        y_start = std::max(0, std::min(y_start, mask_height));
        y_end = std::max(0, std::min(y_end, mask_height));
        if (x_end <= x_start || y_end <= y_start) {
            return {};
        }

        const size_t query_offset = static_cast<size_t>(query) * static_cast<size_t>(mask_height) * static_cast<size_t>(mask_width);
        if (query_offset + static_cast<size_t>(mask_height) * static_cast<size_t>(mask_width) > masks.size()) {
            return {};
        }

        std::vector<uint8_t> resized_mask(static_cast<size_t>(box_width) * static_cast<size_t>(box_height), 0);
        int active_pixels = 0;
        for (int y = 0; y < box_height; ++y) {
            const int src_y = y_start + std::min(y_end - y_start - 1,
                                                 static_cast<int>((static_cast<int64_t>(y) * (y_end - y_start)) / box_height));
            for (int x = 0; x < box_width; ++x) {
                const int src_x = x_start + std::min(x_end - x_start - 1,
                                                     static_cast<int>((static_cast<int64_t>(x) * (x_end - x_start)) / box_width));
                const float value = masks[query_offset + static_cast<size_t>(src_y) * static_cast<size_t>(mask_width) +
                                          static_cast<size_t>(src_x)];
                const bool active = value > 0.0f;
                resized_mask[static_cast<size_t>(y) * static_cast<size_t>(box_width) + static_cast<size_t>(x)] = active ? 1 : 0;
                active_pixels += active ? 1 : 0;
            }
        }
        if (active_pixels <= 0) {
            return {};
        }

        std::vector<DocumentPoint> boundary_points;
        boundary_points.reserve(static_cast<size_t>(box_width + box_height) * 2);
        const auto is_active = [&](const int x, const int y) {
            if (x < 0 || y < 0 || x >= box_width || y >= box_height) {
                return false;
            }
            return resized_mask[static_cast<size_t>(y) * static_cast<size_t>(box_width) + static_cast<size_t>(x)] != 0;
        };
        for (int y = 0; y < box_height; ++y) {
            for (int x = 0; x < box_width; ++x) {
                if (!is_active(x, y)) {
                    continue;
                }
                if (!is_active(x - 1, y) || !is_active(x + 1, y) || !is_active(x, y - 1) || !is_active(x, y + 1)) {
                    boundary_points.push_back({
                        clamp_float(static_cast<float>(x_min + x), 0.0f, static_cast<float>(image_width)),
                        clamp_float(static_cast<float>(y_min + y), 0.0f, static_cast<float>(image_height)),
                    });
                }
            }
        }
        if (boundary_points.size() < 3) {
            return {};
        }

        auto hull = convex_hull(std::move(boundary_points));
        if (hull.size() < 3) {
            return {};
        }
        const float epsilon = std::max(1.0f, 0.01f * std::max(static_cast<float>(box_width), static_cast<float>(box_height)));
        hull = simplify_polygon(hull, epsilon);
        return hull.size() >= 3 ? hull : std::vector<DocumentPoint>{};
    }

    std::vector<int> build_order_sequence(const std::vector<float>& order_logits, const int query_count) {
        std::vector<int> order_seq(static_cast<size_t>(query_count), 0);
        if (query_count <= 0 || order_logits.size() < static_cast<size_t>(query_count * query_count)) {
            for (int i = 0; i < query_count; ++i) {
                order_seq[static_cast<size_t>(i)] = i;
            }
            return order_seq;
        }

        std::vector<std::pair<float, int>> votes;
        votes.reserve(static_cast<size_t>(query_count));
        for (int i = 0; i < query_count; ++i) {
            float vote = 0.0f;
            for (int j = i + 1; j < query_count; ++j) {
                vote += sigmoid(order_logits[static_cast<size_t>(i * query_count + j)]);
            }
            for (int j = 0; j < i; ++j) {
                vote += 1.0f - sigmoid(order_logits[static_cast<size_t>(j * query_count + i)]);
            }
            votes.emplace_back(vote, i);
        }
        std::sort(votes.begin(), votes.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first == rhs.first) {
                return lhs.second < rhs.second;
            }
            return lhs.first < rhs.first;
        });
        for (int rank = 0; rank < static_cast<int>(votes.size()); ++rank) {
            order_seq[static_cast<size_t>(votes[static_cast<size_t>(rank)].second)] = rank;
        }
        return order_seq;
    }

    bool visual_reading_order_less(const DocumentLayoutRegion& lhs, const DocumentLayoutRegion& rhs) {
        const float lhs_height = std::max(1.0f, lhs.box.y1 - lhs.box.y0);
        const float rhs_height = std::max(1.0f, rhs.box.y1 - rhs.box.y0);
        const float y_tolerance = std::max(8.0f, 0.18f * std::min(lhs_height, rhs_height));
        if (std::abs(lhs.box.y0 - rhs.box.y0) > y_tolerance) {
            return lhs.box.y0 < rhs.box.y0;
        }
        if (std::abs(lhs.box.x0 - rhs.box.x0) > 4.0f) {
            return lhs.box.x0 < rhs.box.x0;
        }
        if (lhs.order != rhs.order) {
            return lhs.order < rhs.order;
        }
        return lhs.score > rhs.score;
    }
} // namespace

bool DocumentLayoutEngine::init(const std::string& mmproj_path, const std::vector<int>& devices) {
    stop();
    set_devices(devices.empty() ? std::vector<int>{0} : devices);

    if (!load_layout_module(mmproj_path)) {
        stop();
        return false;
    }

    capture_module_info();
    return true;
}

void DocumentLayoutEngine::stop() {
    layout_module_.reset();
    module_info_ = nlohmann::json::object();
}

void DocumentLayoutEngine::detect_layout(DocumentPreprocessResult& document) const {
    if (!layout_module_) {
        return;
    }

    std::lock_guard<std::mutex> guard(infer_mutex_);
    if (layout_module_->GetOutputNum() < 3) {
        LOG_WARN("Document layout module output count is unexpected: {}", layout_module_->GetOutputNum());
        return;
    }

    const std::string input_name = layout_module_->GetInputName(0);
    const std::string logits_name = layout_module_->GetOutputName(0);
    const std::string boxes_name = layout_module_->GetOutputName(1);
    const std::string order_logits_name = layout_module_->GetOutputName(2);
    const std::string masks_name = layout_module_->GetOutputNum() > 3 ? layout_module_->GetOutputName(3) : std::string();

    const auto run_layout_for_image = [&](const PdfPageImage& image) {
        const auto pixel_values = build_pixel_values(image);
        CHECK_TCIM_STATUS(TcimUtil::set_module_input(layout_module_, input_name, pixel_values.data()));
        CHECK_TCIM_STATUS(layout_module_->Run());
        CHECK_TCIM_STATUS(layout_module_->Sync());

        const auto logits_output = layout_module_->GetDevOutput(logits_name).ToHost(true);
        const auto boxes_output = layout_module_->GetDevOutput(boxes_name).ToHost(true);
        const auto order_logits_output = layout_module_->GetDevOutput(order_logits_name).ToHost(true);
        const auto logits_shape = layout_module_->GetOutputInfo(logits_name).Shape();
        const auto boxes_shape = layout_module_->GetOutputInfo(boxes_name).Shape();
        const auto order_logits_shape = layout_module_->GetOutputInfo(order_logits_name).Shape();
        std::vector<int64_t> masks_shape;
        std::vector<float> masks;
        if (!masks_name.empty()) {
            const auto masks_output = layout_module_->GetDevOutput(masks_name).ToHost(true);
            masks_shape = layout_module_->GetOutputInfo(masks_name).Shape();
            if (masks_shape.size() == 4 && logits_shape.size() == 3 && masks_shape[0] == 1 && masks_shape[1] == logits_shape[1]) {
                masks = TcimUtil::tensor_to_float_vec(masks_output.Buffer().Data(), TcimUtil::numel(masks_shape));
            } else {
                LOG_WARN("Document layout mask output shape is unexpected: {}", shape_to_string(masks_shape));
            }
        }
        if (logits_shape.size() != 3 || boxes_shape.size() != 3 || boxes_shape[2] != 4 ||
            order_logits_shape.size() != 3 || logits_shape[0] != 1 || boxes_shape[0] != 1 ||
            order_logits_shape[0] != 1 || logits_shape[1] != boxes_shape[1] ||
            order_logits_shape[1] != logits_shape[1] || order_logits_shape[2] != logits_shape[1]) {
            LOG_WARN("Document layout output shape is unexpected: logits={}, boxes={}, order_logits={}",
                     shape_to_string(logits_shape), shape_to_string(boxes_shape), shape_to_string(order_logits_shape));
            return std::vector<DocumentLayoutRegion>{};
        }

        const auto logits = TcimUtil::tensor_to_float_vec(logits_output.Buffer().Data(), TcimUtil::numel(logits_shape));
        const auto boxes = TcimUtil::tensor_to_float_vec(boxes_output.Buffer().Data(), TcimUtil::numel(boxes_shape));
        const auto order_logits = TcimUtil::tensor_to_float_vec(order_logits_output.Buffer().Data(), TcimUtil::numel(order_logits_shape));
        const int mask_height = masks_shape.size() == 4 ? static_cast<int>(masks_shape[2]) : 0;
        const int mask_width = masks_shape.size() == 4 ? static_cast<int>(masks_shape[3]) : 0;
        return decode_layout_outputs(logits, boxes, order_logits, masks, static_cast<int>(logits_shape[1]),
                                     static_cast<int>(logits_shape[2]), mask_height, mask_width,
                                     image.width, image.height);
    };

    for (auto& page : document.pages) {
        page.regions = run_layout_for_image(page.image);

        if (page.image.height > page.image.width) {
            const int tile_height = std::max(640, std::min(page.image.height, static_cast<int>(std::lround(page.image.width * 0.85))));
            const int step = std::max(320, static_cast<int>(std::lround(static_cast<double>(tile_height) * 0.55)));
            for (int tile_y0 = step; tile_y0 < page.image.height - 80; tile_y0 += step) {
                if (page.image.height - tile_y0 < 240) {
                    break;
                }
                auto tile = crop_page_image(page.image, tile_y0, std::min(tile_height, page.image.height - tile_y0));
                auto tile_regions = run_layout_for_image(tile);
                for (auto& region : tile_regions) {
                    offset_region(region, 0.0f, static_cast<float>(tile_y0), page.image.width, page.image.height);
                }
                const float band_top = static_cast<float>(tile_y0) + static_cast<float>(tile.height) * 0.18f;
                const float band_bottom = tile_y0 + tile.height >= page.image.height
                                              ? static_cast<float>(page.image.height)
                                              : static_cast<float>(tile_y0) + static_cast<float>(tile.height) * 0.82f;
                keep_regions_in_vertical_band(tile_regions, band_top, band_bottom);
                const auto before_count = page.regions.size();
                merge_regions(page.regions, std::move(tile_regions));
                LOG_INFO("Document layout portrait tile rescan page={}, y0={}, tile_height={}, band_y={}..{}, regions_before={}, regions_after={}",
                         page.image.page_index, tile_y0, tile.height, band_top, band_bottom, before_count, page.regions.size());
            }
        }
        add_projection_fallback_regions(page.image, page.regions);
        LOG_INFO("Document layout detected page={}, regions={}", page.image.page_index, page.regions.size());
    }
}

bool DocumentLayoutEngine::load_layout_module(const std::string& mmproj_path) {
    if (mmproj_path.empty()) {
        LOG_ERROR("Document layout engine cannot load empty mmproj path");
        return false;
    }

    ggml_context* ctx_data = nullptr;
    gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };

    gguf_context* ctx = gguf_init_from_file(mmproj_path.c_str(), params);
    if (!ctx) {
        LOG_ERROR("Document layout engine failed to open mmproj gguf: {}", mmproj_path);
        return false;
    }

    bool loaded = false;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx, i);
        if (name == nullptr || std::string(name) != layout_tensor_name) {
            continue;
        }

        const size_t size = gguf_get_tensor_size(ctx, i);
        const size_t offset = gguf_get_tensor_offset(ctx, i) + gguf_get_meta_size(ctx);
        LOG_INFO("Loading document layout module {}, offset={}, size={}", layout_tensor_name, offset, size);
        layout_module_ = create_infer_engine(mmproj_path, offset, size);
        loaded = layout_module_ != nullptr;
        break;
    }

    gguf_free(ctx);

    if (!loaded) {
        LOG_ERROR("Document layout tensor {} not found or failed to load from {}", layout_tensor_name, mmproj_path);
    }
    return loaded;
}

void DocumentLayoutEngine::capture_module_info() {
    module_info_ = {
        {"name", layout_tensor_name},
        {"inputs", nlohmann::json::array()},
        {"outputs", nlohmann::json::array()},
    };

    if (!layout_module_) {
        return;
    }

    const size_t input_count = layout_module_->GetInputNum();
    LOG_INFO("Document layout module inputs: {}", input_count);
    for (size_t i = 0; i < input_count; ++i) {
        const auto name = layout_module_->GetInputName(static_cast<int>(i));
        const auto info = layout_module_->GetInputInfo(name);
        module_info_["inputs"].push_back(tensor_info_to_json(name, info));
        LOG_INFO("  input[{}] {} shape={} dtype={}", i, name, shape_to_string(info.AsContiguous().Shape()),
                 tcim_dtype_to_string(info.AsContiguous().DataType()));
    }

    const size_t output_count = layout_module_->GetOutputNum();
    LOG_INFO("Document layout module outputs: {}", output_count);
    for (size_t i = 0; i < output_count; ++i) {
        const auto name = layout_module_->GetOutputName(static_cast<int>(i));
        const auto info = layout_module_->GetOutputInfo(name);
        module_info_["outputs"].push_back(tensor_info_to_json(name, info));
        LOG_INFO("  output[{}] {} shape={} dtype={}", i, name, shape_to_string(info.AsContiguous().Shape()),
                 tcim_dtype_to_string(info.AsContiguous().DataType()));
    }
}

std::vector<float> DocumentLayoutEngine::build_pixel_values(const PdfPageImage& image) const {
    if (image.width <= 0 || image.height <= 0 || image.channels < 3 || image.pixels.empty()) {
        throw std::runtime_error("invalid page image for document layout");
    }

    std::vector<float> output(static_cast<size_t>(3 * kLayoutInputSize * kLayoutInputSize), 0.0f);

    const auto sample_channel = [&](const int x, const int y, const int c) -> float {
        const int clamped_x = std::max(0, std::min(x, image.width - 1));
        const int clamped_y = std::max(0, std::min(y, image.height - 1));
        const size_t offset = static_cast<size_t>(clamped_y) * static_cast<size_t>(image.stride) +
                              static_cast<size_t>(clamped_x) * static_cast<size_t>(image.channels) + static_cast<size_t>(c);
        return static_cast<float>(image.pixels[offset]) / 255.0f;
    };

    const float scale_x = static_cast<float>(image.width) / static_cast<float>(kLayoutInputSize);
    const float scale_y = static_cast<float>(image.height) / static_cast<float>(kLayoutInputSize);
    for (int y = 0; y < kLayoutInputSize; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = static_cast<int>(std::floor(src_y));

        for (int x = 0; x < kLayoutInputSize; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = static_cast<int>(std::floor(src_x));

            for (int c = 0; c < 3; ++c) {
                float value = 0.0f;
                float weight_sum = 0.0f;
                for (int ky = -1; ky <= 2; ++ky) {
                    const int sample_y = y0 + ky;
                    const float wy_cubic = cubic_weight(src_y - static_cast<float>(sample_y));
                    for (int kx = -1; kx <= 2; ++kx) {
                        const int sample_x = x0 + kx;
                        const float weight = wy_cubic * cubic_weight(src_x - static_cast<float>(sample_x));
                        value += sample_channel(sample_x, sample_y, c) * weight;
                        weight_sum += weight;
                    }
                }
                if (std::abs(weight_sum) > 1e-6f) {
                    value /= weight_sum;
                }
                output[static_cast<size_t>(c * kLayoutInputSize * kLayoutInputSize + y * kLayoutInputSize + x)] =
                    clamp_float(value, 0.0f, 1.0f);
            }
        }
    }

    return output;
}

std::vector<DocumentLayoutRegion> DocumentLayoutEngine::decode_layout_outputs(const std::vector<float>& logits, const std::vector<float>& boxes,
                                                                              const std::vector<float>& order_logits,
                                                                              const std::vector<float>& masks,
                                                                              const int query_count, const int class_count,
                                                                              const int mask_height, const int mask_width,
                                                                              const int image_width, const int image_height) const {
    std::vector<DocumentLayoutRegion> candidates;
    if (query_count <= 0 || class_count <= 0 ||
        logits.size() < static_cast<size_t>(query_count * class_count) ||
        boxes.size() < static_cast<size_t>(query_count * 4)) {
        LOG_WARN("Document layout output has unexpected size: logits={}, boxes={}, query_count={}, class_count={}",
                 logits.size(), boxes.size(), query_count, class_count);
        return candidates;
    }

    const auto order_seq = build_order_sequence(order_logits, query_count);
    std::vector<std::tuple<float, int, int>> top_scores;
    top_scores.reserve(static_cast<size_t>(query_count * class_count));
    for (int query = 0; query < query_count; ++query) {
        for (int class_id = 0; class_id < class_count; ++class_id) {
            const float score = sigmoid(logits[static_cast<size_t>(query * class_count + class_id)]);
            top_scores.emplace_back(score, query, class_id);
        }
    }
    std::partial_sort(top_scores.begin(),
                      top_scores.begin() + std::min<int>(query_count, static_cast<int>(top_scores.size())),
                      top_scores.end(),
                      [](const auto& lhs, const auto& rhs) {
                          return std::get<0>(lhs) > std::get<0>(rhs);
                      });
    if (static_cast<int>(top_scores.size()) > query_count) {
        top_scores.resize(static_cast<size_t>(query_count));
    }

    for (const auto& item : top_scores) {
        const float score = std::get<0>(item);
        const int query = std::get<1>(item);
        const int class_id = std::get<2>(item);
        if (score < kScoreThreshold) {
            continue;
        }

        const size_t box_offset = static_cast<size_t>(query * 4);
        const float cx = boxes[box_offset + 0] * static_cast<float>(image_width);
        const float cy = boxes[box_offset + 1] * static_cast<float>(image_height);
        const float bw = boxes[box_offset + 2] * static_cast<float>(image_width);
        const float bh = boxes[box_offset + 3] * static_cast<float>(image_height);
        const std::string label = label_for_class(class_id);
        const std::string task_type = task_type_for_label(label);

        DocumentLayoutRegion region;
        region.source_query = query;
        region.class_id = class_id;
        region.order = order_seq[static_cast<size_t>(query)];
        region.label = label;
        region.task_type = task_type;
        region.kind = kind_for_label(label);
        region.score = score;
        region.box.x0 = clamp_float(cx - bw * 0.5f, 0.0f, static_cast<float>(image_width));
        region.box.y0 = clamp_float(cy - bh * 0.5f, 0.0f, static_cast<float>(image_height));
        region.box.x1 = clamp_float(cx + bw * 0.5f, 0.0f, static_cast<float>(image_width));
        region.box.y1 = clamp_float(cy + bh * 0.5f, 0.0f, static_cast<float>(image_height));
        if (region.box.x1 - region.box.x0 < kMinBoxSidePixels ||
            region.box.y1 - region.box.y0 < kMinBoxSidePixels ||
            box_area(region.box) <= 1.0f) {
            continue;
        }
        candidates.push_back(region);
    }

    std::sort(candidates.begin(), candidates.end(), [](const DocumentLayoutRegion& lhs, const DocumentLayoutRegion& rhs) {
        return lhs.score > rhs.score;
    });

    std::vector<DocumentLayoutRegion> kept;
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& existing : kept) {
            if (should_suppress_region(candidate, existing)) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) {
            continue;
        }
        DocumentLayoutRegion region = candidate;
        region.id = static_cast<int>(kept.size());
        kept.push_back(region);
        if (static_cast<int>(kept.size()) >= kMaxRegions) {
            break;
        }
    }

    if (kept.size() > 1) {
        const float image_area = static_cast<float>(image_width) * static_cast<float>(image_height);
        const float area_threshold = image_width > image_height ? 0.82f : 0.93f;
        kept.erase(std::remove_if(kept.begin(), kept.end(), [&](const DocumentLayoutRegion& region) {
            return region.label == "image" && box_area(region.box) > area_threshold * image_area;
        }), kept.end());
    }

    std::vector<DocumentLayoutRegion> merged;
    merged.reserve(kept.size());
    for (size_t i = 0; i < kept.size(); ++i) {
        if (is_preserved_when_contained(kept[i].class_id)) {
            merged.push_back(kept[i]);
            continue;
        }
        bool contained_by_other = false;
        for (size_t j = 0; j < kept.size(); ++j) {
            if (i == j) {
                continue;
            }
            if (is_contained(kept[i].box, kept[j].box)) {
                contained_by_other = true;
                break;
            }
        }
        if (!contained_by_other) {
            merged.push_back(kept[i]);
        }
    }

    std::sort(merged.begin(), merged.end(), visual_reading_order_less);
    for (size_t i = 0; i < merged.size(); ++i) {
        merged[i].id = static_cast<int>(i);
        merged[i].polygon = polygon_from_mask(masks, merged[i].source_query, query_count, mask_height, mask_width,
                                              merged[i].box, image_width, image_height);
        if (merged[i].polygon.size() < 3) {
            merged[i].polygon = box_to_polygon(merged[i].box);
        }
    }

    return merged;
}

HLIE_NAMESPACE_END

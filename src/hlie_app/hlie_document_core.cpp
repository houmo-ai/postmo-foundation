#include "hlie_app/hlie_document_core.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <vector>

HLIE_NAMESPACE_BEGIN
namespace DocumentCore {
namespace {

std::string trim_ascii(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](const unsigned char ch) {
        return std::isspace(ch);
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char ch) {
        return std::isspace(ch);
    }).base();
    return first < last ? std::string(first, last) : std::string{};
}

bool visual_reading_order_less(const DocumentLayoutRegion* lhs, const DocumentLayoutRegion* rhs) {
    // A fixed row bucket keeps the comparison transitive. Pair-dependent
    // tolerances can create comparison cycles and make std::sort undefined.
    constexpr float row_height = 8.0f;
    const auto lhs_row = static_cast<std::int64_t>(std::floor(lhs->box.y0 / row_height));
    const auto rhs_row = static_cast<std::int64_t>(std::floor(rhs->box.y0 / row_height));
    if (lhs_row != rhs_row) {
        return lhs_row < rhs_row;
    }
    if (lhs->box.x0 != rhs->box.x0) {
        return lhs->box.x0 < rhs->box.x0;
    }
    if (lhs->box.y0 != rhs->box.y0) {
        return lhs->box.y0 < rhs->box.y0;
    }
    if (lhs->order != rhs->order) {
        return lhs->order < rhs->order;
    }
    return lhs->id < rhs->id;
}

float box_area(const DocumentBoundingBox& box) {
    return std::max(0.0f, box.x1 - box.x0) * std::max(0.0f, box.y1 - box.y0);
}

float intersection_area(const DocumentBoundingBox& lhs, const DocumentBoundingBox& rhs) {
    const float x0 = std::max(lhs.x0, rhs.x0);
    const float y0 = std::max(lhs.y0, rhs.y0);
    const float x1 = std::min(lhs.x1, rhs.x1);
    const float y1 = std::min(lhs.y1, rhs.y1);
    return std::max(0.0f, x1 - x0) * std::max(0.0f, y1 - y0);
}

bool looks_like_code_block(const std::string& text) {
    return text.find("```") != std::string::npos || text.find("struct ") != std::string::npos ||
        (text.find('{') != std::string::npos && text.find(';') != std::string::npos);
}

bool looks_like_table_markup(const std::string& text) {
    return text.find("<table") != std::string::npos || text.find("<tr") != std::string::npos ||
        text.find("<td") != std::string::npos;
}

bool should_skip_duplicate(const DocumentLayoutRegion& candidate,
                           const std::vector<const DocumentLayoutRegion*>& emitted_regions) {
    if (candidate.text.empty() || !looks_like_table_markup(candidate.text)) {
        return false;
    }
    const float candidate_area = std::max(1.0f, box_area(candidate.box));
    return std::any_of(emitted_regions.begin(), emitted_regions.end(), [&](const auto* emitted) {
        return emitted != nullptr && !emitted->text.empty() && looks_like_code_block(emitted->text) &&
            intersection_area(candidate.box, emitted->box) / candidate_area >= 0.70f;
    });
}

bool is_page_margin_label(const std::string& label) {
    return label == "header" || label == "footer" || label == "number" || label == "footnote";
}

bool is_dark_pixel(const PdfPageImage& image, const int x, const int y) {
    if (x < 0 || y < 0 || x >= image.width || y >= image.height || image.channels < 3 || image.stride <= 0) {
        return false;
    }
    const auto offset = static_cast<std::size_t>(y) * static_cast<std::size_t>(image.stride) +
        static_cast<std::size_t>(x) * static_cast<std::size_t>(image.channels);
    if (offset + 2 >= image.pixels.size()) {
        return false;
    }
    const int r = image.pixels[offset];
    const int g = image.pixels[offset + 1];
    const int b = image.pixels[offset + 2];
    return 0.299f * static_cast<float>(r) + 0.587f * static_cast<float>(g) +
        0.114f * static_cast<float>(b) < 245.0f;
}

bool region_counts_for_coverage(const DocumentLayoutRegion& region) {
    return region.task_type != "abandon" && region.task_type != "skip" && !region.task_type.empty() &&
        region.box.x1 > region.box.x0 && region.box.y1 > region.box.y0;
}

double ratio(const int covered, const int total) {
    return total > 0 ? static_cast<double>(covered) / static_cast<double>(total) : 1.0;
}

} // namespace

void apply_page_margin_policy(DocumentPreprocessResult& result, const bool include_page_margins) {
    if (include_page_margins) {
        return;
    }
    for (auto& page : result.pages) {
        const float top_margin = static_cast<float>(page.image.height) * 0.075f;
        const float bottom_margin = static_cast<float>(page.image.height) * 0.89f;
        for (auto& region : page.regions) {
            const float center_y = (region.box.y0 + region.box.y1) * 0.5f;
            if (is_page_margin_label(region.label) || center_y < top_margin || center_y > bottom_margin) {
                region.task_type = "abandon";
                region.text.clear();
                region.crop = {};
            }
        }
    }
}

nlohmann::json build_page_layout_metrics(const DocumentPage& page, const bool include_page_margins) {
    using nlohmann::json;
    const int width = std::max(1, page.image.width);
    const int height = std::max(1, page.image.height);
    const int min_row_ink = std::max(6, width / 260);
    const int body_y0 = include_page_margins ? 0 : static_cast<int>(std::floor(height * 0.075f));
    const int body_y1 = include_page_margins ? height : static_cast<int>(std::ceil(height * 0.89f));
    const int clamped_body_y0 = std::clamp(body_y0, 0, height);
    const int clamped_body_y1 = std::clamp(body_y1, clamped_body_y0, height);

    std::vector<std::uint8_t> ink_rows(static_cast<std::size_t>(height));
    std::vector<std::uint8_t> covered_rows(static_cast<std::size_t>(height));
    int ink_row_count = 0;
    int body_ink_row_count = 0;
    for (int y = 0; y < height; ++y) {
        int dark_count = 0;
        for (int x = 0; x < width; ++x) {
            dark_count += is_dark_pixel(page.image, x, y) ? 1 : 0;
        }
        if (dark_count >= min_row_ink) {
            ink_rows[static_cast<std::size_t>(y)] = 1;
            ++ink_row_count;
            body_ink_row_count += y >= clamped_body_y0 && y < clamped_body_y1 ? 1 : 0;
        }
    }

    int cover_region_count = 0;
    for (const auto& region : page.regions) {
        if (!region_counts_for_coverage(region)) {
            continue;
        }
        ++cover_region_count;
        const int y0 = std::clamp(static_cast<int>(std::floor(region.box.y0)), 0, height);
        const int y1 = std::clamp(static_cast<int>(std::ceil(region.box.y1)), y0, height);
        std::fill(covered_rows.begin() + y0, covered_rows.begin() + y1, 1);
    }

    int covered_ink_row_count = 0;
    int body_covered_ink_row_count = 0;
    for (int y = 0; y < height; ++y) {
        if (ink_rows[static_cast<std::size_t>(y)] && covered_rows[static_cast<std::size_t>(y)]) {
            ++covered_ink_row_count;
            body_covered_ink_row_count += y >= clamped_body_y0 && y < clamped_body_y1 ? 1 : 0;
        }
    }

    json uncovered_body_bands = json::array();
    const int merge_gap = std::max(3, height / 300);
    int band_start = -1;
    int band_last = -1;
    const auto append_band = [&] {
        uncovered_body_bands.push_back({
            {"y0", band_start}, {"y1", band_last + 1},
            {"bbox_2d", {0, band_start * 1000 / height, 1000, (band_last + 1) * 1000 / height}},
        });
    };
    for (int y = clamped_body_y0; y < clamped_body_y1; ++y) {
        if (!ink_rows[static_cast<std::size_t>(y)] || covered_rows[static_cast<std::size_t>(y)]) {
            continue;
        }
        if (band_start < 0) {
            band_start = band_last = y;
        } else if (y - band_last <= merge_gap) {
            band_last = y;
        } else {
            append_band();
            band_start = band_last = y;
        }
    }
    if (band_start >= 0) {
        append_band();
    }

    return {
        {"page_index", page.image.page_index}, {"width", page.image.width}, {"height", page.image.height},
        {"coverage_mode", "ink_rows"}, {"min_row_ink_pixels", min_row_ink},
        {"body_range", {{"y0", clamped_body_y0}, {"y1", clamped_body_y1},
                        {"bbox_2d", {0, clamped_body_y0 * 1000 / height, 1000, clamped_body_y1 * 1000 / height}}}},
        {"cover_region_count", cover_region_count}, {"ink_rows", ink_row_count},
        {"covered_ink_rows", covered_ink_row_count}, {"ink_coverage", ratio(covered_ink_row_count, ink_row_count)},
        {"body_ink_rows", body_ink_row_count}, {"body_covered_ink_rows", body_covered_ink_row_count},
        {"body_ink_coverage", ratio(body_covered_ink_row_count, body_ink_row_count)},
        {"uncovered_body_bands", uncovered_body_bands},
    };
}

nlohmann::json build_layout_metrics(const DocumentPreprocessResult& result, const bool include_page_margins) {
    using nlohmann::json;
    json pages = json::array();
    int total_ink_rows = 0;
    int total_covered_ink_rows = 0;
    int total_body_ink_rows = 0;
    int total_body_covered_ink_rows = 0;
    int total_uncovered_body_bands = 0;
    for (const auto& page : result.pages) {
        auto metrics = build_page_layout_metrics(page, include_page_margins);
        total_ink_rows += metrics.value("ink_rows", 0);
        total_covered_ink_rows += metrics.value("covered_ink_rows", 0);
        total_body_ink_rows += metrics.value("body_ink_rows", 0);
        total_body_covered_ink_rows += metrics.value("body_covered_ink_rows", 0);
        total_uncovered_body_bands += static_cast<int>(metrics["uncovered_body_bands"].size());
        pages.push_back(std::move(metrics));
    }
    return {
        {"coverage_mode", "ink_rows"}, {"page_count", result.pages.size()},
        {"ink_rows", total_ink_rows}, {"covered_ink_rows", total_covered_ink_rows},
        {"ink_coverage", ratio(total_covered_ink_rows, total_ink_rows)},
        {"body_ink_rows", total_body_ink_rows}, {"body_covered_ink_rows", total_body_covered_ink_rows},
        {"body_ink_coverage", ratio(total_body_covered_ink_rows, total_body_ink_rows)},
        {"uncovered_body_band_count", total_uncovered_body_bands}, {"pages", pages},
    };
}

std::string merge_document_text(const DocumentPreprocessResult& result) {
    std::string merged;
    for (const auto& page : result.pages) {
        std::vector<const DocumentLayoutRegion*> regions;
        regions.reserve(page.regions.size());
        for (const auto& region : page.regions) {
            regions.push_back(&region);
        }
        std::sort(regions.begin(), regions.end(), visual_reading_order_less);

        std::string page_text;
        std::vector<const DocumentLayoutRegion*> emitted;
        for (const auto* region : regions) {
            if (region->text.empty() || should_skip_duplicate(*region, emitted)) {
                continue;
            }
            std::string text = trim_ascii(region->text);
            if (region->label == "doc_title" || region->label == "paragraph_title") {
                while (!text.empty() && text.front() == '#') {
                    text.erase(text.begin());
                }
                text = (region->label == "doc_title" ? "# " : "## ") + trim_ascii(text);
            }
            if (!page_text.empty()) {
                page_text += '\n';
            }
            page_text += text;
            emitted.push_back(region);
        }
        if (!page_text.empty()) {
            if (!merged.empty()) {
                merged += "\n\n";
            }
            merged += page_text;
        }
    }
    return merged;
}

bool is_low_confidence_fallback_region(const DocumentLayoutRegion& region) {
    return region.class_id < 0 || region.score <= 0.21f;
}

bool looks_like_non_transcription_response(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text.find("the text in the image is") != std::string::npos ||
        text.find("no visible text") != std::string::npos || text.find("too blurry") != std::string::npos ||
        text.find("low-resolution") != std::string::npos || text.find("cannot be extracted") != std::string::npos ||
        text.find("provided image") != std::string::npos || text.find("clearly readable text") != std::string::npos;
}

} // namespace DocumentCore
HLIE_NAMESPACE_END

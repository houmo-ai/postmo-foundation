#include "hlie_app/document/hlie_document_core.h"
#include "hlie_app/document/hlie_document_region_cropper.h"

#include <catch_amalgamated.hpp>

#include <algorithm>

namespace {

using namespace hm_HLIE;

PdfPageImage make_image(const int width, const int height, const int row_padding = 0) {
    PdfPageImage image;
    image.width = width;
    image.height = height;
    image.channels = 3;
    image.stride = width * image.channels + row_padding;
    image.pixels.assign(static_cast<std::size_t>(image.stride * height), 255);
    return image;
}

DocumentLayoutRegion region(const int id, const std::string& label, const std::string& task,
                            const DocumentBoundingBox box, const std::string& text = {}) {
    DocumentLayoutRegion value;
    value.id = id;
    value.label = label;
    value.task_type = task;
    value.box = box;
    value.text = text;
    return value;
}

void darken_rows(PdfPageImage& image, const int y0, const int y1) {
    for (int y = y0; y < y1; ++y) {
        for (int x = 0; x < 8; ++x) {
            const auto offset = static_cast<std::size_t>(y * image.stride + x * image.channels);
            image.pixels[offset] = image.pixels[offset + 1] = image.pixels[offset + 2] = 0;
        }
    }
}

} // namespace

TEST_CASE("Document crop clamps padding and respects source stride", "[document][crop]") {
    auto image = make_image(4, 3, 2);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            image.pixels[static_cast<std::size_t>(y * image.stride + x * 3)] = static_cast<unsigned char>(y * 10 + x);
        }
    }
    const auto target = region(1, "text", "text", {-1.0f, 0.8f, 2.2f, 3.8f});
    DocumentPreprocessResult result;
    result.pages.push_back({image, {target}});
    REQUIRE(DocumentRegionCropper::crop_recognizable_regions(
                result, {.padding_pixels = 1, .min_side_pixels = 1}) == 1);
    const auto& crop = result.pages[0].regions[0].crop;

    CHECK(crop.width == 4);
    CHECK(crop.height == 3);
    CHECK(crop.stride == 12);
    CHECK(crop.source_box.x0 == 0.0f);
    CHECK(crop.source_box.y0 == 0.0f);
    CHECK(crop.source_box.x1 == 4.0f);
    CHECK(crop.source_box.y1 == 3.0f);
    REQUIRE(crop.pixels.size() == 36);
    CHECK(crop.pixels[12] == 10);
}

TEST_CASE("Document crop skips abandoned invalid and tiny regions", "[document][crop]") {
    auto image = make_image(10, 10);
    auto valid = region(1, "text", "text", {1, 1, 8, 8});
    auto abandoned = region(2, "footer", "abandon", {1, 1, 8, 8});
    auto tiny = region(3, "text", "text", {1, 1, 2, 2});
    DocumentPreprocessResult result;
    result.pages.push_back({image, {valid, abandoned, tiny}});

    const int count = DocumentRegionCropper::crop_recognizable_regions(result, {.padding_pixels = 0, .min_side_pixels = 2});
    CHECK(count == 1);
    CHECK_FALSE(result.pages[0].regions[0].crop.pixels.empty());
    CHECK(result.pages[0].regions[1].crop.pixels.empty());
    CHECK(result.pages[0].regions[2].crop.pixels.empty());
}

TEST_CASE("Document crop rejects truncated or inconsistent image storage", "[document][crop]") {
    auto truncated = make_image(10, 10);
    truncated.pixels.resize(12);
    const auto target = region(1, "text", "text", {0, 0, 8, 8});
    DocumentPreprocessResult truncated_result;
    truncated_result.pages.push_back({std::move(truncated), {target}});
    CHECK(DocumentRegionCropper::crop_recognizable_regions(
              truncated_result, {.padding_pixels = 0, .min_side_pixels = 1}) == 0);
    CHECK(truncated_result.pages[0].regions[0].crop.pixels.empty());

    auto short_stride = make_image(10, 10);
    short_stride.stride = 8;
    DocumentPreprocessResult short_stride_result;
    short_stride_result.pages.push_back({std::move(short_stride), {target}});
    CHECK(DocumentRegionCropper::crop_recognizable_regions(
              short_stride_result, {.padding_pixels = 0, .min_side_pixels = 1}) == 0);
    CHECK(short_stride_result.pages[0].regions[0].crop.pixels.empty());
}

TEST_CASE("Document margin policy removes semantic and geometric margins", "[document][margin]") {
    DocumentPreprocessResult result;
    auto page = DocumentPage{make_image(100, 100), {
        region(1, "header", "text", {0, 20, 50, 30}, "header"),
        region(2, "text", "text", {0, 0, 50, 5}, "top"),
        region(3, "text", "text", {0, 20, 50, 30}, "body"),
        region(4, "text", "text", {0, 92, 50, 98}, "bottom"),
    }};
    page.regions[0].crop.pixels = {1, 2, 3};
    result.pages.push_back(std::move(page));

    DocumentCore::apply_page_margin_policy(result, false);
    CHECK(result.pages[0].regions[0].task_type == "abandon");
    CHECK(result.pages[0].regions[0].text.empty());
    CHECK(result.pages[0].regions[0].crop.pixels.empty());
    CHECK(result.pages[0].regions[1].task_type == "abandon");
    CHECK(result.pages[0].regions[2].text == "body");
    CHECK(result.pages[0].regions[3].task_type == "abandon");
}

TEST_CASE("Document coverage measures visible ink rather than blank page height", "[document][coverage]") {
    DocumentPage page;
    page.image = make_image(100, 100);
    darken_rows(page.image, 20, 23);
    darken_rows(page.image, 50, 53);
    page.regions.push_back(region(1, "text", "text", {0, 19, 100, 24}));

    const auto metrics = DocumentCore::build_page_layout_metrics(page, false);
    CHECK(metrics["ink_rows"] == 6);
    CHECK(metrics["covered_ink_rows"] == 3);
    CHECK(metrics["body_ink_coverage"].get<double>() == Catch::Approx(0.5));
    REQUIRE(metrics["uncovered_body_bands"].size() == 1);
    CHECK(metrics["uncovered_body_bands"][0]["y0"] == 50);
    CHECK(metrics["uncovered_body_bands"][0]["y1"] == 53);
}

TEST_CASE("Document markdown merge orders titles and suppresses overlapping table duplicates", "[document][merge]") {
    DocumentPreprocessResult result;
    DocumentPage first;
    first.image = make_image(200, 200);
    first.regions = {
        region(3, "text", "text", {10, 80, 190, 100}, "paragraph"),
        region(1, "doc_title", "text", {10, 10, 190, 30}, "## Title "),
        region(2, "code", "code", {10, 40, 190, 75}, "```cpp\nstruct Row {};\n```"),
        region(4, "table", "table", {20, 45, 180, 70}, "<table><tr><td>duplicate</td></tr></table>"),
    };
    DocumentPage second;
    second.image = make_image(200, 200);
    second.regions = {region(5, "paragraph_title", "text", {10, 10, 190, 30}, "# Next")};
    result.pages = {first, second};

    const auto markdown = DocumentCore::merge_document_text(result);
    CHECK(markdown == "# Title\n```cpp\nstruct Row {};\n```\nparagraph\n\n## Next");
    CHECK(markdown.find("duplicate") == std::string::npos);
}

TEST_CASE("Document reading order remains top-to-bottom across staggered rows", "[document][merge]") {
    DocumentPreprocessResult result;
    DocumentPage page;
    page.image = make_image(300, 100);
    page.regions = {
        region(1, "text", "text", {200, 0, 280, 30}, "A"),
        region(2, "text", "text", {100, 10, 180, 40}, "B"),
        region(3, "text", "text", {0, 20, 80, 50}, "C"),
    };
    result.pages.push_back(std::move(page));

    CHECK(DocumentCore::merge_document_text(result) == "A\nB\nC");
}

TEST_CASE("Document fallback OCR filters explanatory hallucinations", "[document][ocr-filter]") {
    CHECK(DocumentCore::is_low_confidence_fallback_region(region(1, "text", "text", {0, 0, 1, 1})));
    CHECK(DocumentCore::looks_like_non_transcription_response("The text in the image is too blurry."));
    CHECK_FALSE(DocumentCore::looks_like_non_transcription_response("Invoice number 1024"));
}

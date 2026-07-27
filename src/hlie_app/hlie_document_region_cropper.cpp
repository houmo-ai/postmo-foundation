#include "hlie_app/hlie_document_region_cropper.h"

#include <algorithm>
#include <cmath>

HLIE_NAMESPACE_BEGIN

namespace {
    int clamp_int(const int value, const int lo, const int hi) {
        return std::max(lo, std::min(value, hi));
    }

} // namespace

bool DocumentRegionCropper::should_crop_region(const DocumentLayoutRegion& region) {
    return region.task_type != "abandon" && region.task_type != "skip";
}

DocumentRegionCrop DocumentRegionCropper::crop_region(const PdfPageImage& image, const DocumentLayoutRegion& region,
                                                      const DocumentCropOptions& options) {
    DocumentRegionCrop crop;
    if (image.width <= 0 || image.height <= 0 || image.channels <= 0 || image.stride <= 0 || image.pixels.empty()) {
        return crop;
    }

    const auto row_bytes = static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.channels);
    const auto source_stride = static_cast<std::size_t>(image.stride);
    const auto source_height = static_cast<std::size_t>(image.height);
    if (source_stride < row_bytes || source_height > image.pixels.size() / source_stride) {
        return crop;
    }

    const auto bounds = region.box;
    const int padding = std::max(0, options.padding_pixels);
    const int x0 = clamp_int(static_cast<int>(std::floor(bounds.x0)) - padding, 0, image.width);
    const int y0 = clamp_int(static_cast<int>(std::floor(bounds.y0)) - padding, 0, image.height);
    const int x1 = clamp_int(static_cast<int>(std::ceil(bounds.x1)) + padding, 0, image.width);
    const int y1 = clamp_int(static_cast<int>(std::ceil(bounds.y1)) + padding, 0, image.height);
    const int crop_width = x1 - x0;
    const int crop_height = y1 - y0;
    if (crop_width < std::max(1, options.min_side_pixels) || crop_height < std::max(1, options.min_side_pixels)) {
        return crop;
    }

    crop.width = crop_width;
    crop.height = crop_height;
    crop.channels = image.channels;
    crop.stride = crop.width * crop.channels;
    crop.mode = "bbox";
    crop.source_box = {
        static_cast<float>(x0),
        static_cast<float>(y0),
        static_cast<float>(x1),
        static_cast<float>(y1),
    };
    crop.pixels.resize(static_cast<size_t>(crop.stride) * static_cast<size_t>(crop.height));

    for (int y = 0; y < crop.height; ++y) {
        const auto* src = image.pixels.data() + static_cast<size_t>(y0 + y) * source_stride +
                          static_cast<size_t>(x0) * static_cast<size_t>(image.channels);
        auto* dst = crop.pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(crop.stride);
        std::copy(src, src + static_cast<size_t>(crop.stride), dst);
    }

    return crop;
}

int DocumentRegionCropper::crop_recognizable_regions(DocumentPreprocessResult& document, const DocumentCropOptions& options) {
    int crop_count = 0;
    for (auto& page : document.pages) {
        for (auto& region : page.regions) {
            region.crop = {};
            if (!should_crop_region(region)) {
                continue;
            }
            region.crop = crop_region(page.image, region, options);
            if (!region.crop.pixels.empty()) {
                ++crop_count;
            }
        }
    }
    return crop_count;
}

HLIE_NAMESPACE_END

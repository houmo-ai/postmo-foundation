#include "hlie_app/hlie_document_image_encoder.h"

#include <cstdlib>
#include <stdexcept>

#include "base64.hpp"

extern "C" unsigned char* stbi_write_png_to_mem(const unsigned char* pixels, int stride_bytes,
                                                int x, int y, int n, int* out_len, const char* parameters);

HLIE_NAMESPACE_BEGIN

std::string DocumentImageEncoder::rgb_to_png_data_url(const raw_buffer& pixels, const int width, const int height,
                                                      const int channels, const int stride) {
    if (width <= 0 || height <= 0 || channels <= 0 || stride <= 0 || pixels.empty()) {
        return {};
    }

    int png_len = 0;
    unsigned char* png_data = stbi_write_png_to_mem(pixels.data(), stride, width, height, channels, &png_len, nullptr);
    if (png_data == nullptr || png_len <= 0) {
        if (png_data != nullptr) {
            std::free(png_data);
        }
        throw std::runtime_error("Failed to encode document image crop to PNG");
    }

    std::string data_url = "data:image/png;base64,";
    data_url += base64::encode(reinterpret_cast<const char*>(png_data), png_len);
    std::free(png_data);
    return data_url;
}

std::string DocumentImageEncoder::crop_to_png_data_url(const DocumentRegionCrop& crop) {
    return rgb_to_png_data_url(crop.pixels, crop.width, crop.height, crop.channels, crop.stride);
}

std::string DocumentImageEncoder::page_to_png_data_url(const PdfPageImage& image) {
    return rgb_to_png_data_url(image.pixels, image.width, image.height, image.channels, image.stride);
}

HLIE_NAMESPACE_END

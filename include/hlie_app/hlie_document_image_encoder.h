#ifndef HLIE_DOCUMENT_IMAGE_ENCODER_H
#define HLIE_DOCUMENT_IMAGE_ENCODER_H

#include <string>

#include "hlie_app/hlie_document_types.h"

HLIE_NAMESPACE_BEGIN

class DocumentImageEncoder {
public:
    static std::string crop_to_png_data_url(const DocumentRegionCrop& crop);
    static std::string page_to_png_data_url(const PdfPageImage& image);

private:
    static std::string rgb_to_png_data_url(const raw_buffer& pixels, int width, int height, int channels, int stride);
};

HLIE_NAMESPACE_END

#endif // HLIE_DOCUMENT_IMAGE_ENCODER_H

#ifndef HLIE_DOCUMENT_REGION_CROPPER_H
#define HLIE_DOCUMENT_REGION_CROPPER_H

#include "hlie_app/hlie_document_types.h"

HLIE_NAMESPACE_BEGIN

struct DocumentCropOptions {
    int padding_pixels = 4;
    int min_side_pixels = 2;
};

class DocumentRegionCropper {
public:
    static int crop_recognizable_regions(DocumentPreprocessResult& document, const DocumentCropOptions& options = {});

private:
    static bool should_crop_region(const DocumentLayoutRegion& region);
    static DocumentRegionCrop crop_region(const PdfPageImage& image, const DocumentLayoutRegion& region,
                                          const DocumentCropOptions& options);
};

HLIE_NAMESPACE_END

#endif // HLIE_DOCUMENT_REGION_CROPPER_H

#ifndef HLIE_PDF_RENDERER_H
#define HLIE_PDF_RENDERER_H

#include <cstdint>
#include <string>
#include <vector>

#include <server/server-http.h>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

struct PdfRenderOptions {
    int dpi = 200;
    int max_long_side = 3500;
    int max_pages = 50;
};

struct PdfPageImage {
    int page_index = 0;
    int width = 0;
    int height = 0;
    int channels = 3;
    int stride = 0;
    raw_buffer pixels;
};

class PdfiumPdfRenderer {
public:
    static std::vector<PdfPageImage> render(const raw_buffer& pdf_data, const PdfRenderOptions& options = {},
                                            int* total_page_count = nullptr);
};

HLIE_NAMESPACE_END

#endif // HLIE_PDF_RENDERER_H

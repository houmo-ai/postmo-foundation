#include "hlie_app/document/hlie_pdf_renderer.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>

#include "hlie_base/hlie_logger.h"

#ifdef HLIE_ENABLE_PDFIUM
#include "fpdfview.h"
#endif

HLIE_NAMESPACE_BEGIN

namespace {
#ifdef HLIE_ENABLE_PDFIUM
    std::once_flag pdfium_init_once;

    void ensure_pdfium_initialized() {
        std::call_once(pdfium_init_once, [] {
            FPDF_LIBRARY_CONFIG config;
            config.version = 3;
            config.m_pUserFontPaths = nullptr;
            config.m_pIsolate = nullptr;
            config.m_v8EmbedderSlot = 0;
            FPDF_InitLibraryWithConfig(&config);
        });
    }

    int scale_dimension(const double points, const int dpi) {
        return std::max(1, static_cast<int>(std::lround(points * static_cast<double>(dpi) / 72.0)));
    }
#endif
} // namespace

std::vector<PdfPageImage> PdfiumPdfRenderer::render(const raw_buffer& pdf_data, const PdfRenderOptions& options,
                                                    int* total_page_count) {
#ifndef HLIE_ENABLE_PDFIUM
    (void) pdf_data;
    (void) options;
    (void) total_page_count;
    throw std::runtime_error("PDFium is not enabled in this build");
#else
    if (pdf_data.empty()) {
        throw std::runtime_error("PDF data is empty");
    }

    ensure_pdfium_initialized();

    FPDF_DOCUMENT document = FPDF_LoadMemDocument64(pdf_data.data(), pdf_data.size(), nullptr);
    if (!document) {
        throw std::runtime_error("Failed to load PDF document");
    }

    std::vector<PdfPageImage> pages;
    const int page_count = FPDF_GetPageCount(document);
    if (total_page_count != nullptr) {
        *total_page_count = page_count;
    }
    const int render_page_count = options.max_pages > 0 ? std::min(page_count, options.max_pages) : page_count;
    pages.reserve(std::max(0, render_page_count));

    try {
        for (int page_index = 0; page_index < render_page_count; ++page_index) {
            FPDF_PAGE page = FPDF_LoadPage(document, page_index);
            if (!page) {
                throw std::runtime_error("Failed to load PDF page " + std::to_string(page_index));
            }

            int width = scale_dimension(FPDF_GetPageWidthF(page), options.dpi);
            int height = scale_dimension(FPDF_GetPageHeightF(page), options.dpi);
            if (options.max_long_side > 0) {
                const int long_side = std::max(width, height);
                if (long_side > options.max_long_side) {
                    const double ratio = static_cast<double>(options.max_long_side) / static_cast<double>(long_side);
                    width = std::max(1, static_cast<int>(std::lround(static_cast<double>(width) * ratio)));
                    height = std::max(1, static_cast<int>(std::lround(static_cast<double>(height) * ratio)));
                }
            }

            const int bgra_stride = width * 4;
            raw_buffer bgra(static_cast<size_t>(bgra_stride) * static_cast<size_t>(height), 0xFF);
            FPDF_BITMAP bitmap = FPDFBitmap_CreateEx(width, height, FPDFBitmap_BGRA, bgra.data(), bgra_stride);
            if (!bitmap) {
                FPDF_ClosePage(page);
                throw std::runtime_error("Failed to create PDFium bitmap");
            }

            FPDFBitmap_FillRect(bitmap, 0, 0, width, height, 0xFFFFFFFF);
            FPDF_RenderPageBitmap(bitmap, page, 0, 0, width, height, 0, FPDF_ANNOT);
            FPDFBitmap_Destroy(bitmap);
            FPDF_ClosePage(page);

            PdfPageImage image;
            image.page_index = page_index;
            image.width = width;
            image.height = height;
            image.channels = 3;
            image.stride = width * image.channels;
            image.pixels.resize(static_cast<size_t>(image.stride) * static_cast<size_t>(height));

            for (int y = 0; y < height; ++y) {
                const uint8_t* src = bgra.data() + static_cast<size_t>(y) * static_cast<size_t>(bgra_stride);
                uint8_t* dst = image.pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(image.stride);
                for (int x = 0; x < width; ++x) {
                    dst[x * 3 + 0] = src[x * 4 + 2];
                    dst[x * 3 + 1] = src[x * 4 + 1];
                    dst[x * 3 + 2] = src[x * 4 + 0];
                }
            }

            pages.push_back(std::move(image));
        }
    } catch (...) {
        FPDF_CloseDocument(document);
        throw;
    }

    FPDF_CloseDocument(document);
    LOG_INFO("Rendered PDF pages: {}, total_pages={}, max_pages={}", pages.size(), page_count, options.max_pages);
    return pages;
#endif
}

HLIE_NAMESPACE_END

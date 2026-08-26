# hi.cpp Agent Notes

## Build

- Preferred CMake executable on this machine:
  `D:\apps\jetbrains\CLion\bin\cmake\win\x64\bin\cmake.exe`
- Common Release build command:
  `D:\apps\jetbrains\CLion\bin\cmake\win\x64\bin\cmake.exe --build D:\workspace\wtt\houmo\llm\hi.cpp\cmake-build-release-visual-studio --target houmo-llm-hi-cpp-win --config Release`
- Reconfigure the existing build directory when CMake files or new globbed `.cpp` files change:
  `D:\apps\jetbrains\CLion\bin\cmake\win\x64\bin\cmake.exe -S D:\workspace\wtt\houmo\llm\hi.cpp -B D:\workspace\wtt\houmo\llm\hi.cpp\cmake-build-release-visual-studio`
- The generated Visual Studio build directory is:
  `D:\workspace\wtt\houmo\llm\hi.cpp\cmake-build-release-visual-studio`

## Notes

- `src/hlie_base/CMakeLists.txt` uses `file(GLOB ...)` for base sources. After adding new `.cpp` files, rerun CMake configure if the generated project does not pick them up automatically.

## Document PDF/OCR Pipeline

- The PDF path is `PDF -> PDFium render -> ppdoclayoutv3 layout -> region crop -> GLM-OCR/VL OCR -> markdown merge`.
- The document pipeline is loaded from the model mmproj GGUF when it contains the `ppdoclayoutv3.hmm` tensor. Treat the layout module lifecycle as part of the model lifecycle: load with the LLM model, stop/detach it when unloading or evicting the LLM model, and avoid leaving it live after the base model is reset.
- PDF input sniffing should happen before normal LLM routing. Multipart `application/pdf` files are the primary path. Do not expand `image_url` `data:application/pdf;base64,...` unless explicitly requested.
- For OpenAI-compatible requests, keep the request surface compatible first. Custom outputs such as `document.markdown` and `document_pipeline.layout_metrics` are acceptable as extensions when OpenAI does not define the PDF OCR result shape.
- Default behavior should ignore page margins. Page headers, footers, page numbers, and footnotes can pollute markdown and should stay disabled unless a request/config flag such as `document_include_page_margins=true` asks to preserve them.
- When debugging incomplete PDF OCR, separate the problem into three layers:
  - Layout coverage: inspect `document_pipeline.layout_metrics.body_ink_coverage` and `uncovered_body_bands`. A low value means layout/crop is missing visible body ink before OCR even runs.
  - OCR quality: inspect individual `document.pages[].regions[].text`. Hallucinated strings such as `The text in the image is`, `No visible text`, or `too blurry` usually come from tiny/low-information fallback crops.
  - Merge quality: if region text is good but markdown is duplicated or out of order, fix visual ordering or merge de-duplication rather than changing layout.
- Coverage should be measured against visible ink rows, not full page height. The goal is to cover all body text ink, not blank bottom margins.
- Projection fallback is useful when ppdoclayoutv3 misses lower body text on tall PDF pages, but it must be conservative:
  - Add fallback only for uncovered body bands, not arbitrary whitespace.
  - Avoid very short fallback crops; tiny strips tend to make the OCR model hallucinate.
  - Low-confidence fallback regions should use strict OCR prompts and discard non-transcription responses.
  - Watch `attempted_regions`; a sudden jump often means fallback is too fragmented.
- Tile rescans help portrait/A4 pages because the layout model sees more local detail. Keep tile merge/NMS strict enough to avoid duplicate large regions.
- Code blocks and tables can overlap in layout output. If a table crop is geometrically inside an already emitted code-block transcription, prefer merge-stage de-duplication so raw region JSON remains useful for debugging.
- Use the Python demo as a behavior reference, but do not copy it blindly. In observed tests, the demo saves final `markdown.md`, `result.json`, and `summary.json` after full-pipeline processing; it is not stream-first for PDF full OCR. Its formatter maps `doc_title` to `#` and `paragraph_title` to `##`.
- When testing, compare both human-visible markdown and metrics. A good regression check is: rendered page count matches the PDF, `recognized_regions` is close to `attempted_regions`, `body_ink_coverage` is high, no page numbers appear by default, and no hallucinated explanatory OCR text leaks into markdown.
- Release builds must be compiled with PDFium enabled for PDF support. On Windows, ensure `HLIE_ENABLE_PDFIUM=ON`, ship `pdfium.dll` plus its required component DLLs/data, and fail packaging if PDFium is expected but missing. On Linux, package `libpdfium.so` under `third_party/pdfium/lib` and headers under `third_party/pdfium/include`.

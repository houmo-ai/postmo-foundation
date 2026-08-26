# Third-Party Notices

HLIECpp includes or links the third-party components listed below. These
components remain subject to their respective license terms. The presence of a
notice here does not imply endorsement by the copyright holders.

## Core and runtime dependencies

| Component | Version/source | Usage | License | License material |
| --- | --- | --- | --- | --- |
| zlib | 1.3.2 | statically linked gzip support | zlib License | `zlib/LICENSE.txt` |
| cpp-httplib | 0.43.3 | source/static HTTP implementation | MIT | `cpp-httplib/LICENSE.txt` |
| spdlog | 1.15.3 | header/source logging implementation | MIT | `spdlog/LICENSE.txt` |
| inih / INIReader | bundled CI SDK | statically linked configuration parser | BSD-3-Clause | `inih/LICENSE.txt` |
| nlohmann/json | 3.12.0 | header-only JSON implementation | MIT | `nlohmann-json/LICENSE.MIT` |
| base64.hpp | bundled with the llama SDK | header-only Base64 implementation | Unlicense | `base64/UNLICENSE.txt` |
| stb_image_write | bundled with the stable-diffusion SDK | header-only image writer | MIT or Public Domain | `stb/LICENSE.txt` |
| libsamplerate | bundled CI SDK | shared sample-rate conversion runtime | BSD-2-Clause | `libsamplerate/COPYING` |
| kaldi-native-fbank | bundled CI SDK | shared audio feature runtime | Apache-2.0 | `kaldi-native-fbank/LICENSE` |
| uWebSockets | bundled CI SDK | header/source WebSocket implementation | Apache-2.0 | `uwebsockets/LICENSE` |
| uSockets | bundled with uWebSockets | statically linked socket implementation | Apache-2.0 | `usockets/LICENSE` |
| libuv | bundled with uWebSockets | shared event-loop runtime | MIT | `libuv/LICENSE` |

Where a CI SDK does not expose an upstream version in its archive name, the SDK
archive and checksum should be treated as the versioned distribution unit.

## Eigen

- Component: Eigen
- Bundled source version: `5.0.1-dev+master`
- Usage: header-only; compiled into HLIECpp
- License: Mozilla Public License 2.0
- License text: `eigen/MPL-2.0.txt`
- Project: https://eigen.tuxfamily.org/

HLIECpp application code is not relicensed under MPL-2.0. Changes made directly
to MPL-covered Eigen files remain governed by MPL-2.0.

## PDFium runtime bundle

The PDFium SDK used by the Windows package currently contains PDFium and
component runtime files including ICU data/runtime, LLVM libc++, Abseil,
HarfBuzz, zlib, and Chromium PartitionAllocator components. Linux packages may
contain the equivalent shared-library components supplied by the PDFium SDK.

| Component | Typical packaged artifact | License material |
| --- | --- | --- |
| PDFium | `pdfium.dll`, `libpdfium.so` | `pdfium/UPSTREAM-LICENSE.txt` |
| Chromium components | PartitionAllocator component libraries | `pdfium/CHROMIUM-LICENSE.txt` |
| ICU | `icuuc.dll`, `icudtl.dat` | `pdfium/ICU-LICENSE.txt` |
| LLVM libc++ | `libc++.dll` or equivalent runtime | `pdfium/LLVM-LICENSE.txt` |
| Abseil | `third_party_abseil-cpp_absl.dll` or equivalent | `pdfium/ABSEIL-LICENSE.txt` |
| HarfBuzz | `third_party_harfbuzz.dll` or equivalent | `pdfium/HARFBUZZ-COPYING.txt` |
| zlib | `third_party_zlib.dll` or equivalent | `pdfium/ZLIB-LICENSE.txt` |

The PDFium SDK supplier remains responsible for identifying any additional
components introduced into its binary archive. CI packaging must be reviewed
when the SDK file inventory changes.

## License source references

The preserved texts were obtained from the corresponding upstream projects:

- PDFium distribution license: the `LICENSE` file supplied with the current SDK
- Chromium license: Chromium
- MPL-2.0: Mozilla
- ICU license: Unicode ICU
- LLVM license and LLVM exception: LLVM Project
- HarfBuzz license: HarfBuzz
- Abseil license: Abseil
- zlib license: the vendored zlib distribution

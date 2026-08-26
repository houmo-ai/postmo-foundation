#include <cstdarg>
#include <cstdio>
#include <cstdlib>

// llama.cpp's server headers route nlohmann assertions through ggml_abort.
// Focused tests do not link the GGML runtime, so preserve the fail-fast
// contract with this tiny assertion-only replacement.
extern "C" [[noreturn]] void ggml_abort(const char* file, const int line, const char* format, ...) {
    std::fprintf(stderr, "GGML assertion failed at %s:%d: ", file, line);
    std::va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::abort();
}

#ifndef HLIE_GGUF_BUNDLE_H
#define HLIE_GGUF_BUNDLE_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gguf.h"
#include "hlie_base/hlie_base.h"
#include "tcim/tcim_runtime.h"

HLIE_NAMESPACE_BEGIN

struct HLIEGgufTensorSpan {
    std::string name;
    size_t offset = 0;
    size_t size = 0;
};

/**
 * Read-only view of a GGUF file used as a bundle of HMM modules and tensors.
 * The class owns GGUF metadata only; large tensor payloads remain file-backed.
 */
class HLIE_API HLIEGgufBundle {
public:
    HLIEGgufBundle() = default;
    ~HLIEGgufBundle();

    HLIEGgufBundle(const HLIEGgufBundle&) = delete;
    HLIEGgufBundle& operator=(const HLIEGgufBundle&) = delete;

    bool open(const std::string& path, std::string* error = nullptr);
    void close();

    [[nodiscard]] bool is_open() const { return ctx_ != nullptr; }
    [[nodiscard]] const std::string& path() const { return path_; }
    [[nodiscard]] gguf_context* context() const { return ctx_; }

    [[nodiscard]] std::optional<HLIEGgufTensorSpan> find_tensor(const std::string& name) const;
    [[nodiscard]] bool has_tensor(const std::string& name) const;
    [[nodiscard]] std::vector<std::string> tensor_names() const;

    bool get_string(const std::string& key, std::string& value, bool required = true,
                    std::string* error = nullptr) const;
    bool get_int32(const std::string& key, int32_t& value, bool required = true,
                   std::string* error = nullptr) const;

    bool load_fp16_tensor(const std::string& name, const std::vector<int64_t>& shape,
                          tcim::Tensor& value, std::string* error = nullptr) const;

private:
    static void set_error(std::string* error, const std::string& message);

    std::string path_;
    size_t file_size_ = 0;
    ggml_context* data_ctx_ = nullptr;
    gguf_context* ctx_ = nullptr;
};

HLIE_NAMESPACE_END

#endif

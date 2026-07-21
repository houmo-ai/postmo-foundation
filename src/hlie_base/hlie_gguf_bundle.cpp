#include "hlie_base/hlie_gguf_bundle.h"

#include <fstream>
#include <filesystem>
#include <limits>
#include <sstream>

#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

HLIEGgufBundle::~HLIEGgufBundle() {
    close();
}

void HLIEGgufBundle::set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

bool HLIEGgufBundle::open(const std::string& path, std::string* error) {
    close();
    if (path.empty()) {
        set_error(error, "GGUF bundle path is empty");
        return false;
    }

    gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &data_ctx_,
    };
    ctx_ = gguf_init_from_file(path.c_str(), params);
    if (!ctx_) {
        data_ctx_ = nullptr;
        set_error(error, "failed to open GGUF bundle: " + path);
        return false;
    }
    std::error_code file_error;
    const auto file_size = std::filesystem::file_size(path, file_error);
    if (file_error || file_size > std::numeric_limits<size_t>::max()) {
        close();
        set_error(error, "failed to determine GGUF bundle size: " + path);
        return false;
    }
    file_size_ = static_cast<size_t>(file_size);
    const size_t metadata_size = gguf_get_meta_size(ctx_);
    const int64_t tensor_count = gguf_get_n_tensors(ctx_);
    for (int64_t index = 0; index < tensor_count; ++index) {
        const size_t relative_offset = gguf_get_tensor_offset(ctx_, index);
        const size_t tensor_size = gguf_get_tensor_size(ctx_, index);
        if (relative_offset > std::numeric_limits<size_t>::max() - metadata_size) {
            close();
            set_error(error, "GGUF tensor offset overflows file address space");
            return false;
        }
        const size_t absolute_offset = metadata_size + relative_offset;
        if (absolute_offset > file_size_ || tensor_size > file_size_ - absolute_offset) {
            const char* name = gguf_get_tensor_name(ctx_, index);
            const std::string tensor_name = name ? name : "<unnamed>";
            close();
            set_error(error, "GGUF tensor payload exceeds file size: " + tensor_name);
            return false;
        }
    }
    path_ = path;
    return true;
}

void HLIEGgufBundle::close() {
    if (ctx_) {
        gguf_free(ctx_);
        ctx_ = nullptr;
    }
    if (data_ctx_) {
        ggml_free(data_ctx_);
        data_ctx_ = nullptr;
    }
    path_.clear();
    file_size_ = 0;
}

std::optional<HLIEGgufTensorSpan> HLIEGgufBundle::find_tensor(const std::string& name) const {
    if (!ctx_) {
        return std::nullopt;
    }
    const int64_t count = gguf_get_n_tensors(ctx_);
    const size_t metadata_size = gguf_get_meta_size(ctx_);
    for (int64_t index = 0; index < count; ++index) {
        const char* tensor_name = gguf_get_tensor_name(ctx_, index);
        if (tensor_name && name == tensor_name) {
            const size_t relative_offset = gguf_get_tensor_offset(ctx_, index);
            const size_t tensor_size = gguf_get_tensor_size(ctx_, index);
            if (relative_offset > std::numeric_limits<size_t>::max() - metadata_size) {
                return std::nullopt;
            }
            const size_t absolute_offset = metadata_size + relative_offset;
            if (absolute_offset > file_size_ || tensor_size > file_size_ - absolute_offset) {
                return std::nullopt;
            }
            return HLIEGgufTensorSpan{
                name,
                absolute_offset,
                tensor_size,
            };
        }
    }
    return std::nullopt;
}

bool HLIEGgufBundle::has_tensor(const std::string& name) const {
    return find_tensor(name).has_value();
}

std::vector<std::string> HLIEGgufBundle::tensor_names() const {
    std::vector<std::string> names;
    if (!ctx_) {
        return names;
    }
    const int64_t count = gguf_get_n_tensors(ctx_);
    names.reserve(static_cast<size_t>(count));
    for (int64_t index = 0; index < count; ++index) {
        if (const char* name = gguf_get_tensor_name(ctx_, index)) {
            names.emplace_back(name);
        }
    }
    return names;
}

bool HLIEGgufBundle::get_string(const std::string& key, std::string& value,
                                const bool required, std::string* error) const {
    if (!ctx_) {
        set_error(error, "GGUF bundle is not open");
        return false;
    }
    const int64_t index = gguf_find_key(ctx_, key.c_str());
    if (index < 0) {
        if (required) {
            set_error(error, "missing GGUF string metadata: " + key);
        }
        return false;
    }
    if (gguf_get_kv_type(ctx_, index) != GGUF_TYPE_STRING) {
        set_error(error, "GGUF metadata has wrong type: " + key);
        return false;
    }
    value = gguf_get_val_str(ctx_, index);
    return true;
}

bool HLIEGgufBundle::get_int32(const std::string& key, int32_t& value,
                               const bool required, std::string* error) const {
    if (!ctx_) {
        set_error(error, "GGUF bundle is not open");
        return false;
    }
    const int64_t index = gguf_find_key(ctx_, key.c_str());
    if (index < 0) {
        if (required) {
            set_error(error, "missing GGUF int32 metadata: " + key);
        }
        return false;
    }
    if (gguf_get_kv_type(ctx_, index) != GGUF_TYPE_INT32) {
        set_error(error, "GGUF metadata has wrong type: " + key);
        return false;
    }
    value = gguf_get_val_i32(ctx_, index);
    return true;
}

bool HLIEGgufBundle::load_fp16_tensor(const std::string& name,
                                      const std::vector<int64_t>& shape,
                                      tcim::Tensor& value, std::string* error) const {
    const auto span = find_tensor(name);
    if (!span) {
        set_error(error, "missing GGUF tensor: " + name);
        return false;
    }

    size_t elements = 1;
    for (const int64_t dimension : shape) {
        if (dimension <= 0 || elements > std::numeric_limits<size_t>::max() / static_cast<size_t>(dimension)) {
            set_error(error, "invalid or overflowing shape for tensor: " + name);
            return false;
        }
        elements *= static_cast<size_t>(dimension);
    }
    if (elements > std::numeric_limits<size_t>::max() / sizeof(uint16_t) ||
        elements * sizeof(uint16_t) != span->size) {
        std::ostringstream message;
        message << "FP16 tensor size mismatch for " << name << ": expected "
                << elements * sizeof(uint16_t) << ", got " << span->size;
        set_error(error, message.str());
        return false;
    }

    std::ifstream stream(path_, std::ios::binary);
    if (!stream) {
        set_error(error, "failed to open GGUF payload: " + path_);
        return false;
    }
    stream.seekg(static_cast<std::streamoff>(span->offset));
    if (!stream) {
        set_error(error, "failed to seek to tensor: " + name);
        return false;
    }

    const auto info = tcim::TensorInfo::CreateNDInfo(shape, tcim::DataType::FLOAT16);
    tcim::Tensor tensor = tcim::Tensor::CreateHostTensor(info);
    if (tensor.GetInitStatus() != tcim::Status::OK) {
        set_error(error, "failed to allocate host tensor: " + name);
        return false;
    }
    stream.read(static_cast<char*>(tensor.Data()), static_cast<std::streamsize>(span->size));
    if (!stream || static_cast<size_t>(stream.gcount()) != span->size) {
        set_error(error, "failed to read complete tensor payload: " + name);
        return false;
    }
    value = std::move(tensor);
    return true;
}

HLIE_NAMESPACE_END

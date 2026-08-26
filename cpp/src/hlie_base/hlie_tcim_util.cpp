#include "hlie_base/hlie_tcim_util.h"

#include <fstream>
#include <regex>
#include <tcim/tcim_runtime.h>

#include "hlie_base/hlie_logger.h"


HLIE_NAMESPACE_BEGIN


int64_t TcimUtil::numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (const auto s : shape)
        n *= s;
    return n;
}


/**
 * 加载通过pytorch 保存成bin结构的embedding 到CPU
 *
 * @param path
 * @param shape
 * @return
 */
tcim::Tensor TcimUtil::load_fp16_bin(const std::string& path, const std::vector<int64_t>& shape) {
    int64_t total = numel(shape);

    std::vector<uint16_t> buffer(total);

    std::ifstream f(path, std::ios::binary);
    assert(f && "failed to open file");

    f.read((char*)buffer.data(), total * sizeof(uint16_t));
    assert(f && "failed to read file");

    // 将 std::vector<int> 转换为 std::vector<int64_t>
    std::vector<int64_t> shape64(shape.begin(), shape.end());
    tcim::TensorInfo info = tcim::TensorInfo::CreateNDInfo(shape64, tcim::DataType::FLOAT16);
    tcim::Tensor t = tcim::Tensor::CreateHostTensor(info);

    memcpy(t.Data(), buffer.data(), total * sizeof(uint16_t));

    return t;
}

/**
 * 从文件读取指定 offset 和 size 的 fp16 数据到内存
 *
 * @param path 文件路径
 * @param offset 偏移量（字节）
 * @param size 数据大小（字节）
 * @return 读取的 fp16 数据向量
 */
std::vector<uint16_t> TcimUtil::load_fp16_data(const std::string& path, size_t offset, size_t size) {
    size_t num_elements = size / sizeof(uint16_t);
    std::vector<uint16_t> buffer(num_elements);

    std::ifstream f(path, std::ios::binary);
    assert(f && "failed to open file");

    f.seekg(offset);
    assert(f.good() && "failed to seek to offset");

    f.read((char*)buffer.data(), size);
    assert(f && "failed to read file");

    return buffer;
}

/**
 * 从内存数据创建 FP16 Tensor
 *
 * @param data 内存数据指针
 * @param shape Tensor 的形状
 * @return 创建的 Tensor
 */
tcim::Tensor TcimUtil::create_fp16_tensor(const void* data, const std::vector<int64_t>& shape) {
    std::vector<int64_t> shape64(shape.begin(), shape.end());
    tcim::TensorInfo info = tcim::TensorInfo::CreateNDInfo(shape64, tcim::DataType::FLOAT16);
    tcim::Tensor t = tcim::Tensor::CreateHostTensor(info);

    memcpy(t.Data(), data, numel(shape) * sizeof(uint16_t));

    return t;
}

tcim::Tensor TcimUtil::embedding_lookup(const tcim::Tensor& weight, const std::vector<int>& input_ids) {
    const auto& weight_shape = weight.Info().Shape();
    int vocab = weight_shape[0];
    int dim = weight_shape[1];

    int T = input_ids.size();

    for (auto s : weight_shape) {
        LOG_TRACE("embedding_lookup weight shape : {}", s);
    }

    LOG_TRACE("embedding_lookup vocab: {}, dim: {}， input_ids: {}", vocab, dim, T);

    // 使用 weight 的 dtype 和 shape 创建输出 Tensor
    std::vector<int64_t> out_shape = {1, (int64_t)T, (int64_t)dim};
    tcim::TensorInfo info = tcim::TensorInfo::CreateNDInfo(out_shape, weight.Info().DataType());
    tcim::Tensor out = tcim::Tensor::CreateHostTensor(info);

    const uint16_t* w = (const uint16_t*)weight.Data();
    uint16_t* o = (uint16_t*)out.Data();
    LOG_TRACE("embedding_lookup start look");

    for (int i = 0; i < T; i++) {
        int id = input_ids[i];
        LOG_TRACE("embedding_lookup look id: {}", id);
        if (id >= vocab) {
            LOG_ERROR("assert(id >= 0 && id < vocab) failed");
        }
        assert(id >= 0 && id < vocab);

        memcpy(o + i * dim, w + id * dim, dim * sizeof(uint16_t));
    }

    return out;
}

/**
 * Tcim Tensor 拼接 , dim use 1
 *
 * @param inputs
 * @return
 */
tcim::Tensor TcimUtil::concat_seq(const std::vector<tcim::Tensor>& inputs) {
    assert(!inputs.empty());

    // 从第一个 tensor 获取 shape 和 dtype
    const auto& ref_shape = inputs[0].Info().Shape();
    tcim::DataType dtype = inputs[0].Info().DataType();

    int BB = ref_shape[0];
    int DD = ref_shape[2];

    int total_T = 0;
    for (auto& t : inputs) {
        assert(t.shape()[0] == BB);
        assert(t.shape()[2] == DD);
        total_T += t.Info().Shape()[1];
    }

    // 使用获取到的 shape 创建 TensorInfo
    std::vector<int64_t> out_shape = {ref_shape[0], (int64_t)total_T, ref_shape[2]};
    tcim::TensorInfo info = tcim::TensorInfo::CreateNDInfo(out_shape, dtype);
    tcim::Tensor out = tcim::Tensor::CreateHostTensor(info);

    uint16_t* dst = (uint16_t*)out.Data();

    int offset_T = 0;

    for (auto& t : inputs) {
        int T = t.Info().Shape()[1];

        const uint16_t* src = (const uint16_t*)t.Data();

        memcpy(dst + offset_T * DD, src, T * DD * sizeof(uint16_t));

        offset_T += T;
    }

    return out;
}

std::vector<std::string> TcimUtil::get_module_input_names(const std::shared_ptr<tcim::Module>& module) {
    std::vector<std::string> input_names;

    if (!module) {
        return input_names;
    }

    size_t num_inputs = module->GetInputNum();
    input_names.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; ++i) {
        input_names.push_back(module->GetInputName(static_cast<int>(i)));
    }

    return input_names;
}

std::vector<std::string> TcimUtil::get_module_output_names(const std::shared_ptr<tcim::Module>& module) {
    std::vector<std::string> output_names;

    if (!module) {
        return output_names;
    }

    size_t num_outputs = module->GetOutputNum();
    output_names.reserve(num_outputs);

    for (size_t i = 0; i < num_outputs; ++i) {
        output_names.push_back(module->GetOutputName(static_cast<int>(i)));
    }

    return output_names;
}

void TcimUtil::set_input_data(const tcim::Tensor& tensor, const int batch_size, const int value) {
    if (tensor.Info().DataType() == tcim::DataType::INT16) {
        std::vector<int16_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::INT32) {
        std::vector<int32_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::INT8) {
        std::vector<int8_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::FLOAT16) {
        std::vector<uint16_t> data_vec(batch_size, value);
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else {
        throw std::invalid_argument("Unsupported tensor data type");
    }
}

void TcimUtil::set_input_data(const tcim::Tensor& tensor, const std::vector<int32_t>& int32_vec, size_t offset = 0) {
    if (offset >= int32_vec.size()) {
        LOG_ERROR("offset is out of range, offset: {}, int32_vec.size(): {}", offset, int32_vec.size());
        throw std::out_of_range("Offset is out of range");
    }

    if (tensor.Info().DataType() == tcim::DataType::INT16) {
        std::vector<int16_t> data_vec(int32_vec.size() - offset, 0);
        for (size_t i = offset; i < int32_vec.size(); i++) {
            data_vec[i - offset] = static_cast<int16_t>(int32_vec[i]);
        }
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else if (tensor.Info().DataType() == tcim::DataType::INT32) {
        std::vector<int32_t> data_vec(int32_vec.size() - offset, 0);
        for (size_t i = offset; i < int32_vec.size(); i++) {
            data_vec[i - offset] = int32_vec[i];
        }
        tensor.Buffer().CopyFromHost(data_vec.data(), tensor.MemSize());
    } else {
        throw std::invalid_argument("Unsupported tensor data type");
    }
}

tcim::Status TcimUtil::set_module_input(const std::shared_ptr<tcim::Module>& module, const std::string& input_name, const float* data) {

    if (!module) {
        LOG_ERROR("Module is null");
        return tcim::Status::UNINITIALIZED;
    }

    // Step 1: Get expected input tensor info (fp16)
    auto input_tensor_info_fp16 = module->GetInputInfo(input_name).AsContiguous();

    // Step 2: Create fp16 tensor on host
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_tensor_info_fp16);

    // Step 3: Create float32 tensor info with same shape
    auto tensor_info_float32 = tcim::TensorInfo::CreateNDInfo(input_tensor_info_fp16.Shape(), tcim::FLOAT32);
    auto input_tensor_float32 = tcim::Tensor::CreateHostTensor(tensor_info_float32);

    // Step 4: Copy data from host to float32 tensor
    // Note: Use input_tensor_float32.MemSize() for the float32 tensor size
    auto mem_size = input_tensor_float32.MemSize();
    auto status = input_tensor_float32.Buffer().CopyFromHost(data, mem_size);
    if (status != tcim::Status::OK) {
        LOG_ERROR("Failed to copy data from host to float32 tensor");
        return status;
    }

    // Step 5: Cast float32 -> float16
    auto cast_status = input_tensor_float32.CastTo(input_tensor);
    if (cast_status != tcim::Status::OK) {
        LOG_ERROR("Failed to cast FLOAT32 tensor to FLOAT16");
        return cast_status;
    }

    // Step 6: Set input to module
    auto set_input_status = module->SetInput(input_name, input_tensor);
    if (set_input_status != tcim::Status::OK) {
        LOG_ERROR("Failed to set input tensor for the model: {}", input_name);
        return set_input_status;
    }

    return tcim::Status::OK;
}

tcim::Status TcimUtil::set_module_input(const std::shared_ptr<tcim::Module>& module, const std::string& input_name, const uint16_t* data) {

    if (!module) {
        LOG_ERROR("Module is null");
        return tcim::Status::UNINITIALIZED;
    }

    auto input_tensor_info_fp16 = module->GetInputInfo(input_name).AsContiguous();
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_tensor_info_fp16);

    auto mem_size = input_tensor_info_fp16.MemSize();
    auto status = input_tensor.Buffer().CopyFromHost(data, mem_size);
    if (status != tcim::Status::OK) {
        LOG_ERROR("Failed to copy data from host to float32 tensor");
        return status;
    }

    auto set_input_status = module->SetInput(input_name, input_tensor);
    if (set_input_status != tcim::Status::OK) {
        LOG_ERROR("Failed to set input tensor for the model: {}", input_name);
        return set_input_status;
    }

    return tcim::Status::OK;
}

uint16_t TcimUtil::float_to_float16(const float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;

    // 1. 处理 NaN 和 Infinity
    if (exp == 128) {
        if (mantissa == 0)
            return sign | 0x7C00; // Inf
        // NaN: 保持最高位有效载荷，确保它依然是 NaN
        return sign | 0x7C00 | (mantissa >> 13) | (mantissa != 0 && (mantissa >> 13) == 0);
    }

    // 2. 边界处理：溢出
    if (exp > 15)
        return sign | 0x7C00;

    // 3. 处理次正规数 (Subnormal) 或 零
    if (exp < -14) {
        if (exp < -24)
            return sign; // 太小，转为正负0

        // 补充隐含位并根据指数差值进行位移
        mantissa |= 0x800000;
        uint32_t shift = 14 - exp;

        // 舍入逻辑 (Tie-to-even)
        uint32_t mask = (1 << shift) - 1;
        uint32_t sticky = (mantissa & mask) != 0;
        uint32_t round_bit = (mantissa >> (shift - 1)) & 1;
        uint16_t sub = (mantissa >> shift);

        if (round_bit && (sticky || (sub & 1)))
            sub++;
        return sign | sub;
    }

    // 4. 处理正规数 (Normal)
    uint16_t new_exp = (uint16_t)(exp + 15);

    // 经典的 Tie-to-even 舍入
    uint32_t round_bit = (mantissa >> 12) & 1;
    uint32_t sticky = (mantissa & 0xFFF) != 0;
    uint16_t new_mantissa = (uint16_t)(mantissa >> 13);

    if (round_bit && (sticky || (new_mantissa & 1))) {
        new_mantissa++;
        if (new_mantissa > 0x3FF) { // 进位导致指数增加
            new_mantissa = 0;
            new_exp++;
        }
    }

    if (new_exp >= 31)
        return sign | 0x7C00; // 舍入后溢出

    return sign | (new_exp << 10) | new_mantissa;
}

float TcimUtil::float16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            // zero
            f = sign;
        } else {
            // subnormal
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FF;
            exp += (127 - 15);
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // inf / NaN
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        // normal
        exp = exp + (127 - 15);
        f = sign | (exp << 23) | (mant << 13);
    }

    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

std::vector<TensorFP16> TcimUtil::float_to_tensor_vec(const std::vector<float>& values) {
    std::vector<TensorFP16> out(values.size());
    std::transform(values.begin(), values.end(), out.begin(), [](float value) { return static_cast<TensorFP16>(value); });
    return out;
}

std::vector<float> TcimUtil::tensor_to_float_vec(const void* data, size_t count) {
    const auto* typed = static_cast<const TensorFP16*>(data);
    std::vector<float> out(count);
    std::transform(typed, typed + count, out.begin(), [](TensorFP16 value) { return static_cast<float>(value); });
    return out;
}

HLIE_NAMESPACE_END

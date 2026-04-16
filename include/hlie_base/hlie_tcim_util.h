#ifndef HLIE_TCIM_UTIL_H
#define HLIE_TCIM_UTIL_H

#include <filesystem>
#include <string>
#include <vector>


#include "half.hpp"
#include "hlie_base/hlie_base.h"
#include "tcim/tcim_runtime.h"

/**
 * @brief Check TCIM status
 */
inline void CheckTcimStatus(const tcim::Status& status, const char* file = __FILE__, const int line = __LINE__) {
    if (status != tcim::Status::OK) {
        std::ostringstream err_msg;
        err_msg << "TCIM error at " << file << ":" << line << ", status: " << static_cast<int>(status);
        throw std::runtime_error(err_msg.str());
    }
}

#define CHECK_TCIM_STATUS(status) CheckTcimStatus(status, __FILE__, __LINE__)

using TensorFP16 = half_float::half;

HLIE_NAMESPACE_BEGIN

class HLIE_API TcimUtil {
public:
    static int64_t numel(const std::vector<int>& shape);
    static tcim::Tensor load_fp16_bin(const std::string& path, const std::vector<int>& shape);
    static std::vector<uint16_t> load_fp16_data(const std::string& path, size_t offset, size_t size);
    static tcim::Tensor create_fp16_tensor(const void* data, const std::vector<int>& shape);
    static tcim::Tensor embedding_lookup(const tcim::Tensor& weight, const std::vector<int>& input_ids);
    static tcim::Tensor concat_seq(const std::vector<tcim::Tensor>& inputs);

    /**
     * Get all input names of a TCIM module.
     *
     * @param module Pointer to the TCIM module
     * @return Vector of input tensor names
     */
    static std::vector<std::string> get_module_input_names(const std::shared_ptr<tcim::Module>& module);

    /**
     * Get all output names of a TCIM module.
     *
     * @param module Pointer to the TCIM module
     * @return Vector of output tensor names
     */
    static std::vector<std::string> get_module_output_names(const std::shared_ptr<tcim::Module>& module);
    static void set_input_data(const tcim::Tensor& tensor, int batch_size, int value);
    static void set_input_data(const tcim::Tensor& tensor, const std::vector<int32_t>& int32_vec, size_t offset);

    /**
     * Create input tensor for TCIM module from float data.
     * Handles float32 -> float16 conversion automatically.
     *
     * @param module The TCIM module
     * @param input_name Name of the input tensor
     * @param data Pointer to float data (FLOAT32)
     * @return Status indicating success or failure
     */
    static tcim::Status set_module_input(const std::shared_ptr<tcim::Module>& module, const std::string& input_name, const float* data);

    /**
     * Create input tensor for TCIM module from float data.
     * Handles float32 -> float16 conversion automatically.
     *
     * @param module The TCIM module
     * @param input_name Name of the input tensor
     * @param data Pointer to float data (FLOAT16)
     * @return Status indicating success or failure
     */
    static tcim::Status set_module_input(const std::shared_ptr<tcim::Module>& module, const std::string& input_name, const uint16_t* data);

    /**
     * Float32 to float16 conversion (IEEE 754 half-precision)
     *
     * @param f
     * @return
     */
    static uint16_t float_to_float16(float f);

    static float float16_to_float(uint16_t h);
};

HLIE_NAMESPACE_END

#endif // HLIE_GGUF_UTIL_H

#ifndef HLIE_MODEL_BASE_H
#define HLIE_MODEL_BASE_H

#define _USE_MATH_DEFINES
#include "arg.h"
#include "hlie_base/hlie_base.h"
#include "tcim/tcim_runtime.h"


HLIE_NAMESPACE_BEGIN

class HLIEModelBase {
public:
    void set_devices(const std::vector<int>& devices);

protected:
    std::shared_ptr<tcim::Module> create_infer_engine(const std::string& file_name, size_t offset, size_t size,
                                                      std::vector<std::string> dummy_tensor_names = {});
    bool load_infer_model(const char* name, const std::string& model_path, std::shared_ptr<tcim::Module>& model_ptr, ggml_context* ctx_data,
                          gguf_context* ctx, size_t offset, size_t size);

    std::vector<int> device_ids_;
    tcim::Module::WeightManager wm_;
};

HLIE_NAMESPACE_END

#endif

//
// Created by tuo.wang on 2025/8/1.
//

#ifndef HLIE_MODEL_INFO_H
#define HLIE_MODEL_INFO_H

#include <string>

#include "hlie_base/hlie_base.h"

class INIReader;

HLIE_NAMESPACE_BEGIN

class HLIE_API HLIEModelInfo {
public:
    int parse_from_file(const std::string& file_path = "info.json");

    [[nodiscard]] const std::string& get_model_path() const { return model_path_; }
    [[nodiscard]] const std::string& get_model_type() const { return model_type_; }
    [[nodiscard]] int get_batch_num() const { return batch_num_; }
    [[nodiscard]] const std::string& get_model_version() const { return model_version_; }
    [[nodiscard]] const std::string& get_mm_proj_path() const { return mm_proj_path_; }
    [[nodiscard]] const std::string& get_params() const { return params_; }
    [[nodiscard]] int get_xh_version() const { return xh_version_; }
    [[nodiscard]] int get_context_size() const { return context_size_; }
    [[nodiscard]] int get_cores() const { return cores_; }
    [[nodiscard]] bool get_is_multi_model() const { return is_multi_model_; }
    [[nodiscard]] const std::string& get_clip_model() const { return clip_model_; }
    [[nodiscard]] const std::string& get_clip_l_model() const { return clip_l_model_; }
    [[nodiscard]] const std::string& get_t5_model() const { return t5_model_; }
    [[nodiscard]] const std::string& get_mmdit_model() const { return mmdit_model_; }
    [[nodiscard]] const std::string& get_vae_model() const { return vae_model_; }
    [[nodiscard]] const std::string& get_devices() const { return devices_; }

private:
    std::string model_path_;
    std::string model_type_;
    std::string model_version_;
    std::string mm_proj_path_;
    std::string params_;
    int xh_version_{1};
    int context_size_{2048};
    int cores_{0};
    int batch_num_{1};
    bool is_multi_model_{false};
    std::string clip_model_;
    std::string clip_l_model_;
    std::string t5_model_;
    std::string mmdit_model_;
    std::string vae_model_;
    std::string devices_;
};

HLIE_NAMESPACE_END

#endif // HLIE_MODEL_INFO_H

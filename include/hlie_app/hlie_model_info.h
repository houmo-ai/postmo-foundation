//
// Created by tuo.wang on 2025/8/1.
//

#ifndef HLIE_MODEL_INFO_H
#define HLIE_MODEL_INFO_H

#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"

class INIReader;

HLIE_NAMESPACE_BEGIN

struct HLIEModelCapabilities {
    bool text = true;
    bool vision = false;
    bool audio = false;
};

class HLIE_API HLIEModelInfo {
public:
    int parse_from_file(const std::string& file_path = "info.json");

    int parse_from_gguf_filename(const std::string& file_path);

    // Parse routing identity from a filename without opening the GGUF file.
    int parse_identity_from_gguf_filename(const std::string& file_path);

    static std::string parse_mmproj_vl_spec(const std::string& name);

    static std::string build_model_key(const HLIEModelInfo& model_info);

    static void ScanLocalModelsFromDir(std::vector<HLIEModelInfo>& result, const std::string& dir);

    static int estimate_mem(const std::string& model_name);
    static int estimate_empirical_mem(const std::string& model_name);
    static int estimate_mem_from_gguf(const std::string& model_path, int context_size = 0, int batch_num = 1);
    static int estimate_device_num(const std::string& model_name);


    void build_llm(const std::string& model_path, const std::string& model_type, int context_size, int cores, int batch_num, int xh_version,
                   bool is_multi_model, const std::string& params, const std::string& mm_proj_path, const std::string& devices);

    void build_sd(const std::string& model_type, const std::string& vae_model, const std::string& t5_model, const std::string& clip_model,
                  const std::string& clip_l_model, const std::string& mmdit_model, const std::string& devices);

    void set_devices(const std::string& devices) { devices_ = devices; }
    void set_service_kind(const std::string& service_kind);
    void set_mm_proj_path(const std::string& mm_proj_path, bool inspect_metadata = true);

    [[nodiscard]] int to_json_file(const std::string& file_path) const;

    [[nodiscard]] const std::string& get_model_name() const { return model_name_; }
    [[nodiscard]] const std::string& get_model_path() const { return model_path_; }
    // Legacy mixed-semantics field retained for info.json and API compatibility.
    [[nodiscard]] const std::string& get_model_type() const { return model_type_; }
    [[nodiscard]] const std::string& get_service_kind() const { return service_kind_; }
    [[nodiscard]] const std::string& get_model_family() const { return model_family_; }
    [[nodiscard]] const std::string& get_architecture() const { return architecture_; }
    [[nodiscard]] const HLIEModelCapabilities& get_capabilities() const { return capabilities_; }
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
    [[nodiscard]] int get_device_num() const { return device_num_; }
    [[nodiscard]] int get_estimated_mem() const;

    bool is_embedding_model() const;
    bool is_rerank_model() const;
    bool is_mtp() const;

    // ===== 模型类型 =====
    bool is_llm_ = true;
    bool is_multi_model_ = false; // VL = true
    std::string model_key_;

    // ===== VL 相关 =====
    std::string mm_proj_path_;    // mmproj 路径
    std::string vl_spec_;         // vit_448_448 / vit_644_364 / unknown

    // ===== 构建信息 =====
    std::string model_version_;   // v0.6.0
    std::string build_time_;      // 20251202...

private:
    void inspect_main_gguf_metadata();
    void inspect_mmproj_gguf_metadata();
    void infer_service_kind_from_name();

    // ===== 基本信息 =====
    std::string model_type_;      // qwen2.5 / qwen2.5-vl / qwen3-vl
    std::string service_kind_ = "llm";
    std::string model_family_;
    std::string architecture_;
    HLIEModelCapabilities capabilities_;
    std::string model_path_;      // /path/to/model.gguf
    std::string model_name_;
    std::string params_;          // 7b / 4b
    int context_size_ = 0;
    int batch_num_ = 1;
    int cores_ = 2;
    int device_num_ = 1;
    int xh_version_{2};           // xh1 / xh2

    std::string clip_model_;
    std::string clip_l_model_;
    std::string t5_model_;
    std::string mmdit_model_;
    std::string vae_model_;
    std::string devices_;
    int estimated_mem_mb_ = 0;
};

HLIE_NAMESPACE_END

#endif // HLIE_MODEL_INFO_H

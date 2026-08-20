#ifndef HLIE_SD_MODEL_SERVICE_H
#define HLIE_SD_MODEL_SERVICE_H

#include <future>
#include <memory>
#include <string>
#include <vector>

#include <server/server-http.h>

#include "hlie_base/hlie_blocking_queue.h"
#include "hlie_model_service.h"
#include "httplib.h"
#include "json.hpp"
#include "stable-diffusion.h"

HLIE_NAMESPACE_BEGIN

/**
 * sd.cpp 内示例参数，需要根据后续需求进行优化整理
 */

enum SDMode {
    IMG_GEN,
    VID_GEN,
    CONVERT,
    MODE_COUNT
};
struct SDParams {
    int n_threads = -1;
    SDMode mode   = IMG_GEN;
    std::string model_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string control_net_path;
    std::string embedding_dir;
    std::string stacked_id_embed_dir;
    std::string input_id_images_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string tensor_type_rules;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string input_path;
    std::string mask_path;
    std::string control_image_path;
    std::vector<std::string> ref_image_paths;
    std::string houmo_mmdit_path;
    std::string houmo_clip_path;
    std::string houmo_clip_l_path;
    std::string houmo_t5_path;
    std::string houmo_vae_path;
    int hm_sd_device = 0;
    std::string prompt;
    std::string negative_prompt;
    float min_cfg       = 1.0f;
    float cfg_scale     = 4.5f;
    float img_cfg_scale = INFINITY;
    float guidance      = 3.5f;
    float eta           = 0.f;
    float style_ratio   = 20.f;
    int clip_skip       = -1;  // <= 0 represents unspecified
    int width           = 512;
    int height          = 512;
    int batch_count     = 1;

    int video_frames         = 6;
    int motion_bucket_id     = 127;
    int fps                  = 6;
    float augmentation_level = 0.f;

    sample_method_t sample_method = EULER;
    schedule_t schedule           = DEFAULT;
    int sample_steps              = 8;
    float strength                = 0.75f;
    float control_strength        = 0.9f;
    rng_type_t rng_type           = CUDA_RNG;
    int64_t seed                  = 42;
    bool verbose                  = false;
    bool vae_tiling               = false;
    bool control_net_cpu          = false;
    bool normalize_input          = false;
    bool clip_on_cpu              = false;
    bool vae_on_cpu               = false;
    bool diffusion_flash_attn     = false;
    bool canny_preprocess         = false;
    bool color                    = false;
    int upscale_repeats           = 1;

    std::vector<int> skip_layers = {7, 8, 9};
    float slg_scale              = 0.f;
    float skip_layer_start       = 0.01f;
    float skip_layer_end         = 0.2f;

    bool chroma_use_dit_mask = true;
    bool chroma_use_t5_mask  = false;
    int chroma_t5_mask_pad   = 1;
};

/**
 * 生图请求消息体
 */
struct ImagesRequest {
    HLIEQueuedRequestControl control;
    int num;
    std::string prompt;
    int width = 512;
    int height = 512;
    int steps = 8;
    int seed = 42;
    float guidance_scale = 4.5;
    // Todo: Extent Params
};

struct ImagesResult {
    std::shared_ptr<sd_image_t> image;
    bool success = false;
    std::string error_message;
    int64_t predicted_ms = 0;
};

class SDModelService final : public IHLIEModelService {
public:
    SDModelService() = default;
    ~SDModelService() override = default;

    /**
     * 初始化并传递参数给SDCtx
     */
    int init(const std::string& model_dir, const HLIEModelInfo& model_info, const std::shared_ptr<HLIEDeviceService>& device_service) override;

    /**
     * 执行模型推理进行生图
     */
    void run() override;

    void process() override;

    void stop() override;

    int get_state() override;

    int handle_health(nlohmann::ordered_json& resp);

    void handle_models(nlohmann::ordered_json& resp) const;

    server_http_res_ptr handle_image_generations(const server_http_req& req) const;

    std::shared_ptr<SDParams> get_params() const {
        return params_;
    }

private:
    std::future<ImagesResult> generate_image_async(const ImagesRequest& request) const;
    ImagesResult generate_image_sync(const ImagesRequest& request,
                                     const std::function<bool()>& should_stop = {}) const;

    std::shared_ptr<sd_image_t> generate_image(const ImagesRequest& request) const;

    std::shared_ptr<SDParams> params_;
    std::shared_ptr<sd_guidance_params_t> guidance_params;
    std::shared_ptr<sd_img_gen_params_t> img_gen_params;
    std::shared_ptr<sd_ctx_t> sd_ctx_t_;
    
    struct QueueItem {
        ImagesRequest request;
        std::shared_ptr<std::promise<ImagesResult>> promise;
    };

    std::shared_ptr<HLIEBlockingQueue<QueueItem>> blocking_queue_;

    std::unique_ptr<std::thread> worker_thread_;  // 工作线程
    std::atomic<bool> should_stop_{false};       // 线程停止标志
};

HLIE_NAMESPACE_END

#endif

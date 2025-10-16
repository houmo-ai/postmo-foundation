#include "hlie_app/hlie_sd_model_service.h"

#include "base64.hpp"
#include "hlie_app/hlie_model_info.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_exception.h"
#include "hlie_base/hlie_util.h"
#include "stb_image_write.h"

HLIE_NAMESPACE_BEGIN

namespace {
    void sd_log_cb(const sd_log_level_t level, const char* log, void* data) {
        if (const auto* params = static_cast<SDParams*>(data); !log || (!params->verbose && level <= SD_LOG_DEBUG)) {
            return;
        }

        spdlog::level::level_enum log_level;
        switch (level) {
        case SD_LOG_INFO:
            log_level = spdlog::level::level_enum::info;
            break;
        case SD_LOG_WARN:
            log_level = spdlog::level::level_enum::warn;
            break;
        case SD_LOG_ERROR:
            log_level = spdlog::level::level_enum::err;
            break;
        case SD_LOG_DEBUG:
        default:
            log_level = spdlog::level::level_enum::debug;
            break;
        }

        spdlog::log(log_level, log);
    }
} // namespace


int SDModelService::init(const std::string& model_dir, const HLIEModelInfo& model_info) {
    if (model_info.get_xh_version() < 2) {
        return -1;
    }

    blocking_queue_ = std::make_shared<HLIEBlockingQueue<QueueItem>>();

    params_ = std::make_shared<SDParams>();
    params_->houmo_mmdit_path = model_info.get_mmdit_model();
    params_->houmo_clip_path = model_info.get_clip_model();
    params_->houmo_clip_l_path = model_info.get_clip_l_model();
    params_->houmo_t5_path = model_info.get_t5_model();
    params_->houmo_vae_path = model_info.get_vae_model();

    if (!params_->houmo_mmdit_path.empty() && HlieUtil::is_relative_path(params_->houmo_mmdit_path)) {
        params_->houmo_mmdit_path = HlieUtil::join_paths(model_dir, params_->houmo_mmdit_path);
    }
    if (!params_->houmo_clip_path.empty() && HlieUtil::is_relative_path(params_->houmo_clip_path)) {
        params_->houmo_clip_path = HlieUtil::join_paths(model_dir, params_->houmo_clip_path);
    }
    if (!params_->houmo_clip_l_path.empty() && HlieUtil::is_relative_path(params_->houmo_clip_l_path)) {
        params_->houmo_clip_l_path = HlieUtil::join_paths(model_dir, params_->houmo_clip_l_path);
    }
    if (!params_->houmo_t5_path.empty() && HlieUtil::is_relative_path(params_->houmo_t5_path)) {
        params_->houmo_t5_path = HlieUtil::join_paths(model_dir, params_->houmo_t5_path);
    }
    if (!params_->houmo_vae_path.empty() && HlieUtil::is_relative_path(params_->houmo_vae_path)) {
        params_->houmo_vae_path = HlieUtil::join_paths(model_dir, params_->houmo_vae_path);
    }

    if (const auto& devices = model_info.get_devices();
        !devices.empty() && Singleton<HLIEConfigService>::instance()->get_int("server", "replicas", 1) == 1) {
        // 单实例模式可以定制Model Devices
        const auto device_id = HlieUtil::parse_natural_num(devices);
        if (device_id < 0) {
            LOG_ERROR("SD cannot acquire model specific device \"{}\"", devices);
            return -1;
        }

        const std::optional<int> val = Singleton<HLIEDeviceHelper>::instance()->acquire_device(SD, device_id);
        if (!val) {
            return -1;
        }

        params_->hm_sd_device = val.value();
        devices_.push_back(*val);
        LOG_INFO("LLM using model specific device: {}", device_id);
    } else if (const std::optional<int> val = Singleton<HLIEDeviceHelper>::instance()->acquire_device(SD); !val) {
        LOG_ERROR("SD cannot acquire available device");
        return -1;
    } else {
        params_->hm_sd_device = val.value();
        devices_.push_back(*val);
        LOG_INFO("SD using device: {}", val.value());
    }

    sd_set_log_callback(sd_log_cb, params_.get());

    // sd_guidance_params_t
    guidance_params = std::make_shared<sd_guidance_params_t>();
    *guidance_params = {params_->cfg_scale,
                        params_->img_cfg_scale,
                        params_->min_cfg,
                        params_->guidance,
                        {
                            params_->skip_layers.data(),
                            params_->skip_layers.size(),
                            params_->skip_layer_start,
                            params_->skip_layer_end,
                            params_->slg_scale,
                        }};


    // sd_img_gen_params_t
    const sd_image_t input_image = {static_cast<uint32_t>(params_->width), static_cast<uint32_t>(params_->height), 3, nullptr};

    const sd_image_t mask_image = {static_cast<uint32_t>(params_->width), static_cast<uint32_t>(params_->height), 1, nullptr};

    img_gen_params = std::make_shared<sd_img_gen_params_t>();
    *img_gen_params = {
        nullptr,
        nullptr,
        params_->clip_skip,
        *guidance_params,
        input_image,
        nullptr,
        0,
        mask_image,
        params_->width,
        params_->height,
        params_->sample_method,
        params_->sample_steps,
        params_->eta,
        params_->strength,
        params_->seed,
        params_->batch_count,
        nullptr,
        params_->control_strength,
        params_->style_ratio,
        params_->normalize_input,
        params_->input_id_images_path.c_str(),
    };

    // sd_ctx_t_
    constexpr bool vae_decode_only = true;
    constexpr bool free_params_immediately = true;
    const sd_ctx_params_t sd_ctx_params = {
        params_->model_path.c_str(),
        params_->clip_l_path.c_str(),
        params_->clip_g_path.c_str(),
        params_->t5xxl_path.c_str(),
        params_->diffusion_model_path.c_str(),
        params_->vae_path.c_str(),
        params_->taesd_path.c_str(),
        params_->control_net_path.c_str(),
        params_->lora_model_dir.c_str(),
        params_->embedding_dir.c_str(),
        params_->stacked_id_embed_dir.c_str(),
        params_->houmo_clip_path.c_str(),
        params_->houmo_clip_l_path.c_str(),
        params_->houmo_t5_path.c_str(),
        params_->houmo_mmdit_path.c_str(),
        params_->houmo_vae_path.c_str(),
        vae_decode_only,
        params_->vae_tiling,
        free_params_immediately,
        params_->n_threads,
        params_->wtype,
        params_->rng_type,
        params_->schedule,
        params_->clip_on_cpu,
        params_->control_net_cpu,
        params_->vae_on_cpu,
        params_->diffusion_flash_attn,
        params_->chroma_use_dit_mask,
        params_->chroma_use_t5_mask,
        params_->chroma_t5_mask_pad,
        params_->hm_sd_device
    };

    sd_ctx_t* sd_ctx = new_sd_ctx(&sd_ctx_params);
    if (!sd_ctx) {
        LOG_ERROR("Failed to init sd_ctx");
        Singleton<HLIEDeviceHelper>::instance()->release_devices(devices_);
        devices_.clear();
        return -1;
    }

    sd_ctx_t_ = std::shared_ptr<sd_ctx_t>(sd_ctx, free_sd_ctx);
    return 0;
}

void SDModelService::run() {
    if (worker_thread_) {
        LOG_WARN("SDModelService thread is already running.");
        return;
    }

    state_.store(SERVER_STATE_READY);
    should_stop_ = false;
    worker_thread_ = std::make_unique<std::thread>(&SDModelService::process, this);
    LOG_INFO("SDModelService thread started.");
}

void SDModelService::process() {
    try {
        while (!should_stop_) {
            const auto [request, promise] = blocking_queue_->pop(); // 阻塞直到有数据或队列关闭
            if (should_stop_) {
                break; // 防止 stop() 后仍有残留任务
            }

            try {
                auto start = std::chrono::steady_clock::now();
                auto image_result = generate_image(request);
                auto end = std::chrono::steady_clock::now();
                // 构造结果
                ImagesResult result;
                result.image = image_result;
                result.success = image_result != nullptr;
                result.predicted_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                LOG_INFO("generate_image time: {} ms", result.predicted_ms);

                // 通知调用者
                promise->set_value(result);
            } catch (const std::exception& e) {
                // 错误处理
                ImagesResult result;
                result.success = false;
                result.error_message = e.what();
                promise->set_value(result);
            }
        }
    } catch (const std::runtime_error&) {
        LOG_WARN("SDModelService process exited");
    }
}

void SDModelService::stop() {
    if (!worker_thread_) {
        return;
    }

    should_stop_ = true;
    if (blocking_queue_) {
        blocking_queue_->close(); // 通知队列关闭，唤醒可能阻塞的 pop()
    }

    if (worker_thread_->joinable()) {
        worker_thread_->join();
    }
    worker_thread_.reset();
    sd_ctx_t_.reset();

    Singleton<HLIEDeviceHelper>::instance()->release_devices(devices_);
    devices_.clear();
}

int SDModelService::get_state() {
    return state_;
}

int SDModelService::handle_health(nlohmann::ordered_json& resp) {
    // Todo: 获取服务状态
    return 0;
}

void SDModelService::handle_models(nlohmann::ordered_json& resp) const {
    // 检查resp是否已经有models数组
    if (!resp.contains("models")) {
        resp["models"] = nlohmann::json::array();
    }

    // 添加SD3模型信息
    resp["models"].push_back({
        {"name", SD},
        {"model", SD},
    });
}

std::future<ImagesResult> SDModelService::generate_image_async(const ImagesRequest& request) const {
    const auto promise = std::make_shared<std::promise<ImagesResult>>();
    auto future = promise->get_future();

    // 创建队列项
    QueueItem item;
    item.request = request;
    item.promise = promise;

    if (blocking_queue_) {
        blocking_queue_->push(item);
    }

    return future;
}

ImagesResult SDModelService::generate_image_sync(const ImagesRequest& request) const {
    auto future = generate_image_async(request);
    return future.get();
}

std::shared_ptr<sd_image_t> SDModelService::generate_image(const ImagesRequest& request) const {
    guidance_params->txt_cfg = request.guidance_scale;
    sd_img_gen_params_t local_img_gen_params = *img_gen_params;
    local_img_gen_params.prompt = request.prompt.c_str();
    local_img_gen_params.guidance = *guidance_params;
    local_img_gen_params.seed = request.seed;
    local_img_gen_params.sample_steps = request.steps;
    LOG_INFO("Start to generate image");
    sd_image_t* result_tmp = nullptr;
    try {
        result_tmp = ::generate_image(sd_ctx_t_.get(), &local_img_gen_params);
    } catch (const std::exception& e) {
        LOG_ERROR("generate_image exception: {}", e.what());
        return nullptr;
    }
    LOG_INFO("generate image done");
    auto result = std::shared_ptr<sd_image_t>(result_tmp);
    return std::move(result);
}

void SDModelService::handle_image_generations(const httplib::Request& req, httplib::Response& res) const {
    try {
        const auto request_json = nlohmann::ordered_json::parse(req.body);
        if (const std::string request_id = res.has_header("X-Request-ID") ? res.get_header_value("X-Request-ID") : "";
            !request_id.empty()) {
            LOG_DEBUG("[Request-{}] start to generate image", request_id);
        }


        ImagesRequest images_request;
        images_request.prompt = request_json.value("prompt", "");
        if (images_request.prompt.empty()) {
            throw CommonException(5, "Prompt cannot be empty.");
        }
        images_request.steps = request_json.value("steps", 8);
        if (images_request.steps <= 0 || images_request.steps > 10) {
            throw CommonException(5, "Sample steps must be greater than 0, and less than or equal to 10.");
        }
        images_request.seed = request_json.value("seed", 42);
        images_request.guidance_scale = request_json.value("guidance_scale", 4.5f);
        if (images_request.guidance_scale < 1.0) {
            throw CommonException(5, "Guidance scale must be greater than 1.0 .");
        }
        if (auto [image, success, error_message, predicted_ms] = generate_image_sync(images_request); success) {
            nlohmann::ordered_json response;

            int png_len;
            unsigned char* png_data = stbi_write_png_to_mem(image->data, 0, static_cast<int>(image->width), static_cast<int>(image->height),
                                                            static_cast<int>(image->channel), &png_len, nullptr);
            if (!png_data) {
                free(image->data);
                throw std::runtime_error("Failed to encode image to PNG.");
            }

            std::string b64_json = base64::encode(reinterpret_cast<const char*>(png_data), png_len);

            free(png_data);
            free(image->data);
            response["created"] =
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            response["data"] = nlohmann::json::array({{{"b64_json", b64_json}}});
            response["timings"] = nlohmann::json::object({{"predicted_ms", predicted_ms}});
            res.set_content(response.dump(), "application/json");
            res.status = 200;
        } else {
            throw CommonException(1, "generate_image_sync failed");
        }
    } catch (const std::exception& ex) {
        LOG_ERROR("handle_image_generations ex: {}", ex.what());
        throw;
    }
}


HLIE_NAMESPACE_END

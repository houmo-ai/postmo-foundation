#include "hlie_app/hlie_sd_model_service.h"

#include "hlie_app/hlie_model_info.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN


int SDModelService::init(const std::string& model_path, const HLIEModelInfo& model_info) {
    // Todo: 获取相关的model_info 判断参数是否满足规范
    if (model_info.get_xh_version() < 2) {
        return -1;
    }

    blocking_queue_ = std::make_shared<HLIEBlockingQueue<QueueItem>>();

    // Todo: 参数校验

    params_ = std::make_shared<SDParams>();
    params_->diffusion_model_path = ""; // Todo: 参数赋值
    // ...

    // sd_guidance_params_t


    // sd_ctx_t_

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
        while(!should_stop_) {
            const auto [request, promise] = blocking_queue_->pop();  // 阻塞直到有数据或队列关闭
            if (should_stop_) {
                break;  // 防止 stop() 后仍有残留任务
            }

            try {
                auto image_result = generate_image(request);

                // 构造结果
                ImagesResult result;
                result.image = image_result;
                result.success = image_result != nullptr;

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
    } catch(const std::runtime_error& ex) {
        // 处理队列关闭的情况
        LOG_ERROR("run exception: {} ", ex.what());
        // Todo: 挽救措施，重新拉起
        Sleep(3000);
        process();
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
}

int SDModelService::get_state() {
    return state_;
}

int SDModelService::handle_health(nlohmann::ordered_json& resp) {
    // Todo: 获取服务状态， 或者slots 信息
    return 0;
}

void SDModelService::handle_models(nlohmann::ordered_json& resp) const {


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

std::shared_ptr<sd_image_t> SDModelService::generate_image(const ImagesRequest& request) {
    // Todo: 请求校验

    sd_image_t* result_tmp;

    // Todo: 生图



    auto result = std::shared_ptr<sd_image_t>(result_tmp);
    return std::move(result);
}

void SDModelService::handle_image_generations(const httplib::Request& req, httplib::Response& res) const {
    try {
        const auto request_json = nlohmann::ordered_json::parse(req.body);

        // Todo: 参数校验
        ImagesRequest images_request;
        images_request.prompt = request_json.value("prompt", "");
        images_request.steps = request_json.value("steps", 20);

        if (auto result = generate_image_sync(images_request); result.success) {
            // 构造响应
            nlohmann::ordered_json response;

            // TODO: 将图像转换为 base64 或其他格式
            response["data"] = "image_data_base64";
            res.set_content(response.dump(), "application/json");
            res.status = 200;
        } else {
            // 错误响应
            nlohmann::ordered_json error_response;
            error_response["error"] = result.error_message;
            res.set_content(error_response.dump(), "application/json");
            res.status = 500;
        }
    } catch (const std::exception& e) {
        nlohmann::ordered_json error_response;
        error_response["error"] = e.what();
        res.set_content(error_response.dump(), "application/json");
        res.status = 400;
    }
}


HLIE_NAMESPACE_END

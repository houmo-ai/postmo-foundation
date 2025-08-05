#include "hlie_app/hlie_workflow.h"

#include <filesystem>

#include "hlie_app/hlie_infer_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_sd_model_service.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_exception.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

int HLIEWorkflow::initialize(const std::string& model_path) {
    if (model_path.empty()) {
        LOG_ERROR("model path is empty");
        return -1;
    }

    const auto info_path = std::filesystem::path(model_path) / "info.json";
    LOG_INFO("model info path: {}", info_path.string());
    HLIEModelInfo model_info;
    if (model_info.parse_from_file(info_path.string()) < 0) {
        LOG_ERROR("model info parse failed");
        return -1;
    }

    const auto model_type = model_info.get_model_type();
    if (model_type.empty()) {
        LOG_ERROR("model_type empty");
        return -1;
    }

    LOG_INFO("model_type: {}", model_type);

    if (model_type == HLIEModelType::SD3) {
        LOG_INFO("Begin to init the SDModelService");
        sd_model_service_ = std::make_shared<SDModelService>();
        sd_model_service_->init(model_path, model_info);
    }

    if (model_type == HLIEModelType::Qwen3 || model_type == HLIEModelType::Qwen25 || model_type == HLIEModelType::DeepSeek) {
        LOG_INFO("Begin to init the LLModelService");
        llm_model_service_ = std::make_shared<LLModelService>();
        llm_model_service_->init(model_path, model_info);
    }

    // 设计上可以在多卡上面同时运行不同类型的模型
    // 同模型多卡的运行也要支持

    return 0;
}

void HLIEWorkflow::run() const {
    if (sd_model_service_) {
        sd_model_service_->run();
    }

    if (llm_model_service_) {
        llm_model_service_->run();
    }
}

void HLIEWorkflow::stop() const {
    if (sd_model_service_) {
        sd_model_service_->stop();
    }

    if (llm_model_service_) {
        llm_model_service_->stop();
    }
}

std::shared_ptr<HLIEInferService> HLIEWorkflow::get_infer_service() const {
    return infer_service_;
}

std::shared_ptr<IHLIEModelService> HLIEWorkflow::get_model_service(const std::string& type) const {
    if (HLIEModelType::LLM == type) {
        return llm_model_service_;
    }

    if (HLIEModelType::SD == type) {
        return sd_model_service_;
    }

    return nullptr;
}

int HLIEWorkflow::cacl_thread_num() const {
    if (llm_model_service_) {
        const auto batch_num = llm_model_service_->get_batch_num();
        return std::max(batch_num, static_cast<int32_t>(std::thread::hardware_concurrency()) - 1);
    }

    if (sd_model_service_) {
        return 1;
    }

    LOG_ERROR("model service not initialized");
    return 1;
}

int HLIEWorkflow::get_model_service_state() const {
    if (llm_model_service_) {
        return llm_model_service_->get_state();
    }

    if (sd_model_service_) {
        return sd_model_service_->get_state();
    }

    LOG_ERROR("model service not initialized");
    return -1;
}

std::vector<std::string> HLIEWorkflow::get_api_keys() const {
    return api_keys_;
}

void HLIEWorkflow::handle_health(httplib::Response& res) const {
    json resp = {
        {"status", "ok"},
    };

    if (llm_model_service_) {
        if (llm_model_service_->handle_health(resp) < 0) {
            throw CommonException(4, "llm model service health check failed");
        }
    }
    if (sd_model_service_) {
        if (sd_model_service_->handle_health(resp) < 0) {
            throw CommonException(4, "sd model service health check failed");
        }
    }

    res_ok(res, resp);
}

void HLIEWorkflow::handle_models(httplib::Response& res) const {
    json resp = {};
    if (llm_model_service_) {
        llm_model_service_->handle_models(resp);
    }
    if (sd_model_service_) {
        sd_model_service_->handle_models(resp);
    }

    res_ok(res, resp);
}

const std::vector<HLIEWorkflow::RouteConfig>& HLIEWorkflow::get_route_configs() const {
    static std::vector<RouteConfig> routes = {
        {{"/health"},
            GET,
            [this](const httplib::Request&, httplib::Response& res) { this->handle_health(res); },
            false // 不需要模型Running
        },
        {
                {"/v1/models", "/models", "/api/tags"},
                GET,
                [this](const httplib::Request&, httplib::Response& res) { this->handle_models(res); },
                false
            },
        {
            {"/v1/chat/completions", "/chat/completions", "/api/chat"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) { llm_model_service_->handle_chat_completions(req, res); },
            true, // 需要模型Running
            LLM // LLM模型类型
        },
        {
            {"/v1/images/generations", "/images/generations"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) { sd_model_service_->handle_image_generations(req, res); },
            true,
            SD
        },
        {
            {"/v1/completions"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) { llm_model_service_->handle_completions_oai(req, res); },
            true,
            LLM
        },
        {
            {"/completion", "/completions"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) {
                llm_model_service_->handle_completions(req, res);
            },
            true,
            LLM
        }};

    return routes;
}


HLIE_NAMESPACE_END

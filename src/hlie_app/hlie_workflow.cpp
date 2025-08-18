#include "hlie_app/hlie_workflow.h"

#include <filesystem>

#include "INIReader.h"
#include "hlie_app/hlie_infer_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_sd_model_service.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_exception.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

namespace {
    std::vector<HLIEModelInfo> load_model_infos_from_directory(const std::string& directory_path) {
        std::vector<HLIEModelInfo> model_infos;

        try {
            // 检查目录是否存在
            if (!std::filesystem::exists(directory_path)) {
                LOG_ERROR("Directory does not exist: {}", directory_path);
                return model_infos;
            }

            if (!std::filesystem::is_directory(directory_path)) {
                LOG_ERROR("Path is not a directory: {}", directory_path);
                return model_infos;
            }

            // 遍历目录中的所有文件
            for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
                // 检查是否为JSON文件
                if (entry.is_regular_file() && entry.path().extension() == ".json") {
                    const std::string file_path = entry.path().string();
                    LOG_INFO("Parsing model info from file: {}", file_path);

                    HLIEModelInfo model_info;
                    if (model_info.parse_from_file(file_path) == 0) {
                        model_infos.push_back(std::move(model_info));
                        LOG_INFO("Successfully parsed model info from: {}", file_path);
                    } else {
                        LOG_ERROR("Failed to parse model info from: {}", file_path);
                    }
                }
            }
        } catch (const std::filesystem::filesystem_error& e) {
            LOG_ERROR("Filesystem error when reading directory {}: {}", directory_path, e.what());
        } catch (const std::exception& e) {
            LOG_ERROR("Exception when reading directory {}: {}", directory_path, e.what());
        }

        LOG_INFO("Loaded {} model info files from directory: {}", model_infos.size(), directory_path);
        return model_infos;
    }
} // namespace

int HLIEWorkflow::initialize(const std::string& model_dir) {
    if (model_dir.empty()) {
        LOG_ERROR("model path is empty");
        return -1;
    }

    const auto model_infos = load_model_infos_from_directory(model_dir);
    if (model_infos.empty()) {
        LOG_ERROR("No valid model info files found in directory: {}", model_dir);
        return -1;
    }

    // 处理每个模型信息
    for (const auto& model_info : model_infos) {
        const auto& model_type = model_info.get_model_type();
        if (model_type.empty()) {
            LOG_WARN("Skipping model with empty type");
            continue;
        }

        LOG_INFO("Processing model_type: {}", model_type);

        // Todo: 目前只支持最多两个不同实例， 后续可以扩展
        if (model_type == HLIEModelType::SD3 && !sd_model_service_) {
            LOG_INFO("Initializing SDModelService");
            sd_model_service_ = std::make_shared<SDModelService>();
            if (sd_model_service_->init(model_dir, model_info) < 0) {
                LOG_ERROR("failed to init the SD model service");
                return -1;
            }
        } else if (!llm_model_service_ &&
                   (model_type == HLIEModelType::Qwen3 || model_type == HLIEModelType::Qwen25 || model_type == HLIEModelType::DeepSeek)) {
            LOG_INFO("Initializing LLModelService");
            llm_model_service_ = std::make_shared<LLModelService>();
            if (llm_model_service_->init(model_dir, model_info) < 0) {
                LOG_ERROR("failed to init the LLM model service");
                return -1;
            }
        }
    }

    // 设计上可以在多卡上面同时运行不同类型的模型

    Singleton<HLIEDeviceHelper>::instance()->print_status();
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
        return static_cast<int32_t>(std::thread::hardware_concurrency()) - 1;
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

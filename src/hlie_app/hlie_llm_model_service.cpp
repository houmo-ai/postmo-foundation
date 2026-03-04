#include "hlie_app/hlie_llm_model_service.h"

#include <INIReader.h>
#include <arg.h>
#include <common.h>
#include <server/server-context.h>
#include <server/info_printer.h>
#include <unordered_set>

#include "hlie_app/hlie_model_info.h"
#include "hlie_app/hlie_device_service.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_singleton.h"
#include "hlie_base/hlie_util.h"
#include "spdlog/fmt/bundled/base.h"

HLIE_NAMESPACE_BEGIN

int LLModelService::init(const std::string& model_dir, const HLIEModelInfo& model_info, const std::shared_ptr<HLIEDeviceService>& device_service) {
    if (model_info.get_model_path().empty()) {
        LOG_ERROR("LLM model path cannot be empty");
        return -1;
    }

    device_service_ = device_service;

    std::unordered_map<std::string, std::string> llm_params;
    // 程序内特定启动参数
    llm_params["--model"] = model_info.get_model_path();
    llm_params["--parallel"] = std::to_string(model_info.get_batch_num());
    llm_params["--ctx-size"] = std::to_string(model_info.get_context_size());
    llm_params["--no-context-shift"] = "";
    llm_params["--no-warmup"] = "";

    // 第二优先级是程序入口参数
    const auto& config_service = Singleton<HLIEConfigService>::instance();
    auto args_from_main = config_service->get_args();
    for (const auto& [key, value] : args_from_main) {
        if (key == "--port" || key == "--device_id" || key == "--instance_id" ||
            key == "--json_info") { // 改为由框架控制，非llama-server参数
            continue;
        }
        llm_params[key] = value;
    }

    // 最高优先级是配置文件
    if (const auto config_reader = config_service->get_reader()) {
        // 获取llama-server同款运行参数
        const std::vector<std::string> keys = config_reader->Keys("llm_params");
        for (const auto& key : keys) {
            const std::string value = config_reader->Get("llm_params", key, "");
            llm_params[key] = value;
        }
    }

    LOG_INFO("Start to merge llama-server params");
    std::vector<char*> argv_vector;
    argv_vector.push_back(const_cast<char*>("houmo-llm"));

    for (const auto& [key, value] : llm_params) {
        LOG_INFO("param: {} = {}", key, value);
        argv_vector.push_back(const_cast<char*>(key.c_str()));
        if (!value.empty()) {
            argv_vector.push_back(const_cast<char*>(value.c_str()));
        }
    }

    // 添加终止符
    argv_vector.push_back(nullptr);

    // 转换为char**数组
    char** argv = argv_vector.data();
    const int argc = static_cast<int>(argv_vector.size() - 1); // 不包括终止符nullptr
    common_params params;
    // 解析llama-server参数，如果有外部相同配置会覆盖上面的Sampling参数
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        LOG_ERROR("Failed to merge llama-server params, try to abandon it.");
    } else {
        LOG_INFO("Successfully merged llama-server params");
    }

    if (params.model.path.empty()) {
        LOG_ERROR("LLM model path cannot be empty from args");
        return -1;
    }

    if (params.n_parallel == 1 && params.kv_unified == false && !params.has_speculative()) {
        LOG_WARN("setting n_parallel = 1 and kv_unified = true (add -kvu to disable this)");

        params.n_parallel = 1;
        params.kv_unified = true;
    }

    if (model_info.is_embedding_model()) {
        params.embedding = true;
    }

    params.model_alias = model_info.get_model_name();

    // 兼容绝对路径与相对路径
    if (HlieUtil::is_relative_path(params.model.path)) {
        params.model.path = HlieUtil::join_paths(model_dir, params.model.path);
    }

    if (model_info.get_is_multi_model() && !model_info.get_mm_proj_path().empty()) {
        params.mmproj.path = model_info.get_mm_proj_path();
        if (HlieUtil::is_relative_path(params.mmproj.path)) {
            params.mmproj.path = HlieUtil::join_paths(model_dir, params.mmproj.path);
        }
        LOG_INFO("LLM mmproj path: {}", params.mmproj.path);
    }

    auto devices = model_info.get_devices();
    if (model_info.get_device_num() == 4) { // 4 chips 目前特殊处理
        devices = "0,1,2,3";
    }

    LOG_INFO("LLM init with devices: {}", devices);

    if (const int replicas = Singleton<HLIEConfigService>::instance()->get_int("server", "replicas", 1); replicas > 1) {
        is_multi_instance_ = true;
        if (const std::optional<int> val = Singleton<HLIEDeviceHelper>::instance()->acquire_device(LLM); !val) {
            LOG_ERROR("LLM cannot acquire available device");
            return -1;
        } else {
            HlieUtil::set_env("HOUMO_DEVICE_ID", std::to_string(val.value()));
            devices_.push_back(*val);
            LOG_INFO("LLM using device: {}", val.value());
        }
    } else { // 单实例模式可以定制Model Devices
#ifndef HOUMO_XH2
        if (const std::optional<int> val = Singleton<HLIEDeviceHelper>::instance()->acquire_device(LLM); !val) {
            LOG_ERROR("LLM cannot acquire available device");
            return -1;
        } else {
            HlieUtil::set_env("HOUMO_DEVICE_ID", std::to_string(val.value()));
            devices_.push_back(*val);
            LOG_INFO("LLM using device: {}", val.value());
        }
#else
        const auto mem_est = HLIEModelInfo::estimate_mem(model_info.get_model_name());
        if (!devices.empty()) {
            if (!HlieUtil::is_valid_devices(devices)) {
                LOG_ERROR("LLM cannot acquire model specific device \"{}\"", devices);
                return -1;
            }

            if (HlieUtil::parse_device(devices, devices_) < 0) {
                LOG_ERROR("LLM parse_device specific device \"{}\" failed", devices);
                return -1;
            }

            if (device_service_->acquire_mem_on_devices(devices_, mem_est) < 0) {
                return -1;
            }

            HlieUtil::set_env("HOUMO_DEVICE_ID", devices);
            LOG_INFO("LLM acquire_mem_on_devices using model specific device: {}", devices);
        } else {
            uint32_t device;
            if (device_service_->acquire_mem(mem_est, device) < 0) {
                return -1;
            }

            devices_.push_back(device);
            HlieUtil::set_env("HOUMO_DEVICE_ID", std::to_string(device));
            LOG_INFO("LLM acquire_mem using model specific device: {}", device);
        }

#endif
    }

    if (const auto log_level = config_service->get_string("log", "level", "debug"); log_level == "debug") {
        common_log_set_verbosity_thold(-1);
    } else if (log_level == "info") {
        common_log_set_verbosity_thold(-2);
    } else if (log_level == "warn") {
        common_log_set_verbosity_thold(-3);
    } else if (log_level == "error") {
        common_log_set_verbosity_thold(-4);
    }

    if (config_service->get_bool("model", "llm_lazy_mode", false)) {
        LOG_INFO("LLAMA_ENABLE_LAZY_MODE = 1");
        HlieUtil::set_env("LLAMA_ENABLE_LAZY_MODE", "1");
    }

    common_init();

    llama_server_ = std::make_unique<server_context>();

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_server_routes_ = std::make_shared<server_routes>(params, *llama_server_, [&] { return is_ready.load(); });
    LOG_INFO("LLM start to load model");
    if (!llama_server_->load_model(params)) {
        Singleton<HLIEDeviceHelper>::instance()->release_devices(devices_);
        devices_.clear();
        return -1;
    }
    LOG_INFO("LLM load model done");

    model_key_ = model_info.model_key_;
    llama_server_->init();
    is_ready.store(true);
    LOG_DEBUG("LLM init done");
    return 0;
}

void LLModelService::run() {
    state_.store(SERVER_STATE_READY);
    LOG_INFO("LLM state: {}", state_.load());

    HLIE_CHECK_NULL(llama_server_);

    if (!info_printer_) {
        info_printer_ = std::make_shared<InfoPrinter>();
    }

    worker_thread_ = std::make_unique<std::thread>(&LLModelService::process, this);
}

void LLModelService::process() {
    llama_server_->start_loop();
}

void LLModelService::stop() {
    if (!worker_thread_) {
        return;
    }

    llama_server_->terminate();
    if (worker_thread_->joinable()) {
        worker_thread_->join();
    }
    worker_thread_.reset();
    if (is_multi_instance_) {
        Singleton<HLIEDeviceHelper>::instance()->release_devices(devices_);
    } else {
#ifdef HOUMO_XH2
        const auto mem_est = HLIEModelInfo::estimate_mem(model_key_);
        device_service_->release_mem_on_devices(devices_, mem_est);
#else
        Singleton<HLIEDeviceHelper>::instance()->release_devices(devices_);
#endif
    }
    devices_.clear();
    llama_memory_breakdown_print(llama_server_->get_llama_context());
}

int LLModelService::get_batch_num() const {
    return batch_num_;
}

server_http_res_ptr LLModelService::handle_health(const server_http_req& req) const {
    auto res = llama_server_routes_->get_health(req);
    return res;
}

server_http_res_ptr LLModelService::handle_models(const server_http_req& req) const {
    auto res = llama_server_routes_->get_models(req);
    return res;
}

server_http_res_ptr LLModelService::handle_chat_completions(const server_http_req& req) const {
    auto res = llama_server_routes_->post_chat_completions(req);
    return res;
}

server_http_res_ptr LLModelService::handle_completions(const server_http_req& req) const {
    auto res = llama_server_routes_->post_completions(req);
    return res;
}

server_http_res_ptr LLModelService::handle_completions_oai(const server_http_req& req) const {
    auto res = llama_server_routes_->post_completions_oai(req);
    return res;
}

server_http_res_ptr LLModelService::handle_embeddings_oai(const server_http_req& req) const {
    return llama_server_routes_->post_embeddings_oai(req);
}

server_http_res_ptr LLModelService::handle_embeddings(const server_http_req& req) const {
    return llama_server_routes_->post_embeddings(req);
}

server_http_res_ptr LLModelService::handle_props(const server_http_req& req) const {
    std::string name = req.get_param("model");
    if (name.empty()) {
        // main instance
        auto res = std::make_unique<server_res_generator_light>();
        res->ok({
            // TODO: add support for this on web UI
            {"role",          "router"},
            {"max_instances", 4}, // dummy value for testing
            // this is a dummy response to make sure webui doesn't break
            {"model_alias", "llama-server"},
            {"model_path",  "none"},
            {"default_generation_settings", {
                {"params", json{}},
                {"n_ctx",  0},
            }},
        });
        return res;
    }

    return llama_server_routes_->get_props(req);
}

int LLModelService::get_state() {
    return state_;
}

HLIE_NAMESPACE_END

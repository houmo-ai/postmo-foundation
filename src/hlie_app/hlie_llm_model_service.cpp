#include "hlie_app/hlie_llm_model_service.h"

#include <INIReader.h>
#include <arg.h>
#include <common.h>
#include <server/server-context.h>
#include <server/info_printer.h>

#include "hlie_app/hlie_device_service.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_rk_performance.h"
#include "hlie_base/hlie_singleton.h"
#include "hlie_base/hlie_util.h"
#include "spdlog/fmt/bundled/base.h"
#include "speculative.h"

namespace {
    void llama_log_cb(ggml_log_level level, const char * log, void * /*user_data*/) {
        spdlog::level::level_enum log_level;
        switch (level) {
        case GGML_LOG_LEVEL_INFO:
            log_level = spdlog::level::level_enum::info;
            break;
        case GGML_LOG_LEVEL_WARN:
            log_level = spdlog::level::level_enum::warn;
            break;
        case GGML_LOG_LEVEL_ERROR:
            log_level = spdlog::level::level_enum::err;
            break;
        case GGML_LOG_LEVEL_DEBUG:
        default:
            log_level = spdlog::level::level_enum::debug;
            break;
        }

        spdlog::log(log_level, log);
    }
}

HLIE_NAMESPACE_BEGIN

int LLModelService::init(const std::string& model_dir, const HLIEModelInfo& model_info, const std::shared_ptr<HLIEDeviceService>& device_service) {
    if (model_info.get_model_path().empty()) {
        LOG_ERROR("LLM model path cannot be empty");
        return -1;
    }

    device_service_ = device_service;

    const auto& config_service = Singleton<HLIEConfigService>::instance();
    HLIELlmParamSet launch_params;
    if (model_info.has_effective_llama_params()) {
        launch_params = model_info.get_effective_llama_params();
        LOG_INFO("Using resolved llama params: model={}, presets={}, hash={}",
                 model_info.model_key_, model_info.get_matched_llama_presets().size(),
                 model_info.get_effective_llama_params_hash());
    } else {
        // Compatibility fallback for direct SDK/service callers that bypass Workflow.
        launch_params.set("--parallel", static_cast<int64_t>(model_info.get_batch_num()));
        launch_params.set("--ctx-size", static_cast<int64_t>(model_info.get_context_size()));
        launch_params.set("--no-context-shift", true);
        launch_params.set("--no-warmup", true);
        for (const auto& [key, value] : config_service->get_args()) {
            if (key.rfind("--", 0) != 0 || HLIELlmParamSet::is_framework_owned(key) ||
                key == "--port" || key == "--device_id" || key == "--instance_id" ||
                key == "--json_info" || key == "--launch_manifest") {
                continue;
            }
            launch_params.set(key, value.empty() ? HLIELlmParamValue{true} : HLIELlmParamValue{value});
        }
        if (const auto config_reader = config_service->get_reader()) {
            for (const auto& key : config_reader->Keys("llm_params")) {
                if (HLIELlmParamSet::is_framework_owned(key)) {
                    LOG_WARN("Ignoring framework-owned [llm_params] option: {}", key);
                    continue;
                }
                const std::string value = config_reader->Get("llm_params", key, "");
                launch_params.set(key, value.empty() ? HLIELlmParamValue{true} : HLIELlmParamValue{value});
            }
        }
        LOG_WARN("LLM model initialized without resolved preset params; using compatibility fallback: {}",
                 model_info.model_key_);
    }
    launch_params.set("--model", model_info.get_model_path());

    LOG_INFO("Start to merge llama-server params");
    std::vector<std::string> argv_storage = launch_params.to_argv();
    std::vector<char*> argv_vector;
    argv_vector.reserve(argv_storage.size());
    for (auto& value : argv_storage) {
        LOG_INFO("llama argv: {}", value);
        argv_vector.push_back(value.data());
    }
    argv_vector.push_back(nullptr);
    char** argv = argv_vector.data();
    const int argc = static_cast<int>(argv_storage.size());
    // 解析llama-server参数，如果有外部相同配置会覆盖上面的Sampling参数
    if (!common_params_parse(argc, argv, params_, LLAMA_EXAMPLE_SERVER)) {
        LOG_ERROR("Failed to parse llama-server params. Check [llm_params] and command-line values. "
                  "For scaled LoRA adapters, use: --lora-scaled = FNAME:SCALE[,FNAME:SCALE...]");
        return -1;
    } else {
        LOG_INFO("Successfully merged llama-server params");
    }

    if (params_.model.path.empty()) {
        LOG_ERROR("LLM model path cannot be empty from args");
        return -1;
    }

    if (model_info.is_embedding_model() || model_info.is_rerank_model()) {
        params_.embedding = true;
        is_embedding_ = true;
        if (model_info.is_rerank_model()) {
            params_.pooling_type = LLAMA_POOLING_TYPE_RANK;
        }
    }

    if (params_.embedding && params_.n_batch > params_.n_ubatch) {
        LOG_WRN("%s: embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", __func__, params_.n_batch, params_.n_ubatch);
        LOG_WRN("%s: setting n_batch = n_ubatch = %d to avoid assertion failure\n", __func__, params_.n_ubatch);
        params_.n_batch = params_.n_ubatch;
    }

    if (params_.n_parallel < 0) {
        LOG_INF("%s: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n", __func__);

        params_.n_parallel = 1;
        params_.kv_unified = true;
    }

    if (model_info.is_mtp()) {
        LOG_INFO("LLM model {} enables MTP speculative decoding", model_info.get_model_name());
        params_.speculative.draft.n_max = 4;
        params_.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    }

    model_key_ = model_info.model_key_.empty() ? HLIEModelInfo::build_model_key(model_info) : model_info.model_key_;
    params_.model.name = model_info.get_model_name();

    // Keep OpenAI response "model" aligned with /v1/models id, especially for mmproj variants.
    if (!model_key_.empty()) {
        params_.model_alias.clear();
        params_.model_alias.insert(model_key_);
    }

    // 兼容绝对路径与相对路径
    if (HlieUtil::is_relative_path(params_.model.path)) {
        params_.model.path = HlieUtil::join_paths(model_dir, params_.model.path);
    }

    if (!model_info.get_mm_proj_path().empty()) {
        params_.mmproj.path = model_info.get_mm_proj_path();
        if (HlieUtil::is_relative_path(params_.mmproj.path)) {
            params_.mmproj.path = HlieUtil::join_paths(model_dir, params_.mmproj.path);
        }
        LOG_INFO("LLM mmproj path: {}", params_.mmproj.path);
    }

    auto devices = model_info.get_devices();

    LOG_INFO("LLM init with devices: {}", devices);

    {
        const auto mem_est = model_info.get_estimated_mem();
        estimated_mem_mb_ = mem_est;
        reservation_id_ = model_info.get_device_reservation_id();
        reservation_owned_externally_ = reservation_id_ != 0;
        LOG_INFO("LLM memory estimate for {} is {}MB", model_info.get_model_name(), mem_est);
        if (!devices.empty()) {
            if (!HlieUtil::is_valid_devices(devices)) {
                LOG_ERROR("LLM cannot acquire model specific device \"{}\"", devices);
                return -1;
            }

            if (HlieUtil::parse_device(devices, devices_) < 0) {
                LOG_ERROR("LLM parse_device specific device \"{}\" failed", devices);
                return -1;
            }

            if (devices_.size() != static_cast<size_t>(model_info.get_device_num())) {
                LOG_ERROR("LLM model {} requires {} devices, but {} devices were configured",
                          model_info.get_model_name(), model_info.get_device_num(), devices_.size());
                return -1;
            }

            if (reservation_id_ != 0) {
                if (!device_service_->reservation_matches(reservation_id_, devices_, mem_est, model_key_)) {
                    LOG_ERROR("LLM reservation {} does not match model/devices", reservation_id_);
                    return -1;
                }
            } else {
                const auto reservation = device_service_->reserve_mem_on_devices(
                    devices_, mem_est, false, model_key_);
                if (!reservation) {
                    return -1;
                }
                reservation_id_ = reservation->id;
            }

            HlieUtil::set_env("HOUMO_DEVICE_ID", devices);
            LOG_INFO("LLM acquire_mem_on_devices using model specific device: {}", devices);
        } else {
            const auto reservation = device_service_->reserve_mem_on_device_count(
                model_info.get_device_num(), mem_est, model_key_);
            if (!reservation) {
                return -1;
            }
            reservation_id_ = reservation->id;
            devices_ = reservation->devices;

            devices = HlieUtil::join_devices(devices_);
            HlieUtil::set_env("HOUMO_DEVICE_ID", devices);
            LOG_INFO("LLM acquired {} device(s): {}", devices_.size(), devices);
        }
    }

    if (const auto log_level = config_service->get_string("log", "level", "debug", true); log_level == "debug") {
        common_log_set_verbosity_thold(4);
    } else if (log_level == "info") {
        common_log_set_verbosity_thold(3);
    } else if (log_level == "warn") {
        common_log_set_verbosity_thold(2);
    } else if (log_level == "error") {
        common_log_set_verbosity_thold(1);
    }

    if (config_service->get_bool("model", "llm_lazy_mode", false)) {
        LOG_INFO("LLAMA_ENABLE_LAZY_MODE = 1");
        HlieUtil::set_env("LLAMA_ENABLE_LAZY_MODE", "1");
    }

    common_init();
    llama_log_set(llama_log_cb, nullptr);
    common_set_log_callback(llama_log_cb, nullptr);

    llama_server_ = std::make_unique<server_context>();

    llama_backend_init();
    llama_numa_init(params_.numa);

    llama_server_routes_ = std::make_shared<server_routes>(params_, *llama_server_);
    LOG_INFO("LLM start to load model");
    if (!llama_server_->load_model(params_)) {
        if (device_service_ && reservation_id_ != 0 &&
            !reservation_owned_externally_) {
            (void)device_service_->release_reservation(reservation_id_);
        }
        reservation_id_ = 0;
        reservation_owned_externally_ = false;
        devices_.clear();
        return -1;
    }

    if (reservation_id_ == 0 ||
        (!reservation_owned_externally_ &&
         !device_service_->commit_reservation(reservation_id_))) {
        LOG_ERROR("LLM failed to commit device reservation {}", reservation_id_);
        if (reservation_id_ != 0 && !reservation_owned_externally_) {
            (void)device_service_->release_reservation(reservation_id_);
        }
        reservation_id_ = 0;
        reservation_owned_externally_ = false;
        devices_.clear();
        return -1;
    }

    LOG_INFO("LLM load model done");

    llama_server_routes_->update_meta(*llama_server_);
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
    HLIERkPerformance::apply_for_current_thread();
    llama_server_->start_loop();
}

void LLModelService::stop() {
    if (worker_thread_) {
        if (llama_server_) {
            llama_server_->terminate();
        }
        if (worker_thread_->joinable()) {
            worker_thread_->join();
        }
        worker_thread_.reset();
    }

    if (reservation_id_ != 0 && device_service_ &&
        !reservation_owned_externally_) {
        (void)device_service_->release_reservation(reservation_id_);
    }
    reservation_id_ = 0;
    reservation_owned_externally_ = false;
    if (!devices_.empty()) {
        devices_.clear();
    }

    llama_server_routes_.reset();
    llama_server_.reset();
    is_ready.store(false);
    // if (is_embedding_) {
    //     return;
    // }
    // llama_memory_breakdown_print(llama_server_->get_llama_context());
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

server_http_res_ptr LLModelService::handle_responses(const server_http_req& req) const {
    auto res = llama_server_routes_->post_responses_oai(req);
    return res;
}

server_http_res_ptr LLModelService::handle_embeddings_oai(const server_http_req& req) const {
    return llama_server_routes_->post_embeddings_oai(req);
}

server_http_res_ptr LLModelService::handle_embeddings(const server_http_req& req) const {
    return llama_server_routes_->post_embeddings(req);
}

server_http_res_ptr LLModelService::handle_rerank(const server_http_req& req) const {
    return llama_server_routes_->post_rerank(req);
}

server_http_res_ptr LLModelService::handle_transcriptions(const server_http_req& req) const {
    auto res = llama_server_routes_->post_transcriptions_oai(req);
    return res;
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

server_http_res_ptr LLModelService::handle_slots(const server_http_req& req) const {
    return llama_server_routes_->get_slots(req);
}

int LLModelService::get_state() {
    return state_;
}

HLIE_NAMESPACE_END

#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/llama_server.hpp"

#include <unordered_set>

#include "INIReader.h"
#include "common.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_singleton.h"
#include "hlie_base/hlie_util.h"
#include "mtmd.h"
#include "spdlog/fmt/bundled/base.h"


HLIE_NAMESPACE_BEGIN

int LLModelService::init(const std::string& model_dir, const HLIEModelInfo& model_info) {
    if (model_info.get_model_path().empty()) {
        LOG_ERROR("LLM model path cannot be empty");
        return -1;
    }

    llama_server_ = std::make_unique<LlamaServer>();

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
        if (key == "--port" || key == "--device_id" || key == "--instance_id") { // 改为由框架控制，非llama-server参数
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

    // 兼容绝对路径与相对路径
    if (HlieUtil::is_relative_path(params.model.path)) {
        params.model.path = HlieUtil::join_paths(model_dir, params.model.path);
    }

    if (model_info.get_is_multi_model() && !model_info.get_mm_proj_path().empty()) {
        params.mmproj.path = model_info.get_mm_proj_path();
        if (HlieUtil::is_relative_path(params.mmproj.path)) {
            params.mmproj.path = HlieUtil::join_paths(model_dir, params.mmproj.path);
        }
    }

    // 用于models请求，返回Qwen2.5-7B/Qwen3-8B ..
    params.model_alias = model_info.get_model_type() + "-" + model_info.get_params();

    if (const auto& devices = model_info.get_devices();
        !devices.empty() && Singleton<HLIEConfigService>::instance()->get_int("server", "replicas", 1) == 1) {
        // 单实例模式可以定制Model Devices
        if (!HlieUtil::is_valid_devices(devices)) {
            LOG_ERROR("LLM cannot acquire model specific device \"{}\"", devices);
            return -1;
        }

        if (!Singleton<HLIEDeviceHelper>::instance()->acquire_devices(LLM, devices)) {
            return -1;
        }

        HlieUtil::set_env("HOUMO_DEVICE_ID", devices);
        LOG_INFO("LLM using model specific device: {}", devices);
    } else if (const std::optional<int> val = Singleton<HLIEDeviceHelper>::instance()->acquire_device(LLM); !val) {
        LOG_ERROR("LLM cannot acquire available device");
        return -1;
    } else {
        HlieUtil::set_env("HOUMO_DEVICE_ID", std::to_string(val.value()));
        LOG_INFO("LLM using device: {}", val.value());
    }

    common_log_set_verbosity_thold(-888); // 负数用于同化llama.cpp日志
    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    if (!llama_server_->load_model(params)) {
        return -1;
    }

    llama_server_->model_status = 0;
    if (llama_server_->initialize() < 0) {
        return -1;
    }

    LOG_DEBUG("LLM init done");
    return 0;
}

void LLModelService::run() {
    state_.store(SERVER_STATE_READY);
    LOG_INFO("LLM state: {}", state_.load());

    HLIE_CHECK_NULL(llama_server_);
    llama_server_->queue_tasks.on_new_task(
        [server = &*llama_server_](server_task&& task) { server->process_single_task(std::forward<server_task>(task)); });

    llama_server_->queue_tasks.on_update_slots([server = &*llama_server_] { server->update_slots(); });

    worker_thread_ = std::make_unique<std::thread>(&LLModelService::process, this);
}

void LLModelService::process() {
    llama_server_->queue_tasks.start_loop();
}

void LLModelService::stop() {
    if (!worker_thread_) {
        return;
    }

    llama_server_->queue_tasks.terminate();
    if (worker_thread_->joinable()) {
        worker_thread_->join();
    }
    worker_thread_.reset();
}


int LLModelService::get_batch_num() const {
    return batch_num_;
}

int LLModelService::handle_health(json& resp) const {
    if (llama_server_) {
        int slots_processing = 0;
        for (auto& slot : llama_server_->slots) {
            if (slot.is_processing()) {
                ++slots_processing;
            }
        }

        const auto slots_idle = llama_server_->slots.size() - slots_processing;
        resp["slots_idle"] = slots_idle;
        resp["slots_processing"] = slots_processing;
    }

    return 0;
}

void LLModelService::handle_models(json& resp) const {
    const auto current_state = static_cast<server_state>(state_.load());
    json model_meta = nullptr;
    if (current_state == SERVER_STATE_READY) {
        model_meta = llama_server_->model_meta();
    }

    const auto params = llama_server_->params_base;

    resp["models"] = {{{"name", params.model_alias.empty() ? params.model.path : params.model_alias},
                       {"model", params.model_alias.empty() ? params.model.path : params.model_alias},
                       {"modified_at", ""},
                       {"size", ""},
                       {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                       {"type", "model"},
                       {"description", ""},
                       {"tags", {""}},
                       {"capabilities", {"completion"}},
                       {"parameters", ""},
                       {"details",
                        {{"parent_model", ""},
                         {"format", "gguf"},
                         {"family", ""},
                         {"families", {""}},
                         {"parameter_size", ""},
                         {"quantization_level", ""}}}}};

    resp["object"] = "list";
    resp["data"] = {{
        {"id", params.model_alias.empty() ? params.model.path : params.model_alias},
        {"object", "model"},
        {"created", std::time(0)},
        {"owned_by", "HLIECpp"},
        {"meta", model_meta},
    }};
}

void LLModelService::handle_chat_completions(const httplib::Request& req, httplib::Response& res) const {
    auto body = json::parse(req.body);
    std::vector<raw_buffer> files;
    json data = oaicompat_chat_params_parse(body, llama_server_->oai_parser_opt, files);

    handle_completions_impl(SERVER_TASK_TYPE_COMPLETION, data, files, req.is_connection_closed, res, OAICOMPAT_TYPE_CHAT);
}

void LLModelService::handle_completions(const httplib::Request& req, httplib::Response& res) const {
    json data = json::parse(req.body);
    std::vector<raw_buffer> files; // dummy
    handle_completions_impl(SERVER_TASK_TYPE_COMPLETION, data, files, req.is_connection_closed, res, OAICOMPAT_TYPE_NONE);
}

void LLModelService::handle_completions_oai(const httplib::Request& req, httplib::Response& res) const {
    json data = oaicompat_completion_params_parse(json::parse(req.body));
    std::vector<raw_buffer> files; // dummy
    handle_completions_impl(SERVER_TASK_TYPE_COMPLETION, data, files, req.is_connection_closed, res, OAICOMPAT_TYPE_COMPLETION);
}

void LLModelService::handle_completions_impl(server_task_type type, json& data, const std::vector<raw_buffer>& files,
                                             const std::function<bool()>& is_connection_closed, httplib::Response& res,
                                             oaicompat_type oaicompat) const {

    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto completion_id = gen_chatcmplid();
    std::unordered_set<int> task_ids;
    try {
        std::vector<server_task> tasks;

        const auto& prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        // SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process files
        mtmd::bitmaps bitmaps;
        const bool has_mtmd = llama_server_->mctx != nullptr;
        {
            if (!has_mtmd && !files.empty()) {
                throw std::runtime_error("This server does not support multimodal");
            }
            for (auto& file : files) {
                mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(llama_server_->mctx, file.data(), file.size()));
                if (!bmp.ptr) {
                    throw std::runtime_error("Failed to load image or audio file");
                }
                // calculate bitmap hash (for KV caching)
                std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
                bmp.set_id(hash.c_str());
                bitmaps.entries.push_back(std::move(bmp));
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;
        if (oaicompat && !prompt.is_string()) {
            throw std::runtime_error("prompt must be a string");
        }

        if (oaicompat && has_mtmd) {
            // multimodal
            std::string prompt_str = prompt.get<std::string>();
            mtmd_input_text inp_txt = {
                prompt_str.c_str(),
                /* add_special */ true,
                /* parse_special */ true,
            };
            mtmd::input_chunks chunks(mtmd_input_chunks_init());
            auto bitmaps_c_ptr = bitmaps.c_ptr();
            int32_t tokenized = mtmd_tokenize(llama_server_->mctx, chunks.ptr.get(), &inp_txt, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
            if (tokenized != 0) {
                throw std::runtime_error("Failed to tokenize prompt");
            }

            server_tokens tmp(chunks, true);
            inputs.push_back(std::move(tmp));
        } else {
            // non-multimodal version
            auto tokenized_prompts = tokenize_input_prompts(llama_server_->vocab, prompt, true, true);
            for (auto& p : tokenized_prompts) {
                auto tmp = server_tokens(p, llama_server_->mctx != nullptr);
                inputs.push_back(std::move(tmp));
            }
        }

        tasks.reserve(inputs.size());

        const std::string request_id = res.has_header("X-Request-ID") ? res.get_header_value("X-Request-ID") : "null";

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = llama_server_->queue_tasks.get_new_id();
            task.index = i;

            task.http_request_id = request_id;

            task.prompt_tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(llama_server_->ctx, llama_server_->params_base, data);
            task.id_selected_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.oaicompat = oaicompat;
            task.params.oaicompat_cmpl_id = completion_id;
            // oaicompat_model is already populated by params_from_json_cmpl

            tasks.push_back(std::move(task));
        }

        task_ids = server_task::get_list_id(tasks);
        llama_server_->queue_results.add_waiting_tasks(tasks);
        llama_server_->queue_tasks.post(std::move(tasks));
    } catch (const std::exception& e) {
        res_error(res, format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    if (bool stream = json_value(data, "stream", false); !stream) {
        llama_server_->receive_multi_results(
            task_ids,
            [&](const std::vector<server_task_result_ptr>& results) {
                try {
                    if (results.size() == 1) {
                        // single result
                        res_ok(res, results[0]->to_json());
                    } else {
                        // multiple results (multitask)
                        json arr = json::array();
                        for (auto& res_data : results) {
                            arr.push_back(res_data->to_json());
                        }
                        res_ok(res, arr);
                    }
                } catch (const nlohmann::json::type_error& ex) {
                    HLIELogger::get_logger()->error("Non-Stream Mode JSON Exception: {}", ex.what());
                    res_error(res, format_error_response(ex.what(), ERROR_TYPE_SERVER));
                } catch (const std::exception& ex) {
                    HLIELogger::get_logger()->error("Non-Stream Mode Exception: {}", ex.what());
                    res_error(res, format_error_response(ex.what(), ERROR_TYPE_SERVER));
                } catch (...) {
                    HLIELogger::get_logger()->error("Non-Stream Mode Unexpected Exception");
                    res_error(res, format_error_response("Stream Model Exception", ERROR_TYPE_SERVER));
                }
            },
            [&](const json& error_data) { res_error(res, error_data); }, is_connection_closed);

        llama_server_->queue_results.remove_waiting_task_ids(task_ids);
    } else {
        const auto chunked_content_provider = [task_ids, oaicompat, this](size_t, httplib::DataSink& sink) {
            llama_server_->receive_cmpl_results_stream(
                task_ids,
                [&](const server_task_result_ptr& result) -> bool {
                    try {
                        json res_json = result->to_json();
                        if (res_json.is_array()) {
                            for (const auto& res_data : res_json) {
                                if (!server_sent_event(sink, "data", res_data)) {
                                    // sending failed (HTTP connection closed), cancel the generation
                                    return false;
                                }
                            }
                            return true;
                        }

                        return server_sent_event(sink, "data", res_json);
                    } catch (const nlohmann::json::type_error& ex) {
                        HLIELogger::get_logger()->error("Stream Mode JSON Exception: {}", ex.what());
                        const auto error_data = format_error_response(ex.what(), ERROR_TYPE_SERVER);
                        return server_sent_event(sink, "error", error_data);
                    } catch (const std::exception& ex) {
                        HLIELogger::get_logger()->error("Stream Mode Exception: {}", ex.what());
                        const auto error_data = format_error_response(ex.what(), ERROR_TYPE_SERVER);
                        return server_sent_event(sink, "error", error_data);
                    } catch (...) {
                        HLIELogger::get_logger()->error("Stream Mode Unexpected Exception");
                        const auto error_data = format_error_response("Unknown error", ERROR_TYPE_SERVER);
                        return server_sent_event(sink, "error", error_data);
                    }
                },
                [&](const json& error_data) { server_sent_event(sink, "error", error_data); },
                [&sink]() {
                    // note: do not use req.is_connection_closed here because req is already destroyed
                    return !sink.is_writable();
                });
            if (oaicompat != OAICOMPAT_TYPE_NONE) {
                static const std::string ev_done = "data: [DONE]\n\n";
                sink.write(ev_done.data(), ev_done.size());
            }
            sink.done();
            return false;
        };

        auto on_complete = [task_ids, this](bool) { llama_server_->queue_results.remove_waiting_task_ids(task_ids); };

        res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
    }
}

int LLModelService::get_state() {
    return state_;
}

HLIE_NAMESPACE_END

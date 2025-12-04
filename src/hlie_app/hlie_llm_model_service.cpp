#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/info_printer.h"
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
        if (key == "--port" || key == "--device_id" || key == "--instance_id" || key == "--json_info") { // 改为由框架控制，非llama-server参数
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

    // 用于models请求，返回Qwen2.5-7B/Qwen3-8B ..
    params.model_alias = HlieUtil::extract_filename(params.model.path);
    if (!HlieUtil::has_model_params(params.model_alias)) {
        params.model_alias += "-" + model_info.get_params();
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

    if (const auto& devices = model_info.get_devices();
        !devices.empty() && Singleton<HLIEConfigService>::instance()->get_int("server", "replicas", 1) == 1) {
        // 单实例模式可以定制Model Devices
        if (!HlieUtil::is_valid_devices(devices)) {
            LOG_ERROR("LLM cannot acquire model specific device \"{}\"", devices);
            return -1;
        }

        if (!Singleton<HLIEDeviceHelper>::instance()->acquire_devices(LLM, devices, devices_)) {
            return -1;
        }

        HlieUtil::set_env("HOUMO_DEVICE_ID", devices);
        LOG_INFO("LLM using model specific device: {}", devices);
    } else if (const std::optional<int> val = Singleton<HLIEDeviceHelper>::instance()->acquire_device(LLM); !val) {
        LOG_ERROR("LLM cannot acquire available device");
        return -1;
    } else {
        HlieUtil::set_env("HOUMO_DEVICE_ID", std::to_string(val.value()));
        devices_.push_back(*val);
        LOG_INFO("LLM using device: {}", val.value());
    }

    common_log_set_verbosity_thold(-888); // 负数用于同化llama.cpp日志
    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    if (!llama_server_->load_model(params)) {
        Singleton<HLIEDeviceHelper>::instance()->release_devices(devices_);
        devices_.clear();
        return -1;
    }

    llama_server_->model_status = 0;
    if (llama_server_->initialize() < 0) {
        Singleton<HLIEDeviceHelper>::instance()->release_devices(devices_);
        devices_.clear();
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

    if (!info_printer_) {
        info_printer_ = std::make_shared<InfoPrinter>();
    }

    auto model_meta = llama_server_->model_meta();
    info_printer_->print_model_info(model_meta);
    info_printer_->print_process_memory_info();

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
    Singleton<HLIEDeviceHelper>::instance()->release_devices(devices_);
    devices_.clear();
    if (llama_server_) {
        llama_memory_breakdown_print(llama_server_->ctx);
    }
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

    bool has_mtmd = llama_server_->mctx != nullptr;
    const auto params = llama_server_->params_base;

    resp["models"] = {{{"name", params.model_alias.empty() ? params.model.path : params.model_alias},
                       {"model", params.model_alias.empty() ? params.model.path : params.model_alias},
                       {"modified_at", ""},
                       {"size", ""},
                       {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                       {"type", "model"},
                       {"description", ""},
                       {"tags", {""}},
                       {"capabilities", has_mtmd ? json({"completion", "multimodal"}) : json({"completion"})},
                       {"parameters", ""},
                       {"details",
                        {{"parent_model", ""},
                         {"format", "gguf"},
                         {"family", ""},
                         {"families", {""}},
                         {"parameter_size", ""},
                         {"quantization_level", ""}}}}};

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
    // need to store the reader as a pointer, so that it won't be destroyed when the handle returns
    // use shared_ptr as it's shared between the chunked_content_provider() and on_complete()
    const auto rd = std::make_shared<server_response_reader>(llama_server_.get());

    try {
        std::vector<server_task> tasks;

        const auto& prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        // SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process prompt
        std::vector<server_tokens> inputs;

        if (oaicompat && llama_server_->mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(llama_server_->mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(llama_server_->vocab, llama_server_->mctx, prompt, true, true);
        }

        tasks.reserve(inputs.size());

        const std::string request_id = res.has_header("X-Request-ID") ? res.get_header_value("X-Request-ID") : "null";

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = llama_server_->queue_tasks.get_new_id();
            task.index = i;

            task.http_request_id = request_id;

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(llama_server_->ctx, llama_server_->params_base, data);
            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.oaicompat = oaicompat;
            task.params.oaicompat_cmpl_id = completion_id;
            // oaicompat_model is already populated by params_from_json_cmpl

            tasks.push_back(std::move(task));
        }

        rd->post_tasks(std::move(tasks));
    } catch (const std::exception& e) {
        res_error(res, format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    if (const bool stream = json_value(data, "stream", false); !stream) {
        // non-stream, wait for the results
        try {
            auto all_results = rd->wait_for_all(is_connection_closed);
            if (all_results.is_terminated) {
                return; // connection is closed
            }

            if (all_results.error) {
                res_error(res, all_results.error->to_json());
                return;
            }

            json arr = json::array();
            for (auto& res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            // if single request, return single object instead of array
            res_ok(res, arr.size() == 1 ? arr[0] : arr);
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
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        server_task_result_ptr first_result = rd->next(is_connection_closed);
        if (first_result == nullptr) {
            return; // connection is closed
        }

        if (first_result->is_error()) {
            res_error(res, first_result->to_json());
            return;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
                    dynamic_cast<server_task_result_cmpl_final*>(first_result.get()) != nullptr);


        // next responses are streamed
        json first_result_json = first_result->to_json();
        const auto chunked_content_provider = [first_result_json, rd, oaicompat](size_t, httplib::DataSink& sink) mutable -> bool {
            // flush the first result as it's not an error
            if (!first_result_json.empty()) {
                if (!server_sent_event(sink, first_result_json)) {
                    sink.done();
                    return false; // sending failed, go to on_complete()
                }
                first_result_json.clear(); // mark as sent
            }

            // receive subsequent results
            const auto result = rd->next([&sink] { return !sink.is_writable(); });
            if (result == nullptr) {
                sink.done();
                return false; // connection is closed, go to on_complete()
            }

            json res_json;
            bool ok = false;
            try {
                // send the results
                res_json = result->to_json();
                if (result->is_error()) {
                    ok = server_sent_event(sink, json{{"error", result->to_json()}});
                    sink.done();
                    return false; // go to on_complete()
                }
            } catch (const nlohmann::json::type_error& ex) {
                HLIELogger::get_logger()->error("Stream Mode JSON Exception: {}", ex.what());
                const auto error_data = format_error_response(ex.what(), ERROR_TYPE_SERVER);
                return server_sent_event(sink, error_data);
            } catch (const std::exception& ex) {
                HLIELogger::get_logger()->error("Stream Mode Exception: {}", ex.what());
                const auto error_data = format_error_response(ex.what(), ERROR_TYPE_SERVER);
                return server_sent_event(sink, error_data);
            } catch (...) {
                HLIELogger::get_logger()->error("Stream Mode Unexpected Exception");
                const auto error_data = format_error_response("Unknown error", ERROR_TYPE_SERVER);
                return server_sent_event(sink, error_data);
            }

            GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr ||
                        dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr);
            ok = server_sent_event(sink, res_json);

            if (!ok) {
                sink.done();
                return false; // sending failed, go to on_complete()
            }

            // check if there is more data
            if (!rd->has_next()) {
                if (oaicompat != OAICOMPAT_TYPE_NONE) {
                    static const std::string ev_done = "data: [DONE]\n\n";
                    sink.write(ev_done.data(), ev_done.size());
                }
                sink.done();
                return false; // no more data, go to on_complete()
            }

            // has next data, continue
            return true;
        };

        auto on_complete = [rd](bool) { rd->stop(); };

        res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
    }
}

void LLModelService::handle_embeddings(const httplib::Request& req, httplib::Response& res, oaicompat_type oaicompat) const {
    if (!llama_server_->params_base.embedding) {
        res_error(res,
                  format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return;
    }

    if (oaicompat != OAICOMPAT_TYPE_NONE && llama_pooling_type(llama_server_->ctx) == LLAMA_POOLING_TYPE_NONE) {
        res_error(res,
                  format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type",
                                        ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        oaicompat = OAICOMPAT_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res_error(res, format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string& format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res_error(
                res,
                format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(llama_server_->vocab, llama_server_->mctx, prompt, true, true);
    for (const auto& tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res_error(res, format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
    }

    int embd_normalize = 2; // default to Euclidean/L2 norm
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (llama_pooling_type(llama_server_->ctx) == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", llama_pooling_type(llama_server_->ctx));
        }
    }

    // create and queue the task
    json responses = json::array();
    server_response_reader rd(llama_server_.get());
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = llama_server_->queue_tasks.get_new_id();
            task.index  = i;
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.oaicompat = oaicompat;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.is_connection_closed);

    // collect results
    if (all_results.is_terminated) {
        return; // connection is closed
    }

    if (all_results.error) {
        res_error(res, all_results.error->to_json());
        return;
    }

    for (auto& res : all_results.results) {
        GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
        responses.push_back(res->to_json());
    }

    // write JSON response
    json root = oaicompat == OAICOMPAT_TYPE_EMBEDDING ? format_embeddings_response_oaicompat(body, responses, use_base64) : json(responses);
    res_ok(res, root);
}

int LLModelService::get_state() {
    return state_;
}

HLIE_NAMESPACE_END

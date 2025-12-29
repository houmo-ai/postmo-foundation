#include "hlie_app/hlie_workflow.h"

#include <filesystem>
#include <server/server-http.h>

#include "INIReader.h"
#include "hlie_app/hlie_infer_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_process_manager.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_exception.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

#ifdef HOUMO_XH2
#include "hlie_app/hlie_device_service.h"
#include "hlie_app/hlie_sd_model_service.h"
#endif

HLIE_NAMESPACE_BEGIN

#define CHECK_MODEL_READY(service)                                                                                                         \
    do {                                                                                                                                   \
        auto res = std::make_unique<server_res_generator>();                                                                               \
        if (!(service) || (service)->get_state() != SERVER_STATE_READY) {                                                                  \
            res->error(format_error_response("Model is not available for now.", ERROR_TYPE_UNAVAILABLE));                                  \
            return res;                                                                                                                    \
        }                                                                                                                                  \
    } while (0)

namespace {
    std::vector<HLIEModelInfo> load_model_info_files_from_directory(const std::string& directory_path) {
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

                    if (HLIEModelInfo model_info; model_info.parse_from_file(file_path) == 0) {
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

    HLIEModelInfo parse_and_build_sd_model(const json& params, const std::unique_ptr<server_res_generator>& res,
                                           const std::string& model_type, const std::string& devices) {
        // SD 模型必须字段检查
        const char* required_fields[] = {"vae_model", "t5_model", "clip_model", "clip_l_model", "mmdit_model"};

        for (const auto& field : required_fields) {
            if (!params.contains(field)) {
                const json resp = {{"ret", 80007}, {"msg", "params invalid, miss someone"}};
                res->error(resp);
                return {};
            }
        }

        // 解析参数
        const auto vae_model = params["vae_model"].get<std::string>();
        const auto t5_model = params["t5_model"].get<std::string>();
        const auto clip_model = params["clip_model"].get<std::string>();
        const auto clip_l_model = params["clip_l_model"].get<std::string>();
        const auto mmdit_model = params["mmdit_model"].get<std::string>();

        // 构建 HLIEModelInfo
        HLIEModelInfo model_info;
        model_info.build_sd(model_type, vae_model, t5_model, clip_model, clip_l_model, mmdit_model, devices);

        return model_info;
    }

    std::string generate_meta_info(const HLIEModelInfo& model_info, const std::string& model_dir,
                                   const std::unique_ptr<server_res_generator>& res) {
        using namespace std::chrono;
        const auto now = system_clock::now();
        const auto t = system_clock::to_time_t(now);

        std::ostringstream oss;
        oss << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
        const auto meta_info_file_name = "model_info" + oss.str() + ".hlif";
        const auto meta_cache_dir = HlieUtil::join_paths(model_dir, "meta_cache");
        std::string json_info_file;
        try {
            if (std::filesystem::create_directories(meta_cache_dir)) {
                LOG_INF("MetaCache dir created successfully.");
            }
            json_info_file = HlieUtil::join_paths(meta_cache_dir, meta_info_file_name);
        } catch (const std::filesystem::filesystem_error& ex) {
            LOG_ERROR("MetaCache Dir created failed, exception: {}", ex.what());
            json_info_file = HlieUtil::join_paths(model_dir, meta_info_file_name);
        }
        if (model_info.to_json_file(json_info_file) < 0) {
            LOG_ERROR("failed to write model info to json file");
            const json resp = {{"ret", 80007}, "failed to write model info to json file"};
            res->error(resp);
            return "";
        }

        return json_info_file;
    }

    void update_status(const common_params& base_params, const int router_port, const std::string& name, const int port) {
        // send a notification to the router server that a model instance is ready
        httplib::Client cli(base_params.hostname, router_port);
        cli.set_connection_timeout(0, 200000); // 200 milliseconds

        httplib::Request req;
        req.method = "POST";
        req.path = "/models/status";
        req.set_header("Content-Type", "application/json");
        if (!base_params.api_keys.empty()) {
            req.set_header("Authorization", "Bearer " + base_params.api_keys[0]);
        }

        json body;
        body["model"] = name;
        body["port"] = port;
        body["value"] = server_model_status_to_string(SERVER_MODEL_STATUS_LOADED);
        req.body = body.dump();

        SRV_INF("notifying router server (port=%d) that model %s is ready\n", router_port, name.c_str());
        if (const auto result = cli.send(req); result.error() != httplib::Error::Success) {
            const auto err_str = httplib::to_string(result.error());
            SRV_ERR("failed to notify router server: %s\n", err_str.c_str());
            exit(1); // force exit
        }
    }
} // namespace

int HLIEWorkflow::initialize(const std::string& model_dir, const std::string& program_path) {
    model_dir_ = model_dir;
    program_path_ = program_path;
    if (model_dir.empty()) {
        LOG_ERROR("model dir is empty");
        return -1;
    }

    const auto& args = Singleton<HLIEConfigService>::instance()->get_args();
    std::vector<HLIEModelInfo> model_infos;
    if (args.find("--json_info") != args.end()) { // 是否为外部接口调用启动服务
        std::string json_path = args.at("--json_info");
        if (HLIEModelInfo model_info; model_info.parse_from_file(args.at("--json_info")) == 0) {
            model_infos.emplace_back(model_info);
            LOG_INFO("Successfully parsed model info from --json_info: {}", json_path);
            is_slave_ = true;
            LOG_INFO("Initializing LLModelService");
            if (model_info.is_llm_) {
                llm_model_service_ = std::make_shared<LLModelService>();
                if (llm_model_service_->init(model_dir, model_info) < 0) {
                    LOG_ERROR("failed to init the LLM model service");
                    return -1;
                }
            }
#ifdef HOUMO_XH2
            else {
                sd_model_service_ = std::make_shared<SDModelService>();
                if (sd_model_service_->init(model_dir, model_info) < 0) {
                    LOG_ERROR("failed to init the SD model service");
                    return -1;
                }
            }

            device_service_ = std::make_shared<HLIEDeviceService>();
#endif
            Singleton<HLIEDeviceHelper>::instance()->print_status();
            start_time_ = time(nullptr);
            return 0;
        }

        LOG_ERROR("Failed to parse model info from --json_info: {}", json_path);
        return -1;
    }

    // 主进程或者是负载均衡模式
    model_infos = load_model_info_files_from_directory(model_dir);
    HLIEModelInfo::ScanLocalModelsFromDir(model_infos, model_dir_);
    for (const auto& model_info : model_infos) {
        ModelInstance model_instance;
        server_model_meta meta;
        meta.name = model_info.get_model_name();
        meta.path = model_info.get_model_path();
        meta.path_mmproj = model_info.get_mm_proj_path();
        model_instance.meta = std::move(meta);
        model_instance.model_info = model_info;
        model_instances_[model_info.get_model_name()] = std::move(model_instance);
    }

    if (model_infos.empty()) {
        LOG_WARN("No valid model info files found in directory: {}", model_dir);
        if (args.find("--instance_id") == args.end()) {
            // Master Process Start in Zero-Load Mode
#ifdef HOUMO_XH2
            device_service_ = std::make_shared<HLIEDeviceService>();
#endif
            Singleton<HLIEDeviceHelper>::instance()->print_status();
            start_time_ = time(nullptr);
            return 8;
        }

        return -1;
    }

    const std::string default_model = Singleton<HLIEConfigService>::instance()->get_string("model", "default");
    const bool enable_sd3 = Singleton<HLIEConfigService>::instance()->get_bool("model", "enable_sd3", false);
    if (default_model.empty() && !enable_sd3) {
        LOG_WARN("No default model: {}", default_model);
        if (args.find("--instance_id") == args.end()) {
            // Master Process Start in Zero-Load Mode
#ifdef HOUMO_XH2
            device_service_ = std::make_shared<HLIEDeviceService>();
#endif
            Singleton<HLIEDeviceHelper>::instance()->print_status();
            start_time_ = time(nullptr);
            return 8;
        }

        return -1;
    }

    bool found_llm = false;
    bool found_sd3 = false;
    HLIEModelInfo default_model_info;
    HLIEModelInfo sd3_model_info;
    if (!default_model.empty()) {
        for (const auto& m : model_infos) {
            LOG_INFO("Scanned model: {}", m.get_model_name());
            if (m.get_model_name() == default_model) {
                LOG_INFO("default_model found: {}", default_model);
                found_llm = true;
                default_model_info = m;
            }
#ifdef HOUMO_XH2
            else if (enable_sd3 && m.get_model_name() == HLIEModelType::SD3) {
                LOG_INFO("SD3 found");
                found_sd3 = true;
                sd3_model_info = m;
            }
#endif
        }

        if (!found_llm) {
            LOG_ERROR("Default llm not found: {}, scanned models count: {}", default_model, model_infos.size());
        }
    }
#ifdef HOUMO_XH2
    else if (enable_sd3) {
        auto it = std::find_if(model_infos.begin(), model_infos.end(),
                               [&](const HLIEModelInfo& info) { return info.get_model_name() == HLIEModelType::SD3; });
        if (it == model_infos.end()) {
            LOG_WARN("No SD3 model found");
            return -1;
        }

        found_sd3 = true;
        sd3_model_info = *it;
    }
#endif

    if (!found_llm && !found_sd3) {
        if (args.find("--instance_id") == args.end()) {
            // Master Process Start in Zero-Load Mode
#ifdef HOUMO_XH2
            device_service_ = std::make_shared<HLIEDeviceService>();
#endif
            Singleton<HLIEDeviceHelper>::instance()->print_status();
            start_time_ = time(nullptr);
            return 8;
        }

        LOG_WARN("No default models found for slave instance");
        return -1;
    }

    if (found_llm) {
        LOG_INFO("Initializing LLModelService");
        llm_model_service_ = std::make_shared<LLModelService>();
        if (llm_model_service_->init(model_dir, default_model_info) < 0) {
            LOG_ERROR("failed to init the LLM model service");
            return -1;
        }
    }

    if (found_sd3) {
#ifdef HOUMO_XH2
        LOG_INFO("Initializing SDModelService");
        sd_model_service_ = std::make_shared<SDModelService>();
        if (sd_model_service_->init(model_dir, sd3_model_info) < 0) {
            LOG_ERROR("failed to init the SD model service");
            return -1;
        }
#else
        LOG_ERROR("failed to init the SD model service, ONLY for XH2");
        return -1;
#endif
    }

#ifdef HOUMO_XH2
    device_service_ = std::make_shared<HLIEDeviceService>();
#endif

    Singleton<HLIEDeviceHelper>::instance()->print_status();
    start_time_ = time(nullptr);
    return 0;
}

void HLIEWorkflow::run() {
    if (llm_model_service_) {
        llm_model_service_->run();

        model_instances_[llm_model_service_->get_model_name()].meta.status = SERVER_MODEL_STATUS_LOADED;

        if (is_slave_) {
            common_params params;
            params.hostname = "127.0.0.1";
            const int master_port = Singleton<HLIEConfigService>::instance()->get_int("server", "port", 7901, true);
            std::thread([=] {
                std::this_thread::sleep_for(std::chrono::minutes(1));
                update_status(params, master_port, llm_model_service_->get_model_name(), port_);
            }).detach();
        }
    }

#ifdef HOUMO_XH2
    if (sd_model_service_) {
        sd_model_service_->run();
        model_instances_[HLIEModelType::SD3].meta.status = SERVER_MODEL_STATUS_LOADED;
        if (is_slave_) {
            common_params params;
            params.hostname = "127.0.0.1";
            const int master_port = Singleton<HLIEConfigService>::instance()->get_int("server", "port", 7901, true);
            std::thread([=] {
                std::this_thread::sleep_for(std::chrono::minutes(1));
                update_status(params, master_port, HLIEModelType::SD3, port_);
            }).detach();
        }
    }
#endif
}

void HLIEWorkflow::stop() const {
#ifdef HOUMO_XH2
    if (sd_model_service_) {
        sd_model_service_->stop();
    }
#endif

    if (llm_model_service_) {
        llm_model_service_->stop();
    }
}

std::shared_ptr<IHLIEModelService> HLIEWorkflow::get_model_service(const std::string& type) const {
    if (HLIEModelType::LLM == type) {
        return llm_model_service_;
    }

#ifdef HOUMO_XH2
    if (HLIEModelType::SD == type) {
        return sd_model_service_;
    }
#endif

    return nullptr;
}

int HLIEWorkflow::cacl_thread_num() const {
    if (llm_model_service_) {
        const auto batch_num = llm_model_service_->get_batch_num();
        return std::max(batch_num, static_cast<int32_t>(std::thread::hardware_concurrency()) - 1);
    }

    return static_cast<int32_t>(std::thread::hardware_concurrency()) - 1;
}

int HLIEWorkflow::get_model_service_state() const {
    if (llm_model_service_) {
        return llm_model_service_->get_state();
    }

#ifdef HOUMO_XH2
    if (sd_model_service_) {
        return sd_model_service_->get_state();
    }
#endif

    LOG_WARN("model service not initialized");
    return -1;
}

std::vector<std::string> HLIEWorkflow::get_api_keys() const {
    return api_keys_;
}

server_http_res_ptr HLIEWorkflow::handle_health(const server_http_req& req) const {
    json resp = {
        {"status", "ok"},
    };

    server_http_res_ptr res = nullptr;
    if (llm_model_service_) {
        res = llm_model_service_->handle_health(req);
    }

#ifdef HOUMO_XH2
    if (sd_model_service_) {
        if (sd_model_service_->handle_health(resp) < 0) {
            throw CommonException(4, "sd model service health check failed");
        }
    }
#endif

    return res;
}

server_http_res_ptr HLIEWorkflow::handle_models(const server_http_req& req) {
    auto res = std::make_unique<server_res_generator>();
    json models_json = json::array();
    const auto all_models = get_all_model_metas();
    std::time_t t = std::time(nullptr);
    for (const auto& meta : all_models) {
        json status{
            {"value", server_model_status_to_string(meta.status)},
            {"args", meta.args},
        };
        if (meta.is_failed()) {
            status["exit_code"] = meta.exit_code;
            status["failed"] = true;
        }
        models_json.push_back(json{
            {"id", meta.name},
            {"object", "model"}, // for OAI-compat
            {"owned_by", "HLIECpp"}, // for OAI-compat
            {"created", t}, // for OAI-compat
            {"in_cache", meta.in_cache},
            {"path", meta.path},
            {"port", meta.port},
            {"status", status},
            // TODO: add other fields, may require reading GGUF metadata
        });
    }
    res->ok({
        {"data", models_json},
        {"object", "list"},
    });
    return res;
}

server_http_res_ptr HLIEWorkflow::start_service(const server_http_req& req) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    auto res = std::make_unique<server_res_generator>();
    if (config_services->get_int("server", "replicas", 1) > 1) {
        // 集群|多实例模式下，不允许外部调用启动模型实例
        res->error(
            format_error_response_ex("In the cluster or multi-instance mode, external calls to start model instances are not allowed.",
                                     ERROR_TYPE_NOT_SUPPORTED, 80005));
        return res;
    }

    auto body = json::parse(req.body);
    std::string model_type = body.value("model_type", LLM);
    std::string devices = body.value("devices", "undefined");
    std::string model_name = body.value("model", std::string());
    std::string model_path = body.value("model_path", std::string());
    LOG_DEBUG("start_service devices: {}, model_type: {}, model_name: {}", devices, model_type, model_name);

    if (model_type != LLM && model_type != SD) {
        res->error(format_error_response_ex("Invalid model type.", ERROR_TYPE_INVALID_REQUEST, 80006));
        return res;
    }

    std::optional<server_model_meta> meta = std::nullopt;
    if (!model_name.empty()) { // 兼容LLAMA新方式， 检查模型是否存在
        meta = get_model_meta(model_name);
        if (!meta.has_value()) {
            res->error(format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }

        if (meta->status == SERVER_MODEL_STATUS_LOADED) {
            res->error(format_error_response("model is already loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    } else if (model_path.empty()) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return res;
    }

    LOG_INFO("ready to acquire devices");
    {
        std::unique_lock lock(mutex_);
        if (devices == "undefined") {
            const std::optional<int> val = Singleton<HLIEDeviceHelper>::instance()->acquire_device(HlieUtil::uuid(), true);
            if (!val) {
                res->error(format_error_response_ex("The devices is not available.", ERROR_TYPE_UNAVAILABLE, 80004));
                return res;
            }
            devices = std::to_string(*val);
        } else if (std::vector<int> devices_vec;
                   !Singleton<HLIEDeviceHelper>::instance()->acquire_devices(HlieUtil::uuid(), devices, devices_vec, true)) {
            res->error(format_error_response_ex("The devices is not available.", ERROR_TYPE_UNAVAILABLE, 80004));
            return res;
        }
    }

    LOG_INFO("acquire devices successfully with {}", devices);

    auto instance_id = 0;
    std::string json_info_file;
    ProcessID process_id = 0;
    if (model_type == LLM) {
        if (!meta) {
            std::filesystem::path model_path_tmp(model_path);
            if (std::error_code ec; !std::filesystem::exists(model_path_tmp, ec) || !std::filesystem::is_regular_file(model_path_tmp, ec)) {
                res->error(format_error_response_ex("model_path file not exist.", ERROR_TYPE_INVALID_REQUEST, 80006));
                return res;
            }

            model_name = model_path_tmp.filename().stem().string();
            meta = get_model_meta(model_name);

            if (!meta) {
                HLIEModelInfo model_info_tmp;
                model_info_tmp.parse_from_gguf_filename(model_path);
                if (body.contains("params") && body["params"].is_object()) {
                    auto& params = body["params"];
                    if (std::string mm_proj_path = params.value("mm_proj_path", std::string{}); !mm_proj_path.empty()) {
                        model_info_tmp.is_multi_model_ = true;
                        model_info_tmp.mm_proj_path_ = mm_proj_path;
                        if (std::error_code ec;
                            !std::filesystem::exists(mm_proj_path, ec) || !std::filesystem::is_regular_file(mm_proj_path, ec)) {
                            res->error(format_error_response_ex("mm_proj_path file not exist.", ERROR_TYPE_INVALID_REQUEST, 80006));
                            return res;
                        }
                    }
                }

                ModelInstance model_instance;
                server_model_meta model_meta;
                model_meta.name = model_info_tmp.get_model_name();
                model_meta.path = model_info_tmp.get_model_path();
                model_meta.path_mmproj = model_info_tmp.get_mm_proj_path();
                model_instance.meta = std::move(model_meta);
                model_instance.model_info = model_info_tmp;
                model_instances_[model_name] = std::move(model_instance);
            }
        }

        const auto& model_info = model_instances_.at(model_name).model_info;
        if (model_info.model_version_.empty()) {
            res->error(format_error_response_ex("failed to parse build_version from model name", ERROR_TYPE_INVALID_REQUEST, 80007));
            return res;
        }

        if (!HlieUtil::version_ge(model_info.model_version_, "v0.7.0")) {
            res->error(format_error_response_ex("build_version must be >= v0.7.0", ERROR_TYPE_INVALID_REQUEST, 80007));
            return res;
        }

        model_instances_[model_name].meta.status = SERVER_MODEL_STATUS_LOADING;
        if (!llm_model_service_) {
            llm_model_service_ = std::make_shared<LLModelService>();
            if (llm_model_service_->init(model_dir_, model_info) < 0) {
                LOG_ERROR("failed to init the LLM model service");
                res->error(format_error_response_ex("failed to init the LLM model service", ERROR_TYPE_SERVER, 80001));
                return res;
            }

            llm_model_service_->run();
            model_instances_[model_name].meta.status = SERVER_MODEL_STATUS_LOADED;
        } else {
            // 子进程实例参数
            json_info_file = generate_meta_info(model_info, model_dir_, res);
            if (json_info_file.empty()) {
                return res;
            }

            instance_id = generate_instance_id();
            std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                                std::string("--device_id"),   devices,
                                                std::string("--json_info"),   json_info_file};
            const auto& args = config_services->get_args();

            // Copy llama-params
            for (const auto& [key, val] : args) {
                args_for_sub_process.push_back(key);
                args_for_sub_process.push_back(val);
            }

            if (auto pid = Singleton<HLIEProcessManager>::instance()->create_process(program_path_, args_for_sub_process, {model_name},
                                                                                     instance_id);
                pid == 0 || pid == -1) {
                LOG_ERROR("Failed to start process for instance_id[{}]", instance_id);
            } else {
                LOG_INFO("Started process PID[{}] instance_id[{}]", pid, instance_id);
            }
        }
    } else if (model_type == SD) {
#ifdef HOUMO_XH2
        if (!body.contains("params") || !body["params"].is_object()) {
            res->error(format_error_response_ex("params invalid for SD3", ERROR_TYPE_INVALID_REQUEST, 80007));
            return res;
        }

        auto& params = body["params"];
        auto model_info = parse_and_build_sd_model(params, res, model_type, devices);
        if (model_info.get_model_type().empty()) {
            return res;
        }

        if (!sd_model_service_) {
            sd_model_service_ = std::make_shared<SDModelService>();
            if (sd_model_service_->init(model_dir_, model_info) < 0) {
                LOG_ERROR("failed to init the SD model service");
                res->error(format_error_response_ex("failed to init the SD model service", ERROR_TYPE_SERVER, 80001));
                return res;
            }

            sd_model_service_->run();
        } else {
            // 子进程实例参数
            json_info_file = generate_meta_info(model_info, model_dir_, res);
            if (json_info_file.empty()) {
                return res;
            }

            instance_id = generate_instance_id();
            std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                                std::string("--device_id"),   devices,
                                                std::string("--json_info"),   json_info_file};
            if (process_id = Singleton<HLIEProcessManager>::instance()->create_process(program_path_, args_for_sub_process, {HLIEModelType::SD3}, instance_id);
                process_id == 0 || process_id == -1) {
                LOG_ERROR("Failed to start process for instance_id[{}]", instance_id);
            } else {
                LOG_INFO("Started process PID[{}] instance_id[{}]", process_id, instance_id);
            }
        }
#else
        res->error(format_error_response_ex("The sd is not available.", ERROR_TYPE_UNAVAILABLE, 80009));
        return res;
#endif
    }

    // TODO: 模型启动参数校验, device 是否可用
    json resp = {
        {"code", 0},
        {"port", port_ + instance_id},
        {"instance_id", instance_id},
        {"process_id", process_id},
        {"json_info_file", json_info_file},
    };

    res->ok(resp);
    return res;
}

std::vector<HLIEWorkflow::RouteConfig> HLIEWorkflow::get_route_configs() {
    auto make_handler = [this](auto func) { return [this, func](const server_http_req& req) { return (this->*func)(req); }; };
    return {{
                {"/health", "/v1/health"},
                GET,
                make_handler(&HLIEWorkflow::handle_health),
                false // 不需要模型Running
            },
            {
                {"/v1/models", "/models", "/api/tags"},
                GET,
                make_handler(&HLIEWorkflow::handle_models),
                false // 不需要模型Running
            },
            {
                {"/v1/chat/completions", "/chat/completions", "/api/chat"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    CHECK_MODEL_READY(llm_model_service_);
                    return llm_model_service_->handle_chat_completions(req);
                },
                true, // 需要模型Running
                LLM // LLM模型类型
            },
#ifdef HOUMO_XH2
            {
                {"/v1/images/generations", "/images/generations"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    CHECK_MODEL_READY(sd_model_service_);
                    return sd_model_service_->handle_image_generations(req);
                },
                true,
                SD // 生图模型类型
            },
            {
                {"/hlie/v1/devices"},
                GET,
                [this](const server_http_req& req) { return device_service_->handle_devices(req); },
                false // 不需要模型Running
            },
#endif
            {
                {"/v1/completions"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    CHECK_MODEL_READY(llm_model_service_);
                    return llm_model_service_->handle_completions_oai(req);
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/completion", "/completions"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    CHECK_MODEL_READY(llm_model_service_);
                    return llm_model_service_->handle_completions(req);
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/embedding", "/embeddings"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    CHECK_MODEL_READY(llm_model_service_);
                    return llm_model_service_->handle_embeddings(req, TASK_RESPONSE_TYPE_NONE);
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/v1/embeddings"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    CHECK_MODEL_READY(llm_model_service_);
                    return llm_model_service_->handle_embeddings(req, TASK_RESPONSE_TYPE_OAI_EMBD);
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/props"},
                GET,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    CHECK_MODEL_READY(llm_model_service_);
                    return llm_model_service_->handle_props(req);
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/hlie/v1/start", "/models/load"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr { return start_service(req); },
                false // 不需要模型Running
            },
            {
                {"/hlie/v1/stop"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr { return stop_service(req); },
                false // 不需要模型Running
            },
            {
                {"/hlie/v1/status"},
                GET,
                [this](const server_http_req& req) -> server_http_res_ptr { return handle_get_status(req); },
                false // 不需要模型Running
            },
            {
                {"/models/status"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr { return handle_post_status(req); },
                false // 不需要模型Running
            },
            {
                {"/hlie/v1/unload_model", "/models/unload"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr { return handle_unload_model(req); },
                false // 不需要模型Running
            }};
}

void HLIEWorkflow::set_server_port(const int port) {
    port_ = port;
    if (llm_model_service_) {
        model_instances_[llm_model_service_->get_model_name()].meta.port = port;
    }
#ifdef HOUMO_XH2
    if (sd_model_service_) {
        model_instances_[HLIEModelType::SD3].meta.port = port;
    }
#endif
}

server_http_res_ptr HLIEWorkflow::stop_service(const server_http_req& req) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    auto res = std::make_unique<server_res_generator>();
    if (config_services->get_int("server", "replicas", 1) > 1) {
        // 集群|多实例模式下，不允许外部调用启动模型实例
        res->error(
            format_error_response_ex("In the cluster or multi-instance mode, external calls to start model instances are not allowed.",
                                     ERROR_TYPE_NOT_SUPPORTED, 80005));
        return res;
    }

    const auto body = json::parse(req.body);
    const auto instance_id = body.at("instance_id").get<int>();
    const auto process_id = body.at("process_id").get<int>();

    std::unique_lock lock(mutex_);

    if (instance_id == 0) {
        if (llm_model_service_) {
            llm_model_service_->stop();
            llm_model_service_.reset();
            model_instances_[llm_model_service_->get_model_name()].meta.status = SERVER_MODEL_STATUS_LOADED;
        }
#ifdef HOUMO_XH2
        if (sd_model_service_) {
            sd_model_service_->stop();
            sd_model_service_.reset();
        }
#endif
        LOG_INFO("Succeed to stop service process for instance_id[{}][{}]", instance_id, process_id);
    } else if (process_id != 0) {
        if (!Singleton<HLIEProcessManager>::instance()->terminate_process(process_id)) {
            LOG_ERROR("Failed to terminate process for instance_id[{}][{}]", instance_id, process_id);
            const json resp = {{"ret", 80001}, "Failed to terminate process for instance_id."};
            res->error(resp);
            return res;
        }

        LOG_INFO("Succeed to terminate process for instance_id[{}][{}]", instance_id, process_id);
    }

    const json resp = {
        {"code", 0},
        {"instance_id", instance_id},
        {"process_id", process_id},
    };

    res->ok(resp);
    return res;
}

server_http_res_ptr HLIEWorkflow::handle_get_status(const server_http_req& req) const {
    const std::vector<HLIEProcessManager::ProcessInfo> process_infos = Singleton<HLIEProcessManager>::instance()->get_running_processes();

    json processors = json::array();
    json models = json::array();
    if (llm_model_service_) {
        models.push_back(llm_model_service_->get_model_name());
    }
    if (sd_model_service_) {
        models.push_back(HLIEModelType::SD3);
    }
    processors.push_back({{"params", ""},
                          {"instance_id", 0},
                          {"port", port_},
                          {"uuid", ""},
                          {"process_id", 0},
                          {"start_time", start_time_},
                          {"models", models}});

    for (const auto& p : process_infos) {
        processors.push_back({
            {"params", p.params},
            {"instance_id", p.instance_id},
            {"port", p.instance_id + port_},
            {"uuid", p.instance_id},
            {"process_id", p.process_id},
            {"start_time", p.start_time},
            {"models", p.model_names},
        });
    }

    const json resp = {{"code", 0}, {"processor", processors}};
    auto res = std::make_unique<server_res_generator>();
    res->ok(resp);
    return res;
}

server_http_res_ptr HLIEWorkflow::handle_post_status(const server_http_req& req) {
    auto res = std::make_unique<server_res_generator>();
    json body = json::parse(req.body);
    std::string model = json_value(body, "model", std::string());
    std::string value = json_value(body, "value", std::string());
    // for now, we only allow updating to LOADED status
    const auto status = server_model_status_from_string(value);
    if (status != SERVER_MODEL_STATUS_LOADED) {
        throw std::runtime_error("invalid status value");
    }

    if (auto meta = get_model_meta(model); meta.has_value()) {
        meta->status = status;
        std::lock_guard lk(mutex_);
        if (const auto it = model_instances_.find(model); it != model_instances_.end()) {
            it->second.meta = meta.value();
            LOG_INFO("Model[{}] loaded", model);
        }
    }

    res->ok({{"success", true}});
    return res;
}

server_http_res_ptr HLIEWorkflow::handle_unload_model(const server_http_req& req) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    auto res = std::make_unique<server_res_generator>();
    if (config_services->get_int("server", "replicas", 1) > 1) {
        // 集群|多实例模式下，不允许外部调用启动模型实例
        res->error(
            format_error_response_ex("In the cluster or multi-instance mode, external calls to start model instances are not allowed.",
                                     ERROR_TYPE_NOT_SUPPORTED, 80005));
        return res;
    }

    const auto body = json::parse(req.body);
    const std::string model_type = body.value("model_type", LLM);
    if (model_type != LLM && model_type != SD) {
        res->error(format_error_response_ex("Invalid model type.", ERROR_TYPE_INVALID_REQUEST, 80006));
        return res;
    }

    std::unique_lock lock(mutex_);
    if (model_type == LLM && llm_model_service_) {
        llm_model_service_->stop();
        llm_model_service_.reset();
    }
#ifdef HOUMO_XH2
    if (model_type == SD && sd_model_service_) {
        sd_model_service_->stop();
        sd_model_service_.reset();
    }
#endif
    LOG_INFO("Succeed to uninstall model");

    const json resp = {
        {"code", 0},
    };

    res->ok(resp);
    return res;
}

std::vector<server_model_meta> HLIEWorkflow::get_all_model_metas() {
    std::unique_lock lk(mutex_);
    std::vector<server_model_meta> result;
    result.reserve(model_instances_.size());
    for (const auto& [name, inst] : model_instances_) {
        result.push_back(inst.meta);
    }
    return result;
}

std::optional<server_model_meta> HLIEWorkflow::get_model_meta(const std::string& name) {
    std::unique_lock lk(mutex_);
    if (const auto it = model_instances_.find(name); it != model_instances_.end()) {
        return it->second.meta;
    }
    return std::nullopt;
}

HLIE_NAMESPACE_END

#include "hlie_app/hlie_workflow.h"

#include <filesystem>

#include "INIReader.h"
#include "hlie_app/hlie_infer_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_process_manager.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_exception.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

#ifdef HOUMO_XH2
#include "hlie_app/hlie_sd_model_service.h"
#include "hlie_app/hlie_device_service.h"
#endif

HLIE_NAMESPACE_BEGIN

#define CHECK_MODEL_READY(service, res)                                                                                                    \
    do {                                                                                                                                   \
        if (!(service) || (service)->get_state() != SERVER_STATE_READY) {                                                                  \
            res_error(res, format_error_response("Model is not available for now.", ERROR_TYPE_UNAVAILABLE));                              \
            return;                                                                                                                        \
        }                                                                                                                                  \
    } while (0)

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

    HLIEModelInfo parse_and_build_llm_model_info(const json& params, httplib::Response& res, const std::string& model_path,
                                                 const std::string& model_type, const std::string& devices) {
        if (!params.contains("context_size") || !params["context_size"].is_number_integer() || !params.contains("batch") ||
            !params["batch"].is_number_integer() || !params.contains("core_num") || !params["core_num"].is_number_integer() ||
            !params.contains("is_mm") || !params["is_mm"].is_boolean() || !params.contains("mm_proj_path") ||
            !params["mm_proj_path"].is_string() || !params.contains("xh_version") || !params["xh_version"].is_number_integer() ||
            !params.contains("size") || !params["size"].is_string()) {
            const json resp = {{"ret", 80007}, {"msg", "params invalid"}};
            res_error(res, resp);
            return {};
        }

        // 解析参数
        const int context_size = params["context_size"].get<int>();
        const int batch = params["batch"].get<int>();
        const int core_num = params["core_num"].get<int>();
        const bool is_mm = params["is_mm"].get<bool>();
        const auto mm_proj_path = params["mm_proj_path"].get<std::string>();
        const int xh_version = params["xh_version"].get<int>();
        const auto size = params["size"].get<std::string>();

        // 构建 HLIEModelInfo
        HLIEModelInfo model_info;
        model_info.build_llm(model_path, model_type, context_size, core_num, batch, xh_version, is_mm, size, mm_proj_path, devices);

        return model_info;
    }

    HLIEModelInfo parse_and_build_sd_model(const json& params, httplib::Response& res, const std::string& model_type,
                                           const std::string& devices) {
        // SD 模型必须字段检查
        const char* required_fields[] = {"vae_model", "t5_model", "clip_model", "clip_l_model", "mmdit_model", "xh_version"};

        for (const auto& field : required_fields) {
            if (const bool is_string_field = std::string(field) != "xh_version"; !params.contains(field) ||
                (is_string_field && !params[field].is_string()) ||
                (!is_string_field && !params[field].is_number_integer()))
            {
                const json resp = {{"ret", 80007}, {"msg", "params invalid"}};
                res_error(res, resp);
                return {};
            }
        }

        // 解析参数
        const auto vae_model = params["vae_model"].get<std::string>();
        const auto t5_model = params["t5_model"].get<std::string>();
        const auto clip_model = params["clip_model"].get<std::string>();
        const auto clip_l_model = params["clip_l_model"].get<std::string>();
        const auto mmdit_model = params["mmdit_model"].get<std::string>();
        const auto xh_version = params["xh_version"].get<int>();

        // 构建 HLIEModelInfo
        HLIEModelInfo model_info;
        model_info.build_sd(model_type, vae_model, t5_model, clip_model, clip_l_model, mmdit_model, xh_version, devices);

        return model_info;
    }

    std::string generate_meta_info(const HLIEModelInfo& model_info ,const std::string& model_dir, httplib::Response& res) {
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
            res_error(res, resp);
            return "";
        }

        return json_info_file;
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
        } else {
            LOG_ERROR("Failed to parse model info from --json_info: {}", json_path);
            return -1;
        }
    } else {
        model_infos = load_model_infos_from_directory(model_dir);
        if (model_infos.empty()) {
            LOG_WARN("No valid model info files found in directory: {}", model_dir);
            if (args.find("--instance_id") == args.end()) {
                // Master Process Start in Zero-Load Mode
                start_time_ = time(nullptr);
                return 8;
            }

            return -1;
        }
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
        if ((model_type == HLIEModelType::SD3 || model_type == SD ) && !sd_model_service_) {
#ifdef HOUMO_XH2
            LOG_INFO("Initializing SDModelService");
            sd_model_service_ = std::make_shared<SDModelService>();
            if (sd_model_service_->init(model_dir, model_info) < 0) {
                LOG_ERROR("failed to init the SD model service");
                return -1;
            }
#else
            LOG_ERROR("failed to init the SD model service, ONLY for XH2");
            return -1;
#endif
        } else if (!llm_model_service_ &&
                   (model_type == LLM || model_type == HLIEModelType::Qwen3 || model_type == HLIEModelType::Qwen25 || model_type == HLIEModelType::DeepSeek)) {
            LOG_INFO("Initializing LLModelService");
            llm_model_service_ = std::make_shared<LLModelService>();
            if (llm_model_service_->init(model_dir, model_info) < 0) {
                LOG_ERROR("failed to init the LLM model service");
                return -1;
            }
        }
    }

#ifdef HOUMO_XH2
    device_service_ = std::make_shared<HLIEDeviceService>();
#endif

    Singleton<HLIEDeviceHelper>::instance()->print_status();
    start_time_ = time(nullptr);
    return 0;
}

void HLIEWorkflow::run() const {
    if (llm_model_service_) {
        llm_model_service_->run();
    }

#ifdef HOUMO_XH2
    if (sd_model_service_) {
        sd_model_service_->run();
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

void HLIEWorkflow::handle_health(httplib::Response& res) const {
    json resp = {
        {"status", "ok"},
    };

    if (llm_model_service_) {
        if (llm_model_service_->handle_health(resp) < 0) {
            throw CommonException(4, "llm model service health check failed");
        }
    }

#ifdef HOUMO_XH2
    if (sd_model_service_) {
        if (sd_model_service_->handle_health(resp) < 0) {
            throw CommonException(4, "sd model service health check failed");
        }
    }
#endif

    res_ok(res, resp);
}

void HLIEWorkflow::handle_models(httplib::Response& res) const {
    json resp = {};
    resp["object"] = "list";
    if (llm_model_service_) {
        llm_model_service_->handle_models(resp);
    } else {
        resp["data"] = json::array();
        resp["models"] = json::array();
    }

#ifdef HOUMO_XH2
    if (sd_model_service_) {
        sd_model_service_->handle_models(resp);
    }
#endif

    res_ok(res, resp);
}

void HLIEWorkflow::start_service(const httplib::Request& req, httplib::Response& res) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    if (config_services->get_int("server", "replicas", 1) > 1) {
        // 集群|多实例模式下，不允许外部调用启动模型实例
        const json resp = {{"ret", 80005},
                           "In the cluster or multi-instance mode, external calls to start model instances are not allowed."};
        res_error(res, resp);
        return;
    }

    auto body = json::parse(req.body);

    auto devices = body.at("devices").get<std::string>();
    auto model_type = body.at("model_type").get<std::string>();

    if (model_type != LLM && model_type != SD) {
        const json resp = {{"ret", 80006}, "Invalid model type."};
        res_error(res, resp);
        return;
    }

    if (!body.contains("params") || !body["params"].is_object()) {
        const json resp = {{"ret", 80007}, {"msg", "params invalid"}};
        res_error(res, resp);
        return;
    }

    LOG_INFO("ready to acquire devices");
    std::unique_lock lock(mutex_);

    auto& params = body["params"];

    if (devices == "undefined") {
        const std::optional<int> val = Singleton<HLIEDeviceHelper>::instance()->acquire_device(HlieUtil::uuid(), true);
        if (!val) {
            const json resp = {{"ret", 80006}, "The devices is not available."};
            res_error(res, resp);
            return;
        }
        devices = std::to_string(*val);
    } else if (std::vector<int> devices_vec;
               !Singleton<HLIEDeviceHelper>::instance()->acquire_devices(HlieUtil::uuid(), devices, devices_vec, true)) {
        const json resp = {{"ret", 80006}, "The devices is not available."};
        res_error(res, resp);
        return;
    }

    LOG_INFO("acquire devices successfully with {}", devices);

    auto instance_id = 0;
    std::string json_info_file;
    ProcessID process_id = 0;
    if (model_type == LLM) {
        auto model_path = body.at("model_path").get<std::string>();
        auto model_info = parse_and_build_llm_model_info(params, res, model_path, model_type, devices);
        if (model_info.get_model_type().empty()) {
            return;
        }

        if (!llm_model_service_) {
             llm_model_service_ = std::make_shared<LLModelService>();
            if (llm_model_service_->init(model_dir_, model_info) < 0) {
                LOG_ERROR("failed to init the LLM model service");
                const json resp = {{"ret", 80007}, "failed to init the LLM model service"};
                res_error(res, resp);
                return;
            }

            llm_model_service_->run();
        } else {
            // 子进程实例参数
            json_info_file = generate_meta_info(model_info, model_dir_, res);
            if (json_info_file.empty()) {
                return;
            }

            instance_id  = generate_instance_id();
            std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                                std::string("--device_id"),   devices,
                                                std::string("--json_info"),   json_info_file};
            const auto& args = config_services->get_args();

            // Copy llama-params
            for (const auto& [key, val] : args) {
                args_for_sub_process.push_back(key);
                args_for_sub_process.push_back(val);
            }

            if (auto pid = Singleton<HLIEProcessManager>::instance()->create_process(program_path_, args_for_sub_process);
                pid == 0 || pid == -1) {
                LOG_ERROR("Failed to start process for instance_id[{}]", instance_id);
            } else {
                LOG_INFO("Started process PID[{}] instance_id[{}]", pid, instance_id);
            }
        }
    } else if (model_type == SD) {
#ifdef HOUMO_XH2
        auto model_info = parse_and_build_sd_model(params, res, model_type, devices);
        if (model_info.get_model_type().empty()) {
            return;
        }

        if (!sd_model_service_) {
             sd_model_service_ = std::make_shared<SDModelService>();
            if (sd_model_service_->init(model_dir_, model_info) < 0) {
                LOG_ERROR("failed to init the SD model service");
                const json resp = {{"ret", 80007}, "failed to init the SD model service"};
                res_error(res, resp);
                return;
            }

            sd_model_service_->run();
        } else {
            // 子进程实例参数
            json_info_file = generate_meta_info(model_info, model_dir_, res);
            if (json_info_file.empty()) {
                return;
            }

            instance_id  = generate_instance_id();
            std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                                std::string("--device_id"),   devices,
                                                std::string("--json_info"),   json_info_file};
            if (process_id = Singleton<HLIEProcessManager>::instance()->create_process(program_path_, args_for_sub_process);
                process_id == 0 || process_id == -1) {
                LOG_ERROR("Failed to start process for instance_id[{}]", instance_id);
            } else {
                LOG_INFO("Started process PID[{}] instance_id[{}]", process_id, instance_id);
            }
        }
#else
        const json resp = {{"ret", 80008}, "The sd is not available."};
        res_error(res, resp);
#endif
    }

    // TODO: 模型启动参数校验, device 是否可用
    json resp = {
        {"code", 0},
        {"port", 7901 + instance_id},
        {"instance_id", instance_id},
        {"process_id", process_id},
        {"json_info_file", json_info_file},
    };

    res_ok(res, resp);
}

const std::vector<HLIEWorkflow::RouteConfig>& HLIEWorkflow::get_route_configs() {
    static std::vector<RouteConfig> routes = {
        {
            {"/health"},
            GET,
            [this](const httplib::Request&, httplib::Response& res) { this->handle_health(res); },
            false // 不需要模型Running
        },
        {
            {"/v1/models", "/models", "/api/tags"},
            GET,
            [this](const httplib::Request&, httplib::Response& res) { this->handle_models(res); },
            false // 不需要模型Running
        },
        {
            {"/v1/chat/completions", "/chat/completions", "/api/chat"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) {
                CHECK_MODEL_READY(llm_model_service_, res);
                llm_model_service_->handle_chat_completions(req, res);
            },
            true, // 需要模型Running
            LLM // LLM模型类型
        },
#ifdef HOUMO_XH2
        {
            {"/v1/images/generations", "/images/generations"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) {
                CHECK_MODEL_READY(sd_model_service_, res);
                sd_model_service_->handle_image_generations(req, res);
            },
            true,
            SD // 生图模型类型
        },
        {
            {"/hlie/v1/devices"},
            GET,
            [this](const httplib::Request& req, httplib::Response& res) { device_service_->handle_devices(req, res); },
            false // 不需要模型Running
        },
#endif
        {
            {"/v1/completions"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) {
                CHECK_MODEL_READY(llm_model_service_, res);
                llm_model_service_->handle_completions_oai(req, res);
            },
            true,
            LLM // LLM模型类型
        },
        {
            {"/completion", "/completions"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) {
                CHECK_MODEL_READY(llm_model_service_, res);
                llm_model_service_->handle_completions(req, res);
            },
            true,
            LLM // LLM模型类型
        },
        {
            {"/embedding", "/embeddings"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) {
                CHECK_MODEL_READY(llm_model_service_, res);
                llm_model_service_->handle_embeddings(req, res, OAICOMPAT_TYPE_NONE);
            },
            true,
            LLM // LLM模型类型
        },
        {
            {"/v1/embeddings"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) {
                CHECK_MODEL_READY(llm_model_service_, res);
                llm_model_service_->handle_embeddings(req, res, OAICOMPAT_TYPE_EMBEDDING);
            },
            true,
            LLM // LLM模型类型
        },
        {
            {"/hlie/v1/start"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) { this->start_service(req, res); },
            false // 不需要模型Running
        },
        {
            {"/hlie/v1/stop"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) { this->stop_service(req, res); },
            false // 不需要模型Running
        },
        {
            {"/hlie/v1/status"},
            GET,
            [this](const httplib::Request& req, httplib::Response& res) { this->handle_status(req, res); },
            false // 不需要模型Running
        },
        {
            {"/hlie/v1/unload_model"},
            POST,
            [this](const httplib::Request& req, httplib::Response& res) { this->handle_unload_model(req, res); },
            false // 不需要模型Running
        }};

    return routes;
}

void HLIEWorkflow::set_server_port(const int port) {
    port_ = port;
}

void HLIEWorkflow::stop_service(const httplib::Request& req, httplib::Response& res) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    if (config_services->get_int("server", "replicas", 1) > 1) {
        // 集群|多实例模式下，不允许外部调用启动模型实例
        const json resp = {{"ret", 80005},
                           "In the cluster or multi-instance mode, external calls to start model instances are not allowed."};
        res_error(res, resp);
        return;
    }

    const auto body = json::parse(req.body);
    const auto instance_id = body.at("instance_id").get<int>();
    const auto process_id = body.at("process_id").get<int>();

    std::unique_lock lock(mutex_);

    if (instance_id == 0) {
        if (llm_model_service_) {
            llm_model_service_->stop();
            llm_model_service_.reset();
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
            const json resp = {{"ret", 80005}, "Failed to terminate process for instance_id."};
            res_error(res, resp);
            return;
        }

        LOG_INFO("Succeed to terminate process for instance_id[{}][{}]", instance_id, process_id);
    }

    const json resp = {
        {"code", 0},
        {"instance_id", instance_id},
        {"process_id", process_id},
    };

    res_ok(res, resp);
}

void HLIEWorkflow::handle_status(const httplib::Request& req, httplib::Response& res) const {

    const std::vector<HLIEProcessManager::ProcessInfo> process_infos = Singleton<HLIEProcessManager>::instance()->get_running_processes();

    json processors = json::array();
    processors.push_back({
        {"params", ""},
        {"instance_id", 0},
        {"port", port_},
        {"uuid", ""},
        {"process_id", 0},
        {"start_time", start_time_},
    });

    int id = 1;
    for (const auto& p : process_infos) {
        processors.push_back({
            {"params", p.params},
            {"instance_id", id},
            {"port", id + port_},
            {"uuid", p.instance_id},
            {"process_id", p.process_id},
            {"start_time", p.start_time},
        });

        id++;
    }

    const json resp = {{"code", 0}, {"processor", processors}};

    res_ok(res, resp);
}


void HLIEWorkflow::handle_unload_model(const httplib::Request& req, httplib::Response& res) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    if (config_services->get_int("server", "replicas", 1) > 1) {
        // 集群|多实例模式下，不允许外部调用启动模型实例
        const json resp = {{"ret", 80005},
                           "In the cluster or multi-instance mode, external calls to start model instances are not allowed."};
        res_error(res, resp);
        return;
    }

    const auto body = json::parse(req.body);
    const auto model_type = body.at("model_type").get<std::string>();
    if (model_type != LLM && model_type != SD) {
        const json resp = {{"ret", 80006}, "Invalid model type."};
        res_error(res, resp);
        return;
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

    res_ok(res, resp);
}

HLIE_NAMESPACE_END

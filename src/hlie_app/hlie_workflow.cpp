#include "hlie_app/hlie_workflow.h"

#include <filesystem>
#include <server/server-http.h>

#include "INIReader.h"
#include "hlie_app/hlie_infer_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_process_manager.h"
#include "hlie_app/hlie_server_http_proxy.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

#include "hlie_app/hlie_device_service.h"
#include "hlie_app/hlie_sd_model_service.h"
#include "hlie_app/hlie_whisper_model_service.h"
#include "hlie_app/hlie_cosyvoice_model_service.h"

HLIE_NAMESPACE_BEGIN

#define CHECK_MODEL_READY_IMPL(service, check_name)                                                                                             \
    do {                                                                                                                                   \
        auto res = std::make_unique<server_res_generator_light>();                                                                               \
        if (!(service) || (service)->get_state() != SERVER_STATE_READY) {                                                                  \
            res->error(format_error_response("Model is not available for now.", ERROR_TYPE_UNAVAILABLE));                                  \
            return res;                                                                                                                    \
        }                                                                                                                                  \
        if ((check_name)) {                                                                                                                \
            const json body = json::parse(req.body);                                                                                       \
            const std::string name = json_value(body, "model", std::string());                                                             \
            if (!router_validate_model(name, res)) {                                                                                       \
                return res;                                                                                                                \
            }                                                                                                                              \
        }                                                                                                                                  \
    } while (0)

#define CHECK_MODEL_READY(service) CHECK_MODEL_READY_IMPL(service, true)
#define CHECK_SERVICE(service) CHECK_MODEL_READY_IMPL(service, false)

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

    HLIEModelInfo parse_and_build_sd_model(const json& params, const std::unique_ptr<server_res_generator_light>& res,
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
                                   const std::unique_ptr<server_res_generator_light>& res) {
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

    void update_status(const int router_port, const std::string& name, const int port,
                       const server_model_status status = SERVER_MODEL_STATUS_LOADED) {
        // send a notification to the router server that a model instance is ready
        httplib::Client cli("127.0.0.1", router_port);
        cli.set_connection_timeout(0, 200000); // 200 milliseconds
        httplib::Request req;
        req.method = "POST";
        req.path = "/models/status";
        req.set_header("Content-Type", "application/json");

        json body;
        body["model"] = name;
        body["port"] = port;
        body["value"] = server_model_status_to_string(status);
        req.body = body.dump();

        LOG_INFO("notifying router server (port={}) that model {} at [{}] is ready", router_port, name.c_str(), port);
        if (const auto result = cli.send(req); result.error() != httplib::Error::Success) {
            const auto err_str = httplib::to_string(result.error());
            LOG_ERR("failed to notify router server: %s\n", err_str.c_str());
            // exit(1); // force exit
        }
    }

    bool is_equal_before_vit(const std::string& full_name, const std::string& name) {
        if (const size_t pos = full_name.find("_vit_"); pos != std::string::npos) {
            const std::string before_vit = full_name.substr(0, pos);
            return before_vit == name;
        }
        // 如果没有 _vit_，直接比较全名
        return full_name == name;
    }
} // namespace

int HLIEWorkflow::initialize(const std::string& model_dir, const std::string& program_path) {
    model_dir_ = model_dir;
    program_path_ = program_path;
    if (model_dir.empty()) {
        LOG_ERROR("model dir is empty");
        return -1;
    }

    device_service_ = std::make_shared<HLIEDeviceService>();
    if (device_service_->init() < 0) {
        LOG_ERROR("failed to init the HLIE device service");
        return -1;
    }

    const auto& args = Singleton<HLIEConfigService>::instance()->get_args();
    replicas_mode_ =  Singleton<HLIEConfigService>::instance()->get_int("server", "replicas", 1) > 1;
    master_port_ = Singleton<HLIEConfigService>::instance()->get_int("server", "port", 7901, true);
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
                if (llm_model_service_->init(model_dir, model_info, device_service_) < 0) {
                    LOG_ERROR("failed to init the LLM model service");
                    return -1;
                }
            }
            else {
                sd_model_service_ = std::make_shared<SDModelService>();
                if (sd_model_service_->init(model_dir, model_info, device_service_) < 0) {
                    LOG_ERROR("failed to init the SD model service");
                    return -1;
                }
            }

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
    for (auto& model_info : model_infos) {
        ModelInstance model_instance;
        server_model_meta meta;
        meta.name = model_info.get_model_name();
        model_instance.path = model_info.get_model_path();
        model_instance.path_mmproj = model_info.get_mm_proj_path();
        model_instance.meta = std::move(meta);

        auto model_key = model_info.get_model_name();
        if (!model_info.get_mm_proj_path().empty()) {
            model_key += "_" + model_info.vl_spec_;
        }

        model_info.model_key_ = model_key;
        model_instance.key = model_key;
        model_instance.model_info = model_info;
        model_instances_[model_key] = std::move(model_instance);
        LOG_INFO("Scanned Local Model: {}", model_key);
    }

    if (model_infos.empty()) {
        LOG_WARN("No valid model info files found in directory: {}", model_dir);
        if (args.find("--instance_id") == args.end()) {
            // Master Process Start in Zero-Load Mode
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
            } else if (enable_sd3 && m.get_model_name() == HLIEModelType::SD3) {
                LOG_INFO("SD3 found");
                found_sd3 = true;
                sd3_model_info = m;
            }
        }

        if (!found_llm) {
            LOG_ERROR("Default llm not found: {}, scanned models count: {}", default_model, model_infos.size());
        }
    } else if (enable_sd3) {
        auto it = std::find_if(model_infos.begin(), model_infos.end(),
                               [&](const HLIEModelInfo& info) { return info.get_model_name() == HLIEModelType::SD3; });
        if (it == model_infos.end()) {
            LOG_WARN("No SD3 model found");
            return -1;
        }

        found_sd3 = true;
        sd3_model_info = *it;
    }

    if (!found_llm && !found_sd3) {
        if (args.find("--instance_id") == args.end()) {
            // Master Process Start in Zero-Load Mode
            Singleton<HLIEDeviceHelper>::instance()->print_status();
            start_time_ = time(nullptr);
            return 8;
        }

        LOG_WARN("No default models found for slave instance");
        return -1;
    }

    if (found_llm) {
        if (HlieUtil::contains_ignore_case(default_model_info.get_model_name(), WHISPER)) {
            LOG_INFO("Initializing WhisperModelService");
            whisper_model_service_ = std::make_shared<WhisperModelService>();
            if (whisper_model_service_->init(model_dir, default_model_info, device_service_) < 0) {
                LOG_ERROR("failed to init the whisper model service");
                whisper_model_service_.reset();
                return -1;
            }
        } else if (HlieUtil::contains_ignore_case(default_model_info.get_model_name(), HLIEModelType::COSYVOICE)) {
            LOG_INFO("Initializing CosyVoiceModelService");
            cosy_voice_model_service_ = std::make_shared<CosyVoiceModelService>();
            if (cosy_voice_model_service_->init(model_dir, default_model_info, device_service_) < 0) {
                LOG_ERROR("failed to init the cosyvoice model service");
                cosy_voice_model_service_.reset();
                return -1;
            }
        } else {
            LOG_INFO("Initializing LLModelService");
            if (const std::string default_mmproj = Singleton<HLIEConfigService>::instance()->get_string("model", "default_mmproj");
                !default_mmproj.empty()) {
                auto mmproj_path = default_mmproj + ".gguf";
                if (HlieUtil::is_relative_path(mmproj_path)) {
                    mmproj_path = HlieUtil::join_paths(model_dir, mmproj_path);
                }

                std::filesystem::path mmproj_path_tmp(mmproj_path);
                if (std::error_code ec;
                    !std::filesystem::exists(mmproj_path_tmp, ec) || !std::filesystem::is_regular_file(mmproj_path_tmp, ec)) {
                    LOG_ERROR("default_mmproj file does not exist");
                    return -1;
                }

                default_model_info.mm_proj_path_ = mmproj_path;
            }

            llm_model_service_ = std::make_shared<LLModelService>();
            if (llm_model_service_->init(model_dir, default_model_info, device_service_) < 0) {
                LOG_ERROR("failed to init the LLM model service");
                llm_model_service_.reset();
                return -1;
            }
        }
    }

    if (found_sd3) {
        LOG_INFO("Initializing SDModelService");
        sd_model_service_ = std::make_shared<SDModelService>();
        if (sd_model_service_->init(model_dir, sd3_model_info, device_service_) < 0) {
            LOG_ERROR("failed to init the SD model service");
            sd_model_service_.reset();
            return -1;
        }
    }

    Singleton<HLIEDeviceHelper>::instance()->print_status();
    start_time_ = time(nullptr);
    return 0;
}

void HLIEWorkflow::run() {
    if (llm_model_service_) {
        llm_model_service_->run();

        model_instances_[llm_model_service_->get_model_key()].meta.status = SERVER_MODEL_STATUS_LOADED;

        if (is_slave_ && !replicas_mode_) {
            std::thread([&] {
                model_instances_[llm_model_service_->get_model_key()].meta.port = port_;
                std::this_thread::sleep_for(std::chrono::seconds(3));
                update_status(master_port_, llm_model_service_->get_model_key(), port_);
            }).detach();
        }
    }

    if (sd_model_service_) {
        sd_model_service_->run();
        model_instances_[HLIEModelType::SD3].meta.status = SERVER_MODEL_STATUS_LOADED;
        if (is_slave_ && !replicas_mode_) {
            std::thread([&] {
                model_instances_[HLIEModelType::SD3].meta.port = port_;
                std::this_thread::sleep_for(std::chrono::seconds(3));
                update_status(master_port_, HLIEModelType::SD3, port_);
            }).detach();
        }
    }

    if (whisper_model_service_) {
        whisper_model_service_->run();
        model_instances_[whisper_model_service_->get_model_key()].meta.status = SERVER_MODEL_STATUS_LOADED;
        if (is_slave_ && !replicas_mode_) {
            std::thread([&] {
                model_instances_[whisper_model_service_->get_model_key()].meta.port = port_;
                std::this_thread::sleep_for(std::chrono::seconds(3));
                update_status(master_port_, whisper_model_service_->get_model_key(), port_);
            }).detach();
        }
    }

    if (cosy_voice_model_service_) {
        cosy_voice_model_service_->run();
        model_instances_[cosy_voice_model_service_->get_model_key()].meta.status = SERVER_MODEL_STATUS_LOADED;
        if (is_slave_ && !replicas_mode_) {
            std::thread([&] {
                model_instances_[cosy_voice_model_service_->get_model_key()].meta.port = port_;
                std::this_thread::sleep_for(std::chrono::seconds(3));
                update_status(master_port_, cosy_voice_model_service_->get_model_key(), port_);
            }).detach();
        }
    }
}

void HLIEWorkflow::stop() {
    if (sd_model_service_) {
        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, HLIEModelType::SD3, port_, SERVER_MODEL_STATUS_UNLOADED);
        }
        sd_model_service_->stop();
        sd_model_service_.reset();
    }

    if (whisper_model_service_) {
        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, whisper_model_service_->get_model_key(), port_, SERVER_MODEL_STATUS_UNLOADED);
        }
        whisper_model_service_->stop();
        whisper_model_service_.reset();
    }


    if (llm_model_service_) {
        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, llm_model_service_->get_model_key(), port_, SERVER_MODEL_STATUS_UNLOADED);
        }
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

    if (HLIEModelType::WHISPER == type) {
        return whisper_model_service_;
    }

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

    if (sd_model_service_) {
        return sd_model_service_->get_state();
    }

    if (whisper_model_service_) {
        return whisper_model_service_->get_state();
    }

    LOG_WARN("model service not initialized");
    return -1;
}

std::vector<std::string> HLIEWorkflow::get_api_keys() const {
    return api_keys_;
}

server_http_res_ptr HLIEWorkflow::handle_health(const server_http_req& req) const {
    if (llm_model_service_) {
        return llm_model_service_->handle_health(req);
    }

    auto res = std::make_unique<server_res_generator_light>();
    res->ok({{"status", "ok"}});
    return res;
}

server_http_res_ptr HLIEWorkflow::handle_models(const server_http_req& req) {
    auto res = std::make_unique<server_res_generator_light>();
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
            {"aliases", meta.aliases},
            {"tags", meta.tags},
            {"object", "model"}, // for OAI-compat
            {"owned_by", "HLIECpp"}, // for OAI-compat
            {"created", t}, // for OAI-compat
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
    auto res = std::make_unique<server_res_generator_light>();
    if (replicas_mode_) {
        // 集群|多实例模式下，不允许外部调用启动模型实例
        res->error(
            format_error_response_ex("In the cluster or multi-instance mode, external calls to start model instances are not allowed.",
                                     ERROR_TYPE_NOT_SUPPORTED, 80005));
        return res;
    }

    auto body = json::parse(req.body);
    const std::string model_type = body.value("model_type", LLM);
    std::string devices = body.value("devices", "undefined");
    std::string model_name = body.value("model", std::string());
    const std::string model_path = body.value("model_path", std::string());
    LOG_DEBUG("start_service devices: {}, model_type: {}, model_name: {}", devices, model_type, model_name);

    if (model_type != HLIEModelType::LLM && model_type != HLIEModelType::SD && model_type != HLIEModelType::WHISPER &&
        model_type != HLIEModelType::COSYVOICE) {
        res->error(format_error_response_ex("Invalid model type.", ERROR_TYPE_INVALID_REQUEST, 80006));
        return res;
    }

    if (model_name.empty() && model_type == SD) {
        model_name = HLIEModelType::SD3;
    }

    std::optional<server_model_meta> meta = std::nullopt;
    int mem_est;
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
        mem_est = HLIEModelInfo::estimate_mem(model_name);
    } else if (model_path.empty()) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return res;
    } else {
        std::filesystem::path model_path_tmp(model_path);
        if (std::error_code ec; !std::filesystem::exists(model_path_tmp, ec) || !std::filesystem::is_regular_file(model_path_tmp, ec)) {
            res->error(format_error_response_ex("model_path file not exist.", ERROR_TYPE_INVALID_REQUEST, 80006));
            return res;
        }

        model_name = model_path_tmp.filename().stem().string();
        meta = get_model_meta(model_name);
        if (meta && meta->status == SERVER_MODEL_STATUS_LOADED) {
            res->error(format_error_response("model is already loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        mem_est = HLIEModelInfo::estimate_mem(model_name);
    }

    LOG_INFO("ready to acquire devices");
    {
        std::unique_lock lock(mutex_);
        if (devices == "undefined") {
            uint32_t device_id = 0;
            if (device_service_->acquire_mem(mem_est, device_id, true) < 0) {
                res->error(format_error_response_ex("The devices is not available.", ERROR_TYPE_UNAVAILABLE, 80004));
                return res;
            }
            devices = std::to_string(device_id);
        } else {
            std::vector<int> devices_vec;
            if (HlieUtil::parse_device(devices, devices_vec) < 0) {
                res->error(format_error_response_ex("The devices parsed failed.", ERROR_TYPE_UNAVAILABLE, 80004));
                return res;
            }
            if (device_service_->acquire_mem_on_devices(devices_vec, mem_est, false, true) < 0) {
                res->error(format_error_response_ex("The devices is not available.", ERROR_TYPE_UNAVAILABLE, 80004));
                return res;
            }
        }
    }

    LOG_INFO("acquire devices successfully with {}", devices);
    auto instance_id = 0;
    std::string json_info_file;
    ProcessID process_id = 0;
    if (model_type == LLM) {
        if (!meta) {
            HLIEModelInfo model_info_tmp;
            model_info_tmp.parse_from_gguf_filename(model_path);
            if (body.contains("params") && body["params"].is_object()) {
                auto& params = body["params"];
                if (std::string mm_proj_path = params.value("mm_proj_path", std::string{}); !mm_proj_path.empty()) {
                    model_info_tmp.is_multi_model_ = true;
                    model_info_tmp.mm_proj_path_ = mm_proj_path;
                    model_info_tmp.vl_spec_ = HLIEModelInfo::parse_mmproj_vl_spec(mm_proj_path);
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
            model_instance.path = model_info_tmp.get_model_path();
            model_instance.path_mmproj = model_info_tmp.get_mm_proj_path();
            model_instance.meta = std::move(model_meta);
            auto model_key = model_info_tmp.get_model_name();
            if (!model_info_tmp.get_mm_proj_path().empty()) {
                model_key += "_" + model_info_tmp.vl_spec_;
            }
            model_info_tmp.model_key_ = model_key;
            model_instance.model_info = model_info_tmp;
            model_instance.key = model_key;
            model_instances_[model_key] = std::move(model_instance);
            LOG_INFO("Service Starting Add Model: {} {}", model_key, model_info_tmp.get_mm_proj_path());
        }

        auto model_inst = get_model_inst(model_name);
        if (!model_inst) {
            res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }

        auto& model_info = model_inst->get().model_info;
        if (model_info.model_version_.empty()) {
            res->error(format_error_response_ex("failed to parse build_version from model name", ERROR_TYPE_INVALID_REQUEST, 80007));
            return res;
        }

        if (!HlieUtil::version_ge(model_info.model_version_, "v1.0.0")) {
            res->error(format_error_response_ex("build_version must be >= v1.0.0", ERROR_TYPE_INVALID_REQUEST, 80007));
            return res;
        }

        model_info.set_devices(devices);

        model_inst->get().meta.status = SERVER_MODEL_STATUS_LOADING;
        if (!llm_model_service_) {
            llm_model_service_ = std::make_shared<LLModelService>();
            if (llm_model_service_->init(model_dir_, model_info, device_service_) < 0) {
                LOG_ERROR("failed to init the LLM model service");
                res->error(format_error_response_ex("failed to init the LLM model service", ERROR_TYPE_SERVER, 80001));
                llm_model_service_->stop();
                llm_model_service_.reset();
                return res;
            }

            llm_model_service_->run();
            model_inst->get().meta.status = SERVER_MODEL_STATUS_LOADED;
            model_inst->get().meta.port = port_;
            if (port_ != master_port_ && !replicas_mode_) {
                update_status(master_port_, llm_model_service_->get_model_key(), port_);
            }
        } else {
            if (port_ != master_port_) {
                LOG_ERROR("slave cannot start new slave");
                res->error(format_error_response_ex("slave cannot start new slave", ERROR_TYPE_SERVER, 80001));
                return res;
            }

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
            model_instances_[HLIEModelType::SD3].meta.status = SERVER_MODEL_STATUS_LOADING;
            if (sd_model_service_->init(model_dir_, model_info, device_service_) < 0) {
                LOG_ERROR("failed to init the SD model service");
                model_instances_[HLIEModelType::SD3].meta.status = SERVER_MODEL_STATUS_UNLOADED;
                res->error(format_error_response_ex("failed to init the SD model service", ERROR_TYPE_SERVER, 80001));
                sd_model_service_->stop();
                sd_model_service_.reset();
                return res;
            }

            sd_model_service_->run();
            model_instances_[HLIEModelType::SD3].meta.status = SERVER_MODEL_STATUS_LOADED;
            model_instances_[HLIEModelType::SD3].meta.port = port_;
            if (port_ != master_port_ && !replicas_mode_) {
                update_status(master_port_, HLIEModelType::SD3, port_);
            }
        } else {
            if (port_ != master_port_) {
                LOG_ERROR("slave cannot start new slave");
                res->error(format_error_response_ex("slave cannot start new slave", ERROR_TYPE_SERVER, 80001));
                return res;
            }

            // 子进程实例参数
            json_info_file = generate_meta_info(model_info, model_dir_, res);
            if (json_info_file.empty()) {
                return res;
            }

            instance_id = generate_instance_id();
            std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                                std::string("--device_id"),   devices,
                                                std::string("--json_info"),   json_info_file};
            if (process_id = Singleton<HLIEProcessManager>::instance()->create_process(program_path_, args_for_sub_process,
                                                                                       {HLIEModelType::SD3}, instance_id);
                process_id == 0 || process_id == -1) {
                LOG_ERROR("Failed to start process for instance_id[{}]", instance_id);
            } else {
                LOG_INFO("Started process PID[{}] instance_id[{}]", process_id, instance_id);
            }
        }
    } else if (model_type == WHISPER) {
        if (start_whisper(meta, res, req, model_name, model_path, devices) < 0) {
            LOG_ERROR("failed to start the WHISPER model");
            return res;
        }
    } else if (model_type == HLIEModelType::COSYVOICE) {
        if (start_cosyvoice(meta, res, req, model_name, model_path, devices) < 0) {
            LOG_ERROR("failed to start the COSYVOICE model");
            return res;
        }
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

int HLIEWorkflow::start_whisper(std::optional<server_model_meta>& meta, std::unique_ptr<server_res_generator_light>& res,
    const server_http_req& req, const std::string& model_name, const std::string& model_path, const std::string& devices) {

    if (!meta) {
        auto body = json::parse(req.body);
        HLIEModelInfo model_info_tmp;
        model_info_tmp.parse_from_gguf_filename(model_path);

        ModelInstance model_instance;
        server_model_meta model_meta;
        model_meta.name = model_info_tmp.get_model_name();
        model_instance.path = model_info_tmp.get_model_path();
        model_instance.meta = std::move(model_meta);
        auto model_key = model_info_tmp.get_model_name();
        model_info_tmp.model_key_ = model_key;
        model_instance.model_info = model_info_tmp;
        model_instance.key = model_key;
        model_instances_[model_key] = std::move(model_instance);
        LOG_INFO("Service Starting Add WHISPER Model: {}", model_key);
    }

    auto model_inst = get_model_inst(model_name);
    if (!model_inst) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return -1;
    }

    auto& model_info = model_inst->get().model_info;
    if (model_info.model_version_.empty()) {
        res->error(format_error_response_ex("failed to parse build_version from model name", ERROR_TYPE_INVALID_REQUEST, 80007));
        return -1;
    }

    if (!HlieUtil::version_ge(model_info.model_version_, "v1.0.0")) {
        res->error(format_error_response_ex("build_version must be >= v1.0.0", ERROR_TYPE_INVALID_REQUEST, 80007));
        return -1;
    }

    model_info.set_devices(devices);

    model_inst->get().meta.status = SERVER_MODEL_STATUS_LOADING;
    const int master_port = Singleton<HLIEConfigService>::instance()->get_int("server", "port", 7901, true);
    if (!whisper_model_service_) {
        whisper_model_service_ = std::make_shared<WhisperModelService>();
        if (whisper_model_service_->init(model_dir_, model_info, device_service_) < 0) {
            LOG_ERROR("failed to init the Whisper model service");
            res->error(format_error_response_ex("failed to init the Whisper model service", ERROR_TYPE_SERVER, 80001));
            whisper_model_service_->stop();
            whisper_model_service_.reset();
            return -1;
        }

        whisper_model_service_->run();
        model_inst->get().meta.status = SERVER_MODEL_STATUS_LOADED;
        model_inst->get().meta.port = port_;
        if (port_ != master_port && !replicas_mode_) {
            update_status(master_port_, whisper_model_service_->get_model_key(), port_);
        }
    } else {
        if (port_ != master_port) {
            LOG_ERROR("slave cannot start new slave");
            res->error(format_error_response_ex("slave cannot start new slave", ERROR_TYPE_SERVER, 80001));
            return -1;
        }

        // 子进程实例参数
        std::string json_info_file = generate_meta_info(model_info, model_dir_, res);
        if (json_info_file.empty()) {
            return -1;
        }

        auto instance_id = generate_instance_id();
        std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                            std::string("--device_id"),   devices,
                                            std::string("--json_info"),   json_info_file};
        const auto& args = Singleton<HLIEConfigService>::instance()->get_args();

        // Copy llama-params
        for (const auto& [key, val] : args) {
            args_for_sub_process.push_back(key);
            args_for_sub_process.push_back(val);
        }

        if (auto pid =
                Singleton<HLIEProcessManager>::instance()->create_process(program_path_, args_for_sub_process, {model_name}, instance_id);
            pid == 0 || pid == -1) {
            LOG_ERROR("Failed to start process for instance_id[{}]", instance_id);
        } else {
            LOG_INFO("Started process PID[{}] instance_id[{}]", pid, instance_id);
        }
    }

    return 0;
}

int HLIEWorkflow::start_cosyvoice(std::optional<server_model_meta>& meta, std::unique_ptr<server_res_generator_light>& res,
    const server_http_req& req, const std::string& model_name, const std::string& model_path, const std::string& devices) {

    if (!meta) {
        auto body = json::parse(req.body);
        HLIEModelInfo model_info_tmp;
        model_info_tmp.parse_from_gguf_filename(model_path);

        ModelInstance model_instance;
        server_model_meta model_meta;
        model_meta.name = model_info_tmp.get_model_name();
        model_instance.path = model_info_tmp.get_model_path();
        model_instance.meta = std::move(model_meta);
        auto model_key = model_info_tmp.get_model_name();
        model_info_tmp.model_key_ = model_key;
        model_instance.model_info = model_info_tmp;
        model_instance.key = model_key;
        model_instances_[model_key] = std::move(model_instance);
        LOG_INFO("Service Starting Add COSYVOICE Model: {}", model_key);
    }

    auto model_inst = get_model_inst(model_name);
    if (!model_inst) {
        res->error(format_error_response("model and model_path not found", ERROR_TYPE_NOT_FOUND));
        return -1;
    }

    auto& model_info = model_inst->get().model_info;
    if (model_info.model_version_.empty()) {
        res->error(format_error_response_ex("failed to parse build_version from model name", ERROR_TYPE_INVALID_REQUEST, 80007));
        return -1;
    }

    if (!HlieUtil::version_ge(model_info.model_version_, "v1.0.0")) {
        res->error(format_error_response_ex("build_version must be >= v1.0.0", ERROR_TYPE_INVALID_REQUEST, 80007));
        return -1;
    }

    model_info.set_devices(devices);

    model_inst->get().meta.status = SERVER_MODEL_STATUS_LOADING;
    const int master_port = Singleton<HLIEConfigService>::instance()->get_int("server", "port", 7901, true);
    if (!cosy_voice_model_service_) {
        cosy_voice_model_service_ = std::make_shared<CosyVoiceModelService>();
        if (cosy_voice_model_service_->init(model_dir_, model_info, device_service_) < 0) {
            LOG_ERROR("failed to init the COSYVOICE model service");
            res->error(format_error_response_ex("failed to init the COSYVOICE model service", ERROR_TYPE_SERVER, 80001));
            cosy_voice_model_service_->stop();
            cosy_voice_model_service_.reset();
            return -1;
        }

        cosy_voice_model_service_->run();
        model_inst->get().meta.status = SERVER_MODEL_STATUS_LOADED;
        model_inst->get().meta.port = port_;
        if (port_ != master_port && !replicas_mode_) {
            update_status(master_port_, cosy_voice_model_service_->get_model_key(), port_);
        }
    } else {
        if (port_ != master_port) {
            LOG_ERROR("slave cannot start new slave");
            res->error(format_error_response_ex("slave cannot start new slave", ERROR_TYPE_SERVER, 80001));
            return -1;
        }

        // 子进程实例参数
        std::string json_info_file = generate_meta_info(model_info, model_dir_, res);
        if (json_info_file.empty()) {
            return -1;
        }

        auto instance_id = generate_instance_id();
        std::vector args_for_sub_process = {std::string("--instance_id"), std::to_string(instance_id),
                                            std::string("--device_id"),   devices,
                                            std::string("--json_info"),   json_info_file};
        const auto& args = Singleton<HLIEConfigService>::instance()->get_args();

        // Copy llama-params
        for (const auto& [key, val] : args) {
            args_for_sub_process.push_back(key);
            args_for_sub_process.push_back(val);
        }

        if (auto pid =
                Singleton<HLIEProcessManager>::instance()->create_process(program_path_, args_for_sub_process, {model_name}, instance_id);
            pid == 0 || pid == -1) {
            LOG_ERROR("Failed to start process for instance_id[{}]", instance_id);
        } else {
            LOG_INFO("Started process PID[{}] instance_id[{}]", pid, instance_id);
        }
    }

    return 0;
}

bool HLIEWorkflow::router_validate_model(const std::string& name, const std::unique_ptr<server_res_generator_light>& res) {
    if (name.empty()) {
        res->error(format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }

    const auto meta = get_model_inst(name);
    if (!meta.has_value()) {
        res->error(format_error_response("model not found", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }

    if (meta->get().meta.status != SERVER_MODEL_STATUS_LOADED) {
        res->error(format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }

    return true;
}

server_http_res_ptr HLIEWorkflow::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, int port) {
    // if (update_last_used) {
    //     std::unique_lock<std::mutex> lk(mutex);
    //     mapping[name].meta.last_used = ggml_time_ms();
    // }
    LOG_INFO("proxying request to model {} on port {}", name.c_str(), port);
    auto proxy = std::make_unique<ServerHttpProxy>(
            method,
            "127.0.0.1",
            port,
            req.path,
            req.headers,
            req.body,
            req.should_stop);
    return proxy;
}

server_http_res_ptr HLIEWorkflow::route_llm_request(const server_http_req& req, const RequestHandler& handler) {
    auto res = std::make_unique<server_res_generator_light>();
    const json body = json::parse(req.body);
    const std::string name = json_value(body, "model", std::string());
    if (name.empty()) {
        res->error(format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const auto inst = get_model_inst(name);
    if (!inst.has_value()) {
        res->error(format_error_response("model not found", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const int master_port = Singleton<HLIEConfigService>::instance()->get_int("server", "port", 7901, true);
    LOG_INFO("LLM Request route info, model [{}] status [{}] port [{}]" , name, static_cast<int>(inst->get().meta.status), inst->get().meta.port);
    if (inst->get().meta.status == SERVER_MODEL_STATUS_LOADED && (inst->get().meta.port == master_port || master_port != port_)) {
        return handler(req);
    }

    const bool auto_load = Singleton<HLIEConfigService>::instance()->get_bool("model", "auto_load", true, true);
    // 如果模型未加载，尝试加载模型，
    if (inst->get().meta.status == SERVER_MODEL_STATUS_UNLOADED) {
        if (auto_load) {
            std::lock_guard lk(slave_mutex_);
            inst->get().meta.status = SERVER_MODEL_STATUS_LOADING;
            auto rest_res = start_service(req);
            if (rest_res.get()->status != 200) {
                // 加载失败
                return rest_res;
            }
        } else {
            res->error(format_error_response("model not loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    const auto inst1 = get_model_inst(name);
    if (inst1 && inst1->get().meta.status == SERVER_MODEL_STATUS_LOADING) {
        std::unique_lock<std::mutex> lk(slave_mutex_);
        slave_cv_.wait(lk, [this, &name]() {
            const auto inst_inner = get_model_inst(name);
            if (inst_inner) {
                return inst_inner->get().meta.status != SERVER_MODEL_STATUS_LOADING;
            }

            return false;
        });
    }

    if (inst1->get().meta.status == SERVER_MODEL_STATUS_UNLOADED) {
        // 加载失败
        res->error("failed to load model: " + name);
        return res;
    }

    // 如果已加载则直接调用
    return proxy_request(req, "POST", name, inst->get().meta.port);
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
                    return route_llm_request(
                        req, std::bind(&LLModelService::handle_chat_completions, llm_model_service_, std::placeholders::_1));
                },
                true, // 需要模型Running
                LLM // LLM模型类型
            },
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
                {"/v1/audio/transcriptions"},
                POST,
                nullptr,
                true,
                WHISPER, // 生图模型类型
                [this](const httplib::Request& req, httplib::Response& res) {
                    if (!whisper_model_service_ || whisper_model_service_->get_state() != SERVER_STATE_READY) {                                                                  \
                        auto error_data =  format_error_response_ex("Model is not available for now.", ERROR_TYPE_UNAVAILABLE, 80004);
                        const json final_response{{"error", error_data}};
                        res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
                        res.status = json_value(error_data, "code", 500);
                        return;
                    }

                    whisper_model_service_->handle_transcriptions(req, res);
                },
            },
        {
                {"/v1/audio/speech"},
                POST,
                nullptr,
                true,
                HLIEModelType::COSYVOICE,
                [this](const httplib::Request& req, httplib::Response& res) {
                    if (!cosy_voice_model_service_ || cosy_voice_model_service_->get_state() != SERVER_STATE_READY) {
                        auto error_data = format_error_response_ex("Model is not available for now.", ERROR_TYPE_UNAVAILABLE, 80004);
                        const json final_response{{"error", error_data}};
                        res.set_content(safe_json_to_str(final_response), MIMETYPE_JSON);
                        res.status = json_value(error_data, "code", 500);
                        return;
                    }

                    cosy_voice_model_service_->handle_speech(req, res);
                },
            },
            {
                {"/hlie/v1/devices"},
                GET,
                [this](const server_http_req& req) { return device_service_->handle_devices(req); },
                false // 不需要模型Running
            },
            {
                {"/v1/completions"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm_request(
                        req, std::bind(&LLModelService::handle_completions_oai, llm_model_service_, std::placeholders::_1));
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/completion", "/completions"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm_request(
                        req, std::bind(&LLModelService::handle_completions, llm_model_service_, std::placeholders::_1));
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/v1/responses", "/responses"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm_request(req, std::bind(&LLModelService::handle_responses, llm_model_service_, std::placeholders::_1));
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/embedding", "/embeddings"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm_request(
                        req, std::bind(&LLModelService::handle_embeddings, llm_model_service_, std::placeholders::_1));
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/v1/embeddings"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm_request(
                        req, std::bind(&LLModelService::handle_embeddings_oai, llm_model_service_, std::placeholders::_1));
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/rerank", "/reranking", "/v1/rerank", "/v1/reranking"},
                POST,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    return route_llm_request(req,
                                             std::bind(&LLModelService::handle_rerank, llm_model_service_, std::placeholders::_1));
                },
                true,
                LLM // LLM模型类型
            },
            {
                {"/props"},
                GET,
                [this](const server_http_req& req) -> server_http_res_ptr {
                    CHECK_SERVICE(llm_model_service_);
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
        model_instances_[llm_model_service_->get_model_key()].meta.port = port;
    }

    if (sd_model_service_) {
        model_instances_[HLIEModelType::SD3].meta.port = port;
    }

    if (whisper_model_service_) {
        model_instances_[whisper_model_service_->get_model_key()].meta.port = port;
    }
}

server_http_res_ptr HLIEWorkflow::stop_service(const server_http_req& req) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    auto res = std::make_unique<server_res_generator_light>();
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

    if (instance_id == 0) {
        std::unique_lock lock(mutex_);
        if (llm_model_service_) {
            model_instances_[llm_model_service_->get_model_key()].meta.status = SERVER_MODEL_STATUS_UNLOADED;
            llm_model_service_->stop();
            llm_model_service_.reset();
        }

        if (sd_model_service_) {
            model_instances_[HLIEModelType::SD3].meta.status = SERVER_MODEL_STATUS_UNLOADED;
            sd_model_service_->stop();
            sd_model_service_.reset();
        }

        if (whisper_model_service_) {
            model_instances_[whisper_model_service_->get_model_key()].meta.status = SERVER_MODEL_STATUS_UNLOADED;
            whisper_model_service_->stop();
            whisper_model_service_.reset();
        }

        LOG_INFO("Succeed to stop service process for instance_id[{}][{}]", instance_id, process_id);
    } else if (process_id != 0) {
        const auto& process_manager = Singleton<HLIEProcessManager>::instance();
        const auto& models = process_manager->get_working_models(instance_id);

        for (const auto& model : models) {
            auto model_inst = get_model_inst(model);
            if (model_inst) {
                model_inst->get().meta.status = SERVER_MODEL_STATUS_UNLOADED;
            }
        }

        if (!process_manager->terminate_process(process_id)) {
            LOG_ERROR("Failed to terminate process for instance_id[{}][{}], maybe has been killed or something", instance_id, process_id);
            const json resp = {{"ret", 80001}, "Failed to terminate process for instance_id, maybe has been killed or something"};
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
        models.push_back(llm_model_service_->get_model_key());
    }

    if (sd_model_service_) {
        models.push_back(HLIEModelType::SD3);
    }

    if (whisper_model_service_) {
        models.push_back(whisper_model_service_->get_model_key());
    }

    if (cosy_voice_model_service_) {
        models.push_back(cosy_voice_model_service_->get_model_key());
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
    auto res = std::make_unique<server_res_generator_light>();
    res->ok(resp);
    return res;
}

server_http_res_ptr HLIEWorkflow::handle_post_status(const server_http_req& req) {
    auto res = std::make_unique<server_res_generator_light>();
    const json body = json::parse(req.body);
    std::string model = json_value(body, "model", std::string());
    const std::string value = json_value(body, "value", std::string());
    const int port = json_value(body, "port", 0);

    const auto status = server_model_status_from_string(value);

    if (const auto inst = get_model_inst(model); inst.has_value()) {
        std::lock_guard lk(mutex_);
        inst->get().meta.status = status;
        inst->get().meta.port = port;
        LOG_INFO("Model[{}] status updated to {}, port: {}", static_cast<int>(status), model, port);

        slave_cv_.notify_all();

        if (status == SERVER_MODEL_STATUS_UNLOADED) { // 更新进程管理中的信息状态
            Singleton<HLIEProcessManager>::instance()->erase_model(model);
        } else if (status == SERVER_MODEL_STATUS_LOADED) {
            const int master_port = Singleton<HLIEConfigService>::instance()->get_int("server", "port", 7901, true);
            const int instance_id = port - master_port;
            Singleton<HLIEProcessManager>::instance()->add_model(model, instance_id);
        }
    }

    res->ok({{"success", true}});
    return res;
}

server_http_res_ptr HLIEWorkflow::handle_unload_model(const server_http_req& req) {
    const auto& config_services = Singleton<HLIEConfigService>::instance();
    auto res = std::make_unique<server_res_generator_light>();
    if (config_services->get_int("server", "replicas", 1) > 1) {
        // 集群|多实例模式下，不允许外部调用启动模型实例
        res->error(
            format_error_response_ex("In the cluster or multi-instance mode, external calls to start model instances are not allowed.",
                                     ERROR_TYPE_NOT_SUPPORTED, 80005));
        return res;
    }

    const auto body = json::parse(req.body);
    const std::string model_type = body.value("model_type", LLM);
    if (model_type != LLM && model_type != SD && model_type != WHISPER && model_type != HLIEModelType::COSYVOICE) {
        res->error(format_error_response_ex("Invalid model type.", ERROR_TYPE_INVALID_REQUEST, 80006));
        return res;
    }

    std::unique_lock lock(mutex_);
    if (model_type == LLM && llm_model_service_) {
        model_instances_[llm_model_service_->get_model_key()].meta.status = SERVER_MODEL_STATUS_UNLOADED;
        llm_model_service_->stop();

        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, llm_model_service_->get_model_key(), port_, SERVER_MODEL_STATUS_UNLOADED);
        }

        llm_model_service_.reset();
    } else if (model_type == SD && sd_model_service_) {
        model_instances_[HLIEModelType::SD3].meta.status = SERVER_MODEL_STATUS_UNLOADED;
        sd_model_service_->stop();

        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, HLIEModelType::SD3, port_, SERVER_MODEL_STATUS_UNLOADED);
        }

        sd_model_service_.reset();
    } else if (model_type == WHISPER && whisper_model_service_) {
        model_instances_[whisper_model_service_->get_model_key()].meta.status = SERVER_MODEL_STATUS_UNLOADED;
        whisper_model_service_->stop();

        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, whisper_model_service_->get_model_key(), port_, SERVER_MODEL_STATUS_UNLOADED);
        }

        whisper_model_service_.reset();
    } else if (model_type == HLIEModelType::COSYVOICE && cosy_voice_model_service_) {
        model_instances_[cosy_voice_model_service_->get_model_key()].meta.status = SERVER_MODEL_STATUS_UNLOADED;
        cosy_voice_model_service_->stop();

        if (is_slave_ && !replicas_mode_) {
            update_status(master_port_, cosy_voice_model_service_->get_model_key(), port_, SERVER_MODEL_STATUS_UNLOADED);
        }

        cosy_voice_model_service_.reset();
    }

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

std::optional<server_model_meta> HLIEWorkflow::get_model_meta(const std::string& model_name) {
    std::unique_lock lk(mutex_);

    if (const auto it = model_instances_.find(model_name); it != model_instances_.end()) {
        return it->second.meta;
    }

    for (const auto& [name, inst] : model_instances_) {
        if (inst.path_mmproj.empty()) {
            continue;
        }

        if (is_equal_before_vit(name, model_name)) {
            return inst.meta;
        }
    }

    return std::nullopt;
}

std::optional<std::reference_wrapper<HLIEWorkflow::ModelInstance>> HLIEWorkflow::get_model_inst(const std::string& model_name) {
    std::unique_lock lk(mutex_);

    if (const auto it = model_instances_.find(model_name); it != model_instances_.end()) {
        return std::ref(it->second);
    }

    for (auto& [name, inst] : model_instances_) {
        if (inst.path_mmproj.empty()) {
            continue;
        }

        if (is_equal_before_vit(name, model_name)) {
            return std::ref(inst);
        }
    }

    return std::nullopt;
}

HLIE_NAMESPACE_END

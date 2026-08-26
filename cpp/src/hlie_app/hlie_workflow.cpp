#include "hlie_app/hlie_workflow.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <map>
#include <optional>
#include <server/server-http.h>
#include <thread>
#include <unordered_set>

#include "INIReader.h"
#include "hlie_app/catalog/hlie_model_catalog.h"
#include "hlie_app/composition/hlie_application_composition_root.h"
#include "hlie_app/http/hlie_worker_status_http_publisher.h"
#include "hlie_app/lifecycle/hlie_application_lifecycle_service.h"
#include "hlie_app/startup/hlie_embedded_startup_service.h"
#include "hlie_app/control/hlie_control_plane_reconciler.h"
#include "hlie_app/control/hlie_model_load_service.h"
#include "hlie_app/control/hlie_model_readiness_service.h"
#include "hlie_app/discovery/hlie_model_discovery_service.h"
#include "hlie_app/asr/hlie_asr_application_service.h"
#include "hlie_app/speech/hlie_speech_application_service.h"
#include "hlie_app/startup/hlie_default_model_initialization_service.h"
#include "hlie_app/inference/hlie_admission_guarded_response.h"
#include "hlie_app/document/hlie_document_application_service.h"
#include "hlie_app/inference/hlie_inference_http_error_mapper.h"
#include "hlie_app/http/hlie_control_http_error_mapper.h"
#include "hlie_core/resource/hlie_model_residency_manager.h"
#include "hlie_app/hlie_model_service_factory.h"
#include "hlie_app/hlie_process_manager.h"
#include "hlie_app/runtime/hlie_cancellation_metrics.h"
#include "hlie_app/hlie_replica_config.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_exception.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_gguf_bundle.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

#include "hlie_app/hlie_device_service.h"

HLIE_NAMESPACE_BEGIN

namespace {
    std::string normalize_model_name(const std::string& name) {
        const auto first = std::find_if_not(name.begin(), name.end(), [](const unsigned char c) { return std::isspace(c); });
        const auto last = std::find_if_not(name.rbegin(), name.rend(), [](const unsigned char c) { return std::isspace(c); }).base();
        if (first >= last) {
            return {};
        }

        std::string normalized(first, last);
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return normalized;
    }

    std::string trim_copy(const std::string& value) {
        const auto first = std::find_if_not(value.begin(), value.end(), [](const unsigned char c) { return std::isspace(c); });
        const auto last = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char c) { return std::isspace(c); }).base();
        if (first >= last) {
            return {};
        }
        return {first, last};
    }

    std::vector<std::string> split_model_list(const std::string& value) {
        std::vector<std::string> result;
        std::stringstream ss(value);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim_copy(token);
            if (!token.empty()) {
                result.push_back(token);
            }
        }
        return result;
    }

    HLIEModelResidencyPolicy build_residency_policy() {
        const auto& config = Singleton<HLIEConfigService>::instance();
        HLIEModelResidencyPolicy policy;
        policy.auto_unload = config->get_bool("model", "auto_unload", false, true);
        policy.idle_ttl_sec = config->get_int("model", "idle_ttl_sec", 0, true);
        policy.max_loaded_models = config->get_int("model", "max_loaded_models", 0, true);
        policy.default_model = config->get_string("model", "default", "", true);

        const auto pinned = config->get_string("model", "pinned_models", "", true);
        for (const auto& model : split_model_list(pinned)) {
            policy.pinned_models.insert(model);
        }

        return policy;
    }

    ProxyTimeoutOptions build_proxy_timeout_options() {
        const auto& config = Singleton<HLIEConfigService>::instance();
        ProxyTimeoutOptions options;
        options.connect_timeout_ms = config->get_int(
            "proxy", "connect_timeout_ms", options.connect_timeout_ms, true);
        options.non_stream_read_timeout_sec = config->get_int(
            "proxy", "non_stream_read_timeout_sec", options.non_stream_read_timeout_sec, true);
        options.stream_idle_timeout_sec = config->get_int(
            "proxy", "stream_idle_timeout_sec", options.stream_idle_timeout_sec, true);
        options.write_timeout_sec = config->get_int(
            "proxy", "write_timeout_sec", options.write_timeout_sec, true);
        options.max_request_timeout_sec = config->get_int(
            "proxy", "max_request_timeout_sec", options.max_request_timeout_sec, true);
        return options.normalized();
    }

    bool json_boolish(const json& body, const std::string& key) {
        if (!body.contains(key) || body.at(key).is_null()) {
            return false;
        }
        const auto& value = body.at(key);
        if (value.is_boolean()) {
            return value.get<bool>();
        }
        if (value.is_string()) {
            const std::string s = normalize_model_name(value.get<std::string>());
            return s == "true" || s == "1" || s == "yes";
        }
        return false;
    }

    std::vector<HLIEModelInfo> load_sd_model_manifests_from_directory(const std::string& directory_path) {
        std::vector<HLIEModelInfo> model_infos;

        try {
            // Check whether the model info directory exists.
            if (!std::filesystem::exists(directory_path)) {
                LOG_ERROR("Directory does not exist: {}", directory_path);
                return model_infos;
            }

            if (!std::filesystem::is_directory(directory_path)) {
                LOG_ERROR("Path is not a directory: {}", directory_path);
                return model_infos;
            }

            // Iterate all model info files in the directory.
            for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
                // Only parse JSON files.
                if (entry.is_regular_file() && entry.path().extension() == ".json") {
                    const std::string file_path = entry.path().string();
                    LOG_INFO("Parsing model info from file: {}", file_path);

                    if (HLIEModelInfo model_info; model_info.parse_from_file(file_path) == 0) {
                        if (model_info.get_service_kind() != "sd") {
                            LOG_WARN("Ignoring non-SD JSON model manifest; non-SD models must be discovered from GGUF: {}",
                                     file_path);
                            continue;
                        }
                        model_infos.push_back(std::move(model_info));
                        LOG_INFO("Successfully parsed SD model manifest from: {}", file_path);
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

        LOG_INFO("Loaded {} SD model manifest files from directory: {}", model_infos.size(), directory_path);
        return model_infos;
    }

    bool validate_discovered_gguf(const std::string& model_dir, const std::string& configured_path) {
        if (configured_path.empty()) {
            return true;
        }

        std::filesystem::path path(configured_path);
        std::string extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension != ".gguf") {
            return true;
        }
        if (path.is_relative()) {
            path = std::filesystem::path(model_dir) / path;
        }

        HLIEGgufBundle bundle;
        std::string error;
        if (!bundle.open(path.string(), &error)) {
            LOG_WARN("Ignoring incomplete or invalid GGUF during discovery: path={}, reason={}",
                     path.string(), error);
            return false;
        }
        return true;
    }

    bool is_discovered_model_complete(const std::string& model_dir, const HLIEModelInfo& model_info) {
        return validate_discovered_gguf(model_dir, model_info.get_model_path()) &&
               validate_discovered_gguf(model_dir, model_info.get_mm_proj_path());
    }

    std::optional<HLIEModelLoadRequest> parse_model_load_request(
        const json& body,
        const std::unique_ptr<server_res_generator_light>& response) {
        HLIEModelLoadRequest request;
        request.model_type = body.value("model_type", std::string{LLM});
        request.model = body.value("model", std::string{});
        request.model_path = body.value("model_path", std::string{});

        const std::string devices = body.value("devices", "undefined");
        if (devices != "undefined") {
            std::vector<int> parsed_devices;
            if (HlieUtil::parse_device(devices, parsed_devices) < 0) {
                response->error(format_error_response_ex(
                    "The devices parsed failed.", ERROR_TYPE_UNAVAILABLE,
                    80004));
                return std::nullopt;
            }
            request.requested_device_ids = std::move(parsed_devices);
        }

        if (body.contains("params") && body.at("params").is_object()) {
            const auto& params = body.at("params");
            request.mmproj_path = params.value(
                "mm_proj_path", std::string{});
            if (request.model_type == SD) {
                const char* required_fields[] = {
                    "vae_model", "t5_model", "clip_model",
                    "clip_l_model", "mmdit_model"};
                for (const auto* field : required_fields) {
                    if (!params.contains(field)) {
                        response->error({
                            {"ret", 80007},
                            {"msg", "params invalid, miss someone"}});
                        return std::nullopt;
                    }
                }
                request.sd_artifacts = HLIESdModelArtifacts{
                    params.at("vae_model").get<std::string>(),
                    params.at("t5_model").get<std::string>(),
                    params.at("clip_model").get<std::string>(),
                    params.at("clip_l_model").get<std::string>(),
                    params.at("mmdit_model").get<std::string>()};
            }
        } else if (request.model_type == SD) {
            response->error(format_error_response_ex(
                "params invalid for SD3", ERROR_TYPE_INVALID_REQUEST,
                80007));
            return std::nullopt;
        }
        return request;
    }

    bool is_equal_before_vit(const std::string& full_name, const std::string& name) {
        if (const size_t pos = full_name.find("_vit_"); pos != std::string::npos) {
            const std::string before_vit = full_name.substr(0, pos);
            return before_vit == name;
        }

        if (const size_t pos = full_name.find("_unknown"); pos != std::string::npos) {
            const std::string before_unknown = full_name.substr(0, pos);
            return before_unknown == name;
        }

        // If _vit_ is absent, compare the full name directly.
        return full_name == name;
    }

} // namespace

HLIEWorkflow::HLIEWorkflow() = default;
HLIEWorkflow::~HLIEWorkflow() = default;

int HLIEWorkflow::initialize(const std::string& model_dir, const std::string& program_path) {
    model_dir_ = model_dir;
    program_path_ = program_path;
    if (model_dir.empty()) {
        LOG_ERROR("model dir is empty");
        return -1;
    }
    replicas_mode_ = HLIEReplicaConfig::load().enabled();
    master_port_ = Singleton<HLIEConfigService>::instance()->get_int(
        "server", "port", 7901, true);

    HLIEApplicationCompositionContext composition_context{
        model_service_manager_, runtime_directory_, model_dir_, program_path_,
        build_residency_policy()};
    composition_context.supervisor_provider = [this] {
        return worker_supervisor_.get();
    };
    composition_context.supervisor_ensurer = [this] {
        if (!worker_supervisor_) {
            worker_supervisor_ = std::make_unique<HLIEWorkerSupervisor>(
                *Singleton<HLIEProcessManager>::instance(),
                HLIEWorkerSupervisor::configured_options(master_port_),
                static_cast<IHLIERuntimeEventSink&>(*model_controller_));
        }
        return worker_supervisor_.get();
    };
    composition_context.embedded_identity_allocator =
        [this](const HLIEModelLaunchTicket& ticket) {
            return runtime_directory_.allocate_embedded_identity(ticket);
        };
    composition_context.local_port_provider = [this] { return port_; };
    composition_context.master_port_provider = [this] {
        return master_port_;
    };
    composition_context.auto_load_provider = [] {
        return Singleton<HLIEConfigService>::instance()->get_bool(
            "model", "auto_load", true, true);
    };
    composition_context.model_list_process_provider = [this] {
        std::vector<HLIEModelListProcessSnapshot> result;
        const auto process_infos =
            Singleton<HLIEProcessManager>::instance()->get_running_processes();
        result.reserve(process_infos.size());
        for (const auto& process : process_infos) {
            result.push_back({
                process.instance_id,
                static_cast<std::uint64_t>(process.process_id),
                process.instance_id + port_,
                process.model_names,
                process.device_ids,
            });
        }
        return result;
    };
    composition_context.model_list_local_provider = [this] {
        std::vector<HLIEModelListLocalInstance> result;
        for (const auto& instance : get_embedded_model_instances()) {
            std::vector<int> devices;
            if (const auto it = instance.meta.loaded_info.find("device_ids");
                it != instance.meta.loaded_info.end() && it->is_array()) {
                try {
                    devices = it->get<std::vector<int>>();
                } catch (...) {
                    LOG_WARN("Ignoring malformed runtime device snapshot: model={}",
                             instance.key);
                }
            }
            result.push_back({instance.key, std::move(devices)});
        }
        return result;
    };
    composition_context.replicas_mode_provider = [this] {
        return replicas_mode_;
    };
    composition_context.loaded_model_by_type_provider = [this](
        const std::string& model_type) -> std::optional<std::string> {
        if (!model_controller_) {
            return std::nullopt;
        }
        for (const auto& instance : model_controller_->get_all_instances()) {
            if (instance.meta.status != SERVER_MODEL_STATUS_LOADED &&
                instance.meta.status != SERVER_MODEL_STATUS_SLEEPING) {
                continue;
            }
            const auto actual_kind = instance.model_info.get_service_kind();
            const bool matches = actual_kind == model_type ||
                (model_type == HLIEModelType::LLM &&
                 HLIEModelServiceFactory::uses_llm_primary_service(
                     instance.model_info)) ||
                (model_type == HLIEModelType::TTS &&
                 actual_kind == HLIEModelType::COSYVOICE);
            if (matches) {
                return instance.key;
            }
        }
        return std::nullopt;
    };
    composition_context.model_unloaded_callback = [this](
        const std::string& model_key) {
        if (worker_control_service_) {
            (void)worker_control_service_->publish_unloaded(
                model_key, worker_launch_ticket_);
        }
    };
    composition_context.stop_service_worker_provider = [this](
        const std::uint64_t process_id)
        -> std::optional<HLIEStopServiceWorkerSnapshot> {
        const auto process = Singleton<HLIEProcessManager>::instance()->
            get_process(static_cast<ProcessID>(process_id));
        if (!process) {
            return std::nullopt;
        }
        return HLIEStopServiceWorkerSnapshot{
            static_cast<std::uint64_t>(process->process_id),
            process->instance_id,
            master_port_ + process->instance_id,
            process->model_names,
            process->launch_ticket,
        };
    };
    composition_context.observability_cancellation_lag_provider = [] {
        std::vector<HLIEObservabilityCancellationLag> result;
        const auto& metrics = HLIECancellationMetrics::instance();
        for (const auto lane : {HLIEBackendCancellationLane::WHISPER,
                               HLIEBackendCancellationLane::COSYVOICE,
                               HLIEBackendCancellationLane::STABLE_DIFFUSION}) {
            for (const auto reason : {
                     HLIEQueuedRequestStopReason::CANCELLED,
                     HLIEQueuedRequestStopReason::DEADLINE}) {
                const auto snapshot = metrics.snapshot(lane, reason);
                result.push_back({
                    std::string(HLIECancellationMetrics::lane_name(lane)),
                    std::string(HLIECancellationMetrics::reason_name(reason)),
                    snapshot.observations,
                    snapshot.total_lag_ms,
                    snapshot.max_lag_ms,
                });
            }
        }
        return result;
    };
    composition_context.worker_spawn_policy = [this] {
        return port_ == master_port_;
    };
    composition_context.proxy_timeout_provider = [] {
        return build_proxy_timeout_options();
    };
    composition_context.draining_provider = [this] {
        return is_draining();
    };
    composition_context.residency_policy_provider = [] {
        return build_residency_policy();
    };
    composition_context.external_live_owner_provider = [] {
        std::vector<HLIEModelLaunchTicket> owners;
        const auto running_processes =
            Singleton<HLIEProcessManager>::instance()->get_running_processes();
        owners.reserve(running_processes.size());
        for (const auto& process : running_processes) {
            if (process.launch_ticket.valid()) {
                owners.push_back(process.launch_ticket);
            }
        }
        return owners;
    };
    composition_context.discovery_options_provider = [] {
        const auto& config = Singleton<HLIEConfigService>::instance();
        HLIEModelDiscoveryOptions options;
        options.scan_interval = std::chrono::seconds(config->get_int(
            "model", "discovery_scan_interval_sec", 30, true));
        options.stability_window = std::chrono::seconds(config->get_int(
            "model", "discovery_stability_sec", 10, true));
        options.scan_on_request = config->get_bool(
            "model", "discovery_scan_on_models_request", false, true);
        options.request_cooldown = std::chrono::seconds(config->get_int(
            "model", "discovery_request_cooldown_sec", 5, true));
        options.request_wait_timeout = std::chrono::seconds(config->get_int(
            "model", "discovery_request_wait_sec", 15, true));
        return options;
    };
    composition_context.prepare_catalog_update =
        [this]() -> std::optional<HLIEModelDiscoveryService::PreparedCatalogUpdate> {
            auto model_infos = std::make_shared<std::vector<HLIEModelInfo>>(
                load_sd_model_manifests_from_directory(model_dir_));
            HLIEModelInfo::ScanLocalModelsFromDir(*model_infos, model_dir_);
            return HLIEModelDiscoveryService::PreparedCatalogUpdate{
                [this, model_infos = std::move(model_infos)] {
                    int discovered_count = 0;
                    for (auto& model_info : *model_infos) {
                        if (!is_discovered_model_complete(model_dir_, model_info)) {
                            continue;
                        }
                        if (model_controller_->register_model_if_absent(
                                model_info, model_info.get_estimated_mem())) {
                            ++discovered_count;
                        }
                    }
                    return discovered_count;
                }};
        };
    composition_context.bootstrap_prepare_launch_configuration = [this] {
        return initialize_llm_preset_catalogs();
    };
    composition_context.bootstrap_model_provider = [this] {
        auto models = load_sd_model_manifests_from_directory(model_dir_);
        HLIEModelInfo::ScanLocalModelsFromDir(models, model_dir_);
        models.erase(
            std::remove_if(
                models.begin(), models.end(),
                [this](const HLIEModelInfo& model) {
                    return !is_discovered_model_complete(model_dir_, model);
                }),
            models.end());
        return models;
    };
    composition_context.replicas_mode = replicas_mode_;
    composition_context.llm_parameter_resolver =
        [this](HLIEModelInfo& model_info) {
            return resolve_llm_launch_params(model_info);
        };
    composition_context.model_load_succeeded =
        [this](const HLIEModelLoadReceipt& receipt) {
            if (receipt.ready &&
                receipt.runtime_kind == HLIERuntimeKind::Embedded) {
                notify_local_model_started(
                    receipt.notify_model_key,
                    receipt.enable_asr_websocket);
            }
        };
    composition_context.llm_admission = &llm_admission_;
    composition_context.qwen3_tts_admission = &qwen3_tts_admission_;
    composition_context.asr_realtime_listener_allowed = [this] {
        return !is_draining() && !(is_slave_ && !replicas_mode_);
    };
    composition_context.embedded_startup_ready =
        [this](const std::string& model_key) {
            schedule_slave_loaded_status(model_key);
        };
    composition_context.worker_status_publisher =
        std::make_unique<HLIEHttpWorkerStatusPublisher>();
    composition_context.worker_control_dependencies.dynamic_worker_provider =
        [this] { return is_slave_; };
    composition_context.worker_control_dependencies.replicas_mode_provider =
        [this] { return replicas_mode_; };
    composition_context.worker_control_dependencies.local_port_provider =
        [this] { return port_; };
    composition_context.worker_control_dependencies.master_port_provider =
        [this] { return master_port_; };
    composition_context.worker_control_dependencies.launch_ticket_provider =
        [this] { return worker_launch_ticket_; };
    composition_context.worker_control_dependencies.local_start_time_provider =
        [this] { return start_time_; };
    composition_context.worker_control_dependencies.local_models_provider =
        [this] { return get_public_local_model_names(); };
    composition_context.worker_control_dependencies.worker_processes_provider =
        [this] { return get_worker_process_snapshots(); };
    composition_context.worker_control_dependencies.erase_model =
        [](const std::string& model_key) {
            Singleton<HLIEProcessManager>::instance()->erase_model(model_key);
        };
    composition_context.worker_control_dependencies.begin_draining =
        [this] { begin_draining(); };
    composition_context.worker_control_dependencies.process_shutdown =
        [this] {
            if (process_shutdown_callback_) {
                process_shutdown_callback_();
            }
        };
    composition_context.worker_control_dependencies.process_shutdown_available =
        [this] { return static_cast<bool>(process_shutdown_callback_); };

    auto composition = HLIEApplicationCompositionRoot::compose(
        std::move(composition_context));
    if (!composition) {
        return -1;
    }
    device_service_ = std::move(composition->device_service);
    model_catalog_ = std::move(composition->model_catalog);
    resource_scheduler_ = std::move(composition->resource_scheduler);
    runtime_factory_ = std::move(composition->runtime_factory);
    backend_launcher_ = std::move(composition->backend_launcher);
    model_controller_ = std::move(composition->model_controller);
    inference_endpoint_resolver_ =
        std::move(composition->inference_endpoint_resolver);
    inference_dispatcher_ = std::move(composition->inference_dispatcher);
    control_plane_reconciler_ =
        std::move(composition->control_plane_reconciler);
    model_discovery_service_ =
        std::move(composition->model_discovery_service);
    observability_application_service_ =
        std::move(composition->observability_application_service);
    model_load_service_ = std::move(composition->model_load_service);
    model_readiness_service_ =
        std::move(composition->model_readiness_service);
    model_list_application_service_ =
        std::move(composition->model_list_application_service);
    model_unload_application_service_ =
        std::move(composition->model_unload_application_service);
    stop_service_application_service_ =
        std::move(composition->stop_service_application_service);
    document_application_service_ =
        std::move(composition->document_application_service);
    llm_application_service_ =
        std::move(composition->llm_application_service);
    asr_application_service_ =
        std::move(composition->asr_application_service);
    speech_application_service_ =
        std::move(composition->speech_application_service);
    default_model_initialization_service_ =
        std::move(composition->default_model_initialization_service);
    bootstrap_application_service_ =
        std::move(composition->bootstrap_application_service);
    embedded_startup_target_provider_ =
        std::move(composition->embedded_startup_target_provider);
    embedded_startup_service_ =
        std::move(composition->embedded_startup_service);
    worker_status_publisher_ =
        std::move(composition->worker_status_publisher);
    worker_control_service_ =
        std::move(composition->worker_control_service);
    worker_control_application_service_ =
        std::move(composition->worker_control_application_service);
    application_lifecycle_service_ =
        std::move(composition->application_lifecycle_service);
    initialize_api_router();

    if (!bootstrap_application_service_) {
        LOG_ERROR("Bootstrap application service is unavailable");
        return -1;
    }
    const auto& config = Singleton<HLIEConfigService>::instance();
    const auto bootstrap = bootstrap_application_service_->initialize({
        model_dir_,
        config->get_args(),
        config->get_string("model", "default"),
        config->get_string("model", "default_mmproj"),
        config->get_bool("model", "enable_sd3", false),
    });
    worker_launch_ticket_ = bootstrap.launch_ticket;
    is_slave_ = bootstrap.worker_process();
    if (bootstrap.started()) {
        start_time_ = time(nullptr);
    }
    return bootstrap.legacy_return_code();
}

bool HLIEWorkflow::run() {
    const auto& config_service = Singleton<HLIEConfigService>::instance();
    if (!application_lifecycle_service_) {
        LOG_ERROR("Application lifecycle service is unavailable");
        return false;
    }
    const auto started = application_lifecycle_service_->start({
        is_slave_,
        calculate_http_thread_count(),
        config_service->get_int(
            "admission", "control_plane_reserved_threads", 2, true),
        config_service->get_int(
            "admission", "llm_max_in_flight", 64, true),
        config_service->get_int(
            "admission", "qwen3_tts_max_in_flight", 1, true),
    });
    if (!started.started()) {
        LOG_ERROR("Application lifecycle startup did not reach Running: status={}",
                  static_cast<int>(started.status));
        (void)application_lifecycle_service_->stop();
        return false;
    }
    return true;
}

bool HLIEWorkflow::publish_serving(
    const std::function<bool()>& publisher) {
    if (!application_lifecycle_service_) {
        LOG_ERROR("Application lifecycle service is unavailable");
        return false;
    }
    return application_lifecycle_service_->try_publish_serving(publisher)
        .published();
}

void HLIEWorkflow::begin_draining() {
    if (application_lifecycle_service_) {
        application_lifecycle_service_->begin_draining();
    }
}

void HLIEWorkflow::stop() {
    if (application_lifecycle_service_) {
        (void)application_lifecycle_service_->stop();
    }
}

bool HLIEWorkflow::is_draining() const {
    return application_lifecycle_service_ &&
        application_lifecycle_service_->draining();
}

void HLIEWorkflow::schedule_slave_loaded_status(const std::string& model_key) {
    if (!worker_control_service_ || !is_slave_ || replicas_mode_) {
        return;
    }

    (void)model_controller_->update_port(model_key, port_);
    (void)worker_control_service_->schedule_loaded(model_key,
                                                   worker_launch_ticket_);
}

int HLIEWorkflow::calculate_http_thread_count() const {
    const auto hardware_threads = std::thread::hardware_concurrency();
    const int baseline = hardware_threads > 1
        ? static_cast<int>(hardware_threads - 1)
        : 2;
    if (model_controller_) {
        int active_batch_num = 0;
        for (const auto& instance : model_controller_->get_all_instances()) {
            // The Controller also exposes catalog entries that are not loaded.
            // Only a live or in-flight model should influence HTTP admission;
            // otherwise zero-load startup inherits an arbitrary catalog batch.
            if (instance.meta.status != SERVER_MODEL_STATUS_LOADING &&
                instance.meta.status != SERVER_MODEL_STATUS_LOADED &&
                instance.meta.status != SERVER_MODEL_STATUS_SLEEPING) {
                continue;
            }
            if (HLIEModelServiceFactory::uses_llm_primary_service(
                    instance.model_info)) {
                active_batch_num = std::max(active_batch_num,
                                            instance.model_info.get_batch_num());
            }
        }
        if (active_batch_num > 0) {
            return std::max({2, active_batch_num, baseline});
        }
    }

    return std::max(2, baseline);
}

int HLIEWorkflow::get_model_service_state() const {
    if (model_controller_) {
        bool ready = false;
        for (const auto& instance : model_controller_->get_all_instances()) {
            if (instance.meta.status == SERVER_MODEL_STATUS_LOADING) {
                return SERVER_STATE_LOADING_MODEL;
            }
            ready = ready || instance.meta.status == SERVER_MODEL_STATUS_LOADED ||
                instance.meta.status == SERVER_MODEL_STATUS_SLEEPING;
        }
        if (ready) {
            return SERVER_STATE_READY;
        }
    }

    LOG_DEBUG("model service not initialized");
    return -1;
}

bool HLIEWorkflow::authorize_api_key(const std::string& api_key) const {
    return std::find(api_keys_.begin(), api_keys_.end(), api_key) !=
        api_keys_.end();
}

server_http_res_ptr HLIEWorkflow::handle_model_discovery_refresh(const server_http_req& req) {
    auto res = std::make_unique<server_res_generator_light>();
    if (is_draining()) {
        res->error(format_error_response_ex(
            "service is draining", ERROR_TYPE_UNAVAILABLE, 503));
        return res;
    }
    json body = json::object();
    if (!req.body.empty()) {
        body = json::parse(req.body);
    }

    const bool wait = !body.contains("wait") || json_boolish(body, "wait");
    const uint64_t generation = model_discovery_service_
        ? model_discovery_service_->request_scan(true)
        : 0;
    if (generation == 0) {
        res->ok({
            {"code", 0},
            {"generation", 0},
            {"status", "skipped"},
        });
        return res;
    }

    const bool completed = wait && model_discovery_service_ &&
        model_discovery_service_->wait_for(generation);
    res->ok({
        {"code", 0},
        {"generation", generation},
        {"status", wait ? (completed ? "completed" : "timeout") : "queued"},
    });
    return res;
}

server_http_res_ptr HLIEWorkflow::start_service(
    const server_http_req& req,
    const std::optional<HLIEModelLaunchTicket>& preclaimed_launch) {
    auto response = std::make_unique<server_res_generator_light>();
    if (!model_load_service_) {
        if (preclaimed_launch) {
            model_controller_->rollback_model_start(
                *preclaimed_launch, "model_load_service_unavailable");
        }
        response->error(format_error_response_ex(
            "model load service is unavailable", ERROR_TYPE_SERVER, 80001));
        return response;
    }

    const auto body = json::parse(req.body);
    auto request = parse_model_load_request(body, response);
    if (!request) {
        if (preclaimed_launch) {
            model_controller_->rollback_model_start(
                *preclaimed_launch, "model_load_request_invalid");
        }
        return response;
    }

    auto result = model_load_service_->load(
        std::move(*request), port_, preclaimed_launch);
    if (const auto* error = result.error_if()) {
        return HLIEControlHttpErrorMapper::map(*error);
    }

    const auto* receipt = result.value_if();
    if (!receipt) {
        response->error(format_error_response_ex(
            "model load returned no result", ERROR_TYPE_SERVER, 80001));
        return response;
    }
    if (receipt->ready &&
        receipt->runtime_kind == HLIERuntimeKind::Embedded) {
        notify_local_model_started(
            receipt->notify_model_key, receipt->enable_asr_websocket);
    }

    response->ok({
        {"code", 0},
        {"port", receipt->port},
        {"instance_id", receipt->instance_id},
        {"process_id", receipt->process_id},
        {"json_info_file", receipt->launch_manifest_path},
    });
    return response;
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

    if (meta->meta.status != SERVER_MODEL_STATUS_LOADED) {
        res->error(format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }

    return true;
}

server_http_res_ptr HLIEWorkflow::dispatch_inference(
    const std::string& model_key,
    const HLIEInferenceOperation operation,
    const server_http_req& req,
    const std::string& method) {
    auto res = std::make_unique<server_res_generator_light>();
    if (!inference_dispatcher_) {
        res->error(format_error_response_ex(
            "inference dispatcher is unavailable", ERROR_TYPE_UNAVAILABLE, 80004));
        return res;
    }
    return HLIEInferenceHttpErrorMapper::map(inference_dispatcher_->dispatch(
        model_key, operation, req, method));
}

std::optional<std::string> HLIEWorkflow::find_ready_model_key(
    const std::string& service_kind,
    const std::optional<HLIERuntimeKind> runtime_kind) const {
    if (!model_controller_) {
        return std::nullopt;
    }
    for (const auto& instance : model_controller_->get_all_instances()) {
        const auto& actual_kind = instance.model_info.get_service_kind();
        const bool service_kind_matches = actual_kind == service_kind ||
            ((service_kind == HLIEModelType::COSYVOICE ||
              service_kind == HLIEModelType::TTS) &&
             (actual_kind == HLIEModelType::COSYVOICE ||
              actual_kind == HLIEModelType::TTS));
        if (instance.meta.status == SERVER_MODEL_STATUS_LOADED &&
            service_kind_matches) {
            if (runtime_kind) {
                const auto ticket = model_controller_->get_launch_ticket(
                    instance.key);
                const auto runtime = ticket
                    ? runtime_directory_.find(*ticket) : nullptr;
                if (!runtime || runtime->identity().kind != *runtime_kind) {
                    continue;
                }
            }
            return instance.key;
        }
    }
    return std::nullopt;
}

void HLIEWorkflow::initialize_api_router() {
    HLIEApiRouteHandlers handlers;
    handlers.live = [this](const server_http_req& req) {
        (void)req;
        return observability_application_service_->handle_live();
    };
    handlers.ready = [this](const server_http_req& req) {
        (void)req;
        return observability_application_service_->handle_ready();
    };
    handlers.metrics = [this](const server_http_req& req) {
        (void)req;
        return observability_application_service_->handle_metrics();
    };
    handlers.health = [this](const server_http_req& req) {
        (void)req;
        return observability_application_service_->handle_health();
    };
    handlers.models = [this](const server_http_req& req) {
        (void)req;
        return model_list_application_service_->handle();
    };
    handlers.refresh_models = [this](const server_http_req& req) {
        return handle_model_discovery_refresh(req);
    };
    handlers.llm = [this](const server_http_req& req,
                          const HLIEInferenceOperation operation,
                          const std::string& method) {
        return llm_application_service_->handle(req, operation, method);
    };
    handlers.image_generations = [this](const server_http_req& req) {
        auto validation = std::make_unique<server_res_generator_light>();
        const json body = json::parse(req.body);
        const std::string requested_model = json_value(
            body, "model", std::string());
        if (!router_validate_model(requested_model, validation)) {
            return server_http_res_ptr(std::move(validation));
        }
        const auto model_key = find_ready_model_key(
            HLIEModelType::SD, HLIERuntimeKind::Embedded);
        if (!model_key) {
            auto response = std::make_unique<server_res_generator_light>();
            response->error(format_error_response(
                "Model is not available for now.", ERROR_TYPE_UNAVAILABLE));
            return server_http_res_ptr(std::move(response));
        }
        return dispatch_inference(
            *model_key, HLIEInferenceOperation::ImageGenerations, req);
    };
    handlers.transcriptions = [this](const server_http_req& req) {
        if (!asr_application_service_) {
            auto response = std::make_unique<server_res_generator_light>();
            response->error(format_error_response_ex(
                "ASR application service is unavailable",
                ERROR_TYPE_UNAVAILABLE, 80004));
            return server_http_res_ptr(std::move(response));
        }
        return asr_application_service_->handle_transcriptions(req);
    };
    handlers.speech = [this](const httplib::Request& req,
                             httplib::Response& res) {
        if (!speech_application_service_) {
            const json error{{"error", {
                {"code", 503},
                {"message", "speech application service is unavailable"},
                {"type", "unavailable_error"},
                {"ret", 80004},
            }}};
            res.status = 503;
            res.set_content(error.dump(), MIMETYPE_JSON);
            return;
        }
        speech_application_service_->handle(req, res);
    };
    handlers.devices = [this](const server_http_req& req) {
        return device_service_->handle_devices(req);
    };
    handlers.properties = [this](const server_http_req& req) {
        return llm_application_service_->handle_properties(req);
    };
    handlers.slots = [this](const server_http_req& req) {
        return llm_application_service_->handle_slots(req);
    };
    handlers.start_model = [this](const server_http_req& req) {
        return start_service(req);
    };
    handlers.stop_model = [this](const server_http_req& req) {
        return stop_service_application_service_->handle(req);
    };
    handlers.get_status = [this](const server_http_req& req) {
        (void)req;
        return worker_control_application_service_->handle_get_status();
    };
    handlers.post_status = [this](const server_http_req& req) {
        return worker_control_application_service_->handle_post_status(req);
    };
    handlers.worker_shutdown = [this](const server_http_req& req) {
        return worker_control_application_service_->handle_shutdown(req);
    };
    handlers.qwen3_tts_debug = [this](const server_http_req& req,
                                      const HLIEInferenceOperation operation) {
        auto permit = qwen3_tts_admission_.try_acquire();
        if (!permit) {
            throw InferenceQueueFullException(
                "Qwen3-TTS admission limit reached");
        }
        const auto model_key = find_ready_model_key(
            HLIEModelType::QWEN3_TTS, HLIERuntimeKind::Embedded);
        if (!model_key) {
            auto response = std::make_unique<server_res_generator_light>();
            response->error(format_error_response(
                "Model is not available for now.", ERROR_TYPE_UNAVAILABLE));
            return server_http_res_ptr(std::move(response));
        }
        auto response = dispatch_inference(*model_key, operation, req);
        return attach_admission_permit(std::move(response), permit);
    };
    handlers.unload_model = [this](const server_http_req& req) {
        return model_unload_application_service_->handle(req);
    };
    api_router_ = std::make_unique<HLIEApiRouter>(std::move(handlers));
}

std::vector<HLIEHttpRouteConfig> HLIEWorkflow::get_route_configs() {
    return api_router_ ? api_router_->get_route_configs()
                       : std::vector<HLIEHttpRouteConfig>{};
}
void HLIEWorkflow::set_server_port(const int port) {
    port_ = port;
    if (!is_slave_) {
        const bool master_port_changed = master_port_ != port;
        master_port_ = port;
        if (model_controller_ && (!worker_supervisor_ || master_port_changed)) {
            const bool restart_supervisor = worker_supervisor_ &&
                worker_supervisor_->running();
            worker_supervisor_.reset();
            worker_supervisor_ = std::make_unique<HLIEWorkerSupervisor>(
                *Singleton<HLIEProcessManager>::instance(),
                HLIEWorkerSupervisor::configured_options(master_port_),
                static_cast<IHLIERuntimeEventSink&>(*model_controller_));
            if (restart_supervisor) {
                worker_supervisor_->start();
            }
        }
    }
    for (const auto& instance : get_embedded_model_instances()) {
        if (!model_controller_->update_port(instance.key, port)) {
            LOG_DEBUG("Ignored port update for stale embedded model: key={}, port={}",
                      instance.key, port);
        }
    }
}

bool HLIEWorkflow::initialize_llm_preset_catalogs() {
    const auto& config = Singleton<HLIEConfigService>::instance();
    llm_presets_enabled_ = config->get_bool("model", "llm_presets_enabled", true, true);
    llm_preset_strict_ = config->get_bool("model", "llm_preset_strict", true, true);
    if (!llm_presets_enabled_) {
        LOG_INFO("LLM preset catalogs are disabled");
        return true;
    }

    const auto bundle = config->get_config_bundle();
    const auto resolve_catalog_path = [&](const std::string& configured_path, const std::filesystem::path& bundle_path) {
        if (!bundle_path.empty()) {
            return bundle_path.string();
        }
        std::filesystem::path path(configured_path);
        if (path.is_relative()) {
            const std::filesystem::path config_path(config->get_config_path());
            path = config_path.parent_path() / path;
        }
        return path.lexically_normal().string();
    };
    const auto load_catalog = [&](HLIELlmPresetCatalog& catalog, const std::string& path,
                                  const std::string& source, const bool required) {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec) || ec) {
            if (required) {
                LOG_WARN("LLM {} catalog not found, continuing without it: {}", source, path);
            } else {
                LOG_INFO("LLM {} catalog not found, continuing without local overrides: {}", source, path);
            }
            return true;
        }

        std::string error;
        if (catalog.load_file(path, source, &error)) {
            LOG_INFO("Loaded LLM {} catalog: path={}, version={}, rules={}",
                     source, path, catalog.catalog_version(), catalog.rules().size());
            return true;
        }

        if (llm_preset_strict_) {
            LOG_ERROR("Failed to load LLM {} catalog in strict mode: path={}, error={}", source, path, error);
            return false;
        }
        LOG_WARN("Ignoring invalid LLM {} catalog: path={}, error={}", source, path, error);
        return true;
    };

    const std::string preset_path = resolve_catalog_path(config->get_string(
        "model", "llm_preset_catalog", "model_presets.json", true), bundle.model_presets);
    const std::string override_path = resolve_catalog_path(config->get_string(
        "model", "llm_override_catalog", "model_overrides.json", true), bundle.model_overrides);
    return load_catalog(llm_preset_catalog_, preset_path, "official", true) &&
           load_catalog(llm_override_catalog_, override_path, "local", false);
}

bool HLIEWorkflow::resolve_llm_launch_params(HLIEModelInfo& model_info) {
    if (!HLIEModelServiceFactory::uses_llm_primary_service(model_info)) {
        return true;
    }
    if (model_info.has_effective_llama_params()) {
        return true;
    }

    HLIELlmParamSet base;
    base.set("--parallel", static_cast<int64_t>(model_info.get_batch_num()));
    base.set("--ctx-size", static_cast<int64_t>(model_info.get_context_size()));
    base.set("--no-context-shift", true);
    base.set("--no-warmup", true);

    const std::vector<const HLIELlmPresetCatalog*> catalogs = llm_presets_enabled_
        ? std::vector<const HLIELlmPresetCatalog*>{&llm_preset_catalog_, &llm_override_catalog_}
        : std::vector<const HLIELlmPresetCatalog*>{};
    auto resolution = HLIELlmPresetResolver::resolve(model_info, base, catalogs);

    const auto& config = Singleton<HLIEConfigService>::instance();
    HLIELlmParamSet cli_layer;
    for (const auto& [key, value] : config->get_args()) {
        if (key.rfind("--", 0) != 0 || HLIELlmParamSet::is_framework_owned(key) ||
            key == "--device_id" || key == "--instance_id" || key == "--json_info" ||
            key == "--launch_manifest" || key == "--port") {
            continue;
        }
        if (value.empty()) {
            cli_layer.set(key, true);
        } else {
            cli_layer.set(key, value);
        }
    }
    if (!cli_layer.empty()) {
        resolution.params.merge_layer(cli_layer);
        resolution.sources.push_back("cli");
    }

    HLIELlmParamSet sys_layer;
    if (const auto reader = config->get_reader()) {
        for (const auto& key : reader->Keys("llm_params")) {
            if (HLIELlmParamSet::is_framework_owned(key)) {
                LOG_WARN("Ignoring framework-owned [llm_params] option: {}", key);
                continue;
            }
            const std::string value = reader->Get("llm_params", key, "");
            if (value.empty()) {
                sys_layer.set(key, true);
            } else {
                sys_layer.set(key, value);
            }
        }
    }
    if (!sys_layer.empty()) {
        resolution.params.merge_layer(sys_layer);
        resolution.sources.push_back("sys_cfg");
    }
    resolution.params_hash = resolution.params.stable_hash();

    model_info.set_effective_llama_params(std::move(resolution.params),
                                          std::move(resolution.matched_presets),
                                          std::move(resolution.sources),
                                          std::move(resolution.params_hash));
    if (!model_info.model_key_.empty()) {
        (void)model_controller_->update_model_info(model_info.model_key_, model_info);
    }
    LOG_INFO("Resolved llama params: model_key={}, role={}, presets={}, hash={}",
             model_info.model_key_, model_info.get_role_name(),
             model_info.get_matched_llama_presets().size(), model_info.get_effective_llama_params_hash());
    return true;
}

std::vector<std::string> HLIEWorkflow::get_public_local_model_names() {
    std::vector<std::string> models;
    const auto append_resolved_model = [this, &models](const std::string& key) {
        if (key.empty()) {
            return;
        }
        const auto instance = get_model_inst(key);
        const std::string public_name =
            instance ? instance->model_info.get_model_name() : key;
        if (!public_name.empty()) {
            models.push_back(public_name);
        }
    };
    const auto append_key = [&models](const std::string& key) {
        if (!key.empty()) {
            models.push_back(key);
        }
    };
    for (const auto& instance : get_embedded_model_instances()) {
        if (instance.model_info.get_service_kind() == HLIEModelType::SD) {
            append_key(HLIEModelType::SD3);
        } else {
            append_resolved_model(instance.key);
        }
    }
    return models;
}

std::vector<HLIEWorkflow::ModelInstance>
HLIEWorkflow::get_embedded_model_instances() const {
    std::vector<ModelInstance> result;
    if (!model_controller_) {
        return result;
    }
    for (const auto& instance : model_controller_->get_all_instances()) {
        if (instance.meta.status != SERVER_MODEL_STATUS_LOADED &&
            instance.meta.status != SERVER_MODEL_STATUS_SLEEPING) {
            continue;
        }
        const auto ticket = model_controller_->get_launch_ticket(instance.key);
        const auto runtime = ticket ? runtime_directory_.find(*ticket) : nullptr;
        if (runtime && runtime->identity().kind == HLIERuntimeKind::Embedded) {
            result.push_back(instance);
        }
    }
    return result;
}

std::vector<HLIEWorkerProcessorSnapshot>
HLIEWorkflow::get_worker_process_snapshots() {
    const auto public_model_names = [this](
        const std::vector<std::string>& names) {
        std::vector<std::string> result;
        std::unordered_set<std::string> seen;
        for (const auto& name : names) {
            const auto model_inst = get_model_inst(name);
            const std::string public_name =
                model_inst ? model_inst->model_info.get_model_name() : name;
            if (!public_name.empty() && seen.insert(public_name).second) {
                result.push_back(public_name);
            }
        }
        return result;
    };

    std::vector<HLIEWorkerProcessorSnapshot> snapshots;
    const auto process_infos =
        Singleton<HLIEProcessManager>::instance()->get_running_processes();
    snapshots.reserve(process_infos.size());
    for (const auto& process : process_infos) {
        snapshots.push_back({
            process.params,
            process.instance_id,
            process.instance_id + port_,
            process.uuid,
            static_cast<std::uint64_t>(process.process_id),
            process.launch_ticket.generation,
            process.start_time,
            public_model_names(process.model_names),
        });
    }
    return snapshots;
}

void HLIEWorkflow::notify_local_model_started(
    const std::string& notify_model_key,
    const bool enable_asr_websocket) {
    if (enable_asr_websocket && asr_application_service_) {
        asr_application_service_->ensure_realtime();
    }

    if (worker_control_service_) {
        (void)worker_control_service_->publish_loaded(
            notify_model_key, worker_launch_ticket_);
    }
}

std::optional<server_model_meta> HLIEWorkflow::get_model_meta(const std::string& model_name) {
    return model_controller_->get_meta(model_name);
}

std::optional<HLIEWorkflow::ModelInstance> HLIEWorkflow::get_model_inst(const std::string& model_name) {
    return model_controller_->get_instance_snapshot(model_name);
}

HLIE_NAMESPACE_END

#include "hlie_app/runtime/hlie_default_model_backend_launcher.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <utility>

#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_model_service_manager.h"
#include "hlie_app/runtime/hlie_worker_supervisor.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

HLIE_NAMESPACE_BEGIN

HLIEDefaultModelBackendLauncher::HLIEDefaultModelBackendLauncher(
    HLIEModelServiceManager& service_manager,
    std::shared_ptr<HLIEDeviceService> device_service,
    std::string model_dir,
    std::string program_path,
    SupervisorProvider supervisor_provider,
    PortProvider local_port_provider,
    WorkerSpawnPolicy worker_spawn_policy)
    : service_manager_(service_manager),
      device_service_(std::move(device_service)),
      model_dir_(std::move(model_dir)),
      program_path_(std::move(program_path)),
      supervisor_provider_(std::move(supervisor_provider)),
      local_port_provider_(std::move(local_port_provider)),
      worker_spawn_policy_(std::move(worker_spawn_policy)) {}

HLIEBackendLaunchArtifact HLIEDefaultModelBackendLauncher::launch(
    const HLIEBackendLaunchRequest& request) {
    HLIEBackendLaunchArtifact result;
    if (!request.launch_ticket.valid() || request.service_kind.empty() ||
        request.model_info.model_key_.empty() || !device_service_) {
        return result;
    }
    const auto normalize_kind = [](const std::string& kind) {
        return kind == HLIEModelType::TTS
            ? std::string(HLIEModelType::COSYVOICE)
            : kind;
    };
    if (normalize_kind(request.service_kind) !=
        normalize_kind(request.model_info.get_service_kind())) {
        LOG_ERROR("Backend launcher rejected service-kind mismatch: model={}, requested={}, actual={}",
                  request.launch_ticket.model_key, request.service_kind,
                  request.model_info.get_service_kind());
        return result;
    }

    if (!service_manager_.get_by_type(request.service_kind)) {
        if (!service_manager_.start_by_model_info(
                model_dir_, request.model_info, device_service_)) {
            result.failure = HLIEBackendLaunchFailure::LocalServiceStartFailed;
            return result;
        }

        result.failure = HLIEBackendLaunchFailure::None;
        result.runtime_kind = HLIERuntimeKind::Embedded;
        result.ready_port = local_port_provider_ ? local_port_provider_() : 0;
        result.notify_model_key = request.launch_ticket.model_key;
        result.enable_asr_websocket =
            request.service_kind == HLIEModelType::LLM ||
            request.service_kind == HLIEModelType::WHISPER;
        result.embedded_runtime = {
            request.launch_ticket,
            request.model_info.get_capabilities(),
            request.service_kind,
            request.service_kind == HLIEModelType::LLM &&
                service_manager_.document_pipeline() != nullptr,
        };
        return result;
    }

    if (!worker_spawn_policy_ || !worker_spawn_policy_()) {
        result.failure = HLIEBackendLaunchFailure::WorkerSpawnNotAllowed;
        return result;
    }
    auto* supervisor = supervisor_provider_ ? supervisor_provider_() : nullptr;
    if (!supervisor) {
        result.failure = HLIEBackendLaunchFailure::WorkerPortUnavailable;
        return result;
    }

    result.launch_manifest_path = write_launch_manifest(request.model_info);
    if (result.launch_manifest_path.empty()) {
        result.failure = HLIEBackendLaunchFailure::LaunchManifestFailed;
        return result;
    }

    const auto port_lease = supervisor->reserve_port();
    if (!port_lease) {
        HLIEWorkerSupervisor::discard_launch_manifest(
            result.launch_manifest_path, "worker_port_unavailable");
        result.launch_manifest_path.clear();
        result.failure = HLIEBackendLaunchFailure::WorkerPortUnavailable;
        return result;
    }

    std::vector<std::string> arguments = {
        "--instance_id", std::to_string(port_lease->instance_id),
        "--device_id", request.model_info.get_devices(),
        "--launch_manifest", result.launch_manifest_path,
    };
    if (request.include_parent_arguments) {
        for (const auto& [key, value] :
             Singleton<HLIEConfigService>::instance()->get_args()) {
            arguments.push_back(key);
            arguments.push_back(value);
        }
    }
    arguments.insert(arguments.end(), {
        "--launch_model_key", request.launch_ticket.model_key,
        "--launch_generation", std::to_string(request.launch_ticket.generation),
        "--launch_token", request.launch_ticket.token,
    });

    const auto worker = supervisor->spawn({
        *port_lease,
        program_path_,
        std::move(arguments),
        {request.launch_ticket.model_key},
        result.launch_manifest_path,
        request.launch_ticket,
    });
    if (!worker) {
        result.launch_manifest_path.clear();
        result.failure = HLIEBackendLaunchFailure::WorkerSpawnFailed;
        return result;
    }

    result.failure = HLIEBackendLaunchFailure::None;
    result.runtime_kind = HLIERuntimeKind::Process;
    result.instance_id = worker->instance_id;
    result.process_id = worker->runtime_id;
    result.process_runtime = {
        {worker->runtime_id, HLIERuntimeKind::Process,
         worker->launch_ticket, worker->instance_id, worker->port},
        request.model_info.get_capabilities(),
        worker->model_names,
        worker->ready,
    };
    LOG_INFO("Backend launcher started Worker: model={}, generation={}, runtime_id={}, instance_id={}",
             request.launch_ticket.model_key,
             request.launch_ticket.generation,
             worker->runtime_id,
             worker->instance_id);
    return result;
}

bool HLIEDefaultModelBackendLauncher::cleanup_unadopted(
    const HLIEBackendLaunchArtifact& artifact,
    const std::string& reason) {
    if (!artifact.started()) {
        return true;
    }
    if (artifact.runtime_kind == HLIERuntimeKind::Process) {
        auto* supervisor = supervisor_provider_ ? supervisor_provider_() : nullptr;
        if (!supervisor || !artifact.process_runtime.identity.valid()) {
            return false;
        }
        const auto& identity = artifact.process_runtime.identity;
        const auto stopped = supervisor->request_stop({
            identity.runtime_id,
            identity.instance_id,
            identity.endpoint_port,
            artifact.process_runtime.model_names,
            identity.launch_ticket,
            artifact.process_runtime.ready,
        }, reason);
        return stopped.confirmed;
    }

    const auto& kind = artifact.embedded_runtime.service_kind;
    const bool native_stop_confirmed = service_manager_.stop_and_reset_by_type(
        kind, artifact.embedded_runtime.owns_document_companion);
    const bool primary_gone = !service_manager_.get_by_type(kind);
    const bool companion_gone =
        !artifact.embedded_runtime.owns_document_companion ||
        !service_manager_.document_pipeline();
    return native_stop_confirmed && primary_gone && companion_gone;
}

std::string HLIEDefaultModelBackendLauncher::write_launch_manifest(
    const HLIEModelInfo& model_info) const {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto t = system_clock::to_time_t(now);
    std::tm local_time{};
#ifdef _WIN32
    if (localtime_s(&local_time, &t) != 0) {
        return {};
    }
#else
    if (localtime_r(&t, &local_time) == nullptr) {
        return {};
    }
#endif
    std::ostringstream timestamp;
    timestamp << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    const auto filename =
        "model_launch_" + timestamp.str() + "_" + HlieUtil::uuid() + ".hlif";
    const auto cache_dir = HlieUtil::join_paths(model_dir_, "meta_cache");
    std::string path;
    try {
        (void)std::filesystem::create_directories(cache_dir);
        path = HlieUtil::join_paths(cache_dir, filename);
    } catch (const std::filesystem::filesystem_error& error) {
        LOG_WARN("Backend launcher could not create meta cache; using model directory: error={}",
                 error.what());
        path = HlieUtil::join_paths(model_dir_, filename);
    }
    if (model_info.to_json_file(path, true) >= 0) {
        return path;
    }

    LOG_ERROR("Backend launcher failed to write launch manifest: model={}",
              model_info.model_key_);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    return {};
}

HLIE_NAMESPACE_END

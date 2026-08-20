#include "hlie_app/control/hlie_model_load_service.h"

#include <algorithm>
#include <filesystem>
#include <utility>

#include "hlie_app/hlie_model_info.h"
#include "hlie_app/hlie_model_memory_estimator.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

HLIE_NAMESPACE_BEGIN

namespace {

constexpr auto kLlm = "llm";
constexpr auto kStableDiffusion = "sd";
constexpr auto kWhisper = "whisper";
constexpr auto kCosyVoice = "cosyvoice";
constexpr auto kLegacyTts = "tts";
constexpr auto kQwen3Tts = "qwen3_tts";
constexpr auto kSd3 = "sd3";

template <typename Callback>
class ScopeExit {
public:
    explicit ScopeExit(Callback callback) : callback_(std::move(callback)) {}
    ~ScopeExit() noexcept {
        if (active_) {
            try {
                callback_();
            } catch (...) {
            }
        }
    }

    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;

    void release() noexcept { active_ = false; }

private:
    Callback callback_;
    bool active_ = true;
};

template <typename Callback>
ScopeExit<Callback> make_scope_exit(Callback callback) {
    return ScopeExit<Callback>(std::move(callback));
}

HLIEModelLoadResult load_failure(
    const HLIEControlErrorCode code,
    const HLIEControlErrorStage stage,
    std::string message,
    const int native_code = 0,
    const bool retryable = false) {
    return HLIEModelLoadResult::failure({
        code, stage, std::move(message), native_code, retryable});
}

} // namespace

HLIEModelLoadService::HLIEModelLoadService(
    HLIEModelController& controller,
    HLIEResourceScheduler& scheduler,
    HLIEModelDiscoveryService* discovery,
    std::string model_directory,
    const bool replicas_mode,
    ResidencyPolicyProvider residency_policy_provider,
    LlmParameterResolver llm_parameter_resolver,
    HLIEBackendRegistry backend_registry)
    : controller_(controller),
      scheduler_(scheduler),
      discovery_(discovery),
      model_directory_(std::move(model_directory)),
      replicas_mode_(replicas_mode),
      residency_policy_provider_(std::move(residency_policy_provider)),
      llm_parameter_resolver_(std::move(llm_parameter_resolver)),
      backend_registry_(std::move(backend_registry)) {}

HLIEModelLoadResult HLIEModelLoadService::load(
    HLIEModelLoadRequest request,
    const int local_port,
    const std::optional<HLIEModelLaunchTicket>& preclaimed) {
    bool preclaim_rollback_pending = preclaimed.has_value();
    auto preclaim_rollback = make_scope_exit(
        [this, &preclaimed, &preclaim_rollback_pending] {
            if (preclaim_rollback_pending && preclaimed) {
                controller_.rollback_model_start(
                    *preclaimed, "model_load_service_failed");
            }
        });

    if (replicas_mode_) {
        return load_failure(
            HLIEControlErrorCode::Unsupported,
            HLIEControlErrorStage::Validation,
            "In the cluster or multi-instance mode, external calls to start "
            "model instances are not allowed.",
            80005);
    }

    std::string model_type = request.model_type.empty()
        ? std::string{kLlm}
        : std::move(request.model_type);
    if (model_type == kLegacyTts) {
        model_type = kCosyVoice;
    }
    if (!backend_registry_.find_by_alias(model_type)) {
        return load_failure(
            HLIEControlErrorCode::InvalidArgument,
            HLIEControlErrorStage::Validation,
            "Invalid model type.", 80006);
    }

    std::string model_name = std::move(request.model);
    const std::string model_path = std::move(request.model_path);
    if (model_name.empty() && model_type == kStableDiffusion) {
        model_name = kSd3;
    }

    std::optional<server_model_meta> meta;
    std::optional<HLIEModelController::ModelInstance> resolved_model;
    std::optional<HLIEModelInfo> path_llm_model_info;
    std::string resolved_model_key;
    int estimated_memory_mb = 0;
    int required_device_count = HLIEModelInfo::estimate_device_num(model_name);

    const auto prepare_path_llm_model_info = [&]() -> HLIEControlError {
        if (path_llm_model_info) {
            return {};
        }

        HLIEModelInfo model_info;
        if (model_info.parse_from_gguf_filename(model_path) != 0) {
            return {HLIEControlErrorCode::InvalidArgument,
                    HLIEControlErrorStage::Catalog,
                    "failed to parse model_path.", 80006, false};
        }
        if (!request.mmproj_path.empty()) {
            if (std::error_code error;
                !std::filesystem::exists(request.mmproj_path, error) ||
                !std::filesystem::is_regular_file(request.mmproj_path, error)) {
                return {HLIEControlErrorCode::InvalidArgument,
                        HLIEControlErrorStage::Catalog,
                        "mm_proj_path file not exist.", 80006, false};
            }
            model_info.set_mm_proj_path(request.mmproj_path, true);
        }

        resolved_model_key = HLIEModelInfo::build_model_key(model_info);
        path_llm_model_info = std::move(model_info);
        return {};
    };

    const auto path_conflicts = [&](const HLIEModelController::ModelInstance& existing) {
        if (!path_llm_model_info) {
            return false;
        }
        const bool model_path_matches = HlieUtil::paths_refer_to_same_file(
            existing.path, path_llm_model_info->get_model_path(),
            model_directory_);
        const bool mmproj_path_matches = HlieUtil::paths_refer_to_same_file(
            existing.path_mmproj, path_llm_model_info->get_mm_proj_path(),
            model_directory_);
        if (model_path_matches && mmproj_path_matches) {
            return false;
        }
        LOG_ERROR(
            "Model key path conflict: key={}, registered_model_path={}, "
            "requested_model_path={}, registered_mmproj={}, requested_mmproj={}",
            existing.key, existing.path,
            path_llm_model_info->get_model_path(), existing.path_mmproj,
            path_llm_model_info->get_mm_proj_path());
        return true;
    };

    const auto find_existing_llm_by_paths = [&]()
        -> std::optional<HLIEModelController::ModelInstance> {
        if (model_type != kLlm || model_path.empty()) {
            return std::nullopt;
        }
        for (const auto& instance : controller_.get_all_instances()) {
            if (HlieUtil::paths_refer_to_same_file(
                    instance.path, model_path, model_directory_) &&
                HlieUtil::paths_refer_to_same_file(
                    instance.path_mmproj, request.mmproj_path,
                    model_directory_)) {
                return instance;
            }
        }
        return std::nullopt;
    };

    if (!model_name.empty()) {
        meta = controller_.get_meta(model_name);
        if (!meta && discovery_) {
            const auto generation = discovery_->request_scan();
            (void)discovery_->wait_for(generation);
            meta = controller_.get_meta(model_name);
        }
        if (!meta) {
            return load_failure(
                HLIEControlErrorCode::NotFound,
                HLIEControlErrorStage::Catalog,
                "model is not found");
        }
        if (meta->status == SERVER_MODEL_STATUS_LOADED) {
            return load_failure(
                HLIEControlErrorCode::AlreadyExists,
                HLIEControlErrorStage::Validation,
                "model is already loaded");
        }
        resolved_model = controller_.get_instance_snapshot(model_name);
        if (!resolved_model) {
            return load_failure(
                HLIEControlErrorCode::NotFound,
                HLIEControlErrorStage::Catalog,
                "model is not found");
        }
        resolved_model_key = resolved_model->key;
        estimated_memory_mb = resolved_model->model_info.get_estimated_mem();
        required_device_count = resolved_model->model_info.get_device_num();
    } else if (model_path.empty()) {
        return load_failure(
            HLIEControlErrorCode::NotFound,
            HLIEControlErrorStage::Catalog,
            "model and model_path not found");
    } else {
        const std::filesystem::path model_path_value(model_path);
        if (std::error_code error;
            !std::filesystem::exists(model_path_value, error) ||
            !std::filesystem::is_regular_file(model_path_value, error)) {
            return load_failure(
                HLIEControlErrorCode::InvalidArgument,
                HLIEControlErrorStage::Catalog,
                "model_path file not exist.", 80006);
        }

        model_name = model_path_value.filename().stem().string();
        if (model_type == kLlm) {
            resolved_model = find_existing_llm_by_paths();
            if (!resolved_model) {
                const auto error = prepare_path_llm_model_info();
                if (!error.message.empty()) {
                    return HLIEModelLoadResult::failure(error);
                }
                resolved_model = controller_.get_exact_instance_snapshot(
                    *path_llm_model_info);
            } else {
                resolved_model_key = resolved_model->key;
            }
            if (resolved_model) {
                if (path_conflicts(*resolved_model)) {
                    return load_failure(
                        HLIEControlErrorCode::Conflict,
                        HLIEControlErrorStage::Catalog,
                        "model key is already registered with different model "
                        "or mmproj paths.",
                        80006);
                }
                meta = resolved_model->meta;
                required_device_count =
                    resolved_model->model_info.get_device_num();
            } else {
                required_device_count = path_llm_model_info->get_device_num();
            }
        } else {
            required_device_count =
                HLIEModelInfo::estimate_device_num(model_name);
            meta = controller_.get_meta(model_name);
        }
        if (meta && meta->status == SERVER_MODEL_STATUS_LOADED) {
            return load_failure(
                HLIEControlErrorCode::AlreadyExists,
                HLIEControlErrorStage::Validation,
                "model is already loaded");
        }
        if (resolved_model) {
            estimated_memory_mb =
                resolved_model->model_info.get_estimated_mem();
        } else if (path_llm_model_info) {
            estimated_memory_mb = path_llm_model_info->get_estimated_mem();
        } else {
            HLIEModelSpec path_spec;
            path_spec.identity.model_name = model_name;
            path_spec.identity.model_path = model_path;
            path_spec.backend_alias = model_type;
            path_spec.legacy_model_type = model_type;
            path_spec.required_device_count = required_device_count;
            path_spec.is_llm = model_type != kStableDiffusion;

            HLIEModelMemoryObservation observation;
            observation.empirical_mb =
                HLIEModelInfo::estimate_empirical_mem(model_name);
            if (observation.empirical_mb <= 0) {
                observation.artifact_mb = HLIEModelInfo::estimate_mem_from_gguf(
                    model_path, 0, 1);
            }
            estimated_memory_mb = HLIEModelMemoryEstimator::estimate(
                path_spec, observation).total_mb;
        }
    }

    const bool is_qwen3_audio_llm =
        HlieUtil::contains_ignore_case(model_name, "qwen3_asr") ||
        HlieUtil::contains_ignore_case(model_name, "qwen3-asr") ||
        HlieUtil::contains_ignore_case(model_name, "qwen3_forcedaligner") ||
        HlieUtil::contains_ignore_case(model_name, "qwen3-forcedaligner");
    if (model_type == kWhisper && is_qwen3_audio_llm) {
        LOG_WARN(
            "Model {} was requested as whisper, overriding model_type to llm",
            model_name);
        model_type = kLlm;
    }

    if (model_type == kLlm && !model_path.empty() &&
        resolved_model_key.empty()) {
        const auto error = prepare_path_llm_model_info();
        if (!error.message.empty()) {
            return HLIEModelLoadResult::failure(error);
        }
        resolved_model = controller_.get_exact_instance_snapshot(
            *path_llm_model_info);
        if (resolved_model) {
            if (path_conflicts(*resolved_model)) {
                return load_failure(
                    HLIEControlErrorCode::Conflict,
                    HLIEControlErrorStage::Catalog,
                    "model key is already registered with different model or "
                    "mmproj paths.",
                    80006);
            }
            meta = resolved_model->meta;
            required_device_count =
                resolved_model->model_info.get_device_num();
            estimated_memory_mb =
                resolved_model->model_info.get_estimated_mem();
            if (meta->status == SERVER_MODEL_STATUS_LOADED) {
                return load_failure(
                    HLIEControlErrorCode::AlreadyExists,
                    HLIEControlErrorStage::Validation,
                    "model is already loaded");
            }
        }
    }

    const std::string routing_model_name = resolved_model_key.empty()
        ? model_name
        : resolved_model_key;
    if (residency_policy_provider_) {
        scheduler_.update_residency_policy(residency_policy_provider_());
    }

    const auto try_evict_plan = [&](const HLIEResidencyPlan& plan) {
        if (!plan.should_evict()) {
            return false;
        }
        const auto instance = controller_.get_instance_snapshot(
            plan.candidate->key);
        if (!instance) {
            LOG_WARN(
                "Model residency eviction candidate disappeared before unload: "
                "key={}",
                plan.candidate->key);
            return false;
        }
        return controller_.evict_runtime(
            instance->key, plan.reason).evicted();
    };

    const auto try_auto_evict_for_resource = [&] {
        const auto plan = scheduler_.plan_for_resource_pressure(
            controller_.get_all_runtime_snapshots(), routing_model_name);
        if (!plan.candidate) {
            LOG_WARN(
                "Model residency has no LRU eviction candidate: "
                "requested_model={}, required_mem_mb={}, auto_unload={}",
                routing_model_name, estimated_memory_mb,
                plan.policy.auto_unload);
        }
        return try_evict_plan(plan);
    };

    {
        const auto plan = scheduler_.plan_for_loaded_limit(
            controller_.get_all_runtime_snapshots(), routing_model_name);
        if (plan.candidate && !try_evict_plan(plan) &&
            plan.policy.auto_unload) {
            return load_failure(
                HLIEControlErrorCode::CapacityExceeded,
                HLIEControlErrorStage::Reservation,
                "max_loaded_models limit reached, but no model could be "
                "evicted.",
                80004, true);
        }
    }

    HLIEPlacementIntent placement;
    placement.required_device_count = required_device_count;
    placement.allow_eviction = true;
    if (request.requested_device_ids) {
        auto requested_devices = std::move(*request.requested_device_ids);
        std::sort(requested_devices.begin(), requested_devices.end());
        requested_devices.erase(
            std::unique(requested_devices.begin(), requested_devices.end()),
            requested_devices.end());
        if (requested_devices.size() !=
            static_cast<std::size_t>(required_device_count)) {
            LOG_WARN(
                "Model {} requires {} devices, but {} devices were requested",
                model_name, required_device_count, requested_devices.size());
            return load_failure(
                HLIEControlErrorCode::InvalidArgument,
                HLIEControlErrorStage::Reservation,
                "The requested device count does not match the model chip "
                "count.",
                80004);
        }
        placement.requested_device_ids = std::move(requested_devices);
        placement.allow_auto_selection = false;
    } else {
        placement.allow_auto_selection = true;
    }

    const auto reserve = [&] {
        return scheduler_.reserve(
            routing_model_name, estimated_memory_mb, placement);
    };
    auto startup_reservation = reserve();
    if (!startup_reservation && try_auto_evict_for_resource()) {
        startup_reservation = reserve();
    }
    if (!startup_reservation) {
        LOG_WARN(
            "Device reservation rejected: model={}, estimated_mb={}, "
            "required_device_count={}",
            routing_model_name, estimated_memory_mb, required_device_count);
        return load_failure(
            HLIEControlErrorCode::CapacityExceeded,
            HLIEControlErrorStage::Reservation,
            "The devices is not available, estimated model memory is " +
                std::to_string(estimated_memory_mb) + "MB.",
            80004, true);
    }
    auto reservation_rollback = make_scope_exit([&] {
        if (startup_reservation) {
            controller_.rollback_pending_reservation(*startup_reservation);
        }
    });

    const std::string devices = HlieUtil::join_devices(
        startup_reservation->reservation().devices);
    LOG_INFO("Acquired model devices: model={}, devices={}",
             routing_model_name, devices);

    HLIEModelLoadReceipt receipt;
    const auto launch_model = [&](HLIEModelInfo model_info,
                                  const std::string& service_kind,
                                  const bool include_parent_arguments,
                                  const std::optional<HLIEModelLaunchTicket>& preclaim)
        -> std::optional<HLIEControlError> {
        model_info.model_key_ = model_info.model_key_.empty()
            ? routing_model_name
            : model_info.model_key_;
        const auto result = controller_.load_model(
            *startup_reservation, std::move(model_info), service_kind,
            include_parent_arguments, preclaim);
        if (preclaim && preclaimed && preclaim->matches(*preclaimed)) {
            preclaim_rollback_pending = false;
        }
        if (!result.accepted()) {
            switch (result.status) {
            case HLIEControllerLoadStatus::BusyOrMissing:
            case HLIEControllerLoadStatus::StalePreclaim:
                return HLIEControlError{
                    HLIEControlErrorCode::Conflict,
                    HLIEControlErrorStage::RuntimeStart,
                    "model is already loading", 0, false};
            case HLIEControllerLoadStatus::ReservationBindFailed:
                return HLIEControlError{
                    HLIEControlErrorCode::Internal,
                    HLIEControlErrorStage::Reservation,
                    "failed to bind model device reservation", 80001, false};
            case HLIEControllerLoadStatus::BackendStartFailed:
                switch (result.backend.failure) {
                case HLIEBackendLaunchFailure::WorkerSpawnNotAllowed:
                    return HLIEControlError{
                        HLIEControlErrorCode::BackendStartFailed,
                        HLIEControlErrorStage::RuntimeStart,
                        "slave cannot start new slave", 80001, false};
                case HLIEBackendLaunchFailure::LaunchManifestFailed:
                    return HLIEControlError{
                        HLIEControlErrorCode::InvalidArgument,
                        HLIEControlErrorStage::RuntimeStart,
                        "failed to write model launch manifest", 80007, false};
                case HLIEBackendLaunchFailure::WorkerPortUnavailable:
                    return HLIEControlError{
                        HLIEControlErrorCode::Unavailable,
                        HLIEControlErrorStage::RuntimeStart,
                        "no available worker port", 80004, true};
                case HLIEBackendLaunchFailure::WorkerSpawnFailed:
                    return HLIEControlError{
                        HLIEControlErrorCode::BackendStartFailed,
                        HLIEControlErrorStage::RuntimeStart,
                        "failed to start model subprocess", 80001, false};
                case HLIEBackendLaunchFailure::LocalServiceStartFailed:
                    return HLIEControlError{
                        HLIEControlErrorCode::BackendStartFailed,
                        HLIEControlErrorStage::RuntimeStart,
                        "failed to initialize model service", 80001, false};
                case HLIEBackendLaunchFailure::InvalidRequest:
                case HLIEBackendLaunchFailure::None:
                    return HLIEControlError{
                        HLIEControlErrorCode::BackendStartFailed,
                        HLIEControlErrorStage::RuntimeStart,
                        "invalid model launch request", 80001, false};
                }
                break;
            case HLIEControllerLoadStatus::CleanupUnconfirmed:
                return HLIEControlError{
                    HLIEControlErrorCode::Unavailable,
                    HLIEControlErrorStage::RuntimeStart,
                    "model launch cleanup was not confirmed", 80004, true};
            case HLIEControllerLoadStatus::RuntimeAdoptionFailed:
                return HLIEControlError{
                    HLIEControlErrorCode::Internal,
                    HLIEControlErrorStage::RuntimeStart,
                    "failed to adopt model runtime", 80001, false};
            case HLIEControllerLoadStatus::BackendLauncherUnavailable:
                return HLIEControlError{
                    HLIEControlErrorCode::Internal,
                    HLIEControlErrorStage::RuntimeStart,
                    "model backend launcher is unavailable", 80001, false};
            case HLIEControllerLoadStatus::Ready:
            case HLIEControllerLoadStatus::AwaitingReady:
                break;
            }
        }

        receipt.instance_id = result.backend.instance_id;
        receipt.process_id = result.backend.process_id;
        receipt.launch_manifest_path = result.backend.launch_manifest_path;
        receipt.notify_model_key = result.backend.notify_model_key;
        receipt.enable_asr_websocket = result.backend.enable_asr_websocket;
        receipt.ready = result.status == HLIEControllerLoadStatus::Ready;
        receipt.runtime_kind = result.backend.runtime_kind;
        receipt.model_key = routing_model_name;
        receipt.port = local_port + receipt.instance_id;
        return std::nullopt;
    };

    if (model_type == kLlm) {
        if (!resolved_model && path_llm_model_info) {
            const bool inserted = controller_.register_model_if_absent(
                *path_llm_model_info, estimated_memory_mb);
            resolved_model = controller_.get_exact_instance_snapshot(
                *path_llm_model_info);
            LOG_INFO(
                "Resolved LLM model for load: key={}, catalog_action={}, "
                "model_path={}, mmproj={}",
                resolved_model_key, inserted ? "insert" : "reuse",
                path_llm_model_info->get_model_path(),
                path_llm_model_info->get_mm_proj_path());
        }

        resolved_model = path_llm_model_info
            ? controller_.get_exact_instance_snapshot(*path_llm_model_info)
            : (resolved_model_key.empty()
                   ? std::nullopt
                   : controller_.get_instance_snapshot(resolved_model_key));
        if (!resolved_model) {
            return load_failure(
                HLIEControlErrorCode::NotFound,
                HLIEControlErrorStage::Catalog,
                "model and model_path not found");
        }
        if (path_conflicts(*resolved_model)) {
            return load_failure(
                HLIEControlErrorCode::Conflict,
                HLIEControlErrorStage::Catalog,
                "model key is already registered with different model or "
                "mmproj paths.",
                80006);
        }

        const std::string model_key = resolved_model->key;
        const bool owns_preclaim = preclaimed &&
            preclaimed->model_key == model_key;
        if (resolved_model->meta.status == SERVER_MODEL_STATUS_LOADED) {
            return load_failure(
                HLIEControlErrorCode::AlreadyExists,
                HLIEControlErrorStage::Validation,
                "model is already loaded");
        }
        if (resolved_model->meta.status == SERVER_MODEL_STATUS_LOADING &&
            !owns_preclaim) {
            return load_failure(
                HLIEControlErrorCode::Conflict,
                HLIEControlErrorStage::Validation,
                "model is already loading");
        }
        if (resolved_model->meta.status == SERVER_MODEL_STATUS_SLEEPING) {
            return load_failure(
                HLIEControlErrorCode::Conflict,
                HLIEControlErrorStage::Validation,
                "model is sleeping and cannot be started again");
        }
        if (owns_preclaim &&
            (resolved_model->meta.status != SERVER_MODEL_STATUS_LOADING ||
             !controller_.is_current_launch(*preclaimed))) {
            return load_failure(
                HLIEControlErrorCode::StaleOperation,
                HLIEControlErrorStage::Validation,
                "preclaimed model is no longer loading");
        }

        auto model_info = controller_.prepare_model_info_with_devices(
            model_key, devices);
        if (!model_info) {
            return load_failure(
                HLIEControlErrorCode::NotFound,
                HLIEControlErrorStage::Catalog,
                "model and model_path not found");
        }
        if (!llm_parameter_resolver_ ||
            !llm_parameter_resolver_(*model_info)) {
            return load_failure(
                HLIEControlErrorCode::InvalidArgument,
                HLIEControlErrorStage::Validation,
                "failed to resolve llama params", 80007);
        }
        if (!model_info->model_key_.empty()) {
            (void)controller_.update_model_info(
                model_info->model_key_, *model_info);
        }
        if (model_info->model_version_.empty()) {
            return load_failure(
                HLIEControlErrorCode::InvalidArgument,
                HLIEControlErrorStage::Validation,
                "failed to parse build_version from model name", 80007);
        }
        if (!HlieUtil::version_ge(model_info->model_version_, "v1.0.0")) {
            return load_failure(
                HLIEControlErrorCode::InvalidArgument,
                HLIEControlErrorStage::Validation,
                "build_version must be >= v1.0.0", 80007);
        }
        if (auto error = launch_model(
                std::move(*model_info), kLlm, true,
                owns_preclaim ? preclaimed : std::nullopt)) {
            return HLIEModelLoadResult::failure(std::move(*error));
        }
    } else if (model_type == kStableDiffusion) {
        if (!request.sd_artifacts || !request.sd_artifacts->valid()) {
            return load_failure(
                HLIEControlErrorCode::InvalidArgument,
                HLIEControlErrorStage::Validation,
                "params invalid for SD3", 80007);
        }
        HLIEModelInfo model_info;
        model_info.build_sd(
            model_type, request.sd_artifacts->vae_model,
            request.sd_artifacts->t5_model,
            request.sd_artifacts->clip_model,
            request.sd_artifacts->clip_l_model,
            request.sd_artifacts->mmdit_model, devices);
        model_info.model_key_ = kSd3;
        if (auto error = launch_model(
                std::move(model_info), kStableDiffusion, false,
                std::nullopt)) {
            return HLIEModelLoadResult::failure(std::move(*error));
        }
    } else if (model_type == kWhisper || model_type == kCosyVoice ||
               model_type == kQwen3Tts) {
        if (!meta) {
            HLIEModelInfo discovered;
            const int parse_result =
                discovered.parse_from_gguf_filename(model_path);
            if (model_type == kQwen3Tts && parse_result < 0) {
                return load_failure(
                    HLIEControlErrorCode::InvalidArgument,
                    HLIEControlErrorStage::Catalog,
                    "invalid Qwen3-TTS model filename", 80007);
            }
            discovered.set_service_kind(model_type);
            controller_.register_model(
                discovered, discovered.get_estimated_mem());
            LOG_INFO(
                "Registered model for dynamic load: key={}, service_kind={}",
                discovered.model_key_, model_type);
        }

        const auto instance = controller_.get_instance_snapshot(model_name);
        if (!instance) {
            return load_failure(
                HLIEControlErrorCode::NotFound,
                HLIEControlErrorStage::Catalog,
                "model and model_path not found");
        }
        auto model_info = controller_.prepare_model_info_with_devices(
            instance->key, devices);
        if (!model_info) {
            return load_failure(
                HLIEControlErrorCode::NotFound,
                HLIEControlErrorStage::Catalog,
                "model and model_path not found");
        }
        if (model_info->model_version_.empty() ||
            !HlieUtil::version_ge(model_info->model_version_, "v1.0.0")) {
            return load_failure(
                HLIEControlErrorCode::InvalidArgument,
                HLIEControlErrorStage::Validation,
                "build_version must be >= v1.0.0", 80007);
        }
        if (auto error = launch_model(
                std::move(*model_info), model_type, true, std::nullopt)) {
            return HLIEModelLoadResult::failure(std::move(*error));
        }
    }

    preclaim_rollback.release();
    return HLIEModelLoadResult::success(std::move(receipt));
}

HLIE_NAMESPACE_END

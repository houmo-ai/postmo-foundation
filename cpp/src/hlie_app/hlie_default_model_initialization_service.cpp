#include "hlie_app/startup/hlie_default_model_initialization_service.h"

#include <filesystem>
#include <optional>
#include <utility>

#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

HLIE_NAMESPACE_BEGIN

namespace {

constexpr const char* kSd3ModelName = "sd3";

HLIEDefaultModelInitializationResult unavailable_result(
    const std::string& reason, const bool zero_load_allowed) {
    if (zero_load_allowed) {
        LOG_WARN("{}; starting in zero-load mode", reason);
        return {HLIEDefaultModelInitializationStatus::ZeroLoad};
    }
    LOG_ERROR("{}; zero-load mode is not allowed for this instance", reason);
    return {HLIEDefaultModelInitializationStatus::Failed};
}

bool is_regular_file(
    const HLIEDefaultModelInitializationDependencies& dependencies,
    const std::string& path) {
    if (dependencies.is_regular_file) {
        return dependencies.is_regular_file(path);
    }
    std::error_code error;
    return std::filesystem::exists(path, error) && !error &&
        std::filesystem::is_regular_file(path, error) && !error;
}

} // namespace

HLIEDefaultModelInitializationService::
HLIEDefaultModelInitializationService(
    HLIEDefaultModelInitializationDependencies dependencies)
    : dependencies_(std::move(dependencies)) {}

HLIEDefaultModelInitializationResult
HLIEDefaultModelInitializationService::initialize(
    const std::vector<HLIEModelInfo>& models,
    const HLIEDefaultModelInitializationOptions& options) const {
    if (models.empty()) {
        return unavailable_result("No valid model info files found",
                                  options.zero_load_allowed);
    }
    if (options.default_model.empty() && !options.enable_sd3) {
        return unavailable_result("No default model is configured",
                                  options.zero_load_allowed);
    }

    std::optional<HLIEModelInfo> primary;
    std::optional<HLIEModelInfo> sd3;
    if (!options.default_model.empty()) {
        for (const auto& model : models) {
            LOG_DEBUG("Scanned startup model: key={}, name={}, service_kind={}",
                      model.model_key_, model.get_model_name(),
                      model.get_service_kind());
            if (model.model_key_ == options.default_model ||
                (!primary &&
                 model.get_model_name() == options.default_model)) {
                primary = model;
            } else if (options.enable_sd3 &&
                       model.get_model_name() == kSd3ModelName) {
                sd3 = model;
            }
        }
        if (primary) {
            LOG_INFO("Default startup model selected: key={}, service_kind={}",
                     primary->model_key_, primary->get_service_kind());
        } else {
            LOG_ERROR("Default model not found: name={}, scanned_models={}",
                      options.default_model, models.size());
        }
    } else {
        for (const auto& model : models) {
            if (model.get_model_name() == kSd3ModelName) {
                sd3 = model;
                break;
            }
        }
        if (!sd3) {
            LOG_WARN("No SD3 model found for enabled startup configuration");
            return {HLIEDefaultModelInitializationStatus::Failed};
        }
    }

    if (!primary && !sd3) {
        return unavailable_result("No configured startup model was found",
                                  options.zero_load_allowed);
    }

    HLIEDefaultModelInitializationResult result{
        HLIEDefaultModelInitializationStatus::Initialized};
    if (primary) {
        const bool uses_llm = dependencies_.uses_llm_primary_service &&
            dependencies_.uses_llm_primary_service(*primary);
        if (uses_llm) {
            if (!options.default_mmproj.empty()) {
                std::string mmproj_path = options.default_mmproj + ".gguf";
                if (HlieUtil::is_relative_path(mmproj_path)) {
                    mmproj_path = HlieUtil::join_paths(
                        options.model_directory, mmproj_path);
                }
                if (!is_regular_file(dependencies_, mmproj_path)) {
                    LOG_ERROR("Default mmproj file does not exist: {}",
                              mmproj_path);
                    return {HLIEDefaultModelInitializationStatus::Failed};
                }
                if (!dependencies_.apply_mmproj) {
                    LOG_ERROR("Default mmproj adapter is unavailable");
                    return {HLIEDefaultModelInitializationStatus::Failed};
                }
                dependencies_.apply_mmproj(*primary, mmproj_path);
                if (dependencies_.contains_model &&
                    dependencies_.register_model &&
                    !dependencies_.contains_model(primary->model_key_)) {
                    dependencies_.register_model(*primary);
                }
            }
            if (!dependencies_.resolve_llm_launch_params ||
                !dependencies_.resolve_llm_launch_params(*primary)) {
                LOG_ERROR("Failed to resolve llama params for default model: {}",
                          primary->model_key_);
                return {HLIEDefaultModelInitializationStatus::Failed};
            }
        }

        LOG_INFO("Initializing default model Service: key={}, service_kind={}",
                 primary->model_key_, primary->get_service_kind());
        if (!dependencies_.load_primary ||
            !dependencies_.load_primary(*primary)) {
            LOG_ERROR("Failed to initialize default model Service: key={}, service_kind={}",
                      primary->model_key_, primary->get_service_kind());
            return {HLIEDefaultModelInitializationStatus::Failed};
        }
        result.primary_initialized = true;
    }

    if (sd3) {
        LOG_INFO("Initializing configured SD3 Service");
        if (!dependencies_.load_sd3 || !dependencies_.load_sd3(*sd3)) {
            LOG_ERROR("Failed to initialize configured SD3 Service");
            result.status = HLIEDefaultModelInitializationStatus::Failed;
            return result;
        }
        result.sd3_initialized = true;
    }
    return result;
}

HLIE_NAMESPACE_END

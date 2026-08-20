#ifndef HLIE_APP_STARTUP_DEFAULT_MODEL_INITIALIZATION_SERVICE_H
#define HLIE_APP_STARTUP_DEFAULT_MODEL_INITIALIZATION_SERVICE_H

#include <functional>
#include <string>
#include <vector>

#include "hlie_app/hlie_model_info.h"

HLIE_NAMESPACE_BEGIN

enum class HLIEDefaultModelInitializationStatus {
    Initialized,
    ZeroLoad,
    Failed,
};

struct HLIEDefaultModelInitializationOptions {
    std::string model_directory;
    std::string default_model;
    std::string default_mmproj;
    bool enable_sd3 = false;
    bool zero_load_allowed = false;
};

struct HLIEDefaultModelInitializationResult {
    HLIEDefaultModelInitializationStatus status =
        HLIEDefaultModelInitializationStatus::Failed;
    bool primary_initialized = false;
    bool sd3_initialized = false;

    [[nodiscard]] bool initialized() const noexcept {
        return status == HLIEDefaultModelInitializationStatus::Initialized;
    }
};

struct HLIEDefaultModelInitializationDependencies {
    std::function<bool(const HLIEModelInfo&)> uses_llm_primary_service;
    std::function<void(HLIEModelInfo&, const std::string&)> apply_mmproj;
    std::function<bool(HLIEModelInfo&)> resolve_llm_launch_params;
    std::function<bool(const std::string&)> contains_model;
    std::function<void(HLIEModelInfo&)> register_model;
    std::function<bool(const HLIEModelInfo&)> load_primary;
    std::function<bool(const HLIEModelInfo&)> load_sd3;
    std::function<bool(const std::string&)> is_regular_file;
};

// Owns the legacy startup selection and Service initialization phase. Runtime
// start/publication remains in HLIEEmbeddedStartupService.
class HLIEDefaultModelInitializationService final {
public:
    explicit HLIEDefaultModelInitializationService(
        HLIEDefaultModelInitializationDependencies dependencies);

    [[nodiscard]] HLIEDefaultModelInitializationResult initialize(
        const std::vector<HLIEModelInfo>& models,
        const HLIEDefaultModelInitializationOptions& options) const;

private:
    HLIEDefaultModelInitializationDependencies dependencies_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_STARTUP_DEFAULT_MODEL_INITIALIZATION_SERVICE_H

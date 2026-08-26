#ifndef HLIE_APP_BOOTSTRAP_APPLICATION_SERVICE_H
#define HLIE_APP_BOOTSTRAP_APPLICATION_SERVICE_H

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "hlie_app/startup/hlie_default_model_initialization_service.h"
#include "hlie_core/domain/hlie_model_launch_ticket.h"

HLIE_NAMESPACE_BEGIN

enum class HLIEBootstrapStatus {
    Ready,
    ZeroLoad,
    Failed,
};

enum class HLIEBootstrapProcessMode {
    Main,
    Worker,
};

struct HLIEBootstrapRequest {
    std::string model_directory;
    std::unordered_map<std::string, std::string> arguments;
    std::string default_model;
    std::string default_mmproj;
    bool enable_sd3 = false;
};

struct HLIEBootstrapResult {
    HLIEBootstrapStatus status = HLIEBootstrapStatus::Failed;
    HLIEBootstrapProcessMode process_mode = HLIEBootstrapProcessMode::Main;
    HLIEModelLaunchTicket launch_ticket;

    [[nodiscard]] bool started() const noexcept {
        return status != HLIEBootstrapStatus::Failed;
    }

    [[nodiscard]] bool worker_process() const noexcept {
        return process_mode == HLIEBootstrapProcessMode::Worker;
    }

    [[nodiscard]] int legacy_return_code() const noexcept {
        switch (status) {
        case HLIEBootstrapStatus::Ready:
            return 0;
        case HLIEBootstrapStatus::ZeroLoad:
            return 8;
        case HLIEBootstrapStatus::Failed:
            return -1;
        }
        return -1;
    }
};

struct HLIEBootstrapApplicationDependencies {
    std::function<bool()> prepare_launch_configuration;
    std::function<std::optional<HLIEModelInfo>(const std::string&)>
        parse_manifest;
    std::function<void(const std::string&)> discard_one_time_manifest;
    std::function<std::vector<HLIEModelInfo>()> discover_models;
    std::function<bool(HLIEModelInfo&)> resolve_llm_launch_params;
    std::function<void(HLIEModelInfo&)> register_model;
    std::function<bool(const HLIEModelInfo&)> initialize_worker_model;
    std::function<void()> initialize_discovery_baseline;
    std::function<HLIEDefaultModelInitializationResult(
        const std::vector<HLIEModelInfo>&,
        const HLIEDefaultModelInitializationOptions&)>
        initialize_defaults;
};

// Owns process-role selection and the initialization phase of startup.
// Starting Services and publishing Runtime generations remain in the later
// process run phase.
class HLIEBootstrapApplicationService final {
public:
    explicit HLIEBootstrapApplicationService(
        HLIEBootstrapApplicationDependencies dependencies);

    [[nodiscard]] HLIEBootstrapResult initialize(
        const HLIEBootstrapRequest& request) const;

private:
    HLIEBootstrapApplicationDependencies dependencies_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_BOOTSTRAP_APPLICATION_SERVICE_H

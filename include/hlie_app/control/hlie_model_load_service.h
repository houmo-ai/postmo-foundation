#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "hlie_app/control/hlie_model_controller.h"
#include "hlie_app/discovery/hlie_model_discovery_service.h"
#include "hlie_app/resource/hlie_resource_scheduler.h"
#include "hlie_core/backend/hlie_backend_registry.h"
#include "hlie_core/domain/hlie_control_result.h"

HLIE_NAMESPACE_BEGIN

struct HLIESdModelArtifacts {
    std::string vae_model;
    std::string t5_model;
    std::string clip_model;
    std::string clip_l_model;
    std::string mmdit_model;

    [[nodiscard]] bool valid() const noexcept {
        return !vae_model.empty() && !t5_model.empty() &&
               !clip_model.empty() && !clip_l_model.empty() &&
               !mmdit_model.empty();
    }
};

// Transport-neutral request produced by the HTTP adapter. An absent device
// list means automatic placement; a present list means explicit placement.
struct HLIEModelLoadRequest {
    std::string model_type = "llm";
    std::string model;
    std::string model_path;
    std::string mmproj_path;
    std::optional<std::vector<int>> requested_device_ids;
    std::optional<HLIESdModelArtifacts> sd_artifacts;
};

struct HLIEModelLoadReceipt {
    int port = 0;
    int instance_id = 0;
    std::uint64_t process_id = 0;
    std::string launch_manifest_path;
    std::string model_key;
    std::string notify_model_key;
    bool enable_asr_websocket = false;
    bool ready = false;
    HLIERuntimeKind runtime_kind = HLIERuntimeKind::Embedded;
};

using HLIEModelLoadResult = HLIEControlResult<HLIEModelLoadReceipt>;

// Owns dynamic model-load orchestration above ModelController. It performs
// catalog/path resolution, residency planning and reservation acquisition, but
// never parses HTTP, starts a concrete Service or mutates Registry directly.
class HLIEModelLoadService {
public:
    using ResidencyPolicyProvider =
        std::function<HLIEModelResidencyPolicy()>;
    using LlmParameterResolver = std::function<bool(HLIEModelInfo&)>;

    HLIEModelLoadService(
        HLIEModelController& controller,
        HLIEResourceScheduler& scheduler,
        HLIEModelDiscoveryService* discovery,
        std::string model_directory,
        bool replicas_mode,
        ResidencyPolicyProvider residency_policy_provider,
        LlmParameterResolver llm_parameter_resolver,
        HLIEBackendRegistry backend_registry =
            HLIEBackendRegistry::houmo_defaults());

    [[nodiscard]] HLIEModelLoadResult load(
        HLIEModelLoadRequest request,
        int local_port,
        const std::optional<HLIEModelLaunchTicket>& preclaimed =
            std::nullopt);

private:
    HLIEModelController& controller_;
    HLIEResourceScheduler& scheduler_;
    HLIEModelDiscoveryService* discovery_ = nullptr;
    std::string model_directory_;
    bool replicas_mode_ = false;
    ResidencyPolicyProvider residency_policy_provider_;
    LlmParameterResolver llm_parameter_resolver_;
    HLIEBackendRegistry backend_registry_;
};

HLIE_NAMESPACE_END

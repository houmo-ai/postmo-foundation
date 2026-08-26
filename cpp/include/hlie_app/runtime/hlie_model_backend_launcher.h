#ifndef HLIE_APP_MODEL_BACKEND_LAUNCHER_H
#define HLIE_APP_MODEL_BACKEND_LAUNCHER_H

#include <cstdint>
#include <string>

#include "hlie_app/hlie_model_info.h"
#include "hlie_app/runtime/hlie_model_runtime_factory.h"

HLIE_NAMESPACE_BEGIN

enum class HLIEBackendLaunchFailure {
    None,
    InvalidRequest,
    LocalServiceStartFailed,
    WorkerSpawnNotAllowed,
    LaunchManifestFailed,
    WorkerPortUnavailable,
    WorkerSpawnFailed,
};

struct HLIEBackendLaunchRequest {
    HLIEModelLaunchTicket launch_ticket;
    HLIEModelInfo model_info;
    std::string service_kind;
    bool include_parent_arguments = true;
};

struct HLIEBackendLaunchArtifact {
    HLIEBackendLaunchFailure failure = HLIEBackendLaunchFailure::InvalidRequest;
    HLIERuntimeKind runtime_kind = HLIERuntimeKind::Embedded;
    HLIEEmbeddedRuntimeCreateRequest embedded_runtime;
    HLIEProcessRuntimeCreateRequest process_runtime;
    int ready_port = 0;
    int instance_id = 0;
    std::uint64_t process_id = 0;
    std::string launch_manifest_path;
    std::string notify_model_key;
    bool enable_asr_websocket = false;

    [[nodiscard]] bool started() const noexcept {
        return failure == HLIEBackendLaunchFailure::None;
    }
};

// App-layer port for the side effect that starts a concrete embedded service
// or Worker. Controller owns transaction ordering and lifecycle publication.
class IHLIEModelBackendLauncher {
public:
    virtual ~IHLIEModelBackendLauncher() = default;

    [[nodiscard]] virtual HLIEBackendLaunchArtifact launch(
        const HLIEBackendLaunchRequest& request) = 0;
    [[nodiscard]] virtual bool cleanup_unadopted(
        const HLIEBackendLaunchArtifact& artifact,
        const std::string& reason) = 0;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_MODEL_BACKEND_LAUNCHER_H

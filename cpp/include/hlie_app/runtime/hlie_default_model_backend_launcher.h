#ifndef HLIE_APP_DEFAULT_MODEL_BACKEND_LAUNCHER_H
#define HLIE_APP_DEFAULT_MODEL_BACKEND_LAUNCHER_H

#include <functional>
#include <memory>
#include <string>

#include "hlie_app/runtime/hlie_model_backend_launcher.h"

HLIE_NAMESPACE_BEGIN

class HLIEDeviceService;
class HLIEModelServiceManager;
class HLIEWorkerSupervisor;

class HLIEDefaultModelBackendLauncher final
    : public IHLIEModelBackendLauncher {
public:
    using SupervisorProvider = std::function<HLIEWorkerSupervisor*()>;
    using PortProvider = std::function<int()>;
    using WorkerSpawnPolicy = std::function<bool()>;

    HLIEDefaultModelBackendLauncher(
        HLIEModelServiceManager& service_manager,
        std::shared_ptr<HLIEDeviceService> device_service,
        std::string model_dir,
        std::string program_path,
        SupervisorProvider supervisor_provider,
        PortProvider local_port_provider,
        WorkerSpawnPolicy worker_spawn_policy);

    [[nodiscard]] HLIEBackendLaunchArtifact launch(
        const HLIEBackendLaunchRequest& request) override;
    [[nodiscard]] bool cleanup_unadopted(
        const HLIEBackendLaunchArtifact& artifact,
        const std::string& reason) override;

private:
    [[nodiscard]] std::string write_launch_manifest(
        const HLIEModelInfo& model_info) const;

    HLIEModelServiceManager& service_manager_;
    std::shared_ptr<HLIEDeviceService> device_service_;
    std::string model_dir_;
    std::string program_path_;
    SupervisorProvider supervisor_provider_;
    PortProvider local_port_provider_;
    WorkerSpawnPolicy worker_spawn_policy_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_DEFAULT_MODEL_BACKEND_LAUNCHER_H

#ifndef HLIE_APP_DEFAULT_MODEL_RUNTIME_FACTORY_H
#define HLIE_APP_DEFAULT_MODEL_RUNTIME_FACTORY_H

#include <functional>

#include "hlie_app/runtime/hlie_model_runtime_factory.h"

HLIE_NAMESPACE_BEGIN

class HLIEModelServiceManager;
class HLIEWorkerSupervisor;

class HLIEDefaultModelRuntimeFactory final : public IHLIEModelRuntimeFactory {
public:
    using SupervisorProvider = std::function<HLIEWorkerSupervisor*()>;
    using EmbeddedIdentityAllocator =
        std::function<HLIERuntimeIdentity(const HLIEModelLaunchTicket&)>;

    HLIEDefaultModelRuntimeFactory(
        HLIEModelServiceManager& service_manager,
        SupervisorProvider supervisor_provider,
        EmbeddedIdentityAllocator embedded_identity_allocator);

    [[nodiscard]] std::shared_ptr<IHLIEModelRuntime> create_embedded(
        const HLIEEmbeddedRuntimeCreateRequest& request) override;
    [[nodiscard]] std::shared_ptr<IHLIEModelRuntime> create_process(
        const HLIEProcessRuntimeCreateRequest& request) override;

private:
    HLIEModelServiceManager& service_manager_;
    SupervisorProvider supervisor_provider_;
    EmbeddedIdentityAllocator embedded_identity_allocator_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_DEFAULT_MODEL_RUNTIME_FACTORY_H

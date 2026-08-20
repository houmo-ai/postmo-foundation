#ifndef HLIE_APP_STARTUP_DEFAULT_EMBEDDED_STARTUP_TARGET_PROVIDER_H
#define HLIE_APP_STARTUP_DEFAULT_EMBEDDED_STARTUP_TARGET_PROVIDER_H

#include <functional>

#include "hlie_app/startup/hlie_embedded_startup_service.h"

HLIE_NAMESPACE_BEGIN

class HLIEModelServiceManager;

// Compatibility adapter from the current concrete ServiceManager slots to
// neutral embedded startup targets. Multi-SDK/provider selection is not part
// of this adapter.
class HLIEDefaultEmbeddedStartupTargetProvider final
    : public IHLIEEmbeddedStartupTargetProvider {
public:
    using LocalPortProvider = std::function<int()>;

    HLIEDefaultEmbeddedStartupTargetProvider(
        HLIEModelServiceManager& service_manager,
        LocalPortProvider local_port_provider);

    [[nodiscard]] std::vector<HLIEEmbeddedStartupTarget> targets(
        HLIEEmbeddedStartupGroup group) const override;

private:
    HLIEModelServiceManager& service_manager_;
    LocalPortProvider local_port_provider_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_STARTUP_DEFAULT_EMBEDDED_STARTUP_TARGET_PROVIDER_H

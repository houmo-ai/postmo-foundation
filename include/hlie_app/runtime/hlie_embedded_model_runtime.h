#ifndef HLIE_APP_EMBEDDED_MODEL_RUNTIME_H
#define HLIE_APP_EMBEDDED_MODEL_RUNTIME_H

#include <memory>
#include <string>

#include "hlie_core/runtime/hlie_model_runtime.h"

HLIE_NAMESPACE_BEGIN

class HLIEModelServiceManager;

// Control-plane view of an in-process ServiceManager bundle. Concrete service
// types are confined to the implementation; the public contract is SDK-free.
class HLIEEmbeddedModelRuntime final : public IHLIEModelRuntime {
public:
    HLIEEmbeddedModelRuntime(HLIEModelServiceManager& service_manager,
                             HLIERuntimeIdentity identity,
                             HLIEModelCapabilities capabilities,
                             std::string service_kind,
                             bool owns_document_companion = false);
    ~HLIEEmbeddedModelRuntime() override;

    HLIEEmbeddedModelRuntime(const HLIEEmbeddedModelRuntime&) = delete;
    HLIEEmbeddedModelRuntime& operator=(const HLIEEmbeddedModelRuntime&) = delete;

    [[nodiscard]] HLIERuntimeIdentity identity() const override;
    [[nodiscard]] HLIEModelCapabilities capabilities() const override;
    [[nodiscard]] HLIERuntimeHealth health() const override;
    void begin_draining() override;
    [[nodiscard]] HLIERuntimeStopResult stop(
        const HLIERuntimeStopRequest& request) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_EMBEDDED_MODEL_RUNTIME_H

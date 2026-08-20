#ifndef HLIE_APP_MODEL_RUNTIME_FACTORY_H
#define HLIE_APP_MODEL_RUNTIME_FACTORY_H

#include <memory>
#include <string>
#include <vector>

#include "hlie_core/runtime/hlie_model_runtime.h"

HLIE_NAMESPACE_BEGIN

struct HLIEEmbeddedRuntimeCreateRequest {
    HLIEModelLaunchTicket launch_ticket;
    HLIEModelCapabilities capabilities;
    std::string service_kind;
    bool owns_document_companion = false;
};

struct HLIEProcessRuntimeCreateRequest {
    HLIERuntimeIdentity identity;
    HLIEModelCapabilities capabilities;
    std::vector<std::string> model_names;
    bool ready = false;
};

// App composition port. It creates transport adapters only; Controller owns
// publication, lifecycle state and reservation transactions.
class IHLIEModelRuntimeFactory {
public:
    virtual ~IHLIEModelRuntimeFactory() = default;

    [[nodiscard]] virtual std::shared_ptr<IHLIEModelRuntime> create_embedded(
        const HLIEEmbeddedRuntimeCreateRequest& request) = 0;
    [[nodiscard]] virtual std::shared_ptr<IHLIEModelRuntime> create_process(
        const HLIEProcessRuntimeCreateRequest& request) = 0;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_MODEL_RUNTIME_FACTORY_H

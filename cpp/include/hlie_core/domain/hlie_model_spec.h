#pragma once

#include <string>

#include "hlie_core/domain/hlie_model_capability.h"
#include "hlie_core/domain/hlie_model_runtime_snapshot.h"

HLIE_NAMESPACE_BEGIN

// SDK/filesystem-free canonical input consumed by control-plane policy.
// The app catalog is responsible for observing files and building this value.
struct HLIEModelSpec {
    HLIEModelIdentity identity;
    std::string backend_alias;
    std::string legacy_model_type;
    std::string model_family;
    std::string architecture;
    HLIEModelCapabilities capabilities;
    HLIEModelRole role = HLIEModelRole::Unknown;
    int context_size = 0;
    int batch_size = 1;
    int required_device_count = 1;
    bool is_llm = true;
};

HLIE_NAMESPACE_END

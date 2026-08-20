#pragma once

#include "hlie_base/hlie_base.h"
#include "hlie_core/backend/hlie_backend_registry.h"

HLIE_NAMESPACE_BEGIN

// App composition façade for the current immutable backend table. Filesystem,
// config, and GGUF observation happen at callers; policy remains in hlie-core.
class HLIE_API HLIEModelMemoryEstimator {
public:
    [[nodiscard]] static HLIEModelMemoryEstimate estimate(
        const HLIEModelSpec& spec,
        const HLIEModelMemoryObservation& observation);
};

HLIE_NAMESPACE_END

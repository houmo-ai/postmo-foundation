#include "hlie_app/hlie_model_memory_estimator.h"

HLIE_NAMESPACE_BEGIN

namespace {

} // namespace

HLIEModelMemoryEstimate HLIEModelMemoryEstimator::estimate(
    const HLIEModelSpec& spec,
    const HLIEModelMemoryObservation& observation) {
    const auto descriptor = houmo_default_registry().resolve(spec);
    if (descriptor) {
        return descriptor->estimate_memory(spec, observation);
    }

    // houmo_defaults always contains LLM and SD fallbacks. If a future custom
    // table violates that invariant, reuse the neutral default policy instead
    // of recreating priority or heuristic logic in app.
    return HLIEBackendDescriptor{}.estimate_memory(spec, observation);
}

HLIE_NAMESPACE_END

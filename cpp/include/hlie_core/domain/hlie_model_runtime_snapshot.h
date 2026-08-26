#ifndef HLIE_CORE_MODEL_RUNTIME_SNAPSHOT_H
#define HLIE_CORE_MODEL_RUNTIME_SNAPSHOT_H

#include <cstdint>
#include <ctime>
#include <string>

#include "hlie_core/domain/hlie_model_launch_ticket.h"

HLIE_NAMESPACE_BEGIN

// Framework lifecycle vocabulary. Sleeping is retained only as a lossless
// compatibility projection of the current server_model_status read model; new
// control-plane transitions should use the explicit draining/stopping states.
enum class HLIEModelLifecycleState {
    Unloaded,
    Reserving,
    Starting,
    Ready,
    Sleeping,
    Draining,
    Stopping,
    Failed,
    Orphaned,
};

struct HLIEModelIdentity {
    std::string model_key;
    std::string model_name;
    std::string display_name;
    std::string model_path;
    std::string projector_path;
    std::string backend_id;
};

// Immutable, SDK/HTTP-free read model published by the current Registry.
// Launch tokens intentionally remain in the Registry ticket API and are not
// exposed through general status/health snapshots.
struct HLIEModelRuntimeSnapshot {
    HLIEModelIdentity identity;
    HLIEModelLifecycleState state = HLIEModelLifecycleState::Unloaded;
    HLIEModelGeneration generation = 0;
    int port = 0;
    int failure_code = 0;
    std::time_t last_access_time = 0;
    std::time_t last_load_time = 0;
    std::uint64_t hit_count = 0;
    std::uint64_t active_requests = 0;
    int estimated_mem_mb = 0;
    bool pinned = false;
    bool evictable = true;
};

[[nodiscard]] constexpr bool hlie_model_state_is_loading(
    const HLIEModelLifecycleState state) noexcept {
    return state == HLIEModelLifecycleState::Reserving ||
           state == HLIEModelLifecycleState::Starting;
}

[[nodiscard]] constexpr bool hlie_model_state_is_ready(
    const HLIEModelLifecycleState state) noexcept {
    return state == HLIEModelLifecycleState::Ready;
}

[[nodiscard]] constexpr bool hlie_model_state_protects_reservation(
    const HLIEModelLifecycleState state) noexcept {
    return state == HLIEModelLifecycleState::Reserving ||
           state == HLIEModelLifecycleState::Starting ||
           state == HLIEModelLifecycleState::Ready ||
           state == HLIEModelLifecycleState::Sleeping ||
           state == HLIEModelLifecycleState::Draining ||
           state == HLIEModelLifecycleState::Stopping ||
           state == HLIEModelLifecycleState::Orphaned;
}

HLIE_NAMESPACE_END

#endif // HLIE_CORE_MODEL_RUNTIME_SNAPSHOT_H

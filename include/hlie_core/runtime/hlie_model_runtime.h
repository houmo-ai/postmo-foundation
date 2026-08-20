#ifndef HLIE_CORE_MODEL_RUNTIME_H
#define HLIE_CORE_MODEL_RUNTIME_H

#include <chrono>
#include <cstdint>
#include <string>

#include "hlie_core/domain/hlie_model_capability.h"
#include "hlie_core/domain/hlie_model_launch_ticket.h"

HLIE_NAMESPACE_BEGIN

using HLIERuntimeId = std::uint64_t;
using HLIERuntimeDeadline = std::chrono::steady_clock::time_point;

struct HLIERuntimeIdentity {
    HLIERuntimeId runtime_id = 0;
    HLIERuntimeKind kind = HLIERuntimeKind::Embedded;
    HLIEModelLaunchTicket launch_ticket;
    int instance_id = 0;
    int endpoint_port = 0;

    [[nodiscard]] bool valid() const noexcept {
        if (runtime_id == 0 || !launch_ticket.valid()) {
            return false;
        }
        if (kind == HLIERuntimeKind::Process) {
            return instance_id > 0 && endpoint_port > 0;
        }
        return instance_id == 0 && endpoint_port == 0;
    }

    [[nodiscard]] bool matches(const HLIERuntimeIdentity& other) const noexcept {
        return valid() && other.valid() && runtime_id == other.runtime_id &&
               kind == other.kind && launch_ticket.matches(other.launch_ticket) &&
               instance_id == other.instance_id && endpoint_port == other.endpoint_port;
    }
};

enum class HLIERuntimeHealthState {
    Starting,
    Ready,
    Draining,
    Stopped,
    Failed,
    Orphaned,
};

struct HLIERuntimeHealth {
    HLIERuntimeIdentity identity;
    HLIERuntimeHealthState state = HLIERuntimeHealthState::Starting;
    int native_code = 0;

    [[nodiscard]] bool ready() const noexcept {
        return identity.valid() && state == HLIERuntimeHealthState::Ready;
    }
};

enum class HLIERuntimeStopReason {
    ManualUnload,
    Eviction,
    Shutdown,
    StartupRollback,
    Recovery,
};

struct HLIERuntimeStopRequest {
    HLIERuntimeDeadline deadline{};
    HLIERuntimeStopReason reason = HLIERuntimeStopReason::ManualUnload;
    bool force = false;

    [[nodiscard]] bool valid() const noexcept {
        return deadline != HLIERuntimeDeadline{};
    }
};

struct HLIERuntimeStopResult {
    HLIERuntimeIdentity identity;
    bool confirmed = false;
    bool graceful = false;
    bool already_stopped = false;
    bool deadline_exceeded = false;
    int native_code = 0;
};

// Control-plane runtime contract. Inference dispatch intentionally remains a
// separate data-plane port so core never needs HTTP DTOs or opaque payloads.
class IHLIEModelRuntime {
public:
    virtual ~IHLIEModelRuntime() = default;

    [[nodiscard]] virtual HLIERuntimeIdentity identity() const = 0;
    [[nodiscard]] virtual HLIEModelCapabilities capabilities() const = 0;
    [[nodiscard]] virtual HLIERuntimeHealth health() const = 0;
    virtual void begin_draining() = 0;
    [[nodiscard]] virtual HLIERuntimeStopResult stop(
        const HLIERuntimeStopRequest& request) = 0;
};

HLIE_NAMESPACE_END

#endif // HLIE_CORE_MODEL_RUNTIME_H

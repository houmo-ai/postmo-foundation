#ifndef HLIE_RUNTIME_EVENT_H
#define HLIE_RUNTIME_EVENT_H

#include <cstdint>

#include "hlie_base/hlie_base.h"
#include "hlie_core/domain/hlie_model_launch_ticket.h"

HLIE_NAMESPACE_BEGIN

enum class HLIERuntimeEventKind {
    Ready,
    ReadyTimeout,
    Exited,
};

/**
 * A generation-fenced fact observed at the runtime/OS boundary.
 *
 * This is deliberately not a lifecycle command. The event sink remains
 * responsible for validating the launch ticket and deciding how Registry and
 * reservation state should change.
 */
struct HLIERuntimeEvent {
    HLIERuntimeEventKind kind = HLIERuntimeEventKind::Exited;
    HLIEModelLaunchTicket launch_ticket;
    std::uint64_t runtime_id = 0;
    int instance_id = 0;
    int port = 0;
    int exit_code = 0;
    bool expected_exit = false;
    bool exit_confirmed = false;
};

class IHLIERuntimeEventSink {
public:
    virtual ~IHLIERuntimeEventSink() = default;
    virtual void on_runtime_event(const HLIERuntimeEvent& event) = 0;
};

HLIE_NAMESPACE_END

#endif // HLIE_RUNTIME_EVENT_H

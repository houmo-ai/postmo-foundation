#ifndef HLIE_WORKER_PORT_ALLOCATOR_H
#define HLIE_WORKER_PORT_ALLOCATOR_H

#include <functional>
#include <optional>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

struct HLIEWorkerPortAssignment {
    int instance_id = 0;
    int port = 0;
    int skipped_candidates = 0;
};

class HLIEWorkerPortAllocator {
public:
    using PortProbe = std::function<bool(int)>;

    [[nodiscard]] static std::optional<HLIEWorkerPortAssignment> find_available(
        int master_port, int first_instance_id, int max_attempts,
        const PortProbe& probe);
    [[nodiscard]] static bool is_tcp_port_available(int port);
};

HLIE_NAMESPACE_END

#endif // HLIE_WORKER_PORT_ALLOCATOR_H

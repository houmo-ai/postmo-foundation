#ifndef HLIE_WORKER_SUPERVISOR_H
#define HLIE_WORKER_SUPERVISOR_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"
#include "hlie_core/runtime/hlie_runtime_event.h"

HLIE_NAMESPACE_BEGIN

class HLIEProcessManager;

struct HLIEWorkerSupervisorOptions {
    int master_port = 7901;
    int port_probe_attempts = 128;
    std::chrono::milliseconds poll_interval{500};
    std::chrono::seconds ready_timeout{120};
    std::chrono::milliseconds graceful_shutdown_timeout{5000};
};

struct HLIEWorkerPortLease {
    std::uint64_t lease_id = 0;
    int instance_id = 0;
    int port = 0;

    [[nodiscard]] bool valid() const noexcept {
        return lease_id != 0 && instance_id > 0 && port > 0;
    }
};

struct HLIEWorkerLaunchRequest {
    HLIEWorkerPortLease port_lease;
    std::string executable;
    std::vector<std::string> arguments;
    std::vector<std::string> model_names;
    std::string launch_manifest_path;
    HLIEModelLaunchTicket launch_ticket;
};

struct HLIEWorkerRuntime {
    std::uint64_t runtime_id = 0;
    int instance_id = 0;
    int port = 0;
    std::vector<std::string> model_names;
    HLIEModelLaunchTicket launch_ticket;
    bool ready = false;

    [[nodiscard]] bool valid() const noexcept {
        return runtime_id != 0 && instance_id > 0 && port > 0;
    }
};

struct HLIEWorkerStopRequest {
    std::string reason;
    std::chrono::milliseconds graceful_timeout{5000};
    bool force = false;
};

struct HLIEWorkerStopResult {
    HLIEWorkerRuntime runtime;
    bool confirmed = false;
    bool graceful = false;
};

struct HLIEWorkerSupervisorMetrics {
    std::uint64_t unexpected_exits = 0;
    std::uint64_t ready_timeouts = 0;
    std::uint64_t graceful_shutdowns = 0;
    std::uint64_t forced_shutdowns = 0;
    std::uint64_t port_conflicts = 0;
};

/**
 * Owns dynamic Worker process supervision and emits runtime facts.
 *
 * The class intentionally has no dependency on Workflow, Registry,
 * DeviceService, reservation ownership, or HTTP response DTOs.
 */
class HLIE_API HLIEWorkerSupervisor {
public:
    HLIEWorkerSupervisor(HLIEProcessManager& process_manager,
                         HLIEWorkerSupervisorOptions options,
                         IHLIERuntimeEventSink& event_sink);
    ~HLIEWorkerSupervisor();

    HLIEWorkerSupervisor(const HLIEWorkerSupervisor&) = delete;
    HLIEWorkerSupervisor& operator=(const HLIEWorkerSupervisor&) = delete;

    [[nodiscard]] static HLIEWorkerSupervisorOptions configured_options(int master_port);

    void start();
    void stop();
    [[nodiscard]] bool running() const;

    [[nodiscard]] std::optional<HLIEWorkerPortLease> reserve_port();
    void release_port_lease(const HLIEWorkerPortLease& lease);

    [[nodiscard]] std::optional<HLIEWorkerRuntime> spawn(
        const HLIEWorkerLaunchRequest& request);
    [[nodiscard]] std::optional<HLIEWorkerRuntime> find_runtime(
        std::uint64_t runtime_id) const;
    [[nodiscard]] std::vector<HLIEWorkerRuntime> running_runtimes() const;

    [[nodiscard]] HLIEWorkerStopResult request_stop(
        const HLIEWorkerRuntime& runtime, const std::string& reason);
    [[nodiscard]] HLIEWorkerStopResult request_stop(
        const HLIEWorkerRuntime& runtime, const HLIEWorkerStopRequest& request);
    [[nodiscard]] std::vector<HLIEWorkerStopResult> shutdown_all(
        const std::string& reason);

    [[nodiscard]] HLIEWorkerSupervisorMetrics metrics() const noexcept;

    static void discard_launch_manifest(const std::string& path,
                                        const char* reason);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

HLIE_NAMESPACE_END

#endif // HLIE_WORKER_SUPERVISOR_H

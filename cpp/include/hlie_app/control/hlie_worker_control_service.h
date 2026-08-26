#ifndef HLIE_APP_CONTROL_WORKER_CONTROL_SERVICE_H
#define HLIE_APP_CONTROL_WORKER_CONTROL_SERVICE_H

#include <atomic>
#include <cstdint>
#include <ctime>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "hlie_app/util/hlie_background_task_group.h"
#include "hlie_core/domain/hlie_model_launch_ticket.h"

HLIE_NAMESPACE_BEGIN

enum class HLIEWorkerModelStatus {
    Unloaded,
    Loading,
    Loaded,
    Sleeping,
};

struct HLIEWorkerStatusUpdate {
    std::string model_key;
    HLIEWorkerModelStatus status = HLIEWorkerModelStatus::Unloaded;
    int port = 0;
    HLIEModelGeneration generation = 0;
    std::string launch_token;
};

struct HLIEWorkerStatusPublication {
    int router_port = 0;
    std::string model_key;
    HLIEWorkerModelStatus status = HLIEWorkerModelStatus::Unloaded;
    int port = 0;
    HLIEModelLaunchTicket launch_ticket;
};

class IHLIEWorkerStatusPublisher {
public:
    virtual ~IHLIEWorkerStatusPublisher() = default;

    [[nodiscard]] virtual bool publish(
        const HLIEWorkerStatusPublication& publication) = 0;
};

struct HLIEWorkerProcessorSnapshot {
    std::string params;
    int instance_id = 0;
    int port = 0;
    std::string uuid;
    std::uint64_t process_id = 0;
    HLIEModelGeneration generation = 0;
    std::time_t start_time = 0;
    std::vector<std::string> models;
};

struct HLIEWorkerStatusSnapshot {
    int local_port = 0;
    std::time_t local_start_time = 0;
    std::vector<std::string> local_models;
    std::vector<HLIEWorkerProcessorSnapshot> workers;
};

enum class HLIEWorkerStatusApplyResult {
    Applied,
    LookupFailed,
    Stale,
    Rejected,
};

enum class HLIEWorkerShutdownResult {
    Accepted,
    AlreadyRequested,
    NotDynamicWorker,
    InvalidTicket,
    CallbackUnavailable,
};

struct HLIEWorkerControlDependencies {
    std::function<bool()> dynamic_worker_provider;
    std::function<bool()> replicas_mode_provider;
    std::function<int()> local_port_provider;
    std::function<int()> master_port_provider;
    std::function<HLIEModelLaunchTicket()> launch_ticket_provider;
    std::function<std::time_t()> local_start_time_provider;
    std::function<std::vector<std::string>()> local_models_provider;
    std::function<std::vector<HLIEWorkerProcessorSnapshot>()>
        worker_processes_provider;

    std::function<std::optional<std::string>(const std::string&)>
        resolve_model_key;
    std::function<bool(const HLIEModelLaunchTicket&,
                       HLIEWorkerModelStatus, int)>
        apply_status;
    std::function<void(const std::string&)> erase_model;

    std::function<void()> begin_draining;
    std::function<bool()> process_shutdown_available;
    std::function<void()> process_shutdown;
};

// Owns the compatibility control protocol used by dynamic Workers. The
// service is deliberately transport-neutral: HTTP parsing and legacy status
// conversion stay in adapters/composition, while this class owns ticket
// fencing, delayed READY publication, status application, and shutdown
// idempotency.
class HLIEWorkerControlService {
public:
    HLIEWorkerControlService(
        IHLIEWorkerStatusPublisher& publisher,
        HLIEWorkerControlDependencies dependencies);
    ~HLIEWorkerControlService();

    HLIEWorkerControlService(const HLIEWorkerControlService&) = delete;
    HLIEWorkerControlService& operator=(const HLIEWorkerControlService&) = delete;

    [[nodiscard]] bool schedule_loaded(const std::string& model_key,
                                       const HLIEModelLaunchTicket& ticket);
    [[nodiscard]] bool publish_loaded(const std::string& model_key,
                                      const HLIEModelLaunchTicket& ticket);
    [[nodiscard]] bool publish_unloaded(const std::string& model_key,
                                        const HLIEModelLaunchTicket& ticket);

    [[nodiscard]] HLIEWorkerStatusApplyResult apply_status(
        HLIEWorkerStatusUpdate update);
    [[nodiscard]] HLIEWorkerStatusSnapshot snapshot() const;
    [[nodiscard]] HLIEWorkerShutdownResult request_shutdown(
        HLIEModelGeneration generation, const std::string& launch_token);

    void stop_and_join();

private:
    [[nodiscard]] bool is_dynamic_worker() const;
    [[nodiscard]] bool can_publish() const;
    [[nodiscard]] bool publish(const std::string& model_key,
                               HLIEWorkerModelStatus status,
                               int port,
                               const HLIEModelLaunchTicket& ticket);
    void request_process_shutdown();

    IHLIEWorkerStatusPublisher& publisher_;
    HLIEWorkerControlDependencies dependencies_;
    BackgroundTaskGroup tasks_;
    std::atomic<bool> status_update_stop_{false};
    std::atomic<bool> process_shutdown_requested_{false};
    mutable std::mutex status_mutex_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_CONTROL_WORKER_CONTROL_SERVICE_H

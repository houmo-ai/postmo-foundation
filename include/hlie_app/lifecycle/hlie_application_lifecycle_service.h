#ifndef HLIE_APP_APPLICATION_LIFECYCLE_SERVICE_H
#define HLIE_APP_APPLICATION_LIFECYCLE_SERVICE_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>

#include "hlie_app/control/hlie_model_controller.h"
#include "hlie_app/startup/hlie_embedded_startup_service.h"

HLIE_NAMESPACE_BEGIN

enum class HLIEApplicationLifecyclePhase {
    Composed,
    Starting,
    Running,
    Publishing,
    Serving,
    StartFailed,
    Draining,
    Stopping,
    Stopped,
};

enum class HLIEApplicationStartStatus {
    Started,
    AlreadyStarted,
    PrimaryStartupAborted,
    Failed,
};

enum class HLIEApplicationServingStatus {
    Published,
    AlreadyPublished,
    NotRunning,
    Failed,
};

struct HLIEApplicationLifecycleOptions {
    bool worker_process = false;
    int http_threads = 1;
    int reserved_control_threads = 2;
    int llm_max_in_flight = 64;
    int qwen3_tts_max_in_flight = 1;
};

struct HLIEApplicationStartResult {
    HLIEApplicationStartStatus status = HLIEApplicationStartStatus::Failed;
    HLIEEmbeddedStartupSummary primary;
    HLIEEmbeddedStartupSummary secondary;
    std::size_t effective_llm_max_in_flight = 0;
    std::size_t effective_qwen3_tts_max_in_flight = 0;

    [[nodiscard]] bool started() const noexcept {
        return status == HLIEApplicationStartStatus::Started ||
               status == HLIEApplicationStartStatus::AlreadyStarted;
    }
};

struct HLIEApplicationServingResult {
    HLIEApplicationServingStatus status =
        HLIEApplicationServingStatus::Failed;

    [[nodiscard]] bool published() const noexcept {
        return status == HLIEApplicationServingStatus::Published ||
               status == HLIEApplicationServingStatus::AlreadyPublished;
    }
};

struct HLIEApplicationStopResult {
    HLIEControllerStopSummary runtimes;
    bool workers_confirmed = true;
    bool residual_services_confirmed = true;
    bool residual_status_publications_confirmed = true;
    bool callbacks_complete = true;
    bool already_stopped = false;

    [[nodiscard]] bool complete() const noexcept {
        return runtimes.complete() && workers_confirmed &&
               residual_services_confirmed &&
               residual_status_publications_confirmed &&
               callbacks_complete;
    }
};

struct HLIEResidualServiceCleanupResult {
    bool services_confirmed = true;
    bool status_publications_confirmed = true;

    [[nodiscard]] bool complete() const noexcept {
        return services_confirmed && status_publications_confirmed;
    }
};

struct HLIEApplicationLifecycleDependencies {
    std::function<HLIEEmbeddedStartupSummary(HLIEEmbeddedStartupGroup)>
        start_embedded_group;
    std::function<void(std::size_t)> configure_llm_admission;
    std::function<void(std::size_t)> configure_qwen3_tts_admission;
    std::function<void()> start_asr_realtime;
    std::function<void()> stop_asr_realtime;
    std::function<void()> start_reconciler;
    std::function<void()> stop_reconciler;
    std::function<void()> start_discovery;
    std::function<void()> stop_discovery;
    std::function<void()> start_worker_supervisor;
    std::function<void()> controller_begin_draining;
    std::function<void()> stop_readiness;
    std::function<HLIEControllerStopSummary()> stop_runtimes;
    std::function<bool(const std::string&)> shutdown_workers;
    std::function<void()> stop_worker_supervisor;
    std::function<HLIEResidualServiceCleanupResult()>
        cleanup_residual_services;
    std::function<void()> stop_worker_control;
};

// Owns process-level startup and shutdown ordering. Concrete services stay in
// the composition root and enter through transport-neutral callbacks.
class HLIEApplicationLifecycleService final {
public:
    explicit HLIEApplicationLifecycleService(
        HLIEApplicationLifecycleDependencies dependencies);

    [[nodiscard]] HLIEApplicationStartResult start(
        const HLIEApplicationLifecycleOptions& options);
    // Serializes externally visible serving publication against draining.
    // The callback must not re-enter this lifecycle service.
    [[nodiscard]] HLIEApplicationServingResult try_publish_serving(
        const std::function<bool()>& publisher);
    void begin_draining() noexcept;
    [[nodiscard]] HLIEApplicationStopResult stop();

    [[nodiscard]] HLIEApplicationLifecyclePhase phase() const noexcept;
    [[nodiscard]] bool draining() const noexcept;

private:
    [[nodiscard]] bool validate_dependencies() const;

    HLIEApplicationLifecycleDependencies dependencies_;

    mutable std::mutex mutex_;
    std::condition_variable state_changed_;
    HLIEApplicationLifecyclePhase phase_ =
        HLIEApplicationLifecyclePhase::Composed;
    HLIEApplicationStartResult start_result_;
    HLIEApplicationStopResult stop_result_;
    std::once_flag draining_once_;
    std::atomic<bool> draining_{false};
    std::atomic<bool> draining_callback_complete_{true};
    bool serving_publication_in_progress_ = false;
    std::mutex serving_publication_mutex_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_APPLICATION_LIFECYCLE_SERVICE_H

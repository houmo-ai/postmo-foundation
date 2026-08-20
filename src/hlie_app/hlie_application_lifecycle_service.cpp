#include "hlie_app/lifecycle/hlie_application_lifecycle_service.h"

#include <algorithm>
#include <exception>
#include <utility>

#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

namespace {

template <typename Callback>
bool invoke_shutdown_callback(const char* name, const Callback& callback) {
    if (!callback) {
        LOG_ERROR("Lifecycle shutdown callback is unavailable: {}", name);
        return false;
    }
    try {
        callback();
        return true;
    } catch (const std::exception& error) {
        LOG_ERROR("Lifecycle shutdown callback failed: callback={}, error={}",
                  name, error.what());
    } catch (...) {
        LOG_ERROR("Lifecycle shutdown callback failed: callback={}, error=unknown",
                  name);
    }
    return false;
}

} // namespace

HLIEApplicationLifecycleService::HLIEApplicationLifecycleService(
    HLIEApplicationLifecycleDependencies dependencies)
    : dependencies_(std::move(dependencies)) {}

bool HLIEApplicationLifecycleService::validate_dependencies() const {
    bool valid = true;
#define HLIE_REQUIRE_LIFECYCLE_CALLBACK(member)                             \
    do {                                                                    \
        if (!dependencies_.member) {                                        \
            LOG_ERROR("Required lifecycle callback is unavailable: {}",    \
                      #member);                                             \
            valid = false;                                                  \
        }                                                                   \
    } while (false)

    HLIE_REQUIRE_LIFECYCLE_CALLBACK(start_embedded_group);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(configure_llm_admission);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(configure_qwen3_tts_admission);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(start_asr_realtime);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(stop_asr_realtime);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(start_reconciler);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(stop_reconciler);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(start_discovery);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(stop_discovery);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(start_worker_supervisor);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(controller_begin_draining);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(stop_readiness);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(stop_runtimes);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(shutdown_workers);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(stop_worker_supervisor);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(cleanup_residual_services);
    HLIE_REQUIRE_LIFECYCLE_CALLBACK(stop_worker_control);

#undef HLIE_REQUIRE_LIFECYCLE_CALLBACK
    return valid;
}

HLIEApplicationStartResult HLIEApplicationLifecycleService::start(
    const HLIEApplicationLifecycleOptions& options) {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        state_changed_.wait(lock, [this] {
            return phase_ != HLIEApplicationLifecyclePhase::Starting;
        });
        if (phase_ == HLIEApplicationLifecyclePhase::Running ||
            phase_ == HLIEApplicationLifecyclePhase::Publishing ||
            phase_ == HLIEApplicationLifecyclePhase::Serving) {
            auto result = start_result_;
            result.status = HLIEApplicationStartStatus::AlreadyStarted;
            return result;
        }
        if (phase_ != HLIEApplicationLifecyclePhase::Composed) {
            if (phase_ == HLIEApplicationLifecyclePhase::Draining ||
                phase_ == HLIEApplicationLifecyclePhase::Stopping ||
                phase_ == HLIEApplicationLifecyclePhase::Stopped) {
                auto result = start_result_;
                result.status = HLIEApplicationStartStatus::Failed;
                return result;
            }
            return start_result_;
        }
        if (!validate_dependencies()) {
            phase_ = HLIEApplicationLifecyclePhase::StartFailed;
            state_changed_.notify_all();
            return start_result_;
        }
        phase_ = HLIEApplicationLifecyclePhase::Starting;
    }

    HLIEApplicationStartResult result;
    try {
        result.primary = dependencies_.start_embedded_group(
            HLIEEmbeddedStartupGroup::Primary);
        if (result.primary.abort_remaining) {
            result.status =
                HLIEApplicationStartStatus::PrimaryStartupAborted;
            std::lock_guard<std::mutex> lock(mutex_);
            start_result_ = result;
            phase_ = HLIEApplicationLifecyclePhase::StartFailed;
            state_changed_.notify_all();
            return result;
        }

        const int reserved_control_threads = std::clamp(
            options.reserved_control_threads, 1,
            std::max(1, options.http_threads - 1));
        const int data_plane_capacity =
            std::max(1, options.http_threads - reserved_control_threads);
        const int configured_llm_limit =
            std::clamp(options.llm_max_in_flight, 1, 4096);
        const int configured_qwen_limit =
            std::clamp(options.qwen3_tts_max_in_flight, 1, 64);
        result.effective_llm_max_in_flight = static_cast<std::size_t>(
            std::min(configured_llm_limit, data_plane_capacity));
        result.effective_qwen3_tts_max_in_flight = static_cast<std::size_t>(
            std::min(configured_qwen_limit, data_plane_capacity));

        dependencies_.configure_llm_admission(
            result.effective_llm_max_in_flight);
        dependencies_.configure_qwen3_tts_admission(
            result.effective_qwen3_tts_max_in_flight);
        LOG_INFO(
            "Inference admission configured: http_threads={}, "
            "reserved_control_threads={}, llm_limit={}, qwen3_tts_limit={}",
            options.http_threads, reserved_control_threads,
            result.effective_llm_max_in_flight,
            result.effective_qwen3_tts_max_in_flight);

        result.secondary = dependencies_.start_embedded_group(
            HLIEEmbeddedStartupGroup::Secondary);
        dependencies_.start_asr_realtime();
        dependencies_.start_reconciler();
        if (!options.worker_process) {
            dependencies_.start_discovery();
            dependencies_.start_worker_supervisor();
        }

        result.status = HLIEApplicationStartStatus::Started;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            start_result_ = result;
            phase_ = draining_.load(std::memory_order_acquire)
                ? HLIEApplicationLifecyclePhase::Draining
                : HLIEApplicationLifecyclePhase::Running;
        }
        state_changed_.notify_all();
        LOG_INFO("Application lifecycle startup completed");
        return result;
    } catch (const std::exception& error) {
        LOG_ERROR("Application lifecycle startup failed: {}", error.what());
    } catch (...) {
        LOG_ERROR("Application lifecycle startup failed: unknown exception");
    }

    result.status = HLIEApplicationStartStatus::Failed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        start_result_ = result;
        phase_ = HLIEApplicationLifecyclePhase::StartFailed;
    }
    state_changed_.notify_all();
    return result;
}

HLIEApplicationServingResult
HLIEApplicationLifecycleService::try_publish_serving(
    const std::function<bool()>& publisher) {
    if (!publisher) {
        LOG_ERROR("Application serving publisher is unavailable");
        return {HLIEApplicationServingStatus::Failed};
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (phase_ == HLIEApplicationLifecyclePhase::Serving) {
        LOG_DEBUG("Application serving state was already published");
        return {HLIEApplicationServingStatus::AlreadyPublished};
    }
    if (phase_ != HLIEApplicationLifecyclePhase::Running ||
        serving_publication_in_progress_) {
        LOG_DEBUG("Application serving publication rejected: phase={}",
                  static_cast<int>(phase_));
        return {HLIEApplicationServingStatus::NotRunning};
    }

    serving_publication_in_progress_ = true;
    phase_ = HLIEApplicationLifecyclePhase::Publishing;
    lock.unlock();
    std::unique_lock<std::mutex> publication_lock(serving_publication_mutex_);

    bool published = false;
    try {
        published = publisher();
    } catch (const std::exception& error) {
        LOG_ERROR("Application serving publisher failed: {}", error.what());
    } catch (...) {
        LOG_ERROR("Application serving publisher failed: unknown exception");
    }

    publication_lock.unlock();
    lock.lock();
    serving_publication_in_progress_ = false;
    const bool can_publish = published && !draining_.load(std::memory_order_acquire) &&
        phase_ == HLIEApplicationLifecyclePhase::Publishing;
    if (!can_publish) {
        phase_ = draining_.load(std::memory_order_acquire)
            ? HLIEApplicationLifecyclePhase::Draining
            : HLIEApplicationLifecyclePhase::Running;
        lock.unlock();
        state_changed_.notify_all();
        if (!published) {
            LOG_ERROR("Application serving publisher reported failure");
        } else {
            LOG_DEBUG("Application serving publication superseded by draining");
        }
        return {published ? HLIEApplicationServingStatus::NotRunning
                          : HLIEApplicationServingStatus::Failed};
    }
    phase_ = HLIEApplicationLifecyclePhase::Serving;
    lock.unlock();
    state_changed_.notify_all();
    LOG_INFO("Application serving state published");
    return {HLIEApplicationServingStatus::Published};
}

void HLIEApplicationLifecycleService::begin_draining() noexcept {
    std::call_once(draining_once_, [this] {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            state_changed_.wait(lock, [this] {
                return phase_ != HLIEApplicationLifecyclePhase::Starting &&
                       !serving_publication_in_progress_;
            });
            if (phase_ == HLIEApplicationLifecyclePhase::Composed ||
                phase_ == HLIEApplicationLifecyclePhase::Running ||
                phase_ == HLIEApplicationLifecyclePhase::Publishing ||
                phase_ == HLIEApplicationLifecyclePhase::Serving) {
                phase_ = HLIEApplicationLifecyclePhase::Draining;
            }
        }
        draining_.store(true, std::memory_order_release);
        LOG_INFO("Application lifecycle entered draining state");
        if (!invoke_shutdown_callback(
                "controller_begin_draining",
                dependencies_.controller_begin_draining)) {
            draining_callback_complete_.store(false,
                                               std::memory_order_release);
        }
    });
}

HLIEApplicationStopResult HLIEApplicationLifecycleService::stop() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        state_changed_.wait(lock, [this] {
            return phase_ != HLIEApplicationLifecyclePhase::Starting &&
                   !serving_publication_in_progress_;
        });
        if (phase_ == HLIEApplicationLifecyclePhase::Stopping) {
            state_changed_.wait(lock, [this] {
                return phase_ == HLIEApplicationLifecyclePhase::Stopped;
            });
            auto result = stop_result_;
            result.already_stopped = true;
            return result;
        }
        if (phase_ == HLIEApplicationLifecyclePhase::Stopped) {
            auto result = stop_result_;
            result.already_stopped = true;
            return result;
        }
        phase_ = HLIEApplicationLifecyclePhase::Stopping;
    }

    HLIEApplicationStopResult result;
    begin_draining();
    result.callbacks_complete =
        draining_callback_complete_.load(std::memory_order_acquire);

    result.callbacks_complete &= invoke_shutdown_callback(
        "stop_discovery", dependencies_.stop_discovery);
    result.callbacks_complete &= invoke_shutdown_callback(
        "stop_reconciler", dependencies_.stop_reconciler);
    result.callbacks_complete &= invoke_shutdown_callback(
        "stop_readiness", dependencies_.stop_readiness);
    result.callbacks_complete &= invoke_shutdown_callback(
        "stop_asr_realtime", dependencies_.stop_asr_realtime);

    if (!dependencies_.stop_runtimes) {
        LOG_ERROR("Lifecycle shutdown callback is unavailable: stop_runtimes");
        result.callbacks_complete = false;
    } else {
        try {
            result.runtimes = dependencies_.stop_runtimes();
        } catch (const std::exception& error) {
            LOG_ERROR(
                "Lifecycle shutdown callback failed: callback=stop_runtimes, "
                "error={}",
                error.what());
            result.callbacks_complete = false;
        } catch (...) {
            LOG_ERROR(
                "Lifecycle shutdown callback failed: callback=stop_runtimes, "
                "error=unknown");
            result.callbacks_complete = false;
        }
    }

    if (!dependencies_.shutdown_workers) {
        LOG_ERROR("Lifecycle shutdown callback is unavailable: shutdown_workers");
        result.workers_confirmed = false;
        result.callbacks_complete = false;
    } else {
        try {
            result.workers_confirmed =
                dependencies_.shutdown_workers("application_stop");
        } catch (const std::exception& error) {
            LOG_ERROR(
                "Lifecycle shutdown callback failed: callback=shutdown_workers, "
                "error={}",
                error.what());
            result.workers_confirmed = false;
            result.callbacks_complete = false;
        } catch (...) {
            LOG_ERROR(
                "Lifecycle shutdown callback failed: callback=shutdown_workers, "
                "error=unknown");
            result.workers_confirmed = false;
            result.callbacks_complete = false;
        }
    }

    result.callbacks_complete &= invoke_shutdown_callback(
        "stop_worker_supervisor", dependencies_.stop_worker_supervisor);

    if (!dependencies_.cleanup_residual_services) {
        LOG_ERROR(
            "Lifecycle shutdown callback is unavailable: "
            "cleanup_residual_services");
        result.residual_services_confirmed = false;
        result.callbacks_complete = false;
    } else {
        try {
            const auto residual =
                dependencies_.cleanup_residual_services();
            result.residual_services_confirmed =
                residual.services_confirmed;
            result.residual_status_publications_confirmed =
                residual.status_publications_confirmed;
        } catch (const std::exception& error) {
            LOG_ERROR(
                "Lifecycle shutdown callback failed: "
                "callback=cleanup_residual_services, error={}",
                error.what());
            result.residual_services_confirmed = false;
            result.residual_status_publications_confirmed = false;
            result.callbacks_complete = false;
        } catch (...) {
            LOG_ERROR(
                "Lifecycle shutdown callback failed: "
                "callback=cleanup_residual_services, error=unknown");
            result.residual_services_confirmed = false;
            result.residual_status_publications_confirmed = false;
            result.callbacks_complete = false;
        }
    }

    result.callbacks_complete &= invoke_shutdown_callback(
        "stop_worker_control", dependencies_.stop_worker_control);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_result_ = result;
        phase_ = HLIEApplicationLifecyclePhase::Stopped;
    }
    state_changed_.notify_all();

    if (result.complete()) {
        LOG_INFO(
            "Application lifecycle shutdown completed: runtimes_stopped={}",
            result.runtimes.stopped);
    } else {
        LOG_ERROR(
            "Application lifecycle shutdown incomplete: "
            "runtime_unconfirmed={}, workers_confirmed={}, "
            "residual_services_confirmed={}, "
            "residual_status_publications_confirmed={}, "
            "callbacks_complete={}",
            result.runtimes.unconfirmed, result.workers_confirmed,
            result.residual_services_confirmed,
            result.residual_status_publications_confirmed,
            result.callbacks_complete);
    }
    return result;
}

HLIEApplicationLifecyclePhase HLIEApplicationLifecycleService::phase()
    const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return phase_;
}

bool HLIEApplicationLifecycleService::draining() const noexcept {
    return draining_.load(std::memory_order_acquire);
}

HLIE_NAMESPACE_END

#include "hlie_app/control/hlie_model_readiness_service.h"

#include <exception>
#include <utility>

#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

HLIEModelReadinessService::HLIEModelReadinessService(
    HLIEModelController& controller,
    LoadFunction load,
    DrainingProvider draining_provider,
    LoadSucceededCallback load_succeeded)
    : controller_(controller),
      load_(std::move(load)),
      draining_provider_(std::move(draining_provider)),
      load_succeeded_(std::move(load_succeeded)) {}

HLIEModelReadinessService::~HLIEModelReadinessService() {
    stop_and_join();
}

bool HLIEModelReadinessService::is_draining() const {
    return draining_provider_ && draining_provider_();
}

HLIEModelReadinessResult HLIEModelReadinessService::ensure_ready(
    HLIEModelReadinessRequest request) {
    if (request.model_key.empty()) {
        return {HLIEModelReadinessStatus::Missing, {}};
    }
    const std::string model_key = request.model_key;
    const auto decision = controller_.ensure_ready(model_key);
    switch (decision.disposition) {
    case HLIEEnsureReadyDisposition::Ready:
        return {HLIEModelReadinessStatus::Ready, model_key};
    case HLIEEnsureReadyDisposition::Missing:
        return {HLIEModelReadinessStatus::Missing, model_key};
    case HLIEEnsureReadyDisposition::Unavailable:
        return {HLIEModelReadinessStatus::Unavailable, model_key};
    case HLIEEnsureReadyDisposition::LaunchRequired:
        if (!decision.launch_ticket || !request.load_request || !load_) {
            if (decision.launch_ticket) {
                controller_.rollback_model_start(
                    *decision.launch_ticket,
                    "readiness_load_request_unavailable");
            }
            return {HLIEModelReadinessStatus::InvalidLoadRequest, model_key};
        }
        {
            const auto launch_ticket = *decision.launch_ticket;
            const int local_port = request.local_port;
            const bool launched = tasks_.launch(
                [this, model_key, launch_ticket, local_port,
                 load_request = std::move(*request.load_request)] {
                    if (is_draining()) {
                        controller_.rollback_model_start(
                            launch_ticket,
                            "readiness_draining_before_auto_load");
                        return;
                    }
                    try {
                        auto result = load_(
                            load_request, local_port, launch_ticket);
                        if (const auto* error = result.error_if()) {
                            controller_.rollback_model_start(
                                launch_ticket, "readiness_auto_load_failed");
                            LOG_ERROR(
                                "Automatic model load failed: model={}, "
                                "generation={}, stage={}, code={}, "
                                "native_code={}, error={}",
                                model_key, launch_ticket.generation,
                                static_cast<int>(error->stage),
                                static_cast<int>(error->code),
                                error->native_code, error->message);
                        } else if (const auto* receipt = result.value_if();
                                   receipt && load_succeeded_) {
                            load_succeeded_(*receipt);
                        }
                    } catch (const std::exception& error) {
                        controller_.rollback_model_start(
                            launch_ticket, "readiness_auto_load_exception");
                        LOG_ERROR(
                            "Automatic model load threw: model={}, "
                            "generation={}, error={}",
                            model_key, launch_ticket.generation, error.what());
                    } catch (...) {
                        controller_.rollback_model_start(
                            launch_ticket,
                            "readiness_auto_load_unknown_exception");
                        LOG_ERROR(
                            "Automatic model load threw an unknown exception: "
                            "model={}, generation={}",
                            model_key, launch_ticket.generation);
                    }
                });
            if (!launched) {
                controller_.rollback_model_start(
                    launch_ticket, "readiness_task_launch_failed");
                return {HLIEModelReadinessStatus::ScheduleFailed, model_key};
            }
        }
        break;
    case HLIEEnsureReadyDisposition::WaitForExistingLaunch:
        break;
    }

    const auto should_stop = [this, &request] {
        return is_draining() ||
               (request.should_stop && request.should_stop());
    };
    const auto wait_result = controller_.wait_until_ready(
        model_key, request.deadline, should_stop);
    switch (wait_result.status) {
    case HLIEWaitReadyStatus::Ready:
        return {HLIEModelReadinessStatus::Ready, model_key};
    case HLIEWaitReadyStatus::Missing:
        return {HLIEModelReadinessStatus::Missing, model_key};
    case HLIEWaitReadyStatus::Unavailable:
        return {HLIEModelReadinessStatus::Unavailable, model_key};
    case HLIEWaitReadyStatus::TimedOut:
        return {HLIEModelReadinessStatus::TimedOut, model_key};
    case HLIEWaitReadyStatus::Cancelled:
        return {HLIEModelReadinessStatus::Cancelled, model_key};
    case HLIEWaitReadyStatus::Orphaned:
        return {HLIEModelReadinessStatus::Orphaned, model_key};
    case HLIEWaitReadyStatus::Failed:
        return {HLIEModelReadinessStatus::Failed, model_key};
    case HLIEWaitReadyStatus::Unloaded:
        return {HLIEModelReadinessStatus::Unloaded, model_key};
    }
    return {HLIEModelReadinessStatus::Unavailable, model_key};
}

void HLIEModelReadinessService::stop_and_join() {
    tasks_.stop_and_join();
}

HLIE_NAMESPACE_END

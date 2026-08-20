#include "hlie_app/startup/hlie_embedded_startup_service.h"

#include <exception>
#include <utility>

#include "hlie_app/control/hlie_model_controller.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

HLIEEmbeddedStartupService::HLIEEmbeddedStartupService(
    HLIEModelController& controller,
    IHLIEEmbeddedStartupTargetProvider& target_provider,
    ReadyCallback ready_callback,
    UnpublishedCleanupCallback unpublished_cleanup)
    : controller_(controller), target_provider_(target_provider),
      ready_callback_(std::move(ready_callback)),
      unpublished_cleanup_(std::move(unpublished_cleanup)) {}

bool HLIEEmbeddedStartupService::cleanup_unpublished_target(
    const HLIEEmbeddedStartupTarget& target, const char* reason) const {
    if (!unpublished_cleanup_) {
        LOG_ERROR(
            "Initialized embedded Service cleanup callback is unavailable: "
            "model={}, service_kind={}, reason={}",
            target.model_key, target.service_kind, reason);
        return false;
    }
    try {
        const bool confirmed = unpublished_cleanup_(target);
        if (!confirmed) {
            LOG_ERROR(
                "Initialized embedded Service cleanup was not confirmed: "
                "model={}, service_kind={}, reason={}",
                target.model_key, target.service_kind, reason);
        }
        return confirmed;
    } catch (const std::exception& error) {
        LOG_ERROR(
            "Initialized embedded Service cleanup failed: model={}, "
            "service_kind={}, reason={}, error={}",
            target.model_key, target.service_kind, reason, error.what());
    } catch (...) {
        LOG_ERROR(
            "Initialized embedded Service cleanup failed: model={}, "
            "service_kind={}, reason={}, error=unknown",
            target.model_key, target.service_kind, reason);
    }
    return false;
}

bool HLIEEmbeddedStartupService::publish_started_target(
    const HLIEEmbeddedStartupTarget& target, bool& model_missing) {
    model_missing = false;
    const auto model_instance =
        controller_.get_instance_snapshot(target.model_key);
    if (!model_instance) {
        model_missing = true;
        LOG_ERROR(
            "Initialized model instance is missing before Runtime publication: "
            "key={}, service_kind={}",
            target.model_key, target.service_kind);
        (void)cleanup_unpublished_target(target, "model_missing");
        return false;
    }

    const auto ticket =
        controller_.try_begin_loading_ticket(target.model_key);
    if (!ticket) {
        LOG_ERROR(
            "Failed to claim initialized model Runtime: key={}, "
            "service_kind={}",
            target.model_key, target.service_kind);
        (void)cleanup_unpublished_target(target, "loading_ticket_unavailable");
        return false;
    }
    const auto published = controller_.publish_embedded_runtime({
        *ticket, model_instance->model_info.get_capabilities(),
        target.service_kind, target.companion_service != nullptr});
    if (!published.published) {
        bool cleanup_confirmed = published.cleanup_confirmed;
        if (!cleanup_confirmed) {
            cleanup_confirmed = cleanup_unpublished_target(
                target, "runtime_publication_cleanup_unconfirmed");
        }
        controller_.reconcile_runtime_publish_failure(
            *ticket, cleanup_confirmed,
            "initialized_runtime_publish_failed");
        return false;
    }

    const int ready_port = target.ready_port.value_or(
        model_instance->meta.port);
    if (!controller_.mark_loaded_if_current(*ticket, ready_port)) {
        const bool stopped = controller_.stop_and_detach_runtime(
            *ticket, HLIERuntimeStopReason::StartupRollback, true);
        controller_.reconcile_runtime_publish_failure(
            *ticket, stopped, "initialized_runtime_stale_publish");
        return false;
    }
    if (ready_callback_) {
        ready_callback_(target.model_key);
    }
    return true;
}

HLIEEmbeddedStartupSummary HLIEEmbeddedStartupService::start_group(
    const HLIEEmbeddedStartupGroup group) {
    HLIEEmbeddedStartupSummary summary;
    for (const auto& target : target_provider_.targets(group)) {
        ++summary.attempted;
        if (!target.primary_service || target.model_key.empty() ||
            target.service_kind.empty()) {
            ++summary.failed;
            LOG_ERROR(
                "Invalid initialized embedded startup target: key={}, "
                "service_kind={}",
                target.model_key, target.service_kind);
            continue;
        }

        target.primary_service->run();
        if (target.companion_service) {
            target.companion_service->run();
        }
        (void)controller_.update_runtime_devices(
            target.model_key, target.primary_service->get_devices());

        bool model_missing = false;
        if (publish_started_target(target, model_missing)) {
            ++summary.started;
            continue;
        }

        ++summary.failed;
        LOG_ERROR(
            "Failed to publish initialized embedded Runtime: key={}, "
            "service_kind={}",
            target.model_key, target.service_kind);
        if (model_missing && target.abort_if_model_missing) {
            summary.abort_remaining = true;
            break;
        }
    }
    return summary;
}

HLIE_NAMESPACE_END

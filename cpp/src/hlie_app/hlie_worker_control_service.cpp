#include "hlie_app/control/hlie_worker_control_service.h"

#include <chrono>
#include <utility>

#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

HLIEWorkerControlService::HLIEWorkerControlService(
    IHLIEWorkerStatusPublisher& publisher,
    HLIEWorkerControlDependencies dependencies)
    : publisher_(publisher), dependencies_(std::move(dependencies)) {}

HLIEWorkerControlService::~HLIEWorkerControlService() {
    stop_and_join();
}

bool HLIEWorkerControlService::is_dynamic_worker() const {
    return dependencies_.dynamic_worker_provider &&
           dependencies_.dynamic_worker_provider();
}

bool HLIEWorkerControlService::can_publish() const {
    return is_dynamic_worker() &&
           !(dependencies_.replicas_mode_provider &&
             dependencies_.replicas_mode_provider());
}

bool HLIEWorkerControlService::publish(
    const std::string& model_key, const HLIEWorkerModelStatus status,
    const int port, const HLIEModelLaunchTicket& ticket) {
    if (!can_publish()) {
        return false;
    }
    const int router_port = dependencies_.master_port_provider
        ? dependencies_.master_port_provider()
        : 0;
    return publisher_.publish({router_port, model_key, status, port, ticket});
}

bool HLIEWorkerControlService::publish_loaded(
    const std::string& model_key, const HLIEModelLaunchTicket& ticket) {
    return publish(model_key, HLIEWorkerModelStatus::Loaded,
                   dependencies_.local_port_provider
                       ? dependencies_.local_port_provider()
                       : 0,
                   ticket);
}

bool HLIEWorkerControlService::publish_unloaded(
    const std::string& model_key, const HLIEModelLaunchTicket& ticket) {
    if (!can_publish()) {
        return false;
    }
    const int local_port = dependencies_.local_port_provider
        ? dependencies_.local_port_provider()
        : 0;
    return publisher_.publish({
        dependencies_.master_port_provider ? dependencies_.master_port_provider() : 0,
        model_key, HLIEWorkerModelStatus::Unloaded, local_port, ticket});
}

bool HLIEWorkerControlService::schedule_loaded(
    const std::string& model_key, const HLIEModelLaunchTicket& ticket) {
    if (!can_publish() || status_update_stop_.load(std::memory_order_acquire)) {
        return false;
    }
    const bool launched = tasks_.launch([this, model_key, ticket] {
        constexpr auto delay = std::chrono::seconds(3);
        constexpr auto poll_interval = std::chrono::milliseconds(100);
        const auto deadline = std::chrono::steady_clock::now() + delay;
        while (std::chrono::steady_clock::now() < deadline) {
            if (status_update_stop_.load(std::memory_order_acquire)) {
                return;
            }
            std::this_thread::sleep_for(poll_interval);
        }

        if (status_update_stop_.load(std::memory_order_acquire)) {
            return;
        }
        if (!publish_loaded(model_key, ticket)) {
            LOG_ERROR(
                "Worker READY publication failed; entering draining: model={}, "
                "generation={}, port={}",
                model_key, ticket.generation,
                dependencies_.local_port_provider
                    ? dependencies_.local_port_provider()
                    : 0);
            if (dependencies_.begin_draining) {
                dependencies_.begin_draining();
            }
            request_process_shutdown();
        }
    });
    if (!launched) {
        LOG_WARN("Failed to schedule slave loaded status update: model={}",
                 model_key);
    }
    return launched;
}

HLIEWorkerStatusApplyResult HLIEWorkerControlService::apply_status(
    HLIEWorkerStatusUpdate update) {
    if (!dependencies_.resolve_model_key || !dependencies_.apply_status) {
        return HLIEWorkerStatusApplyResult::Rejected;
    }
    std::lock_guard lock(status_mutex_);
    const auto resolved_key = dependencies_.resolve_model_key(update.model_key);
    if (!resolved_key) {
        return HLIEWorkerStatusApplyResult::LookupFailed;
    }
    const HLIEModelLaunchTicket ticket{
        *resolved_key, update.generation, std::move(update.launch_token)};
    if (!ticket.valid() ||
        !dependencies_.apply_status(ticket, update.status, update.port)) {
        return HLIEWorkerStatusApplyResult::Stale;
    }
    if (update.status == HLIEWorkerModelStatus::Unloaded &&
        dependencies_.erase_model) {
        dependencies_.erase_model(*resolved_key);
    }
    return HLIEWorkerStatusApplyResult::Applied;
}

HLIEWorkerStatusSnapshot HLIEWorkerControlService::snapshot() const {
    HLIEWorkerStatusSnapshot result;
    if (dependencies_.local_port_provider) {
        result.local_port = dependencies_.local_port_provider();
    }
    if (dependencies_.local_start_time_provider) {
        result.local_start_time = dependencies_.local_start_time_provider();
    }
    if (dependencies_.local_models_provider) {
        result.local_models = dependencies_.local_models_provider();
    }
    if (dependencies_.worker_processes_provider) {
        result.workers = dependencies_.worker_processes_provider();
    }
    return result;
}

void HLIEWorkerControlService::request_process_shutdown() {
    if (!process_shutdown_requested_.exchange(true, std::memory_order_acq_rel) &&
        dependencies_.process_shutdown &&
        (!dependencies_.process_shutdown_available ||
         dependencies_.process_shutdown_available())) {
        dependencies_.process_shutdown();
    }
}

HLIEWorkerShutdownResult HLIEWorkerControlService::request_shutdown(
    const HLIEModelGeneration generation, const std::string& launch_token) {
    if (!is_dynamic_worker() ||
        (dependencies_.replicas_mode_provider &&
         dependencies_.replicas_mode_provider())) {
        return HLIEWorkerShutdownResult::NotDynamicWorker;
    }
    if (!dependencies_.process_shutdown || !dependencies_.launch_ticket_provider ||
        (dependencies_.process_shutdown_available &&
         !dependencies_.process_shutdown_available())) {
        return HLIEWorkerShutdownResult::CallbackUnavailable;
    }
    const auto ticket = dependencies_.launch_ticket_provider();
    const HLIEModelLaunchTicket requested{ticket.model_key, generation,
                                          launch_token};
    if (!ticket.valid() || !ticket.matches(requested)) {
        return HLIEWorkerShutdownResult::InvalidTicket;
    }
    const bool already_requested = process_shutdown_requested_.exchange(
        true, std::memory_order_acq_rel);
    if (dependencies_.begin_draining) {
        dependencies_.begin_draining();
    }
    if (!already_requested) {
        dependencies_.process_shutdown();
        return HLIEWorkerShutdownResult::Accepted;
    }
    return HLIEWorkerShutdownResult::AlreadyRequested;
}

void HLIEWorkerControlService::stop_and_join() {
    status_update_stop_.store(true, std::memory_order_release);
    tasks_.stop_and_join();
}

HLIE_NAMESPACE_END

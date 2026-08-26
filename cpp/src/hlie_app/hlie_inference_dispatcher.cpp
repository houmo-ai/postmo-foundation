#include "hlie_app/inference/hlie_inference_dispatcher.h"

#include <exception>
#include <optional>
#include <utility>
#include <vector>

#include "hlie_app/control/hlie_model_controller.h"
#include "hlie_app/runtime/hlie_runtime_directory.h"

HLIE_NAMESPACE_BEGIN

struct HLIEInferenceSession::State {
    State(HLIEModelController& controller,
          HLIEModelLaunchTicket launch_ticket,
          std::shared_ptr<IHLIEModelRuntime> runtime_handle,
          std::shared_ptr<IHLIEInferenceEndpoint> endpoint_handle,
          std::shared_ptr<std::atomic<std::uint64_t>> active_sessions)
        : controller(&controller), ticket(std::move(launch_ticket)),
          runtime(std::move(runtime_handle)), endpoint(std::move(endpoint_handle)),
          active(std::move(active_sessions)) {
        active->fetch_add(1, std::memory_order_relaxed);
    }

    ~State() {
        // Data-plane adapters may retain Service/proxy state. Drop them before
        // making the generation evictable again.
        endpoint.reset();
        runtime.reset();
        if (controller && ticket.valid()) {
            controller->end_request(ticket);
        }
        if (active) {
            active->fetch_sub(1, std::memory_order_relaxed);
        }
    }

    HLIEModelController* controller = nullptr;
    HLIEModelLaunchTicket ticket;
    std::shared_ptr<IHLIEModelRuntime> runtime;
    std::shared_ptr<IHLIEInferenceEndpoint> endpoint;
    std::shared_ptr<std::atomic<std::uint64_t>> active;
};

namespace {

class SessionGuardedResponse final : public server_http_res {
public:
    SessionGuardedResponse(server_http_res_ptr inner,
                           HLIEInferenceSession session)
        : session_(std::move(session)), inner_(std::move(inner)) {
        content_type = inner_->content_type;
        status = inner_->status;
        data = std::move(inner_->data);
        headers = std::move(inner_->headers);
        if (inner_->is_stream()) {
            next = [this](std::string& output) {
                if (!data.empty()) {
                    output = std::move(data);
                    data.clear();
                    return true;
                }
                return inner_->next(output);
            };
        }
    }

private:
    // Reverse destruction order is deliberate: destroy the upstream response
    // first, then release the generation lease held by session_.
    HLIEInferenceSession session_;
    server_http_res_ptr inner_;
};

} // namespace

HLIEInferenceSession::HLIEInferenceSession(std::shared_ptr<State> state)
    : state_(std::move(state)) {}

bool HLIEInferenceSession::active() const noexcept {
    return state_ && state_->controller && state_->ticket.valid() &&
        state_->runtime && state_->endpoint;
}

HLIERuntimeIdentity HLIEInferenceSession::identity() const {
    return active() ? state_->runtime->identity() : HLIERuntimeIdentity{};
}

std::shared_ptr<IHLIEInferenceEndpoint> HLIEInferenceSession::endpoint() const {
    return active() ? state_->endpoint : nullptr;
}

std::shared_ptr<void> HLIEInferenceSession::lifetime_guard() const {
    return active() ? std::static_pointer_cast<void>(state_) : nullptr;
}

void HLIEInferenceSession::reset() {
    state_.reset();
}

HLIEInferenceDispatcher::HLIEInferenceDispatcher(
    HLIEModelController& controller,
    HLIERuntimeDirectory& runtime_directory,
    IHLIEInferenceEndpointResolver& endpoint_resolver,
    DrainingProvider draining_provider)
    : controller_(controller), runtime_directory_(runtime_directory),
      endpoint_resolver_(endpoint_resolver),
      draining_provider_(std::move(draining_provider)) {}

HLIEInferenceAcquireResult HLIEInferenceDispatcher::acquire(
    const std::string& model_name,
    const HLIEInferenceOperation operation) {
    HLIEInferenceAcquireResult result;
    if (draining_provider_ && draining_provider_()) {
        result.status = HLIEInferenceDispatchStatus::Draining;
        result.detail = "service is draining";
        record(result.status);
        return result;
    }

    const auto instance = controller_.get_instance_snapshot(model_name);
    if (!instance) {
        result.status = HLIEInferenceDispatchStatus::ModelNotFound;
        result.detail = "model not found";
        record(result.status);
        return result;
    }
    result.model_key = instance->key;

    auto ticket = controller_.begin_request_ticket(instance->key);
    if (!ticket) {
        result.status = HLIEInferenceDispatchStatus::ModelNotReady;
        result.detail = "model is not ready for inference";
        record(result.status);
        return result;
    }

    auto runtime = runtime_directory_.find(*ticket);
    if (!runtime) {
        controller_.end_request(*ticket);
        result.status = HLIEInferenceDispatchStatus::RuntimeUnavailable;
        result.detail = "generation runtime is unavailable";
        record(result.status);
        return result;
    }

    const auto identity = runtime->identity();
    HLIEInferenceEndpointRequest endpoint_request;
    endpoint_request.identity = identity;
    endpoint_request.service_kind = instance->model_info.get_service_kind();
    auto endpoint = resolve_endpoint(endpoint_request);
    if (!endpoint || !endpoint->identity().matches(identity)) {
        controller_.end_request(*ticket);
        result.status = HLIEInferenceDispatchStatus::EndpointUnavailable;
        result.detail = "generation endpoint is unavailable";
        record(result.status);
        return result;
    }
    if (!endpoint->supports(operation)) {
        controller_.end_request(*ticket);
        result.status = HLIEInferenceDispatchStatus::UnsupportedOperation;
        result.detail = "operation is not supported by the endpoint";
        record(result.status);
        return result;
    }

    result.session = HLIEInferenceSession(std::make_shared<
        HLIEInferenceSession::State>(controller_, *ticket, std::move(runtime),
                                     std::move(endpoint), active_sessions_));
    result.status = HLIEInferenceDispatchStatus::Dispatched;
    return result;
}

HLIEInferenceDispatchResult HLIEInferenceDispatcher::dispatch(
    const std::string& model_name,
    const HLIEInferenceOperation operation,
    const server_http_req& request,
    const std::string& method) {
    HLIEInferenceDispatchResult result;
    auto acquired = acquire(model_name, operation);
    result.status = acquired.status;
    result.model_key = acquired.model_key;
    result.detail = std::move(acquired.detail);
    if (!acquired.acquired()) {
        return result;
    }

    result.identity = acquired.session.identity();
    auto endpoint = acquired.session.endpoint();
    try {
        auto response = endpoint->dispatch(operation, request, method);
        if (!response) {
            result.status = HLIEInferenceDispatchStatus::EndpointFailure;
            result.detail = "endpoint returned no response";
            record(result.status);
            return result;
        }
        result.response = guard_response(
            std::move(response), std::move(acquired.session));
        result.status = HLIEInferenceDispatchStatus::Dispatched;
        record(result.status);
        return result;
    } catch (const std::exception& error) {
        result.status = HLIEInferenceDispatchStatus::EndpointFailure;
        result.detail = error.what();
    } catch (...) {
        result.status = HLIEInferenceDispatchStatus::EndpointFailure;
        result.detail = "endpoint threw an unknown exception";
    }
    record(result.status);
    return result;
}

HLIEInferenceDispatcherMetrics HLIEInferenceDispatcher::metrics() const {
    return {
        dispatched_.load(std::memory_order_relaxed),
        draining_rejected_.load(std::memory_order_relaxed),
        model_not_found_.load(std::memory_order_relaxed),
        model_not_ready_.load(std::memory_order_relaxed),
        runtime_unavailable_.load(std::memory_order_relaxed),
        endpoint_unavailable_.load(std::memory_order_relaxed),
        unsupported_operation_.load(std::memory_order_relaxed),
        endpoint_failures_.load(std::memory_order_relaxed),
        active_sessions_->load(std::memory_order_relaxed),
        [&] {
            std::lock_guard lock(endpoint_mutex_);
            return static_cast<std::uint64_t>(endpoints_.size());
        }(),
    };
}

std::shared_ptr<IHLIEInferenceEndpoint>
HLIEInferenceDispatcher::resolve_endpoint(
    const HLIEInferenceEndpointRequest& request) {
    if (!request.valid()) {
        return {};
    }
    const auto& model_key = request.identity.launch_ticket.model_key;
    {
        std::lock_guard lock(endpoint_mutex_);
        const auto it = endpoints_.find(model_key);
        if (it != endpoints_.end() && it->second &&
            it->second->identity().matches(request.identity)) {
            return it->second;
        }
    }

    auto resolved = endpoint_resolver_.resolve(request);
    if (!resolved || !resolved->identity().matches(request.identity)) {
        return {};
    }
    std::lock_guard lock(endpoint_mutex_);
    const auto it = endpoints_.find(model_key);
    if (it != endpoints_.end() && it->second &&
        it->second->identity().matches(request.identity)) {
        return it->second;
    }
    endpoints_.insert_or_assign(model_key, resolved);
    return resolved;
}

void HLIEInferenceDispatcher::reconcile_endpoints() {
    std::vector<std::pair<std::string, HLIERuntimeIdentity>> candidates;
    {
        std::lock_guard lock(endpoint_mutex_);
        candidates.reserve(endpoints_.size());
        for (const auto& [model_key, endpoint] : endpoints_) {
            if (endpoint) {
                candidates.emplace_back(model_key, endpoint->identity());
            }
        }
    }

    for (const auto& [model_key, identity] : candidates) {
        if (runtime_directory_.find(identity.launch_ticket)) {
            continue;
        }
        std::lock_guard lock(endpoint_mutex_);
        const auto it = endpoints_.find(model_key);
        if (it != endpoints_.end() && it->second &&
            it->second->identity().matches(identity)) {
            endpoints_.erase(it);
        }
    }
}

void HLIEInferenceDispatcher::observe_native_dispatch(
    const bool successful) noexcept {
    record(successful ? HLIEInferenceDispatchStatus::Dispatched
                      : HLIEInferenceDispatchStatus::EndpointFailure);
}

server_http_res_ptr HLIEInferenceDispatcher::guard_response(
    server_http_res_ptr response,
    HLIEInferenceSession session) {
    if (!response || !session.active()) {
        return response;
    }
    return std::make_unique<SessionGuardedResponse>(
        std::move(response), std::move(session));
}

void HLIEInferenceDispatcher::record(
    const HLIEInferenceDispatchStatus status) noexcept {
    switch (status) {
    case HLIEInferenceDispatchStatus::Dispatched:
        dispatched_.fetch_add(1, std::memory_order_relaxed);
        break;
    case HLIEInferenceDispatchStatus::Draining:
        draining_rejected_.fetch_add(1, std::memory_order_relaxed);
        break;
    case HLIEInferenceDispatchStatus::ModelNotFound:
        model_not_found_.fetch_add(1, std::memory_order_relaxed);
        break;
    case HLIEInferenceDispatchStatus::ModelNotReady:
        model_not_ready_.fetch_add(1, std::memory_order_relaxed);
        break;
    case HLIEInferenceDispatchStatus::RuntimeUnavailable:
        runtime_unavailable_.fetch_add(1, std::memory_order_relaxed);
        break;
    case HLIEInferenceDispatchStatus::EndpointUnavailable:
        endpoint_unavailable_.fetch_add(1, std::memory_order_relaxed);
        break;
    case HLIEInferenceDispatchStatus::UnsupportedOperation:
        unsupported_operation_.fetch_add(1, std::memory_order_relaxed);
        break;
    case HLIEInferenceDispatchStatus::EndpointFailure:
        endpoint_failures_.fetch_add(1, std::memory_order_relaxed);
        break;
    }
}

HLIE_NAMESPACE_END

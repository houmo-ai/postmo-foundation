#ifndef HLIE_APP_INFERENCE_DISPATCHER_H
#define HLIE_APP_INFERENCE_DISPATCHER_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "hlie_app/inference/hlie_inference_endpoint.h"

HLIE_NAMESPACE_BEGIN

class HLIEModelController;
class HLIERuntimeDirectory;

enum class HLIEInferenceDispatchStatus {
    Dispatched,
    Draining,
    ModelNotFound,
    ModelNotReady,
    RuntimeUnavailable,
    EndpointUnavailable,
    UnsupportedOperation,
    EndpointFailure,
};

struct HLIEInferenceDispatcherMetrics {
    std::uint64_t dispatched = 0;
    std::uint64_t draining_rejected = 0;
    std::uint64_t model_not_found = 0;
    std::uint64_t model_not_ready = 0;
    std::uint64_t runtime_unavailable = 0;
    std::uint64_t endpoint_unavailable = 0;
    std::uint64_t unsupported_operation = 0;
    std::uint64_t endpoint_failures = 0;
    std::uint64_t active_sessions = 0;
    std::uint64_t cached_endpoints = 0;
};

class HLIEInferenceSession {
public:
    HLIEInferenceSession() = default;
    ~HLIEInferenceSession() = default;

    HLIEInferenceSession(const HLIEInferenceSession&) = delete;
    HLIEInferenceSession& operator=(const HLIEInferenceSession&) = delete;
    HLIEInferenceSession(HLIEInferenceSession&&) noexcept = default;
    HLIEInferenceSession& operator=(HLIEInferenceSession&&) noexcept = default;

    [[nodiscard]] bool active() const noexcept;
    [[nodiscard]] HLIERuntimeIdentity identity() const;
    [[nodiscard]] std::shared_ptr<IHLIEInferenceEndpoint> endpoint() const;
    [[nodiscard]] std::shared_ptr<void> lifetime_guard() const;
    void reset();

private:
    struct State;
    explicit HLIEInferenceSession(std::shared_ptr<State> state);

    std::shared_ptr<State> state_;

    friend class HLIEInferenceDispatcher;
};

struct HLIEInferenceAcquireResult {
    HLIEInferenceDispatchStatus status =
        HLIEInferenceDispatchStatus::RuntimeUnavailable;
    HLIEInferenceSession session;
    std::string model_key;
    std::string detail;

    [[nodiscard]] bool acquired() const noexcept {
        return status == HLIEInferenceDispatchStatus::Dispatched &&
            session.active();
    }
};

struct HLIEInferenceDispatchResult {
    HLIEInferenceDispatchStatus status =
        HLIEInferenceDispatchStatus::EndpointFailure;
    server_http_res_ptr response;
    HLIERuntimeIdentity identity;
    std::string model_key;
    std::string detail;

    [[nodiscard]] bool dispatched() const noexcept {
        return status == HLIEInferenceDispatchStatus::Dispatched &&
            response != nullptr;
    }
};

// Request-boundary dispatcher. A successful response owns a session until the
// upstream response (including a stream) has been destroyed. No retry is
// performed here because generation requests are not generally replay-safe.
class HLIEInferenceDispatcher {
public:
    using DrainingProvider = std::function<bool()>;

    HLIEInferenceDispatcher(
        HLIEModelController& controller,
        HLIERuntimeDirectory& runtime_directory,
        IHLIEInferenceEndpointResolver& endpoint_resolver,
        DrainingProvider draining_provider = {});

    [[nodiscard]] HLIEInferenceAcquireResult acquire(
        const std::string& model_name,
        HLIEInferenceOperation operation);
    [[nodiscard]] HLIEInferenceDispatchResult dispatch(
        const std::string& model_name,
        HLIEInferenceOperation operation,
        const server_http_req& request,
        const std::string& method = "POST");

    [[nodiscard]] HLIEInferenceDispatcherMetrics metrics() const;
    void reconcile_endpoints();
    void observe_native_dispatch(bool successful) noexcept;

    // Preserves the existing server_http_res streaming ABI. The wrapped
    // response is destroyed before the request session releases its lease.
    [[nodiscard]] static server_http_res_ptr guard_response(
        server_http_res_ptr response,
        HLIEInferenceSession session);

private:
    void record(HLIEInferenceDispatchStatus status) noexcept;
    [[nodiscard]] std::shared_ptr<IHLIEInferenceEndpoint> resolve_endpoint(
        const HLIEInferenceEndpointRequest& request);

    HLIEModelController& controller_;
    HLIERuntimeDirectory& runtime_directory_;
    IHLIEInferenceEndpointResolver& endpoint_resolver_;
    DrainingProvider draining_provider_;

    std::atomic<std::uint64_t> dispatched_{0};
    std::atomic<std::uint64_t> draining_rejected_{0};
    std::atomic<std::uint64_t> model_not_found_{0};
    std::atomic<std::uint64_t> model_not_ready_{0};
    std::atomic<std::uint64_t> runtime_unavailable_{0};
    std::atomic<std::uint64_t> endpoint_unavailable_{0};
    std::atomic<std::uint64_t> unsupported_operation_{0};
    std::atomic<std::uint64_t> endpoint_failures_{0};
    std::shared_ptr<std::atomic<std::uint64_t>> active_sessions_ =
        std::make_shared<std::atomic<std::uint64_t>>(0);
    mutable std::mutex endpoint_mutex_;
    std::unordered_map<std::string, std::shared_ptr<IHLIEInferenceEndpoint>>
        endpoints_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_INFERENCE_DISPATCHER_H

#include <catch_amalgamated.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hlie_app/control/hlie_model_controller.h"
#include "hlie_app/inference/hlie_inference_dispatcher.h"

namespace {

using namespace hm_HLIE;

class DispatchReservationProvider final
    : public IHLIEDeviceReservationProvider {
public:
    DispatchReservationProvider() {
        ledger_.update_observation({0, 8192, 0, 8192});
    }

    std::optional<HLIEDeviceReservation> reserve_resources(
        const HLIEDeviceReservationRequest& request) override {
        return ledger_.reserve(request);
    }
    bool bind_reservation(HLIEDeviceReservationId id, const std::string& key,
                          std::uint64_t generation) override {
        return ledger_.bind(id, key, generation);
    }
    bool commit_reservation(HLIEDeviceReservationId id) override {
        return ledger_.commit(id);
    }
    bool rollback_reservation(HLIEDeviceReservationId id) override {
        return ledger_.rollback(id);
    }
    bool release_reservation(HLIEDeviceReservationId id) override {
        return ledger_.release(id);
    }
    std::vector<HLIEDeviceReservation> rollback_expired_pending_reservations(
        const std::unordered_set<HLIEDeviceReservationId>& protected_ids) override {
        return ledger_.rollback_expired_pending(protected_ids);
    }
    std::vector<HLIEDeviceReservation> get_reservations() const override {
        return ledger_.reservations();
    }
    std::vector<HLIEDeviceLedgerSnapshot> get_ledger_snapshots() const override {
        return ledger_.snapshots();
    }

private:
    HLIEDeviceReservationLedger ledger_;
};

class DispatchRuntime final : public IHLIEModelRuntime {
public:
    explicit DispatchRuntime(HLIERuntimeIdentity identity)
        : identity_(std::move(identity)) {}

    HLIERuntimeIdentity identity() const override { return identity_; }
    HLIEModelCapabilities capabilities() const override { return {}; }
    HLIERuntimeHealth health() const override {
        return {identity_, HLIERuntimeHealthState::Ready, 0};
    }
    void begin_draining() override {}
    HLIERuntimeStopResult stop(const HLIERuntimeStopRequest&) override {
        return {identity_, true, true, false, false, 0};
    }

private:
    HLIERuntimeIdentity identity_;
};

class DispatchRuntimeFactory final : public IHLIEModelRuntimeFactory {
public:
    std::shared_ptr<IHLIEModelRuntime> create_embedded(
        const HLIEEmbeddedRuntimeCreateRequest&) override {
        return {};
    }
    std::shared_ptr<IHLIEModelRuntime> create_process(
        const HLIEProcessRuntimeCreateRequest&) override {
        return {};
    }
};

class ObservedResponse final : public server_http_res {
public:
    ObservedResponse(HLIEModelController& controller, std::string model_key,
                     std::atomic<bool>& destroyed_with_lease)
        : controller_(controller), model_key_(std::move(model_key)),
          destroyed_with_lease_(destroyed_with_lease) {
        next = [remaining = 1](std::string& output) mutable {
            if (remaining-- > 0) {
                output = "chunk";
                return true;
            }
            return false;
        };
    }

    ~ObservedResponse() override {
        const auto snapshot = controller_.get_runtime_snapshot(model_key_);
        destroyed_with_lease_.store(
            snapshot && snapshot->active_requests == 1,
            std::memory_order_relaxed);
    }

private:
    HLIEModelController& controller_;
    std::string model_key_;
    std::atomic<bool>& destroyed_with_lease_;
};

class DispatchEndpoint final : public IHLIEInferenceEndpoint {
public:
    DispatchEndpoint(HLIERuntimeIdentity identity,
                     HLIEModelController& controller,
                     std::string model_key,
                     std::atomic<bool>& destroyed_with_lease)
        : identity_(std::move(identity)), controller_(controller),
          model_key_(std::move(model_key)),
          destroyed_with_lease_(destroyed_with_lease) {}

    HLIERuntimeIdentity identity() const override { return identity_; }
    bool supports(HLIEInferenceOperation operation) const override {
        return operation == HLIEInferenceOperation::ChatCompletions;
    }
    server_http_res_ptr dispatch(HLIEInferenceOperation,
                                 const server_http_req&,
                                 const std::string&) override {
        return std::make_unique<ObservedResponse>(
            controller_, model_key_, destroyed_with_lease_);
    }

private:
    HLIERuntimeIdentity identity_;
    HLIEModelController& controller_;
    std::string model_key_;
    std::atomic<bool>& destroyed_with_lease_;
};

class DispatchEndpointResolver final : public IHLIEInferenceEndpointResolver {
public:
    std::shared_ptr<IHLIEInferenceEndpoint> endpoint;
    std::atomic<int> resolves{0};

    std::shared_ptr<IHLIEInferenceEndpoint> resolve(
        const HLIEInferenceEndpointRequest&) override {
        resolves.fetch_add(1, std::memory_order_relaxed);
        return endpoint;
    }
};

HLIEModelInfo dispatch_model() {
    HLIEModelInfo info;
    info.build_sd("sd", "vae", "t5", "clip", "clip-l", "mmdit", "0");
    return info;
}

struct DispatchFixture {
    std::shared_ptr<DispatchReservationProvider> provider =
        std::make_shared<DispatchReservationProvider>();
    HLIEResourceScheduler scheduler{provider};
    HLIERuntimeDirectory directory;
    DispatchRuntimeFactory runtime_factory;
    HLIEModelController controller{scheduler, directory, runtime_factory};
    DispatchEndpointResolver resolver;
    std::atomic<bool> draining{false};
    HLIEInferenceDispatcher dispatcher{
        controller, directory, resolver,
        [this] { return draining.load(std::memory_order_relaxed); }};
    HLIEModelInfo model = dispatch_model();
    HLIEModelLaunchTicket ticket;
    std::atomic<bool> response_destroyed_with_lease{false};

    DispatchFixture() {
        controller.register_model(model, 512);
        const auto launch = controller.try_begin_loading_ticket(model.model_key_);
        REQUIRE(launch);
        ticket = *launch;
        auto runtime = std::make_shared<DispatchRuntime>(HLIERuntimeIdentity{
            301, HLIERuntimeKind::Embedded, ticket, 0, 0});
        REQUIRE(directory.publish(runtime));
        REQUIRE(controller.mark_loaded_if_current(ticket, 7901));
        resolver.endpoint = std::make_shared<DispatchEndpoint>(
            runtime->identity(), controller, model.model_key_,
            response_destroyed_with_lease);
    }
};

const server_http_req& empty_request() {
    static const std::function<bool()> should_stop = [] { return false; };
    static const server_http_req request{{}, {}, "/v1/chat/completions", {},
                                         "{}", {}, should_stop};
    return request;
}

} // namespace

TEST_CASE("InferenceDispatcher keeps generation lease through stream teardown",
          "[p1][inference-dispatcher]") {
    DispatchFixture fixture;

    auto result = fixture.dispatcher.dispatch(
        fixture.model.model_key_, HLIEInferenceOperation::ChatCompletions,
        empty_request());
    REQUIRE(result.dispatched());
    auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->active_requests == 1);
    CHECK(fixture.dispatcher.metrics().active_sessions == 1);

    std::string chunk;
    CHECK(result.response->next(chunk));
    CHECK(chunk == "chunk");
    result.response.reset();

    snapshot = fixture.controller.get_runtime_snapshot(fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->active_requests == 0);
    CHECK(fixture.response_destroyed_with_lease.load());
    CHECK(fixture.dispatcher.metrics().active_sessions == 0);
    CHECK(fixture.dispatcher.metrics().dispatched == 1);
    CHECK(fixture.dispatcher.metrics().cached_endpoints == 1);
}

TEST_CASE("InferenceDispatcher rejects stale endpoints without leaking leases",
          "[p1][inference-dispatcher]") {
    DispatchFixture fixture;
    auto stale_identity = fixture.resolver.endpoint->identity();
    ++stale_identity.runtime_id;
    fixture.resolver.endpoint = std::make_shared<DispatchEndpoint>(
        stale_identity, fixture.controller, fixture.model.model_key_,
        fixture.response_destroyed_with_lease);

    const auto result = fixture.dispatcher.dispatch(
        fixture.model.model_key_, HLIEInferenceOperation::ChatCompletions,
        empty_request());
    CHECK(result.status ==
          HLIEInferenceDispatchStatus::EndpointUnavailable);
    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->active_requests == 0);
    CHECK(fixture.dispatcher.metrics().active_sessions == 0);
    CHECK(fixture.dispatcher.metrics().endpoint_unavailable == 1);
}

TEST_CASE("InferenceDispatcher fails closed while draining or unsupported",
          "[p1][inference-dispatcher]") {
    DispatchFixture fixture;
    fixture.draining.store(true);
    auto result = fixture.dispatcher.dispatch(
        fixture.model.model_key_, HLIEInferenceOperation::ChatCompletions,
        empty_request());
    CHECK(result.status == HLIEInferenceDispatchStatus::Draining);

    fixture.draining.store(false);
    result = fixture.dispatcher.dispatch(
        fixture.model.model_key_, HLIEInferenceOperation::Embeddings,
        empty_request());
    CHECK(result.status ==
          HLIEInferenceDispatchStatus::UnsupportedOperation);
    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->active_requests == 0);
}

TEST_CASE("Inference session lifetime guard carries lease across async callbacks",
          "[p1][inference-dispatcher][streaming]") {
    DispatchFixture fixture;
    auto acquired = fixture.dispatcher.acquire(
        fixture.model.model_key_,
        HLIEInferenceOperation::ChatCompletions);
    REQUIRE(acquired.acquired());

    auto callback_guard = acquired.session.lifetime_guard();
    REQUIRE(callback_guard);
    acquired.session.reset();
    auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->active_requests == 1);

    callback_guard.reset();
    snapshot = fixture.controller.get_runtime_snapshot(fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->active_requests == 0);
    CHECK(fixture.dispatcher.metrics().active_sessions == 0);
    fixture.dispatcher.observe_native_dispatch(true);
    CHECK(fixture.dispatcher.metrics().dispatched == 1);
    fixture.dispatcher.observe_native_dispatch(false);
    CHECK(fixture.dispatcher.metrics().endpoint_failures == 1);
}

TEST_CASE("Cached inference dispatch stays within request-boundary budget",
          "[p1][inference-dispatcher][performance]") {
    DispatchFixture fixture;
    constexpr int request_count = 2000;
    int completed_requests = 0;
    const auto started = std::chrono::steady_clock::now();
    for (int index = 0; index < request_count; ++index) {
        auto result = fixture.dispatcher.dispatch(
            fixture.model.model_key_,
            HLIEInferenceOperation::ChatCompletions,
            empty_request());
        if (!result.dispatched()) {
            break;
        }
        result.response.reset();
        ++completed_requests;
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    REQUIRE(completed_requests == request_count);
    const auto average_microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() /
        request_count;
    INFO("average cached dispatch overhead: " << average_microseconds << " us");
#if !defined(HLIE_THREAD_SANITIZER)
    CHECK(average_microseconds < 100);
#endif
    CHECK(fixture.resolver.resolves.load() == 1);
    CHECK(fixture.dispatcher.metrics().active_sessions == 0);
}

TEST_CASE("InferenceDispatcher caches endpoints per generation and reconciles stale handles",
          "[p1][inference-dispatcher][performance]") {
    DispatchFixture fixture;
    for (int request_index = 0; request_index < 2; ++request_index) {
        auto result = fixture.dispatcher.dispatch(
            fixture.model.model_key_,
            HLIEInferenceOperation::ChatCompletions,
            empty_request());
        REQUIRE(result.dispatched());
        result.response.reset();
    }
    CHECK(fixture.resolver.resolves.load() == 1);
    CHECK(fixture.dispatcher.metrics().cached_endpoints == 1);

    REQUIRE(fixture.directory.detach(fixture.ticket));
    fixture.dispatcher.reconcile_endpoints();
    CHECK(fixture.dispatcher.metrics().cached_endpoints == 0);
}

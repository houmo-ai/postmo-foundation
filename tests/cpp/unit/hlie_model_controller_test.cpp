#include <catch_amalgamated.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hlie_app/control/hlie_model_controller.h"
#include "hlie_app/control/hlie_model_readiness_service.h"

namespace {

using namespace hm_HLIE;

class ControllerReservationProvider final : public IHLIEDeviceReservationProvider {
public:
    ControllerReservationProvider() {
        ledger_.update_observation({0, 8192, 0, 8192});
    }

    std::optional<HLIEDeviceReservation> reserve_resources(
        const HLIEDeviceReservationRequest& request) override {
        return ledger_.reserve(request);
    }
    bool bind_reservation(const HLIEDeviceReservationId id,
                          const std::string& key,
                          const std::uint64_t generation) override {
        return ledger_.bind(id, key, generation);
    }
    bool commit_reservation(const HLIEDeviceReservationId id) override {
        return ledger_.commit(id);
    }
    bool rollback_reservation(const HLIEDeviceReservationId id) override {
        return ledger_.rollback(id);
    }
    bool release_reservation(const HLIEDeviceReservationId id) override {
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

class ControllerRuntime final : public IHLIEModelRuntime {
public:
    ControllerRuntime(HLIERuntimeIdentity identity, const bool stop_confirmed = true)
        : identity_(std::move(identity)), stop_confirmed_(stop_confirmed) {}

    HLIERuntimeIdentity identity() const override { return identity_; }
    HLIEModelCapabilities capabilities() const override { return {}; }
    HLIERuntimeHealth health() const override {
        return {identity_, state_.load(), 0};
    }
    void begin_draining() override {
        state_.store(HLIERuntimeHealthState::Draining);
    }
    HLIERuntimeStopResult stop(const HLIERuntimeStopRequest&) override {
        if (stop_confirmed_) {
            state_.store(HLIERuntimeHealthState::Stopped);
        }
        return {identity_, stop_confirmed_, stop_confirmed_, false,
                !stop_confirmed_, stop_confirmed_ ? 0 : -77};
    }

private:
    HLIERuntimeIdentity identity_;
    bool stop_confirmed_ = true;
    std::atomic<HLIERuntimeHealthState> state_{HLIERuntimeHealthState::Starting};
};

class ControllerRuntimeFactory final : public IHLIEModelRuntimeFactory {
public:
    std::shared_ptr<IHLIEModelRuntime> next_runtime;
    bool echo_embedded_request = false;

    std::shared_ptr<IHLIEModelRuntime> create_embedded(
        const HLIEEmbeddedRuntimeCreateRequest& request) override {
        if (echo_embedded_request) {
            return std::make_shared<ControllerRuntime>(HLIERuntimeIdentity{
                401, HLIERuntimeKind::Embedded, request.launch_ticket, 0, 0});
        }
        return std::exchange(next_runtime, {});
    }
    std::shared_ptr<IHLIEModelRuntime> create_process(
        const HLIEProcessRuntimeCreateRequest&) override {
        return std::exchange(next_runtime, {});
    }
};

class ControllerBackendLauncher final : public IHLIEModelBackendLauncher {
public:
    HLIEBackendLaunchArtifact next_artifact;
    std::atomic<int> launches{0};
    std::atomic<int> cleanups{0};
    bool cleanup_confirmed = true;
    bool echo_request = false;

    HLIEBackendLaunchArtifact launch(
        const HLIEBackendLaunchRequest& request) override {
        ++launches;
        if (echo_request) {
            next_artifact.embedded_runtime.launch_ticket =
                request.launch_ticket;
            next_artifact.embedded_runtime.capabilities =
                request.model_info.get_capabilities();
            next_artifact.embedded_runtime.service_kind =
                request.service_kind;
        }
        return next_artifact;
    }

    bool cleanup_unadopted(
        const HLIEBackendLaunchArtifact&,
        const std::string&) override {
        ++cleanups;
        return cleanup_confirmed;
    }
};

HLIEPlacementIntent automatic_placement() {
    HLIEPlacementIntent placement;
    placement.required_device_count = 1;
    placement.allow_auto_selection = true;
    return placement;
}

HLIEModelInfo controller_model() {
    HLIEModelInfo info;
    info.build_sd("sd", "vae", "t5", "clip", "clip-l", "mmdit", "0");
    return info;
}

struct ControllerFixture {
    std::shared_ptr<ControllerReservationProvider> provider =
        std::make_shared<ControllerReservationProvider>();
    HLIEResourceScheduler scheduler{provider};
    HLIERuntimeDirectory directory;
    ControllerRuntimeFactory runtime_factory;
    ControllerBackendLauncher backend_launcher;
    std::atomic<std::size_t> notifications{0};
    HLIEModelController controller{scheduler, directory, runtime_factory, [this] {
        notifications.fetch_add(1);
    }, &backend_launcher};

    HLIEModelInfo model = controller_model();

    HLIEModelLaunchTicket start_with_tracked_reservation(
        const bool stop_confirmed = true) {
        controller.register_model(model, 512);
        auto pending = scheduler.reserve(
            model.model_key_, 512, automatic_placement());
        REQUIRE(pending);
        const auto ticket = controller.try_begin_loading_ticket(model.model_key_);
        REQUIRE(ticket);
        REQUIRE(scheduler.bind(*pending, *ticket));
        REQUIRE(scheduler.track(*pending, *ticket));
        auto runtime = std::make_shared<ControllerRuntime>(
            HLIERuntimeIdentity{101, HLIERuntimeKind::Process, *ticket, 1, 19001},
            stop_confirmed);
        REQUIRE(directory.publish(runtime));
        return *ticket;
    }
};

} // namespace

TEST_CASE("ModelController owns READY and confirmed exit reconciliation",
          "[p1][model-controller]") {
    ControllerFixture fixture;
    const auto ticket = fixture.start_with_tracked_reservation();

    fixture.controller.on_runtime_event({
        HLIERuntimeEventKind::Ready, ticket, 101, 1, 19001, 0, false, false});
    auto snapshot = fixture.controller.get_runtime_snapshot(ticket.model_key);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Ready);
    CHECK(fixture.scheduler.metrics().tracked_committed == 1);

    fixture.controller.on_runtime_event({
        HLIERuntimeEventKind::Exited, ticket, 101, 1, 19001, 23, false, true});
    snapshot = fixture.controller.get_runtime_snapshot(ticket.model_key);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Failed);
    CHECK(snapshot->failure_code == 23);
    CHECK(fixture.scheduler.metrics().tracked_committed == 0);
    CHECK(fixture.directory.size() == 0);
    CHECK(fixture.notifications.load() >= 3);
}

TEST_CASE("ModelController retains uncertain runtime ownership as Orphaned",
          "[p1][model-controller]") {
    ControllerFixture fixture;
    const auto ticket = fixture.start_with_tracked_reservation(false);
    fixture.controller.on_runtime_event({
        HLIERuntimeEventKind::Ready, ticket, 101, 1, 19001, 0, false, false});

    fixture.controller.on_runtime_event({
        HLIERuntimeEventKind::Exited, ticket, 101, 1, 19001, -9, false, false});
    auto snapshot = fixture.controller.get_runtime_snapshot(ticket.model_key);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Orphaned);
    CHECK(fixture.scheduler.metrics().tracked_committed == 1);
    CHECK(fixture.directory.find(ticket));
    CHECK_FALSE(fixture.controller.try_begin_loading_ticket(ticket.model_key));

    const auto stopped = fixture.controller.stop_all(
        HLIERuntimeStopReason::Shutdown, std::chrono::milliseconds(1));
    CHECK_FALSE(stopped.complete());
    CHECK(fixture.directory.find(ticket));
    CHECK(fixture.scheduler.metrics().tracked_committed == 1);
}

TEST_CASE("ModelController eviction releases resources only after confirmed stop",
          "[p1][model-controller][eviction]") {
    ControllerFixture fixture;
    const auto ticket = fixture.start_with_tracked_reservation();
    fixture.controller.on_runtime_event({
        HLIERuntimeEventKind::Ready, ticket, 101, 1, 19001, 0, false, false});

    const auto result = fixture.controller.evict_runtime(
        ticket.model_key, "test_eviction");
    REQUIRE(result.evicted());
    CHECK(result.identity.launch_ticket.matches(ticket));
    CHECK_FALSE(result.registry_update_stale);
    CHECK(fixture.directory.size() == 0);
    CHECK(fixture.scheduler.metrics().tracked_committed == 0);
    const auto snapshot = fixture.controller.get_runtime_snapshot(ticket.model_key);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Unloaded);
}

TEST_CASE("ModelController aborts eviction when the Runtime handle is missing",
          "[p1][model-controller][eviction]") {
    ControllerFixture fixture;
    fixture.controller.register_model(fixture.model, 512);
    const auto ticket = fixture.controller.try_begin_loading_ticket(
        fixture.model.model_key_);
    REQUIRE(ticket);
    REQUIRE(fixture.controller.mark_loaded_if_current(*ticket, 19007));

    const auto result = fixture.controller.evict_runtime(
        fixture.model.model_key_, "missing_runtime");
    CHECK(result.status == HLIEControllerEvictionStatus::RuntimeMissing);
    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Ready);
}

TEST_CASE("ModelController ensure_ready coalesces concurrent launch ownership",
          "[p1][model-controller][ensure-ready][stress]") {
    ControllerFixture fixture;
    fixture.controller.register_model(fixture.model, 512);

    std::atomic<int> launchers{0};
    std::atomic<int> waiters{0};
    std::atomic<int> invalid_decisions{0};
    std::vector<std::thread> callers;
    constexpr int caller_count = 16;
    callers.reserve(caller_count);
    for (int index = 0; index < caller_count; ++index) {
        callers.emplace_back([&] {
            const auto decision = fixture.controller.ensure_ready(
                fixture.model.model_key_);
            if (decision.disposition ==
                HLIEEnsureReadyDisposition::LaunchRequired) {
                ++launchers;
                if (!decision.launch_ticket) {
                    ++invalid_decisions;
                }
            } else if (decision.disposition ==
                       HLIEEnsureReadyDisposition::WaitForExistingLaunch) {
                ++waiters;
            } else {
                ++invalid_decisions;
            }
        });
    }
    for (auto& caller : callers) {
        caller.join();
    }

    CHECK(launchers.load() == 1);
    CHECK(waiters.load() == caller_count - 1);
    CHECK(invalid_decisions.load() == 0);
    const auto ticket = fixture.controller.get_launch_ticket(
        fixture.model.model_key_);
    REQUIRE(ticket);
    REQUIRE(fixture.controller.mark_loaded_if_current(*ticket, 19002));
    const auto ready = fixture.controller.wait_until_ready(
        fixture.model.model_key_,
        std::chrono::steady_clock::now() + std::chrono::seconds(1));
    CHECK(ready.status == HLIEWaitReadyStatus::Ready);
    REQUIRE(ready.snapshot);
    CHECK(ready.snapshot->generation == ticket->generation);
}

TEST_CASE("ModelController owns RuntimeFactory publication and rejected cleanup",
          "[p1][model-controller][runtime-factory]") {
    ControllerFixture fixture;
    fixture.controller.register_model(fixture.model, 512);
    const auto ticket = fixture.controller.try_begin_loading_ticket(
        fixture.model.model_key_);
    REQUIRE(ticket);

    auto first = std::make_shared<ControllerRuntime>(
        HLIERuntimeIdentity{201, HLIERuntimeKind::Embedded, *ticket, 0, 0});
    fixture.runtime_factory.next_runtime = first;
    const auto published = fixture.controller.publish_embedded_runtime({
        *ticket, {}, "sd", false});
    REQUIRE(published.published);
    CHECK(published.identity.runtime_id == 201);
    CHECK(fixture.directory.find(*ticket) == first);

    auto duplicate = std::make_shared<ControllerRuntime>(
        HLIERuntimeIdentity{202, HLIERuntimeKind::Embedded, *ticket, 0, 0});
    fixture.runtime_factory.next_runtime = duplicate;
    const auto rejected = fixture.controller.publish_embedded_runtime({
        *ticket, {}, "sd", false});
    CHECK_FALSE(rejected.published);
    CHECK(rejected.cleanup_confirmed);
    CHECK(rejected.identity.runtime_id == 202);
    CHECK(duplicate->health().state == HLIERuntimeHealthState::Stopped);
    CHECK(fixture.directory.find(*ticket) == first);
}

TEST_CASE("ModelController owns embedded load reservation adoption",
          "[p1][model-controller][load-transaction]") {
    ControllerFixture fixture;
    fixture.controller.register_model(fixture.model, 512);
    auto pending = fixture.scheduler.reserve(
        fixture.model.model_key_, 512, automatic_placement());
    REQUIRE(pending);
    const auto begin = fixture.controller.begin_load(
        fixture.model.model_key_, *pending);
    REQUIRE(begin.started());
    const auto ticket = *begin.launch_ticket;

    fixture.runtime_factory.next_runtime =
        std::make_shared<ControllerRuntime>(HLIERuntimeIdentity{
            301, HLIERuntimeKind::Embedded, ticket, 0, 0});
    const auto adopted = fixture.controller.adopt_embedded_runtime(
        *pending, ticket, {ticket, {}, "sd", false}, 19003);
    REQUIRE(adopted.accepted());
    CHECK(adopted.status == HLIEControllerRuntimeAdoptionStatus::Ready);
    CHECK_FALSE(pending->valid());
    CHECK(fixture.scheduler.metrics().tracked_committed == 1);
    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Ready);
    CHECK(snapshot->port == 19003);
}

TEST_CASE("ModelController owns process adoption until READY commit",
          "[p1][model-controller][load-transaction]") {
    ControllerFixture fixture;
    fixture.controller.register_model(fixture.model, 512);
    auto pending = fixture.scheduler.reserve(
        fixture.model.model_key_, 512, automatic_placement());
    REQUIRE(pending);
    const auto begin = fixture.controller.begin_load(
        fixture.model.model_key_, *pending);
    REQUIRE(begin.started());
    const auto ticket = *begin.launch_ticket;
    const HLIERuntimeIdentity identity{
        302, HLIERuntimeKind::Process, ticket, 2, 19004};
    fixture.runtime_factory.next_runtime =
        std::make_shared<ControllerRuntime>(identity);

    const auto adopted = fixture.controller.adopt_process_runtime(
        *pending, ticket, {identity, {}, {ticket.model_key}, false});
    REQUIRE(adopted.accepted());
    CHECK(adopted.status ==
          HLIEControllerRuntimeAdoptionStatus::AwaitingReady);
    CHECK(fixture.scheduler.metrics().tracked_pending == 1);
    fixture.controller.on_runtime_event({
        HLIERuntimeEventKind::Ready, ticket, identity.runtime_id,
        identity.instance_id, identity.endpoint_port, 0, false, false});
    CHECK(fixture.scheduler.metrics().tracked_committed == 1);
    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Ready);
}

TEST_CASE("ModelController sequences complete embedded load transaction",
          "[p1][model-controller][load-transaction]") {
    ControllerFixture fixture;
    fixture.controller.register_model(fixture.model, 512);
    auto pending = fixture.scheduler.reserve(
        fixture.model.model_key_, 512, automatic_placement());
    REQUIRE(pending);

    fixture.backend_launcher.next_artifact.failure =
        HLIEBackendLaunchFailure::None;
    fixture.backend_launcher.next_artifact.runtime_kind =
        HLIERuntimeKind::Embedded;
    fixture.backend_launcher.next_artifact.ready_port = 19005;
    fixture.backend_launcher.next_artifact.embedded_runtime = {
        {}, fixture.model.get_capabilities(), "sd", false};
    fixture.backend_launcher.echo_request = true;
    fixture.runtime_factory.echo_embedded_request = true;

    const auto loaded = fixture.controller.load_model(
        *pending, fixture.model, "sd", false);
    REQUIRE(loaded.accepted());
    CHECK(loaded.status == HLIEControllerLoadStatus::Ready);
    CHECK_FALSE(pending->valid());
    CHECK(fixture.scheduler.metrics().tracked_committed == 1);
    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Ready);
    CHECK(snapshot->port == 19005);
}

TEST_CASE("ModelController rolls back a failed backend start",
          "[p1][model-controller][load-transaction][rollback]") {
    ControllerFixture fixture;
    fixture.controller.register_model(fixture.model, 512);
    auto pending = fixture.scheduler.reserve(
        fixture.model.model_key_, 512, automatic_placement());
    REQUIRE(pending);
    fixture.backend_launcher.next_artifact.failure =
        HLIEBackendLaunchFailure::LocalServiceStartFailed;

    const auto loaded = fixture.controller.load_model(
        *pending, fixture.model, "sd", false);
    CHECK(loaded.status == HLIEControllerLoadStatus::BackendStartFailed);
    CHECK_FALSE(pending->valid());
    CHECK(fixture.scheduler.metrics().tracked_pending == 0);
    CHECK(fixture.scheduler.metrics().tracked_committed == 0);
    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Unloaded);
}

TEST_CASE("ModelController rolls back an unclaimed competing reservation",
          "[p1][model-controller][load-transaction][rollback]") {
    ControllerFixture fixture;
    fixture.controller.register_model(fixture.model, 512);
    const auto owner = fixture.controller.try_begin_loading_ticket(
        fixture.model.model_key_);
    REQUIRE(owner);
    auto competing = fixture.scheduler.reserve(
        fixture.model.model_key_, 512, automatic_placement());
    REQUIRE(competing);

    const auto loaded = fixture.controller.load_model(
        *competing, fixture.model, "sd", false);
    CHECK(loaded.status == HLIEControllerLoadStatus::BusyOrMissing);
    CHECK_FALSE(competing->valid());
    CHECK(fixture.provider->get_reservations().empty());
    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Starting);
    CHECK(snapshot->generation == owner->generation);
}

TEST_CASE("ModelController keeps failed unadopted backend ownership Orphaned",
          "[p1][model-controller][load-transaction][orphaned]") {
    ControllerFixture fixture;
    fixture.controller.register_model(fixture.model, 512);
    auto pending = fixture.scheduler.reserve(
        fixture.model.model_key_, 512, automatic_placement());
    REQUIRE(pending);
    fixture.backend_launcher.next_artifact.failure =
        HLIEBackendLaunchFailure::None;
    fixture.backend_launcher.next_artifact.runtime_kind =
        HLIERuntimeKind::Embedded;
    fixture.backend_launcher.next_artifact.ready_port = 19006;
    fixture.backend_launcher.next_artifact.embedded_runtime = {
        {}, fixture.model.get_capabilities(), "sd", false};
    fixture.backend_launcher.echo_request = true;
    fixture.backend_launcher.cleanup_confirmed = false;

    const auto loaded = fixture.controller.load_model(
        *pending, fixture.model, "sd", false);
    CHECK(loaded.status == HLIEControllerLoadStatus::CleanupUnconfirmed);
    CHECK(fixture.backend_launcher.cleanups.load() == 1);
    CHECK_FALSE(pending->valid());
    CHECK(fixture.scheduler.metrics().tracked_pending == 1);
    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Orphaned);
}

TEST_CASE("ModelReadinessService coalesces concurrent automatic loads",
          "[p1][model-readiness]") {
    ControllerFixture fixture;
    fixture.controller.register_model(fixture.model, 512);
    std::atomic<int> load_count{0};
    HLIEModelReadinessService readiness{
        fixture.controller,
        [&](HLIEModelLoadRequest, int,
            const std::optional<HLIEModelLaunchTicket>& preclaimed) {
            ++load_count;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (!preclaimed || !fixture.controller.mark_loaded_if_current(
                    *preclaimed, 19011)) {
                return HLIEModelLoadResult::failure({
                    HLIEControlErrorCode::StaleOperation,
                    HLIEControlErrorStage::Readiness,
                    "test load became stale", 0, false});
            }
            HLIEModelLoadReceipt receipt;
            receipt.model_key = preclaimed->model_key;
            receipt.ready = true;
            return HLIEModelLoadResult::success(std::move(receipt));
        },
        [] { return false; }};

    constexpr int request_count = 12;
    std::vector<HLIEModelReadinessStatus> statuses(
        request_count, HLIEModelReadinessStatus::Missing);
    std::vector<std::thread> threads;
    threads.reserve(request_count);
    for (int index = 0; index < request_count; ++index) {
        threads.emplace_back([&, index] {
            HLIEModelReadinessRequest request;
            request.model_key = fixture.model.model_key_;
            request.load_request.emplace();
            request.deadline = std::chrono::steady_clock::now() +
                std::chrono::seconds(2);
            statuses[index] = readiness.ensure_ready(
                std::move(request)).status;
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    readiness.stop_and_join();

    CHECK(load_count.load() == 1);
    for (const auto status : statuses) {
        CHECK(status == HLIEModelReadinessStatus::Ready);
    }
    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Ready);
    CHECK(snapshot->port == 19011);
}

TEST_CASE("ModelReadinessService cancellation preserves the current generation",
          "[p1][model-readiness]") {
    ControllerFixture fixture;
    fixture.controller.register_model(fixture.model, 512);
    const auto ticket = fixture.controller.try_begin_loading_ticket(
        fixture.model.model_key_);
    REQUIRE(ticket);
    std::atomic<int> load_count{0};
    HLIEModelReadinessService readiness{
        fixture.controller,
        [&](HLIEModelLoadRequest, int,
            const std::optional<HLIEModelLaunchTicket>&) {
            ++load_count;
            return HLIEModelLoadResult::failure({
                HLIEControlErrorCode::Internal,
                HLIEControlErrorStage::RuntimeStart,
                "unexpected load", 0, false});
        },
        [] { return false; }};

    HLIEModelReadinessRequest request;
    request.model_key = fixture.model.model_key_;
    request.deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(1);
    request.should_stop = [] { return true; };
    const auto result = readiness.ensure_ready(std::move(request));

    CHECK(result.status == HLIEModelReadinessStatus::Cancelled);
    CHECK(load_count.load() == 0);
    CHECK(fixture.controller.is_current_launch(*ticket));
    fixture.controller.rollback_model_start(
        *ticket, "readiness_cancellation_test_cleanup");
}

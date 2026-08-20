#include <catch_amalgamated.hpp>

#include <chrono>
#include <memory>
#include <thread>
#include <unordered_set>
#include <utility>

#include "hlie_app/control/hlie_control_plane_reconciler.h"
#include "hlie_app/control/hlie_model_controller.h"

namespace {

using namespace hm_HLIE;

class ReconcilerReservationProvider final
    : public IHLIEDeviceReservationProvider {
public:
    ReconcilerReservationProvider() {
        ledger_.update_observation({0, 8192, 0, 8192});
        ledger_.set_pending_ttl(std::chrono::seconds(0));
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

class ReconcilerRuntime final : public IHLIEModelRuntime {
public:
    explicit ReconcilerRuntime(HLIERuntimeIdentity identity)
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

class ReconcilerRuntimeFactory final : public IHLIEModelRuntimeFactory {
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

HLIEPlacementIntent reconciler_placement() {
    HLIEPlacementIntent placement;
    placement.required_device_count = 1;
    placement.allow_auto_selection = true;
    return placement;
}

HLIEModelInfo reconciler_model() {
    HLIEModelInfo info;
    info.build_sd("sd", "vae", "t5", "clip", "clip-l", "mmdit", "0");
    return info;
}

struct ReconcilerFixture {
    std::shared_ptr<ReconcilerReservationProvider> provider =
        std::make_shared<ReconcilerReservationProvider>();
    HLIEResourceScheduler scheduler{provider};
    HLIERuntimeDirectory directory;
    ReconcilerRuntimeFactory runtime_factory;
    HLIEModelController controller{scheduler, directory, runtime_factory};
};

} // namespace

TEST_CASE("ControlPlaneReconciler generation-fences expired pending owners",
          "[p1][control-plane-reconciler]") {
    ReconcilerFixture fixture;
    const HLIEModelLaunchTicket owner{"external-model", 7, "owner-token"};
    auto pending = fixture.scheduler.reserve(
        owner.model_key, 256, reconciler_placement());
    REQUIRE(pending);
    REQUIRE(fixture.scheduler.bind(*pending, owner));
    REQUIRE(fixture.scheduler.track(*pending, owner));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    bool owner_is_live = true;
    HLIEControlPlaneReconciler reconciler(
        fixture.scheduler, fixture.controller, nullptr,
        [] { return HLIEModelResidencyPolicy{}; },
        [&] {
            return owner_is_live
                ? std::vector<HLIEModelLaunchTicket>{owner}
                : std::vector<HLIEModelLaunchTicket>{};
        });

    REQUIRE(reconciler.reconcile_once(false));
    CHECK(fixture.scheduler.metrics().expired_pending_protected == 1);
    CHECK(fixture.scheduler.metrics().tracked_pending == 1);

    owner_is_live = false;
    REQUIRE(reconciler.reconcile_once(false));
    const auto metrics = fixture.scheduler.metrics();
    CHECK(metrics.expired_pending_protected == 0);
    CHECK(metrics.expired_pending_rollbacks == 1);
    CHECK(metrics.tracked_pending == 0);
}

TEST_CASE("ControlPlaneReconciler executes idle TTL through ModelController",
          "[p1][control-plane-reconciler][eviction]") {
    ReconcilerFixture fixture;
    auto model = reconciler_model();
    fixture.controller.register_model(model, 512);
    const auto ticket = fixture.controller.try_begin_loading_ticket(model.model_key_);
    REQUIRE(ticket);
    auto runtime = std::make_shared<ReconcilerRuntime>(HLIERuntimeIdentity{
        501, HLIERuntimeKind::Embedded, *ticket, 0, 0});
    REQUIRE(fixture.directory.publish(runtime));
    REQUIRE(fixture.controller.mark_loaded_if_current(*ticket, 7901));

    HLIEControlPlaneReconciler reconciler(
        fixture.scheduler, fixture.controller, nullptr,
        [] {
            HLIEModelResidencyPolicy policy;
            policy.auto_unload = true;
            policy.idle_ttl_sec = 1;
            return policy;
        });

    REQUIRE(reconciler.reconcile_once(true, std::time(nullptr) + 10));
    const auto snapshot = fixture.controller.get_runtime_snapshot(model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Unloaded);
    CHECK(fixture.directory.size() == 0);
    CHECK(reconciler.metrics().idle_ttl_evictions == 1);
    CHECK(reconciler.metrics().policy_refreshes == 1);
}

TEST_CASE("ControlPlaneReconciler stop interrupts a long wait",
          "[p1][control-plane-reconciler][lifecycle]") {
    ReconcilerFixture fixture;
    HLIEControlPlaneReconcilerOptions options;
    options.reconciliation_interval = std::chrono::seconds(5);
    HLIEControlPlaneReconciler reconciler(
        fixture.scheduler, fixture.controller, nullptr,
        [] { return HLIEModelResidencyPolicy{}; }, {}, options);

    reconciler.start();
    REQUIRE(reconciler.metrics().running);
    const auto started = std::chrono::steady_clock::now();
    reconciler.stop();
    const auto elapsed = std::chrono::steady_clock::now() - started;

    CHECK_FALSE(reconciler.metrics().running);
    CHECK(elapsed < std::chrono::seconds(1));
}

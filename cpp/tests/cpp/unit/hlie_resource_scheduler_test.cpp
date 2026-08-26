#include <catch_amalgamated.hpp>

#include <chrono>
#include <memory>
#include <thread>
#include <unordered_set>

#include "hlie_app/resource/hlie_resource_scheduler.h"

HLIE_NAMESPACE_BEGIN
namespace {

class FakeReservationProvider final : public IHLIEDeviceReservationProvider {
public:
    FakeReservationProvider() {
        ledger_.update_observation({0, 8192, 0, 8192});
        ledger_.update_observation({1, 8192, 0, 8192});
    }

    void expire_pending_immediately() {
        ledger_.set_pending_ttl(std::chrono::seconds(0));
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

HLIEModelLaunchTicket ticket(const std::string& key, const std::uint64_t generation) {
    return {key, generation, "token-" + std::to_string(generation)};
}

HLIEPlacementIntent automatic_placement(const int count = 1) {
    HLIEPlacementIntent placement;
    placement.required_device_count = count;
    placement.allow_auto_selection = true;
    return placement;
}

} // namespace

TEST_CASE("resource scheduler enforces typed pending to committed transition",
          "[p1][resource-scheduler]") {
    const auto provider = std::make_shared<FakeReservationProvider>();
    HLIEResourceScheduler scheduler(provider);

    auto pending = scheduler.reserve("model-a", 1024, automatic_placement());
    REQUIRE(pending);
    REQUIRE(pending->valid());
    const auto launch = ticket("model-a", 1);
    REQUIRE(scheduler.bind(*pending, launch));
    REQUIRE(scheduler.track(*pending, launch));
    CHECK_FALSE(pending->valid());

    CHECK_FALSE(scheduler.commit(ticket("model-a", 2)));
    REQUIRE(scheduler.commit(launch));
    auto metrics = scheduler.metrics();
    CHECK(metrics.tracked_pending == 0);
    CHECK(metrics.tracked_committed == 1);

    scheduler.release(launch);
    metrics = scheduler.metrics();
    CHECK(metrics.tracked_committed == 0);
    CHECK(provider->get_reservations().empty());
}

TEST_CASE("resource scheduler canonicalizes an unbound alias exactly once",
          "[p1][resource-scheduler]") {
    const auto provider = std::make_shared<FakeReservationProvider>();
    HLIEResourceScheduler scheduler(provider);
    auto pending = scheduler.reserve("request-alias", 512, automatic_placement());
    REQUIRE(pending);

    const auto canonical = ticket("canonical-model-key", 1);
    REQUIRE(scheduler.bind(*pending, canonical));
    CHECK(pending->reservation().model_key == "canonical-model-key");
    CHECK(pending->reservation().generation == 1);
    CHECK_FALSE(pending->bind(ticket("different-owner", 2)));
    scheduler.rollback(*pending);
}

TEST_CASE("resource scheduler owns local runtime reservation through release",
          "[p1][resource-scheduler]") {
    const auto provider = std::make_shared<FakeReservationProvider>();
    HLIEResourceScheduler scheduler(provider);
    auto pending = scheduler.reserve("local-model", 512, automatic_placement());
    REQUIRE(pending);
    const auto id = pending->id();
    const auto owner = ticket("local-model", 7);
    REQUIRE(scheduler.bind(*pending, owner));

    REQUIRE(scheduler.track(*pending, owner));
    CHECK_FALSE(pending->valid());
    REQUIRE(scheduler.commit(owner));
    const auto grants = provider->get_reservations();
    REQUIRE(grants.size() == 1);
    CHECK(grants.front().id == id);
    CHECK(grants.front().state == HLIEDeviceReservationState::Committed);

    scheduler.release(owner);
    CHECK(provider->get_reservations().empty());
}

TEST_CASE("resource scheduler fences replacement and explicit rollback",
          "[p1][resource-scheduler]") {
    const auto provider = std::make_shared<FakeReservationProvider>();
    HLIEResourceScheduler scheduler(provider);

    auto first = scheduler.reserve("reloadable", 512, automatic_placement());
    REQUIRE(first);
    const auto first_ticket = ticket("reloadable", 1);
    REQUIRE(scheduler.bind(*first, first_ticket));
    REQUIRE(scheduler.track(*first, first_ticket));

    auto second = scheduler.reserve("reloadable", 512, automatic_placement());
    REQUIRE(second);
    const auto second_id = second->id();
    const auto second_ticket = ticket("reloadable", 2);
    REQUIRE(scheduler.bind(*second, second_ticket));
    REQUIRE(scheduler.track(*second, second_ticket));
    REQUIRE(provider->get_reservations().size() == 1);
    CHECK(provider->get_reservations().front().id == second_id);

    scheduler.release(first_ticket);
    CHECK(provider->get_reservations().size() == 1);
    scheduler.release(second_ticket);
    CHECK(provider->get_reservations().empty());

    auto orphan = scheduler.reserve("orphan", 256, automatic_placement());
    REQUIRE(orphan);
    scheduler.rollback(*orphan);
    CHECK_FALSE(orphan->valid());
    CHECK(provider->get_reservations().empty());
}

TEST_CASE("resource scheduler reconciles only unowned expired grants",
          "[p1][resource-scheduler]") {
    const auto provider = std::make_shared<FakeReservationProvider>();
    provider->expire_pending_immediately();
    HLIEResourceScheduler scheduler(provider);
    auto pending = scheduler.reserve("worker", 512, automatic_placement());
    REQUIRE(pending);
    const auto launch = ticket("worker", 3);
    REQUIRE(scheduler.bind(*pending, launch));
    REQUIRE(scheduler.track(*pending, launch));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    scheduler.reconcile_expired_pending({launch});
    CHECK(provider->get_reservations().size() == 1);
    CHECK(scheduler.metrics().expired_pending_protected == 1);

    scheduler.reconcile_expired_pending({});
    CHECK(provider->get_reservations().empty());
    const auto metrics = scheduler.metrics();
    CHECK(metrics.expired_pending_protected == 0);
    CHECK(metrics.expired_pending_rollbacks == 1);
    CHECK(metrics.tracked_pending == 0);
}

TEST_CASE("resource scheduler owns placement validation and residency planning",
          "[p1][resource-scheduler]") {
    const auto provider = std::make_shared<FakeReservationProvider>();
    HLIEModelResidencyPolicy policy;
    policy.auto_unload = true;
    policy.max_loaded_models = 1;
    HLIEResourceScheduler scheduler(provider, policy);

    auto invalid = automatic_placement(2);
    invalid.requested_device_ids = {0};
    invalid.allow_auto_selection = false;
    CHECK_FALSE(scheduler.reserve("invalid", 512, invalid));

    auto explicit_placement = automatic_placement();
    explicit_placement.requested_device_ids = {1};
    explicit_placement.allow_auto_selection = false;
    const auto reservation = scheduler.reserve("explicit", 512, explicit_placement);
    REQUIRE(reservation);
    CHECK(reservation->reservation().devices == std::vector<int>{1});

    HLIEModelRuntimeSnapshot loaded;
    loaded.identity.model_key = "old";
    loaded.identity.model_name = "old";
    loaded.state = HLIEModelLifecycleState::Ready;
    loaded.evictable = true;
    loaded.last_access_time = 10;
    const auto plan = scheduler.plan_for_loaded_limit({loaded}, "new");
    REQUIRE(plan.should_evict());
    CHECK(plan.reason == "max_loaded_models");
    CHECK(plan.candidate->key == "old");
}

HLIE_NAMESPACE_END

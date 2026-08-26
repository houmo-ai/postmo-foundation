#include <catch_amalgamated.hpp>

#include <atomic>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hlie_app/control/hlie_model_controller.h"
#include "hlie_app/startup/hlie_embedded_startup_service.h"

namespace {

using namespace hm_HLIE;

class ReservationProvider final : public IHLIEDeviceReservationProvider {
public:
    ReservationProvider() { ledger_.update_observation({0, 8192, 0, 8192}); }

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
        const std::unordered_set<HLIEDeviceReservationId>& protected_ids)
        override {
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

class FakeService final : public IHLIEModelService {
public:
    explicit FakeService(std::vector<int> devices) {
        devices_ = std::move(devices);
    }

    int init(const std::string&, const HLIEModelInfo&,
             const std::shared_ptr<HLIEDeviceService>&) override {
        return 0;
    }
    void run() override { ++run_count; }
    void process() override {}
    void stop() override { ++stop_count; }
    int get_state() override { return 0; }

    std::atomic<int> run_count{0};
    std::atomic<int> stop_count{0};
};

class FakeRuntime final : public IHLIEModelRuntime {
public:
    explicit FakeRuntime(HLIERuntimeIdentity identity)
        : identity_(std::move(identity)) {}

    HLIERuntimeIdentity identity() const override { return identity_; }
    HLIEModelCapabilities capabilities() const override { return {}; }
    HLIERuntimeHealth health() const override {
        return {identity_, state_, 0};
    }
    void begin_draining() override { state_ = HLIERuntimeHealthState::Draining; }
    HLIERuntimeStopResult stop(const HLIERuntimeStopRequest&) override {
        state_ = HLIERuntimeHealthState::Stopped;
        return {identity_, true, true, false, false, 0};
    }

private:
    HLIERuntimeIdentity identity_;
    HLIERuntimeHealthState state_ = HLIERuntimeHealthState::Starting;
};

class RuntimeFactory final : public IHLIEModelRuntimeFactory {
public:
    bool enabled = true;

    std::shared_ptr<IHLIEModelRuntime> create_embedded(
        const HLIEEmbeddedRuntimeCreateRequest& request) override {
        if (!enabled) {
            return nullptr;
        }
        return std::make_shared<FakeRuntime>(HLIERuntimeIdentity{
            701, HLIERuntimeKind::Embedded, request.launch_ticket, 0, 0});
    }
    std::shared_ptr<IHLIEModelRuntime> create_process(
        const HLIEProcessRuntimeCreateRequest&) override {
        return nullptr;
    }
};

class FakeTargetProvider final : public IHLIEEmbeddedStartupTargetProvider {
public:
    std::vector<HLIEEmbeddedStartupTarget> primary;
    std::vector<HLIEEmbeddedStartupTarget> secondary;

    std::vector<HLIEEmbeddedStartupTarget> targets(
        const HLIEEmbeddedStartupGroup group) const override {
        return group == HLIEEmbeddedStartupGroup::Primary ? primary : secondary;
    }
};

struct Fixture {
    std::shared_ptr<ReservationProvider> provider =
        std::make_shared<ReservationProvider>();
    HLIEResourceScheduler scheduler{provider};
    HLIERuntimeDirectory directory;
    RuntimeFactory runtime_factory;
    HLIEModelController controller{scheduler, directory, runtime_factory};
    HLIEModelInfo model;
    std::shared_ptr<FakeService> service =
        std::make_shared<FakeService>(std::vector<int>{0});

    Fixture() {
        model.build_sd("sd", "vae", "t5", "clip", "clip-l", "mmdit", "0");
        controller.register_model(model, 512);
    }
};

} // namespace

TEST_CASE("EmbeddedStartupService runs and publishes initialized target",
          "[p1][embedded-startup]") {
    Fixture fixture;
    FakeTargetProvider provider;
    provider.secondary.push_back({
        HLIEEmbeddedStartupGroup::Secondary,
        fixture.model.model_key_,
        HLIEModelType::SD,
        fixture.service,
        {},
        19007,
        false,
    });
    std::vector<std::string> ready;
    HLIEEmbeddedStartupService service(
        fixture.controller, provider,
        [&ready](const std::string& key) { ready.push_back(key); });

    const auto summary = service.start_group(HLIEEmbeddedStartupGroup::Secondary);
    REQUIRE(summary.complete());
    CHECK(summary.attempted == 1);
    CHECK(summary.started == 1);
    CHECK(fixture.service->run_count.load() == 1);
    REQUIRE(ready == std::vector<std::string>{fixture.model.model_key_});

    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Ready);
    CHECK(snapshot->port == 19007);
}

TEST_CASE("EmbeddedStartupService preserves primary missing-model abort",
          "[p1][embedded-startup]") {
    Fixture fixture;
    FakeTargetProvider provider;
    auto missing_service = std::make_shared<FakeService>(
        std::vector<int>{0});
    provider.primary.push_back({
        HLIEEmbeddedStartupGroup::Primary,
        "missing-model",
        HLIEModelType::LLM,
        missing_service,
        {},
        std::nullopt,
        true,
    });
    int cleanup_calls = 0;
    HLIEEmbeddedStartupService service(
        fixture.controller, provider, {},
        [&cleanup_calls](const HLIEEmbeddedStartupTarget& target) {
            CHECK(target.service_kind == HLIEModelType::LLM);
            ++cleanup_calls;
            return true;
        });

    const auto summary = service.start_group(HLIEEmbeddedStartupGroup::Primary);
    CHECK_FALSE(summary.complete());
    CHECK(summary.abort_remaining);
    CHECK(summary.failed == 1);
    CHECK(missing_service->run_count.load() == 1);
    CHECK(cleanup_calls == 1);
}

TEST_CASE("EmbeddedStartupService reconciles RuntimeFactory failure",
          "[p1][embedded-startup]") {
    Fixture fixture;
    fixture.runtime_factory.enabled = false;
    FakeTargetProvider provider;
    provider.secondary.push_back({
        HLIEEmbeddedStartupGroup::Secondary,
        fixture.model.model_key_,
        HLIEModelType::SD,
        fixture.service,
        {},
        19007,
        false,
    });
    int cleanup_calls = 0;
    HLIEEmbeddedStartupService service(
        fixture.controller, provider, {},
        [&cleanup_calls](const HLIEEmbeddedStartupTarget& target) {
            CHECK(target.service_kind == HLIEModelType::SD);
            ++cleanup_calls;
            return true;
        });

    const auto summary = service.start_group(HLIEEmbeddedStartupGroup::Secondary);
    CHECK_FALSE(summary.complete());
    CHECK(summary.failed == 1);
    CHECK(cleanup_calls == 1);
    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Unloaded);
}

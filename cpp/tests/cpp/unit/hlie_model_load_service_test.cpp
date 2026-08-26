#include <catch_amalgamated.hpp>

#include <atomic>
#include <memory>
#include <unordered_set>
#include <utility>

#include "hlie_app/control/hlie_model_load_service.h"
#include "hlie_app/http/hlie_control_http_error_mapper.h"
#include "hlie_app/runtime/hlie_model_backend_launcher.h"
#include "hlie_app/runtime/hlie_model_runtime_factory.h"

namespace {

using namespace hm_HLIE;

class LoadReservationProvider final : public IHLIEDeviceReservationProvider {
public:
    LoadReservationProvider() {
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

class LoadRuntime final : public IHLIEModelRuntime {
public:
    explicit LoadRuntime(HLIERuntimeIdentity identity)
        : identity_(std::move(identity)) {}

    HLIERuntimeIdentity identity() const override { return identity_; }
    HLIEModelCapabilities capabilities() const override { return {}; }
    HLIERuntimeHealth health() const override {
        return {identity_, state_, 0};
    }
    void begin_draining() override {
        state_ = HLIERuntimeHealthState::Draining;
    }
    HLIERuntimeStopResult stop(const HLIERuntimeStopRequest&) override {
        state_ = HLIERuntimeHealthState::Stopped;
        return {identity_, true, true, false, false, 0};
    }

private:
    HLIERuntimeIdentity identity_;
    HLIERuntimeHealthState state_ = HLIERuntimeHealthState::Starting;
};

class LoadRuntimeFactory final : public IHLIEModelRuntimeFactory {
public:
    std::shared_ptr<IHLIEModelRuntime> create_embedded(
        const HLIEEmbeddedRuntimeCreateRequest& request) override {
        return std::make_shared<LoadRuntime>(HLIERuntimeIdentity{
            401, HLIERuntimeKind::Embedded, request.launch_ticket, 0, 0});
    }
    std::shared_ptr<IHLIEModelRuntime> create_process(
        const HLIEProcessRuntimeCreateRequest&) override {
        return nullptr;
    }
};

class LoadBackendLauncher final : public IHLIEModelBackendLauncher {
public:
    std::atomic<int> launches{0};

    HLIEBackendLaunchArtifact launch(
        const HLIEBackendLaunchRequest& request) override {
        ++launches;
        HLIEBackendLaunchArtifact artifact;
        artifact.failure = HLIEBackendLaunchFailure::None;
        artifact.runtime_kind = HLIERuntimeKind::Embedded;
        artifact.ready_port = 19005;
        artifact.instance_id = 0;
        artifact.notify_model_key = request.model_info.model_key_;
        artifact.embedded_runtime = {
            request.launch_ticket, request.model_info.get_capabilities(),
            request.service_kind, false};
        return artifact;
    }

    bool cleanup_unadopted(
        const HLIEBackendLaunchArtifact&,
        const std::string&) override {
        return true;
    }
};

HLIEModelInfo load_sd_model() {
    HLIEModelInfo model;
    model.build_sd("sd", "vae", "t5", "clip", "clip-l", "mmdit", "0");
    return model;
}

HLIEModelLoadRequest load_sd_request(const std::string& model_key) {
    HLIEModelLoadRequest request;
    request.model_type = "sd";
    request.model = model_key;
    request.requested_device_ids = std::vector<int>{0};
    request.sd_artifacts = HLIESdModelArtifacts{
        "vae", "t5", "clip", "clip-l", "mmdit"};
    return request;
}

struct LoadServiceFixture {
    std::shared_ptr<LoadReservationProvider> provider =
        std::make_shared<LoadReservationProvider>();
    HLIEResourceScheduler scheduler{provider};
    HLIERuntimeDirectory directory;
    LoadRuntimeFactory runtime_factory;
    LoadBackendLauncher backend_launcher;
    HLIEModelController controller{
        scheduler, directory, runtime_factory, {}, &backend_launcher};
    HLIEModelLoadService service{
        controller, scheduler, nullptr, "", false,
        [] { return HLIEModelResidencyPolicy{}; },
        [](HLIEModelInfo&) { return true; }};
    HLIEModelInfo model = load_sd_model();
};

} // namespace

TEST_CASE("ModelLoadService owns reservation and embedded load transaction",
          "[p1][model-load-service]") {
    LoadServiceFixture fixture;
    fixture.controller.register_model(fixture.model, 512);

    auto result = fixture.service.load(
        load_sd_request(fixture.model.model_key_), 18932);
    REQUIRE(result);
    const auto* receipt = result.value_if();
    REQUIRE(receipt);
    CHECK(receipt->ready);
    CHECK(receipt->runtime_kind == HLIERuntimeKind::Embedded);
    CHECK(receipt->port == 18932);
    CHECK(receipt->notify_model_key == fixture.model.model_key_);
    CHECK(fixture.backend_launcher.launches.load() == 1);
    CHECK(fixture.scheduler.metrics().tracked_pending == 0);
    CHECK(fixture.scheduler.metrics().tracked_committed == 1);

    const auto snapshot = fixture.controller.get_runtime_snapshot(
        fixture.model.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Ready);
    CHECK(snapshot->port == 19005);
}

TEST_CASE("ModelLoadService rejects invalid explicit placement before launch",
          "[p1][model-load-service]") {
    LoadServiceFixture fixture;
    fixture.controller.register_model(fixture.model, 512);
    auto request = load_sd_request(fixture.model.model_key_);
    request.requested_device_ids = std::vector<int>{0, 1};

    const auto result = fixture.service.load(std::move(request), 18932);
    REQUIRE_FALSE(result);
    const auto* error = result.error_if();
    REQUIRE(error);
    CHECK(error->code == HLIEControlErrorCode::InvalidArgument);
    CHECK(error->stage == HLIEControlErrorStage::Reservation);
    CHECK(error->native_code == 80004);
    CHECK(fixture.backend_launcher.launches.load() == 0);
    CHECK(fixture.provider->get_reservations().empty());
}

TEST_CASE("ModelLoadService rejects replica-mode dynamic starts",
          "[p1][model-load-service]") {
    LoadServiceFixture fixture;
    HLIEModelLoadService replica_service{
        fixture.controller, fixture.scheduler, nullptr, "", true,
        [] { return HLIEModelResidencyPolicy{}; },
        [](HLIEModelInfo&) { return true; }};

    const auto result = replica_service.load({}, 18932);
    REQUIRE_FALSE(result);
    const auto* error = result.error_if();
    REQUIRE(error);
    CHECK(error->code == HLIEControlErrorCode::Unsupported);
    CHECK(error->native_code == 80005);
    CHECK(fixture.backend_launcher.launches.load() == 0);
}

TEST_CASE("Control HTTP mapper preserves typed error categories",
          "[p1][control-http]") {
    CHECK(HLIEControlHttpErrorMapper::error_type_for(
              HLIEControlErrorCode::CapacityExceeded) ==
          ERROR_TYPE_UNAVAILABLE);
    CHECK(HLIEControlHttpErrorMapper::error_type_for(
              HLIEControlErrorCode::Conflict) ==
          ERROR_TYPE_INVALID_REQUEST);
    CHECK(HLIEControlHttpErrorMapper::error_type_for(
              HLIEControlErrorCode::NotFound) == ERROR_TYPE_NOT_FOUND);
    CHECK(HLIEControlHttpErrorMapper::error_type_for(
              HLIEControlErrorCode::Unsupported) ==
          ERROR_TYPE_NOT_SUPPORTED);
}

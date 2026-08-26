#include <catch_amalgamated.hpp>

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "hlie_app/control/hlie_worker_control_service.h"

namespace {

using namespace hm_HLIE;

class FakePublisher final : public IHLIEWorkerStatusPublisher {
public:
    bool publish(const HLIEWorkerStatusPublication& publication) override {
        std::lock_guard lock(mutex);
        publications.push_back(publication);
        return publish_result;
    }

    std::mutex mutex;
    std::vector<HLIEWorkerStatusPublication> publications;
    bool publish_result = true;
};

struct Fixture {
    FakePublisher publisher;
    HLIEModelLaunchTicket ticket{"model-key", 11, "launch-token"};
    HLIEWorkerControlDependencies dependencies;
    std::atomic<int> apply_count{0};
    std::atomic<int> erase_count{0};
    std::atomic<int> drain_count{0};
    std::atomic<int> shutdown_count{0};
    bool apply_result = true;

    Fixture() {
        dependencies.dynamic_worker_provider = [] { return true; };
        dependencies.replicas_mode_provider = [] { return false; };
        dependencies.local_port_provider = [] { return 8901; };
        dependencies.master_port_provider = [] { return 7901; };
        dependencies.launch_ticket_provider = [this] { return ticket; };
        dependencies.local_start_time_provider = [] { return std::time_t{123}; };
        dependencies.local_models_provider = [] {
            return std::vector<std::string>{"local-model"};
        };
        dependencies.worker_processes_provider = [] {
            return std::vector<HLIEWorkerProcessorSnapshot>{
                {"--model worker", 1, 8902, "worker-uuid", 42, 11,
                 std::time_t{456}, {"worker-model"}}};
        };
        dependencies.resolve_model_key = [](const std::string& requested)
            -> std::optional<std::string> {
            if (requested == "alias" || requested == "model-key") {
                return "model-key";
            }
            return std::nullopt;
        };
        dependencies.apply_status = [this](
            const HLIEModelLaunchTicket& actual,
            const HLIEWorkerModelStatus,
            const int) {
            ++apply_count;
            return apply_result && actual.matches(ticket);
        };
        dependencies.erase_model = [this](const std::string& key) {
            if (key == "model-key") {
                ++erase_count;
            }
        };
        dependencies.begin_draining = [this] { ++drain_count; };
        dependencies.process_shutdown_available = [] { return true; };
        dependencies.process_shutdown = [this] { ++shutdown_count; };
    }
};

} // namespace

TEST_CASE("WorkerControlService publishes typed loaded and unloaded status",
          "[p1][worker-control]") {
    Fixture fixture;
    HLIEWorkerControlService service(fixture.publisher,
                                     std::move(fixture.dependencies));

    REQUIRE(service.publish_loaded("model-key", fixture.ticket));
    REQUIRE(service.publish_unloaded("model-key", fixture.ticket));
    REQUIRE(fixture.publisher.publications.size() == 2);
    CHECK(fixture.publisher.publications[0].router_port == 7901);
    CHECK(fixture.publisher.publications[0].status ==
          HLIEWorkerModelStatus::Loaded);
    CHECK(fixture.publisher.publications[1].status ==
          HLIEWorkerModelStatus::Unloaded);
    CHECK(fixture.publisher.publications[1].launch_ticket.matches(
        fixture.ticket));
}

TEST_CASE("WorkerControlService fences status callbacks and erases unloaded workers",
          "[p1][worker-control]") {
    Fixture fixture;
    HLIEWorkerControlService service(fixture.publisher,
                                     std::move(fixture.dependencies));

    CHECK(service.apply_status({"alias", HLIEWorkerModelStatus::Loaded, 8901,
                                11, "launch-token"}) ==
          HLIEWorkerStatusApplyResult::Applied);
    CHECK(service.apply_status({"alias", HLIEWorkerModelStatus::Unloaded, 8901,
                                11, "launch-token"}) ==
          HLIEWorkerStatusApplyResult::Applied);
    CHECK(fixture.apply_count.load() == 2);
    CHECK(fixture.erase_count.load() == 1);

    CHECK(service.apply_status({"missing", HLIEWorkerModelStatus::Loaded, 8901,
                                11, "launch-token"}) ==
          HLIEWorkerStatusApplyResult::LookupFailed);
    CHECK(service.apply_status({"alias", HLIEWorkerModelStatus::Loaded, 8901,
                                10, "old-token"}) ==
          HLIEWorkerStatusApplyResult::Stale);
}

TEST_CASE("WorkerControlService shutdown ticket is idempotent",
          "[p1][worker-control]") {
    Fixture fixture;
    HLIEWorkerControlService service(fixture.publisher,
                                     std::move(fixture.dependencies));

    CHECK(service.request_shutdown(10, "old-token") ==
          HLIEWorkerShutdownResult::InvalidTicket);
    CHECK(service.request_shutdown(11, "launch-token") ==
          HLIEWorkerShutdownResult::Accepted);
    CHECK(service.request_shutdown(11, "launch-token") ==
          HLIEWorkerShutdownResult::AlreadyRequested);
    CHECK(fixture.shutdown_count.load() == 1);
    CHECK(fixture.drain_count.load() == 2);
}

TEST_CASE("WorkerControlService exposes a neutral processor snapshot",
          "[p1][worker-control]") {
    Fixture fixture;
    HLIEWorkerControlService service(fixture.publisher,
                                     std::move(fixture.dependencies));
    const auto snapshot = service.snapshot();

    CHECK(snapshot.local_port == 8901);
    CHECK(snapshot.local_start_time == 123);
    REQUIRE(snapshot.local_models == std::vector<std::string>{"local-model"});
    REQUIRE(snapshot.workers.size() == 1);
    CHECK(snapshot.workers.front().port == 8902);
    CHECK(snapshot.workers.front().generation == 11);
}

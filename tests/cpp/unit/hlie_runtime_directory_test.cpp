#include <catch_amalgamated.hpp>

#include <atomic>
#include <thread>
#include <vector>

#include "hlie_app/runtime/hlie_runtime_directory.h"

namespace {

using namespace hm_HLIE;

class FakeRuntime final : public IHLIEModelRuntime {
public:
    explicit FakeRuntime(HLIERuntimeIdentity identity) : identity_(std::move(identity)) {}

    HLIERuntimeIdentity identity() const override { return identity_; }
    HLIEModelCapabilities capabilities() const override { return {}; }
    HLIERuntimeHealth health() const override { return {identity_, state_, 0}; }
    void begin_draining() override { state_ = HLIERuntimeHealthState::Draining; }
    HLIERuntimeStopResult stop(const HLIERuntimeStopRequest&) override {
        state_ = HLIERuntimeHealthState::Stopped;
        return {identity_, true, true, false, false, 0};
    }

private:
    HLIERuntimeIdentity identity_;
    HLIERuntimeHealthState state_ = HLIERuntimeHealthState::Ready;
};

HLIEModelLaunchTicket directory_ticket(const std::string& key,
                                       const HLIEModelGeneration generation) {
    return {key, generation, "directory-token-" + std::to_string(generation)};
}

} // namespace

TEST_CASE("runtime directory publishes and detaches by exact generation",
          "[p1][runtime-directory]") {
    HLIERuntimeDirectory directory;
    const auto first_ticket = directory_ticket("model", 1);
    const auto first_identity = directory.allocate_embedded_identity(first_ticket);
    REQUIRE(first_identity.valid());
    auto first = std::make_shared<FakeRuntime>(first_identity);
    REQUIRE(directory.publish(first));
    CHECK(directory.publish(first));
    CHECK(directory.find(first_ticket) == first);
    CHECK(directory.size() == 1);

    const auto stale = directory_ticket("model", 0);
    CHECK_FALSE(directory.find(stale));
    const auto second_ticket = directory_ticket("model", 2);
    auto second = std::make_shared<FakeRuntime>(
        directory.allocate_embedded_identity(second_ticket));
    REQUIRE(directory.publish(second));
    CHECK_FALSE(directory.find(first_ticket));
    CHECK(directory.find(second_ticket) == second);
    CHECK_FALSE(directory.detach(first_ticket));
    CHECK(directory.detach(second_ticket) == second);
    CHECK(directory.size() == 0);
}

TEST_CASE("runtime directory rejects duplicate generation with different identity",
          "[p1][runtime-directory]") {
    HLIERuntimeDirectory directory;
    const auto ticket = directory_ticket("same-generation", 3);
    auto first = std::make_shared<FakeRuntime>(directory.allocate_embedded_identity(ticket));
    auto duplicate = std::make_shared<FakeRuntime>(directory.allocate_embedded_identity(ticket));
    REQUIRE(directory.publish(first));
    CHECK_FALSE(directory.publish(duplicate));
    CHECK(directory.find(ticket) == first);
    REQUIRE(directory.health_snapshots().size() == 1);
    CHECK(directory.health_snapshots().front().ready());
}

TEST_CASE("runtime directory atomically detaches all ownership handles",
          "[p1][runtime-directory]") {
    HLIERuntimeDirectory directory;
    auto first = std::make_shared<FakeRuntime>(directory.allocate_embedded_identity(
        directory_ticket("first", 1)));
    auto second = std::make_shared<FakeRuntime>(directory.allocate_embedded_identity(
        directory_ticket("second", 1)));
    REQUIRE(directory.publish(first));
    REQUIRE(directory.publish(second));

    const auto detached = directory.detach_all();
    CHECK(detached.size() == 2);
    CHECK(directory.size() == 0);
    CHECK_FALSE(directory.find("first"));
    CHECK_FALSE(directory.find("second"));
}

TEST_CASE("runtime directory fences concurrent independent ownership changes",
          "[p1][runtime-directory][stress]") {
    HLIERuntimeDirectory directory;
    std::atomic<std::size_t> failures{0};
    std::vector<std::thread> workers;
    constexpr std::size_t worker_count = 8;
    constexpr std::size_t iterations = 200;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
                const auto key = "runtime-" + std::to_string(worker) + "-" +
                    std::to_string(iteration);
                const auto ticket = directory_ticket(key, 1);
                auto runtime = std::make_shared<FakeRuntime>(
                    directory.allocate_embedded_identity(ticket));
                if (!directory.publish(runtime) || directory.find(ticket) != runtime ||
                    directory.detach(ticket) != runtime) {
                    ++failures;
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    CHECK(failures.load() == 0);
    CHECK(directory.size() == 0);
}

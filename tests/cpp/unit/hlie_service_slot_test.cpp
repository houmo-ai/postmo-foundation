#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include <catch_amalgamated.hpp>

#include "hlie_core/runtime/hlie_service_slot.h"

namespace {

struct FakeService {
    explicit FakeService(int value) : value(value) {}
    int value = 0;
};

} // namespace

TEST_CASE("Service slot grants only one concurrent load ticket", "[service-slot][concurrency]") {
    hm_HLIE::HLIEServiceSlot<FakeService> slot;
    std::atomic<int> winners{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 32; ++i) {
        threads.emplace_back([&] {
            if (slot.begin_load()) {
                ++winners;
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    CHECK(winners.load() == 1);
    CHECK(slot.is_loading());
}

TEST_CASE("Service slot rejects a stale publisher after detach", "[service-slot][generation]") {
    hm_HLIE::HLIEServiceSlot<FakeService> slot;
    const auto ticket = slot.begin_load();
    REQUIRE(ticket);

    CHECK_FALSE(slot.detach());
    CHECK_FALSE(slot.publish(*ticket, std::make_shared<FakeService>(1)));
    CHECK_FALSE(slot.snapshot());

    const auto next = slot.begin_load();
    REQUIRE(next);
    CHECK(next->epoch > ticket->epoch);
    CHECK(slot.publish(*next, std::make_shared<FakeService>(2)));
    REQUIRE(slot.snapshot());
    CHECK(slot.snapshot()->value == 2);
}

TEST_CASE("Service slot snapshots keep detached services alive", "[service-slot][ownership]") {
    hm_HLIE::HLIEServiceSlot<FakeService> slot;
    const auto ticket = slot.begin_load();
    REQUIRE(ticket);
    CHECK(slot.publish(*ticket, std::make_shared<FakeService>(7)));

    const auto snapshot = slot.snapshot();
    REQUIRE(snapshot);
    const auto detached = slot.detach();
    REQUIRE(detached);
    CHECK_FALSE(slot.snapshot());
    CHECK(snapshot->value == 7);
    CHECK(detached == snapshot);
}

TEST_CASE("Service slot only clears the current load ticket", "[service-slot][generation]") {
    hm_HLIE::HLIEServiceSlot<FakeService> slot;
    const auto first = slot.begin_load();
    REQUIRE(first);
    CHECK(slot.fail_load(*first));
    CHECK_FALSE(slot.is_loading());

    const auto second = slot.begin_load();
    REQUIRE(second);
    CHECK_FALSE(slot.fail_load(*first));
    CHECK(slot.is_loading());
    CHECK(slot.fail_load(*second));
    CHECK_FALSE(slot.is_loading());
}

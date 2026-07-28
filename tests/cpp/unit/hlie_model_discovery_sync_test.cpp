#include "hlie_app/hlie_model_discovery_sync.h"

#include <catch_amalgamated.hpp>

#include <chrono>
#include <future>
#include <thread>

using namespace hm_HLIE;
using namespace std::chrono_literals;

TEST_CASE("Discovery generations only complete after the matching scan", "[model-discovery]") {
    HLIEModelDiscoverySync sync;
    const uint64_t first = sync.request();
    CHECK_FALSE(sync.wait_for(first, 0ms));

    sync.complete_through(first);
    CHECK(sync.wait_for(first, 0ms));

    const uint64_t second = sync.request();
    CHECK(second > first);
    CHECK_FALSE(sync.wait_for(second, 0ms));
    sync.complete_through(second);
    CHECK(sync.wait_for(second, 0ms));
}

TEST_CASE("Stopping discovery releases waiters without reporting completion", "[model-discovery]") {
    HLIEModelDiscoverySync sync;
    const uint64_t generation = sync.request();
    sync.stop();
    CHECK_FALSE(sync.wait_for(generation, 10ms));
}

TEST_CASE("Discovery completion promptly wakes a concurrent waiter", "[model-discovery]") {
    HLIEModelDiscoverySync sync;
    const uint64_t generation = sync.request();
    auto waiter = std::async(std::launch::async, [&] {
        return sync.wait_for(generation, 2s);
    });

    std::this_thread::sleep_for(20ms);
    sync.complete_through(generation);

    REQUIRE(waiter.wait_for(200ms) == std::future_status::ready);
    CHECK(waiter.get());
}

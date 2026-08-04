#include <catch_amalgamated.hpp>

#include <atomic>
#include <chrono>
#include <thread>

#include "hlie_app/util/hlie_background_task_group.h"

using namespace hm_HLIE;
using namespace std::chrono_literals;

TEST_CASE("Background task group joins owned callbacks", "[background-task-group]") {
    std::atomic<bool> finished{false};
    BackgroundTaskGroup tasks;

    REQUIRE(tasks.launch([&] {
        std::this_thread::sleep_for(10ms);
        finished.store(true);
    }));

    tasks.stop_and_join();
    CHECK(finished.load());
    CHECK_FALSE(tasks.launch([] {}));
}

TEST_CASE("Background task group destructor waits for callbacks", "[background-task-group]") {
    std::atomic<bool> finished{false};
    {
        BackgroundTaskGroup tasks;
        REQUIRE(tasks.launch([&] { finished.store(true); }));
    }
    CHECK(finished.load());
}

TEST_CASE("Background task group reaps completed callbacks on the next launch", "[background-task-group]") {
    BackgroundTaskGroup tasks;
    std::atomic<bool> first_finished{false};
    std::atomic<bool> release_second{false};

    REQUIRE(tasks.launch([&] { first_finished.store(true); }));
    while (!first_finished.load()) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(10ms);
    REQUIRE(tasks.tracked_task_count() == 1);

    REQUIRE(tasks.launch([&] {
        while (!release_second.load()) {
            std::this_thread::yield();
        }
    }));
    CHECK(tasks.tracked_task_count() == 1);

    release_second.store(true);
    tasks.stop_and_join();
}

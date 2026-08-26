#include <catch_amalgamated.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

#include "hlie_app/util/hlie_model_load_waiter.h"

using namespace hm_HLIE;
using namespace std::chrono_literals;

TEST_CASE("Model load waiter resumes when loading completes", "[model-load-waiter]") {
    std::atomic<server_model_status> status{SERVER_MODEL_STATUS_LOADING};
    std::mutex mutex;
    std::condition_variable state_changed;

    std::thread loader([&] {
        std::this_thread::sleep_for(10ms);
        status.store(SERVER_MODEL_STATUS_LOADED);
        state_changed.notify_all();
    });

    std::unique_lock lock(mutex);
    const auto result = wait_for_model_load(
        state_changed, lock,
        [&] { return std::optional(status.load()); },
        std::chrono::steady_clock::now() + 1s, {}, 5ms);
    lock.unlock();
    loader.join();

    CHECK(result == ModelLoadWaitResult::LOADED);
}

TEST_CASE("Model load waiter reports failed and missing models", "[model-load-waiter]") {
    std::mutex mutex;
    std::condition_variable state_changed;

    SECTION("loading stopped without reaching loaded") {
        std::atomic<server_model_status> status{SERVER_MODEL_STATUS_UNLOADED};
        std::unique_lock lock(mutex);
        CHECK(wait_for_model_load(
                  state_changed, lock,
                  [&] { return std::optional(status.load()); },
                  std::chrono::steady_clock::now() + 1s) == ModelLoadWaitResult::STOPPED);
    }

    SECTION("model disappeared") {
        std::unique_lock lock(mutex);
        CHECK(wait_for_model_load(
                  state_changed, lock,
                  [] { return std::optional<server_model_status>{}; },
                  std::chrono::steady_clock::now() + 1s) == ModelLoadWaitResult::MISSING);
    }
}

TEST_CASE("Model load waiter enforces its deadline", "[model-load-waiter]") {
    std::mutex mutex;
    std::condition_variable state_changed;
    std::unique_lock lock(mutex);

    const auto started = std::chrono::steady_clock::now();
    const auto result = wait_for_model_load(
        state_changed, lock,
        [] { return std::optional(SERVER_MODEL_STATUS_LOADING); },
        std::chrono::steady_clock::now() + 25ms, {}, 5ms);
    const auto elapsed = std::chrono::steady_clock::now() - started;

    CHECK(result == ModelLoadWaitResult::TIMED_OUT);
    CHECK(elapsed >= 20ms);
    CHECK(elapsed < 500ms);
}

TEST_CASE("Model load waiter rechecks state when a notification is missed", "[model-load-waiter]") {
    std::atomic<server_model_status> status{SERVER_MODEL_STATUS_LOADING};
    std::mutex mutex;
    std::condition_variable state_changed;

    // Change the state without notifying. The bounded recheck interval prevents a
    // lost notification from delaying the request until the overall timeout.
    std::thread loader([&] {
        std::this_thread::sleep_for(10ms);
        status.store(SERVER_MODEL_STATUS_LOADED);
    });

    std::unique_lock lock(mutex);
    const auto started = std::chrono::steady_clock::now();
    const auto result = wait_for_model_load(
        state_changed, lock,
        [&] { return std::optional(status.load()); },
        std::chrono::steady_clock::now() + 1s, {}, 20ms);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    lock.unlock();
    loader.join();

    CHECK(result == ModelLoadWaitResult::LOADED);
    CHECK(elapsed < 250ms);
}

TEST_CASE("Model load waiter stops when the request is cancelled", "[model-load-waiter]") {
    std::atomic<bool> cancelled{false};
    std::mutex mutex;
    std::condition_variable state_changed;

    std::thread client([&] {
        std::this_thread::sleep_for(10ms);
        cancelled.store(true);
    });

    std::unique_lock lock(mutex);
    const auto result = wait_for_model_load(
        state_changed, lock,
        [] { return std::optional(SERVER_MODEL_STATUS_LOADING); },
        std::chrono::steady_clock::now() + 1s,
        [&] { return cancelled.load(); },
        20ms);
    lock.unlock();
    client.join();

    CHECK(result == ModelLoadWaitResult::CANCELLED);
}

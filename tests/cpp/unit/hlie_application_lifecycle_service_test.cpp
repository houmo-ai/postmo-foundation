#include <catch_amalgamated.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "hlie_app/hlie_workflow.h"
#include "hlie_app/lifecycle/hlie_application_lifecycle_service.h"

namespace {

using namespace hm_HLIE;

struct LifecycleFixture {
    std::vector<std::string> events;
    HLIEEmbeddedStartupSummary primary{1, 1, 0, false};
    HLIEEmbeddedStartupSummary secondary{1, 1, 0, false};
    HLIEControllerStopSummary runtime_stop{2, 0, 0};
    bool workers_confirmed = true;
    bool residual_services_confirmed = true;
    bool status_publications_confirmed = true;
    bool throw_during_primary_start = false;

    HLIEApplicationLifecycleDependencies dependencies() {
        HLIEApplicationLifecycleDependencies result;
        result.start_embedded_group = [this](HLIEEmbeddedStartupGroup group) {
            if (group == HLIEEmbeddedStartupGroup::Primary) {
                events.push_back("primary");
                if (throw_during_primary_start) {
                    throw std::runtime_error("primary start failed");
                }
                return primary;
            }
            events.push_back("secondary");
            return secondary;
        };
        result.configure_llm_admission = [this](std::size_t limit) {
            events.push_back("llm:" + std::to_string(limit));
        };
        result.configure_qwen3_tts_admission = [this](std::size_t limit) {
            events.push_back("qwen:" + std::to_string(limit));
        };
        result.start_asr_realtime = [this] {
            events.push_back("asr_start");
        };
        result.stop_asr_realtime = [this] {
            events.push_back("asr_stop");
        };
        result.start_reconciler = [this] {
            events.push_back("reconciler_start");
        };
        result.stop_reconciler = [this] {
            events.push_back("reconciler_stop");
        };
        result.start_discovery = [this] {
            events.push_back("discovery_start");
        };
        result.stop_discovery = [this] {
            events.push_back("discovery_stop");
        };
        result.start_worker_supervisor = [this] {
            events.push_back("supervisor_start");
        };
        result.controller_begin_draining = [this] {
            events.push_back("drain");
        };
        result.stop_readiness = [this] {
            events.push_back("readiness_stop");
        };
        result.stop_runtimes = [this] {
            events.push_back("runtimes_stop");
            return runtime_stop;
        };
        result.shutdown_workers = [this](const std::string& reason) {
            events.push_back("workers_stop:" + reason);
            return workers_confirmed;
        };
        result.stop_worker_supervisor = [this] {
            events.push_back("supervisor_stop");
        };
        result.cleanup_residual_services = [this] {
            events.push_back("residual_cleanup");
            return HLIEResidualServiceCleanupResult{
                residual_services_confirmed,
                status_publications_confirmed,
            };
        };
        result.stop_worker_control = [this] {
            events.push_back("worker_control_stop");
        };
        return result;
    }
};

HLIEApplicationLifecycleOptions main_options() {
    HLIEApplicationLifecycleOptions options;
    options.http_threads = 8;
    options.reserved_control_threads = 2;
    options.llm_max_in_flight = 5000;
    options.qwen3_tts_max_in_flight = 9;
    return options;
}

} // namespace

static_assert(std::is_same_v<
              decltype(std::declval<HLIEWorkflow&>().run()), bool>);

TEST_CASE("ApplicationLifecycle starts main-process services in order",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    fixture.secondary = {1, 0, 1, false};
    HLIEApplicationLifecycleService service(fixture.dependencies());

    const auto result = service.start(main_options());

    CHECK(result.status == HLIEApplicationStartStatus::Started);
    CHECK(result.started());
    CHECK(result.effective_llm_max_in_flight == 6);
    CHECK(result.effective_qwen3_tts_max_in_flight == 6);
    CHECK(result.secondary.failed == 1);
    CHECK(service.phase() == HLIEApplicationLifecyclePhase::Running);
    CHECK(fixture.events == std::vector<std::string>{
                                "primary", "llm:6", "qwen:6", "secondary",
                                "asr_start", "reconciler_start",
                                "discovery_start", "supervisor_start"});
}

TEST_CASE("ApplicationLifecycle worker startup skips discovery and supervisor",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    auto options = main_options();
    options.worker_process = true;
    HLIEApplicationLifecycleService service(fixture.dependencies());

    const auto result = service.start(options);

    CHECK(result.started());
    CHECK(fixture.events == std::vector<std::string>{
                                "primary", "llm:6", "qwen:6", "secondary",
                                "asr_start", "reconciler_start"});
}

TEST_CASE("ApplicationLifecycle concurrent start uses the first options once",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    auto dependencies = fixture.dependencies();
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool entered = false;
    bool release = false;
    dependencies.start_embedded_group =
        [&](HLIEEmbeddedStartupGroup group) {
            if (group == HLIEEmbeddedStartupGroup::Secondary) {
                fixture.events.push_back("secondary");
                return fixture.secondary;
            }
            fixture.events.push_back("primary");
            std::unique_lock<std::mutex> lock(gate_mutex);
            entered = true;
            gate_cv.notify_all();
            gate_cv.wait(lock, [&] { return release; });
            return fixture.primary;
        };
    HLIEApplicationLifecycleService service(std::move(dependencies));

    const auto first_options = main_options();
    auto second_options = main_options();
    second_options.http_threads = 100;
    auto first = std::async(std::launch::async, [&] {
        return service.start(first_options);
    });
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate_cv.wait(lock, [&] { return entered; });
    }
    auto second = std::async(std::launch::async, [&] {
        return service.start(second_options);
    });
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release = true;
    }
    gate_cv.notify_all();

    const auto first_result = first.get();
    const auto second_result = second.get();
    CHECK(first_result.status == HLIEApplicationStartStatus::Started);
    CHECK(second_result.status ==
          HLIEApplicationStartStatus::AlreadyStarted);
    CHECK(second_result.effective_llm_max_in_flight == 6);
    CHECK(std::count(fixture.events.begin(), fixture.events.end(), "primary") ==
          1);
}

TEST_CASE("ApplicationLifecycle primary abort short-circuits startup",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    fixture.primary = {1, 0, 1, true};
    HLIEApplicationLifecycleService service(fixture.dependencies());

    const auto result = service.start(main_options());

    CHECK(result.status ==
          HLIEApplicationStartStatus::PrimaryStartupAborted);
    CHECK_FALSE(result.started());
    CHECK(service.phase() == HLIEApplicationLifecyclePhase::StartFailed);
    CHECK(fixture.events == std::vector<std::string>{"primary"});
}

TEST_CASE("ApplicationLifecycle rejects missing callbacks before effects",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    auto dependencies = fixture.dependencies();
    dependencies.configure_llm_admission = {};
    HLIEApplicationLifecycleService service(std::move(dependencies));

    const auto result = service.start(main_options());

    CHECK(result.status == HLIEApplicationStartStatus::Failed);
    CHECK(service.phase() == HLIEApplicationLifecyclePhase::StartFailed);
    CHECK(fixture.events.empty());
}

TEST_CASE("ApplicationLifecycle shuts down in ownership-safe order",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    HLIEApplicationLifecycleService service(fixture.dependencies());
    REQUIRE(service.start(main_options()).started());
    fixture.events.clear();

    const auto result = service.stop();

    CHECK(result.complete());
    CHECK_FALSE(result.already_stopped);
    CHECK(service.draining());
    CHECK(service.phase() == HLIEApplicationLifecyclePhase::Stopped);
    CHECK(fixture.events == std::vector<std::string>{
                                "drain", "discovery_stop", "reconciler_stop",
                                "readiness_stop", "asr_stop", "runtimes_stop",
                                "workers_stop:application_stop",
                                "supervisor_stop", "residual_cleanup",
                                "worker_control_stop"});
}

TEST_CASE("ApplicationLifecycle propagates incomplete ownership cleanup",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    fixture.runtime_stop = {1, 2, 0};
    fixture.workers_confirmed = false;
    fixture.residual_services_confirmed = false;
    HLIEApplicationLifecycleService service(fixture.dependencies());

    const auto result = service.stop();

    CHECK_FALSE(result.complete());
    CHECK(result.runtimes.unconfirmed == 2);
    CHECK_FALSE(result.workers_confirmed);
    CHECK_FALSE(result.residual_services_confirmed);
    CHECK(result.residual_status_publications_confirmed);
    CHECK(result.callbacks_complete);
}

TEST_CASE("ApplicationLifecycle reports residual status publication failure",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    fixture.status_publications_confirmed = false;
    HLIEApplicationLifecycleService service(fixture.dependencies());

    const auto result = service.stop();

    CHECK_FALSE(result.complete());
    CHECK(result.residual_services_confirmed);
    CHECK_FALSE(result.residual_status_publications_confirmed);
    CHECK(result.callbacks_complete);
}

TEST_CASE("ApplicationLifecycle repeated stop returns the cached result",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    HLIEApplicationLifecycleService service(fixture.dependencies());

    const auto first = service.stop();
    const auto first_events = fixture.events;
    const auto second = service.stop();

    CHECK_FALSE(first.already_stopped);
    CHECK(second.already_stopped);
    CHECK(second.runtimes.stopped == first.runtimes.stopped);
    CHECK(second.complete() == first.complete());
    CHECK(fixture.events == first_events);
    CHECK(service.start(main_options()).status ==
          HLIEApplicationStartStatus::Failed);
}

TEST_CASE("ApplicationLifecycle concurrent stop executes callbacks once",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    auto dependencies = fixture.dependencies();
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool entered = false;
    bool release = false;
    int discovery_stop_calls = 0;
    dependencies.stop_discovery = [&] {
        std::unique_lock<std::mutex> lock(gate_mutex);
        ++discovery_stop_calls;
        entered = true;
        gate_cv.notify_all();
        gate_cv.wait(lock, [&] { return release; });
        fixture.events.push_back("discovery_stop");
    };
    HLIEApplicationLifecycleService service(std::move(dependencies));

    auto first = std::async(std::launch::async, [&] { return service.stop(); });
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate_cv.wait(lock, [&] { return entered; });
    }
    auto second =
        std::async(std::launch::async, [&] { return service.stop(); });
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release = true;
    }
    gate_cv.notify_all();

    const auto first_result = first.get();
    const auto second_result = second.get();
    CHECK(discovery_stop_calls == 1);
    CHECK_FALSE(first_result.already_stopped);
    CHECK(second_result.already_stopped);
    CHECK(first_result.complete());
    CHECK(second_result.complete());
}

TEST_CASE("ApplicationLifecycle start exception still permits full cleanup",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    fixture.throw_during_primary_start = true;
    HLIEApplicationLifecycleService service(fixture.dependencies());
    REQUIRE(service.start(main_options()).status ==
            HLIEApplicationStartStatus::Failed);
    REQUIRE(service.phase() == HLIEApplicationLifecyclePhase::StartFailed);
    fixture.events.clear();

    const auto result = service.stop();

    CHECK(result.complete());
    CHECK(fixture.events == std::vector<std::string>{
                                "drain", "discovery_stop", "reconciler_stop",
                                "readiness_stop", "asr_stop", "runtimes_stop",
                                "workers_stop:application_stop",
                                "supervisor_stop", "residual_cleanup",
                                "worker_control_stop"});
}

TEST_CASE("ApplicationLifecycle continues shutdown after callback exception",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    auto dependencies = fixture.dependencies();
    dependencies.stop_discovery = [&] {
        fixture.events.push_back("discovery_stop");
        throw std::runtime_error("discovery stop failed");
    };
    HLIEApplicationLifecycleService service(std::move(dependencies));

    const auto result = service.stop();

    CHECK_FALSE(result.complete());
    CHECK_FALSE(result.callbacks_complete);
    CHECK(fixture.events.front() == "drain");
    CHECK(fixture.events.back() == "worker_control_stop");
    CHECK(std::find(fixture.events.begin(), fixture.events.end(),
                    "runtimes_stop") != fixture.events.end());
}

TEST_CASE("ApplicationLifecycle begin draining is concurrent and once-only",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    auto dependencies = fixture.dependencies();
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool entered = false;
    bool release = false;
    int drain_calls = 0;
    dependencies.controller_begin_draining = [&] {
        std::unique_lock<std::mutex> lock(gate_mutex);
        ++drain_calls;
        entered = true;
        gate_cv.notify_all();
        gate_cv.wait(lock, [&] { return release; });
    };
    HLIEApplicationLifecycleService service(std::move(dependencies));

    auto first = std::async(std::launch::async, [&] {
        service.begin_draining();
    });
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate_cv.wait(lock, [&] { return entered; });
    }
    auto second = std::async(std::launch::async, [&] {
        service.begin_draining();
    });
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release = true;
    }
    gate_cv.notify_all();
    first.get();
    second.get();

    CHECK(drain_calls == 1);
    CHECK(service.draining());
    CHECK(service.phase() == HLIEApplicationLifecyclePhase::Draining);
}

TEST_CASE("ApplicationLifecycle publishes serving state exactly once",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    HLIEApplicationLifecycleService service(fixture.dependencies());
    REQUIRE(service.start(main_options()).started());

    int publisher_calls = 0;
    const auto first = service.try_publish_serving([&] {
        ++publisher_calls;
        return true;
    });
    const auto second = service.try_publish_serving([&] {
        ++publisher_calls;
        return true;
    });

    CHECK(first.status == HLIEApplicationServingStatus::Published);
    CHECK(second.status == HLIEApplicationServingStatus::AlreadyPublished);
    CHECK(publisher_calls == 1);
    CHECK(service.phase() == HLIEApplicationLifecyclePhase::Serving);
}

TEST_CASE("ApplicationLifecycle rejects serving publication after stop",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    HLIEApplicationLifecycleService service(fixture.dependencies());
    REQUIRE(service.start(main_options()).started());
    REQUIRE(service.stop().complete());

    bool publisher_called = false;
    const auto result = service.try_publish_serving([&] {
        publisher_called = true;
        return true;
    });

    CHECK(result.status == HLIEApplicationServingStatus::NotRunning);
    CHECK_FALSE(publisher_called);
}

TEST_CASE("ApplicationLifecycle serializes serving publication and stop",
          "[p1][application-lifecycle]") {
    LifecycleFixture fixture;
    HLIEApplicationLifecycleService service(fixture.dependencies());
    REQUIRE(service.start(main_options()).started());

    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool publisher_entered = false;
    bool release_publisher = false;
    std::atomic<bool> stop_called{false};

    auto publishing = std::async(std::launch::async, [&] {
        return service.try_publish_serving([&] {
            std::unique_lock<std::mutex> lock(gate_mutex);
            publisher_entered = true;
            gate_cv.notify_all();
            gate_cv.wait(lock, [&] { return release_publisher; });
            return true;
        });
    });
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate_cv.wait(lock, [&] { return publisher_entered; });
    }

    auto stopping = std::async(std::launch::async, [&] {
        stop_called.store(true, std::memory_order_release);
        return service.stop();
    });
    while (!stop_called.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release_publisher = true;
    }
    gate_cv.notify_all();

    const auto publish_result = publishing.get();
    const auto stop_result = stopping.get();
    CHECK(publish_result.status == HLIEApplicationServingStatus::Published);
    CHECK(stop_result.complete());
    CHECK(service.phase() == HLIEApplicationLifecyclePhase::Stopped);
}

#include <catch_amalgamated.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>

#include "hlie_app/hlie_model_registry.h"
#include "hlie_core/resource/hlie_device_reservation_ledger.h"
#include "hlie_core/runtime/hlie_service_slot.h"

namespace {

using namespace hm_HLIE;

class LoggerLevelGuard {
public:
    explicit LoggerLevelGuard(const spdlog::level::level_enum level)
        : logger_(spdlog::default_logger()), previous_(logger_->level()) {
        logger_->set_level(level);
    }

    ~LoggerLevelGuard() {
        logger_->set_level(previous_);
    }

private:
    std::shared_ptr<spdlog::logger> logger_;
    spdlog::level::level_enum previous_;
};

struct StressServiceCounters {
    std::atomic<std::uint64_t> created{0};
    std::atomic<std::uint64_t> destroyed{0};
};

struct StressService {
    StressService(const std::uint64_t generation,
                  const std::shared_ptr<StressServiceCounters>& counters)
        : generation(generation), counters(std::move(counters)) {
        this->counters->created.fetch_add(1);
    }

    ~StressService() {
        counters->destroyed.fetch_add(1);
    }

    std::uint64_t generation = 0;
    std::shared_ptr<StressServiceCounters> counters;
};

HLIEModelRegistry::ModelInstance make_stress_instance() {
    HLIEModelInfo info;
    info.build_sd("sd", "vae", "t5", "clip", "clip-l", "mmdit", "0");
    HLIEModelRegistry::ModelInstance instance;
    instance.key = "p0-stress-model";
    instance.meta.name = instance.key;
    instance.meta.status = SERVER_MODEL_STATUS_UNLOADED;
    instance.model_info = std::move(info);
    instance.estimated_mem_mb = 2048;
    return instance;
}

bool begin_eviction_after_drain(HLIEModelRegistry& registry) {
    for (int attempt = 0; attempt < 100000; ++attempt) {
        if (registry.try_begin_eviction("p0-stress-model")) {
            return true;
        }
        std::this_thread::yield();
    }
    return false;
}

} // namespace

TEST_CASE("P0 control plane survives 1000 request unload reload interleavings",
          "[p0][stress][control-plane]") {
    LoggerLevelGuard quiet_logs(spdlog::level::err);
    HLIEModelRegistry registry;
    registry.upsert(make_stress_instance());
    HLIEServiceSlot<StressService> slot;
    const auto counters = std::make_shared<StressServiceCounters>();

    auto registry_ticket = registry.try_begin_loading_ticket("p0-stress-model");
    auto slot_ticket = slot.begin_load();
    REQUIRE(registry_ticket);
    REQUIRE(slot_ticket);
    REQUIRE(slot.publish(*slot_ticket, std::make_shared<StressService>(0, counters)));
    REQUIRE(registry.mark_loaded_if_current(*registry_ticket, 7901));

    std::atomic<bool> stop{false};
    std::atomic<bool> draining{false};
    std::atomic<std::uint64_t> accepted_requests{0};
    std::atomic<std::uint64_t> invariant_failures{0};
    std::vector<std::thread> request_threads;
    for (int thread_index = 0; thread_index < 8; ++thread_index) {
        request_threads.emplace_back([&] {
            while (!stop.load(std::memory_order_acquire)) {
                if (draining.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                    continue;
                }
                const auto request_ticket = registry.begin_request_ticket("p0-stress-model");
                if (!request_ticket) {
                    std::this_thread::yield();
                    continue;
                }

                const auto service = slot.snapshot();
                if (!service || service->generation > 1000) {
                    invariant_failures.fetch_add(1);
                } else {
                    accepted_requests.fetch_add(1);
                }
                std::this_thread::yield();
                registry.end_request(*request_ticket);
            }
        });
    }

    // A fixed yield count can expire before Windows schedules newly created
    // request threads. Bound startup by wall time so the stress assertion
    // measures the control plane instead of scheduler latency.
    const auto request_start_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (accepted_requests.load() < 100 &&
           std::chrono::steady_clock::now() < request_start_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(accepted_requests.load() >= 100);

    for (std::uint64_t cycle = 1; cycle <= 1000; ++cycle) {
        draining.store(true, std::memory_order_release);
        REQUIRE(begin_eviction_after_drain(registry));

        auto detached = slot.detach();
        REQUIRE(detached);
        REQUIRE(registry.mark_unloaded("p0-stress-model"));
        detached.reset();

        if (cycle % 10 == 0) {
            const auto stale_slot_ticket = slot.begin_load();
            const auto stale_registry_ticket = registry.try_begin_loading_ticket("p0-stress-model");
            REQUIRE(stale_slot_ticket);
            REQUIRE(stale_registry_ticket);

            CHECK_FALSE(slot.detach());
            REQUIRE(registry.mark_unloaded_if_current(*stale_registry_ticket));
            CHECK_FALSE(slot.publish(
                *stale_slot_ticket, std::make_shared<StressService>(cycle, counters)));
            CHECK_FALSE(registry.mark_loaded_if_current(*stale_registry_ticket, 7902));
        }

        slot_ticket = slot.begin_load();
        registry_ticket = registry.try_begin_loading_ticket("p0-stress-model");
        REQUIRE(slot_ticket);
        REQUIRE(registry_ticket);
        REQUIRE(slot.publish(
            *slot_ticket, std::make_shared<StressService>(cycle, counters)));
        REQUIRE(registry.mark_loaded_if_current(*registry_ticket, 7901));
        draining.store(false, std::memory_order_release);
        std::this_thread::yield();
    }

    stop.store(true, std::memory_order_release);
    for (auto& thread : request_threads) {
        thread.join();
    }

    draining.store(true, std::memory_order_release);
    REQUIRE(begin_eviction_after_drain(registry));
    auto final_service = slot.detach();
    REQUIRE(final_service);
    REQUIRE(registry.mark_unloaded("p0-stress-model"));
    final_service.reset();

    const auto final_snapshot = registry.get_instance_snapshot("p0-stress-model");
    REQUIRE(final_snapshot);
    CHECK(final_snapshot->meta.status == SERVER_MODEL_STATUS_UNLOADED);
    CHECK(final_snapshot->active_requests == 0);
    CHECK_FALSE(slot.snapshot());
    CHECK_FALSE(slot.is_loading());
    CHECK(invariant_failures.load() == 0);
    CHECK(accepted_requests.load() >= 100);
    CHECK(counters->created.load() == counters->destroyed.load());
}

TEST_CASE("P0 device ledger survives 4000 concurrent reservation transactions",
          "[p0][stress][resource]") {
    HLIEDeviceReservationLedger ledger(128);
    ledger.update_observation({0, 8192, 0, 8192});
    ledger.update_observation({1, 8192, 0, 8192});

    constexpr int thread_count = 16;
    constexpr int transactions_per_thread = 250;
    std::atomic<int> ready_threads{0};
    std::atomic<bool> start{false};
    std::atomic<bool> monitor_stop{false};
    std::atomic<std::uint64_t> next_generation{1};
    std::atomic<std::uint64_t> successful_transactions{0};
    std::atomic<std::uint64_t> invariant_failures{0};

    std::thread monitor([&] {
        while (!monitor_stop.load(std::memory_order_acquire)) {
            for (const auto& snapshot : ledger.snapshots()) {
                if (snapshot.effective_used_mb > snapshot.total_mb ||
                    snapshot.pending_reserved_mb > snapshot.total_mb ||
                    snapshot.committed_reserved_mb > snapshot.total_mb) {
                    invariant_failures.fetch_add(1);
                }
            }
            std::this_thread::yield();
        }
    });

    std::vector<std::thread> workers;
    for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
        workers.emplace_back([&, thread_index] {
            ready_threads.fetch_add(1);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            for (int transaction = 0; transaction < transactions_per_thread; ++transaction) {
                const auto generation = next_generation.fetch_add(1);
                HLIEDeviceReservationRequest request;
                request.model_key = "stress-" + std::to_string(thread_index) + "-" +
                                    std::to_string(transaction);
                request.generation = generation;
                request.total_mb = transaction % 4 == 0 ? 1024 : 512;
                request.device_count = transaction % 4 == 0 ? 2 : 1;

                const auto reservation = ledger.reserve(request);
                if (!reservation) {
                    std::this_thread::yield();
                    continue;
                }
                if (!ledger.bind(reservation->id, request.model_key, generation)) {
                    invariant_failures.fetch_add(1);
                    (void)ledger.release(reservation->id);
                    continue;
                }

                if (transaction % 3 == 0) {
                    if (!ledger.commit(reservation->id) || !ledger.release(reservation->id)) {
                        invariant_failures.fetch_add(1);
                    }
                } else if (!ledger.rollback(reservation->id)) {
                    invariant_failures.fetch_add(1);
                }
                successful_transactions.fetch_add(1);
            }
        });
    }

    while (ready_threads.load() != thread_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) {
        worker.join();
    }
    monitor_stop.store(true, std::memory_order_release);
    monitor.join();

    ledger.update_observation({0, 8192, 0, 8192});
    ledger.update_observation({1, 8192, 0, 8192});
    const auto snapshots = ledger.snapshots();
    REQUIRE(snapshots.size() == 2);
    CHECK(ledger.reservations().empty());
    CHECK(invariant_failures.load() == 0);
    CHECK(successful_transactions.load() > 0);
    for (const auto& snapshot : snapshots) {
        CHECK(snapshot.pending_reserved_mb == 0);
        CHECK(snapshot.committed_reserved_mb == 0);
        CHECK(snapshot.committed_unobserved_mb == 0);
        CHECK(snapshot.effective_used_mb == 0);
    }
}

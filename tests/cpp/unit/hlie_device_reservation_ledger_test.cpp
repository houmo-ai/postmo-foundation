#include <catch_amalgamated.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <unordered_set>
#include <vector>

#include "hlie_core/resource/hlie_device_reservation_ledger.h"

using namespace hm_HLIE;

TEST_CASE("device ledger prevents concurrent memory oversubscription") {
    HLIEDeviceReservationLedger ledger(200);
    ledger.update_observation({0, 16000, 0, 16000});

    std::atomic<int> winners{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&, i] {
            HLIEDeviceReservationRequest request;
            request.model_key = "model-" + std::to_string(i);
            request.total_mb = 9000;
            if (ledger.reserve(request)) {
                ++winners;
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    CHECK(winners.load() == 1);
    const auto snapshots = ledger.snapshots();
    REQUIRE(snapshots.size() == 1);
    CHECK(snapshots.front().pending_reserved_mb == 9000);
}

TEST_CASE("device reservation keeps unobserved ownership across aggregate HAL growth") {
    HLIEDeviceReservationLedger ledger(200);
    ledger.update_observation({0, 16000, 1000, 15000});

    HLIEDeviceReservationRequest request;
    request.model_key = "model";
    request.total_mb = 6000;
    const auto reservation = ledger.reserve(request);
    REQUIRE(reservation);
    REQUIRE(ledger.commit(reservation->id));
    REQUIRE(ledger.commit(reservation->id));

    auto snapshot = ledger.snapshots().front();
    CHECK(snapshot.observed_used_mb == 1000);
    CHECK(snapshot.committed_reserved_mb == 6000);
    CHECK(snapshot.committed_unobserved_mb == 6000);
    CHECK(snapshot.effective_used_mb == 7000);

    ledger.update_observation({0, 16000, 7000, 9000});
    snapshot = ledger.snapshots().front();
    CHECK(snapshot.committed_reserved_mb == 6000);
    CHECK(snapshot.committed_unobserved_mb == 6000);
    CHECK(snapshot.effective_used_mb == 13000);

    REQUIRE(ledger.release(reservation->id));
    snapshot = ledger.snapshots().front();
    CHECK(snapshot.committed_reserved_mb == 0);
}

TEST_CASE("device reservation covers a smaller resolved runtime estimate") {
    HLIEDeviceReservationLedger ledger(200);
    ledger.update_observation({0, 16000, 1000, 15000});

    HLIEDeviceReservationRequest request;
    request.model_key = "model-a";
    request.requested_devices = {0};
    request.total_mb = 6000;
    const auto reservation = ledger.reserve(request);
    REQUIRE(reservation);

    CHECK(ledger.covers(reservation->id, {0}, 5000, "model-a"));
    CHECK(ledger.covers(reservation->id, {0, 0}, 6000, "model-a"));
    CHECK_FALSE(ledger.covers(reservation->id, {0}, 6001, "model-a"));
    CHECK_FALSE(ledger.covers(reservation->id, {1}, 5000, "model-a"));
    CHECK_FALSE(ledger.covers(reservation->id, {0}, 5000, "model-b"));
}

TEST_CASE("rollback and release are reservation-id scoped and idempotent") {
    HLIEDeviceReservationLedger ledger(0);
    ledger.update_observation({0, 16000, 0, 16000});

    HLIEDeviceReservationRequest first_request;
    first_request.model_key = "first";
    first_request.total_mb = 4000;
    const auto first = ledger.reserve(first_request);
    REQUIRE(first);

    HLIEDeviceReservationRequest second_request;
    second_request.model_key = "second";
    second_request.total_mb = 4000;
    const auto second = ledger.reserve(second_request);
    REQUIRE(second);

    CHECK(ledger.rollback(first->id));
    CHECK_FALSE(ledger.rollback(first->id));
    REQUIRE(ledger.commit(second->id));
    CHECK(ledger.release(second->id));
    CHECK_FALSE(ledger.release(second->id));
    CHECK(ledger.reservations().empty());
}

TEST_CASE("reservation generation binding is pending-only and cannot be retargeted") {
    HLIEDeviceReservationLedger ledger(0);
    ledger.update_observation({0, 16000, 0, 16000});

    HLIEDeviceReservationRequest request;
    request.model_key = "model-a";
    request.total_mb = 1024;
    const auto reservation = ledger.reserve(request);
    REQUIRE(reservation);

    REQUIRE(ledger.bind(reservation->id, "model-a", 7));
    REQUIRE(ledger.bind(reservation->id, "model-a", 7));
    CHECK_FALSE(ledger.bind(reservation->id, "model-a", 8));
    CHECK_FALSE(ledger.bind(reservation->id, "model-b", 7));

    const auto bound = ledger.get(reservation->id);
    REQUIRE(bound);
    CHECK(bound->model_key == "model-a");
    CHECK(bound->generation == 7);
    CHECK(bound->expires_at >= reservation->expires_at);

    REQUIRE(ledger.commit(reservation->id));
    CHECK_FALSE(ledger.bind(reservation->id, "model-a", 7));
    REQUIRE(ledger.release(reservation->id));
}

TEST_CASE("multi-device reservation is all-or-nothing") {
    HLIEDeviceReservationLedger ledger(200);
    ledger.update_observation({0, 8000, 0, 8000});
    ledger.update_observation({1, 8000, 7000, 1000});

    HLIEDeviceReservationRequest request;
    request.model_key = "sharded";
    request.total_mb = 8000;
    request.device_count = 2;
    request.requested_devices = {0, 1};
    CHECK_FALSE(ledger.reserve(request));

    const auto snapshots = ledger.snapshots();
    REQUIRE(snapshots.size() == 2);
    CHECK(snapshots[0].pending_reserved_mb == 0);
    CHECK(snapshots[1].pending_reserved_mb == 0);
}

TEST_CASE("expired pending reservations roll back unless an active owner protects them") {
    HLIEDeviceReservationLedger ledger(0);
    ledger.set_pending_ttl(std::chrono::seconds(30));
    ledger.update_observation({0, 16000, 0, 16000});

    HLIEDeviceReservationRequest request;
    request.model_key = "loading-model";
    request.total_mb = 4096;
    const auto reservation = ledger.reserve(request);
    REQUIRE(reservation);

    CHECK(ledger.rollback_expired_pending(
        {}, reservation->expires_at - std::chrono::milliseconds(1)).empty());

    const std::unordered_set<HLIEDeviceReservationId> protected_ids{reservation->id};
    CHECK(ledger.rollback_expired_pending(
        protected_ids, reservation->expires_at + std::chrono::milliseconds(1)).empty());
    REQUIRE(ledger.get(reservation->id));

    const auto expired = ledger.rollback_expired_pending(
        {}, reservation->expires_at + std::chrono::milliseconds(1));
    REQUIRE(expired.size() == 1);
    CHECK(expired.front().id == reservation->id);
    CHECK_FALSE(ledger.get(reservation->id));
    CHECK(ledger.snapshots().front().pending_reserved_mb == 0);
}

TEST_CASE("committed reservations are never removed by pending TTL reconciliation") {
    HLIEDeviceReservationLedger ledger(0);
    ledger.set_pending_ttl(std::chrono::seconds(30));
    ledger.update_observation({0, 16000, 0, 16000});

    HLIEDeviceReservationRequest request;
    request.model_key = "loaded-model";
    request.total_mb = 4096;
    const auto reservation = ledger.reserve(request);
    REQUIRE(reservation);
    REQUIRE(ledger.commit(reservation->id));

    CHECK(ledger.rollback_expired_pending(
        {}, reservation->expires_at + std::chrono::hours(1)).empty());
    const auto committed = ledger.get(reservation->id);
    REQUIRE(committed);
    CHECK(committed->state == HLIEDeviceReservationState::Committed);
    CHECK(ledger.snapshots().front().committed_reserved_mb == 4096);
}

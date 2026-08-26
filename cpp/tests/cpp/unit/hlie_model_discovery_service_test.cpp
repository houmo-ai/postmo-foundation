#include <catch_amalgamated.hpp>

#include <atomic>
#include <chrono>
#include <optional>
#include <string>
#include <thread>

#include "hlie_app/discovery/hlie_model_discovery_service.h"

using namespace hm_HLIE;
using namespace std::chrono_literals;

TEST_CASE("ModelDiscoveryService parses only a changed stable snapshot",
          "[p1][model-discovery-service]") {
    HLIEModelDiscoveryOptions options;
    options.scan_interval = 0ms;
    options.stability_window = 30ms;
    std::string snapshot = "snapshot-a";
    int prepare_calls = 0;
    int publish_calls = 0;

    HLIEModelDiscoveryService service(
        "models", [&] { return options; },
        [&]() -> std::optional<HLIEModelDiscoveryService::PreparedCatalogUpdate> {
            ++prepare_calls;
            return [&] {
                ++publish_calls;
                return 2;
            };
        },
        [&]() -> std::optional<std::string> { return snapshot; });

    REQUIRE(service.initialize_baseline());
    auto result = service.scan_once();
    CHECK(result.successful);
    CHECK(result.stable);
    CHECK_FALSE(result.parsed);

    snapshot = "snapshot-b";
    result = service.scan_once();
    CHECK(result.successful);
    CHECK_FALSE(result.stable);
    CHECK(prepare_calls == 0);

    std::this_thread::sleep_for(40ms);
    result = service.scan_once();
    CHECK(result.successful);
    CHECK(result.stable);
    CHECK(result.parsed);
    CHECK(result.published_models == 2);
    CHECK(prepare_calls == 1);
    CHECK(publish_calls == 1);

    result = service.scan_once();
    CHECK(result.successful);
    CHECK_FALSE(result.parsed);
    CHECK(prepare_calls == 1);
    CHECK(service.metrics().parse_attempts == 1);
}

TEST_CASE("ModelDiscoveryService verifies the snapshot before publication",
          "[p1][model-discovery-service][two-phase]") {
    HLIEModelDiscoveryOptions options;
    options.scan_interval = 0ms;
    options.stability_window = 0ms;
    std::string snapshot = "snapshot-a";
    int prepare_calls = 0;
    int publish_calls = 0;

    HLIEModelDiscoveryService service(
        "models", [&] { return options; },
        [&]() -> std::optional<HLIEModelDiscoveryService::PreparedCatalogUpdate> {
            ++prepare_calls;
            if (prepare_calls == 1) {
                snapshot = "snapshot-c";
            }
            return [&] {
                ++publish_calls;
                return 1;
            };
        },
        [&]() -> std::optional<std::string> { return snapshot; });

    REQUIRE(service.initialize_baseline());
    snapshot = "snapshot-b";
    auto result = service.scan_once();
    CHECK(result.successful);
    CHECK_FALSE(result.stable);
    CHECK(result.parsed);
    CHECK(publish_calls == 0);

    result = service.scan_once();
    CHECK(result.successful);
    CHECK(result.stable);
    CHECK(result.published_models == 1);
    CHECK(prepare_calls == 2);
    CHECK(publish_calls == 1);
}

TEST_CASE("ModelDiscoveryService completes requested scan generations",
          "[p1][model-discovery-service][lifecycle]") {
    HLIEModelDiscoveryOptions options;
    options.scan_interval = 0ms;
    options.stability_window = 0ms;
    options.scan_on_request = true;
    options.request_cooldown = 0ms;
    options.request_wait_timeout = 1s;
    std::string snapshot = "snapshot-a";
    std::atomic<int> publications{0};

    HLIEModelDiscoveryService service(
        "models", [&] { return options; },
        [&]() -> std::optional<HLIEModelDiscoveryService::PreparedCatalogUpdate> {
            return [&] {
                publications.fetch_add(1, std::memory_order_relaxed);
                return 1;
            };
        },
        [&]() -> std::optional<std::string> { return snapshot; });

    REQUIRE(service.initialize_baseline());
    service.start();
    REQUIRE(service.metrics().running);
    snapshot = "snapshot-b";
    const auto generation = service.request_scan();
    REQUIRE(generation > 0);
    CHECK(service.wait_for(generation));
    CHECK(publications.load(std::memory_order_relaxed) == 1);
    CHECK(service.metrics().completed_generation >= generation);

    const auto started = std::chrono::steady_clock::now();
    service.stop();
    CHECK(std::chrono::steady_clock::now() - started < 1s);
    CHECK_FALSE(service.metrics().running);
    CHECK(service.request_scan(true) == 0);
}

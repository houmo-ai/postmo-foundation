#include <catch_amalgamated.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "hlie_app/models/hlie_stop_service_application_service.h"
#include "json.hpp"

namespace {

using namespace hm_HLIE;

server_http_req request_with_body(std::string body) {
    static const std::function<bool()> should_stop = [] { return false; };
    return {{}, {}, "/hlie/v1/stop", {}, std::move(body), {}, should_stop};
}

nlohmann::json response_body(const server_http_res_ptr& response) {
    REQUIRE(response);
    return nlohmann::json::parse(response->data);
}

struct StopFixture {
    bool replicas_mode = false;
    std::vector<std::string> local_models;
    std::string busy_model;
    std::string failed_eviction;
    std::vector<std::string> claims;
    std::vector<std::string> aborts;
    std::vector<std::string> evictions;

    std::optional<HLIEStopServiceWorkerSnapshot> worker;
    bool ticket_stop_succeeds = true;
    bool fallback_stop_succeeds = true;
    int worker_lookup_count = 0;
    int ticket_stop_count = 0;
    int fallback_stop_count = 0;
    std::vector<HLIEModelLaunchTicket> confirmed_tickets;
    std::vector<std::string> legacy_stops;

    HLIEStopServiceApplicationService service() {
        return HLIEStopServiceApplicationService({
            [this] { return replicas_mode; },
            [this] { return local_models; },
            [this](const std::string& model_key) {
                claims.push_back(model_key);
                return model_key != busy_model;
            },
            [this](const std::string& model_key) {
                aborts.push_back(model_key);
            },
            [this](const std::string& model_key) {
                evictions.push_back(model_key);
                return model_key != failed_eviction;
            },
            [this](const std::uint64_t process_id)
                -> std::optional<HLIEStopServiceWorkerSnapshot> {
                ++worker_lookup_count;
                if (worker && worker->process_id == process_id) {
                    return worker;
                }
                return std::nullopt;
            },
            [this](const HLIEModelLaunchTicket&) {
                ++ticket_stop_count;
                return ticket_stop_succeeds;
            },
            [this](const HLIEStopServiceWorkerSnapshot&) {
                ++fallback_stop_count;
                return fallback_stop_succeeds;
            },
            [this](const HLIEModelLaunchTicket& ticket) {
                confirmed_tickets.push_back(ticket);
            },
            [](const std::string& requested) -> std::optional<std::string> {
                if (requested == "legacy-a" || requested == "legacy-a-alias") {
                    return "legacy-key-a";
                }
                if (requested == "legacy-b") {
                    return "legacy-key-b";
                }
                return std::nullopt;
            },
            [this](const std::string& model_key) {
                legacy_stops.push_back(model_key);
            },
        });
    }
};

HLIEStopServiceWorkerSnapshot ticketed_worker() {
    return {
        77,
        2,
        7903,
        {"worker-model"},
        HLIEModelLaunchTicket{"worker-model", 9, "secret-token"},
    };
}

HLIEStopServiceWorkerSnapshot legacy_worker() {
    return {
        88,
        3,
        7904,
        {"legacy-a", "legacy-a-alias", "missing", "legacy-b"},
        {},
    };
}

} // namespace

TEST_CASE("Stop service application evicts a unique local model batch",
          "[p1][stop-service][application-service]") {
    StopFixture fixture;
    fixture.local_models = {"local-a", "local-b", "local-a", ""};
    auto service = fixture.service();

    const auto response = service.handle(
        request_with_body(R"({"instance_id":0,"process_id":0})"));
    const auto body = response_body(response);

    CHECK(response->status == 200);
    CHECK(body["code"] == 0);
    CHECK(body["instance_id"] == 0);
    CHECK(body["process_id"] == 0);
    CHECK(fixture.claims ==
          std::vector<std::string>{"local-a", "local-b"});
    CHECK(fixture.evictions ==
          std::vector<std::string>{"local-a", "local-b"});
    CHECK(fixture.aborts.empty());
}

TEST_CASE("Stop service application rolls back unexecuted local claims",
          "[p1][stop-service][application-service]") {
    SECTION("busy model aborts every successful claim") {
        StopFixture fixture;
        fixture.local_models = {"local-a", "local-b", "local-c"};
        fixture.busy_model = "local-b";
        auto service = fixture.service();

        const auto response = service.handle(
            request_with_body(R"({"instance_id":0,"process_id":17})"));
        const auto body = response_body(response);
        CHECK(response->status == 503);
        CHECK(body["error"]["ret"] == 80004);
        CHECK(fixture.evictions.empty());
        CHECK(fixture.aborts ==
              std::vector<std::string>{"local-a", "local-c"});
    }

    SECTION("failed eviction keeps current ownership and aborts only the tail") {
        StopFixture fixture;
        fixture.local_models = {"local-a", "local-b", "local-c"};
        fixture.failed_eviction = "local-b";
        auto service = fixture.service();

        const auto response = service.handle(
            request_with_body(R"({"instance_id":0,"process_id":0})"));
        const auto body = response_body(response);
        CHECK(response->status == 500);
        CHECK(body["error"]["ret"] == 80001);
        CHECK(fixture.evictions ==
              std::vector<std::string>{"local-a", "local-b"});
        CHECK(fixture.aborts == std::vector<std::string>{"local-c"});
    }
}

TEST_CASE("Stop service application uses ticketed Worker stop then fallback",
          "[p1][stop-service][application-service]") {
    SECTION("Controller-owned runtime confirms stop") {
        StopFixture fixture;
        fixture.worker = ticketed_worker();
        auto service = fixture.service();

        const auto response = service.handle(
            request_with_body(R"({"instance_id":2,"process_id":77})"));
        CHECK(response->status == 200);
        CHECK(fixture.ticket_stop_count == 1);
        CHECK(fixture.fallback_stop_count == 0);
        REQUIRE(fixture.confirmed_tickets.size() == 1);
        CHECK(fixture.confirmed_tickets.front().matches(
            fixture.worker->launch_ticket));
        CHECK(fixture.legacy_stops.empty());
    }

    SECTION("Supervisor compatibility fallback confirms stop") {
        StopFixture fixture;
        fixture.worker = ticketed_worker();
        fixture.ticket_stop_succeeds = false;
        auto service = fixture.service();

        const auto response = service.handle(
            request_with_body(R"({"instance_id":2,"process_id":77})"));
        CHECK(response->status == 200);
        CHECK(fixture.ticket_stop_count == 1);
        CHECK(fixture.fallback_stop_count == 1);
        CHECK(fixture.confirmed_tickets.size() == 1);
    }
}

TEST_CASE("Stop service application reconciles legacy Worker snapshot",
          "[p1][stop-service][application-service]") {
    StopFixture fixture;
    fixture.worker = legacy_worker();
    auto service = fixture.service();

    const auto response = service.handle(
        request_with_body(R"({"instance_id":3,"process_id":88})"));

    CHECK(response->status == 200);
    CHECK(fixture.ticket_stop_count == 0);
    CHECK(fixture.fallback_stop_count == 1);
    CHECK(fixture.confirmed_tickets.empty());
    CHECK(fixture.legacy_stops ==
          std::vector<std::string>{"legacy-key-a", "legacy-key-b"});
}

TEST_CASE("Stop service application preserves Worker failure and idempotency DTOs",
          "[p1][stop-service][application-service]") {
    SECTION("unconfirmed Worker stop preserves legacy error shape") {
        StopFixture fixture;
        fixture.worker = ticketed_worker();
        fixture.ticket_stop_succeeds = false;
        fixture.fallback_stop_succeeds = false;
        auto service = fixture.service();

        const auto response = service.handle(
            request_with_body(R"({"instance_id":2,"process_id":77})"));
        const auto body = response_body(response);
        CHECK(response->status == 500);
        REQUIRE(body["error"].is_array());
        CHECK(body["error"].at(0) == nlohmann::json::array({"ret", 80001}));
        CHECK(body["error"].at(1) ==
              "Failed to stop process for instance_id");
        CHECK(fixture.confirmed_tickets.empty());
    }

    SECTION("missing Worker is an idempotent success") {
        StopFixture fixture;
        auto service = fixture.service();
        const auto response = service.handle(
            request_with_body(R"({"instance_id":9,"process_id":99})"));
        CHECK(response->status == 200);
        CHECK(response_body(response)["process_id"] == 99);
        CHECK(fixture.worker_lookup_count == 1);
        CHECK(fixture.ticket_stop_count == 0);
        CHECK(fixture.fallback_stop_count == 0);
    }

    SECTION("zero process id is a compatibility no-op") {
        StopFixture fixture;
        auto service = fixture.service();
        const auto response = service.handle(
            request_with_body(R"({"instance_id":9,"process_id":0})"));
        CHECK(response->status == 200);
        CHECK(fixture.worker_lookup_count == 0);
    }
}

TEST_CASE("Stop service application rejects replica mode before parsing",
          "[p1][stop-service][application-service]") {
    StopFixture fixture;
    fixture.replicas_mode = true;
    auto service = fixture.service();

    const auto response = service.handle(request_with_body("not-json"));
    const auto body = response_body(response);
    CHECK(response->status == 501);
    CHECK(body["error"]["ret"] == 80005);
    CHECK(fixture.claims.empty());
    CHECK(fixture.worker_lookup_count == 0);
}

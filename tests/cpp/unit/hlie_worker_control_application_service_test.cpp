#include <catch_amalgamated.hpp>

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "hlie_app/control/hlie_worker_control_application_service.h"
#include "json.hpp"

namespace {

using namespace hm_HLIE;

class ApplicationPublisher final : public IHLIEWorkerStatusPublisher {
public:
    bool publish(const HLIEWorkerStatusPublication&) override { return true; }
};

server_http_req request_with_body(std::string body) {
    static const std::function<bool()> should_stop = [] { return false; };
    return {{}, {}, "/", {}, std::move(body), {}, should_stop};
}

nlohmann::json response_body(const server_http_res_ptr& response) {
    REQUIRE(response);
    return nlohmann::json::parse(response->data);
}

struct ApplicationFixture {
    ApplicationPublisher publisher;
    HLIEModelLaunchTicket ticket{"worker-model", 9, "secret-token"};
    HLIEWorkerControlDependencies dependencies;
    std::atomic<int> apply_count{0};
    std::atomic<int> shutdown_count{0};
    bool dynamic_worker = true;

    ApplicationFixture() {
        dependencies.dynamic_worker_provider = [this] {
            return dynamic_worker;
        };
        dependencies.replicas_mode_provider = [] { return false; };
        dependencies.local_port_provider = [] { return 8901; };
        dependencies.master_port_provider = [] { return 7901; };
        dependencies.launch_ticket_provider = [this] { return ticket; };
        dependencies.local_start_time_provider = [] { return std::time_t{123}; };
        dependencies.local_models_provider = [] {
            return std::vector<std::string>{"local-model"};
        };
        dependencies.worker_processes_provider = [] {
            return std::vector<HLIEWorkerProcessorSnapshot>{
                {"--model worker", 2, 7903, "ignored-uuid", 77, 9,
                 std::time_t{456}, {"worker-model"}}};
        };
        dependencies.resolve_model_key = [](const std::string& requested)
            -> std::optional<std::string> {
            return requested == "alias"
                ? std::optional<std::string>{"worker-model"}
                : std::nullopt;
        };
        dependencies.apply_status = [this](
            const HLIEModelLaunchTicket& actual,
            const HLIEWorkerModelStatus, const int) {
            ++apply_count;
            return actual.matches(ticket);
        };
        dependencies.begin_draining = [] {};
        dependencies.process_shutdown_available = [] { return true; };
        dependencies.process_shutdown = [this] { ++shutdown_count; };
    }
};

} // namespace

TEST_CASE("Worker control application maps the legacy processor DTO",
          "[p1][worker-control][application-service]") {
    ApplicationFixture fixture;
    HLIEWorkerControlService control(fixture.publisher,
                                     std::move(fixture.dependencies));
    HLIEWorkerControlApplicationService application(
        control, {[&fixture] { return fixture.ticket; }});

    const auto response = application.handle_get_status();
    const auto body = response_body(response);

    CHECK(response->status == 200);
    REQUIRE(body["processor"].size() == 2);
    CHECK(body["processor"].at(0)["instance_id"] == 0);
    CHECK(body["processor"].at(0)["port"] == 8901);
    CHECK(body["processor"].at(0)["models"] ==
          nlohmann::json::array({"local-model"}));
    CHECK(body["processor"].at(1)["instance_id"] == 2);
    CHECK(body["processor"].at(1)["uuid"] == 2);
    CHECK(body["processor"].at(1)["process_id"] == 77);
    CHECK(body["processor"].at(1)["generation"] == 9);
}

TEST_CASE("Worker control application preserves status acknowledgement fencing",
          "[p1][worker-control][application-service]") {
    ApplicationFixture fixture;
    HLIEWorkerControlService control(fixture.publisher,
                                     std::move(fixture.dependencies));
    HLIEWorkerControlApplicationService application(
        control, {[&fixture] { return fixture.ticket; }});

    auto applied = application.handle_post_status(request_with_body(
        R"({"model":"alias","value":"loaded","port":8901,"generation":9,"launch_token":"secret-token"})"));
    CHECK(applied->status == 200);
    CHECK(response_body(applied)["success"] == true);
    CHECK(fixture.apply_count.load() == 1);

    auto lookup_failed = application.handle_post_status(request_with_body(
        R"({"model":"missing","value":"loaded","port":8901,"generation":9,"launch_token":"secret-token"})"));
    CHECK(lookup_failed->status == 200);
    CHECK(response_body(lookup_failed)["success"] == true);

    auto stale = application.handle_post_status(request_with_body(
        R"({"model":"alias","value":"loaded","port":8901,"generation":8,"launch_token":"wrong"})"));
    CHECK(stale->status == 400);
    const auto stale_body = response_body(stale);
    CHECK(stale_body["error"]["ret"] == 409);
    CHECK(stale_body["error"]["type"] == "invalid_request_error");
}

TEST_CASE("Worker control application maps shutdown fencing and idempotency",
          "[p1][worker-control][application-service]") {
    ApplicationFixture fixture;
    HLIEWorkerControlService control(fixture.publisher,
                                     std::move(fixture.dependencies));
    HLIEWorkerControlApplicationService application(
        control, {[&fixture] { return fixture.ticket; }});

    auto invalid = application.handle_shutdown(request_with_body(
        R"({"generation":8,"launch_token":"wrong"})"));
    CHECK(invalid->status == 403);
    CHECK(response_body(invalid)["error"]["ret"] == 80009);

    auto accepted = application.handle_shutdown(request_with_body(
        R"({"generation":9,"launch_token":"secret-token"})"));
    CHECK(accepted->status == 200);
    const auto accepted_body = response_body(accepted);
    CHECK(accepted_body["generation"] == 9);
    CHECK(accepted_body["already_requested"] == false);

    auto duplicate = application.handle_shutdown(request_with_body(
        R"({"generation":9,"launch_token":"secret-token"})"));
    CHECK(duplicate->status == 200);
    CHECK(response_body(duplicate)["already_requested"] == true);
    CHECK(fixture.shutdown_count.load() == 1);
}

#include <catch_amalgamated.hpp>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "hlie_app/llm/hlie_llm_application_service.h"
#include "hlie_base/hlie_exception.h"

namespace {

using namespace hm_HLIE;

const std::function<bool()> NEVER_STOP = [] { return false; };

server_http_req make_request(
    std::string body = {}, std::map<std::string, std::string> params = {},
    const std::function<bool()>& should_stop = NEVER_STOP) {
    return {std::move(params), {}, "/v1/chat/completions", {},
            std::move(body), {}, should_stop};
}

server_http_res_ptr response(
    const int status = 200, std::string data = R"({"ok":true})") {
    auto result = std::make_unique<server_http_res>();
    result->status = status;
    result->data = std::move(data);
    return result;
}

nlohmann::json response_body(const server_http_res_ptr& result) {
    REQUIRE(result);
    return nlohmann::json::parse(result->data);
}

struct LlmFixture {
    HLIEAdmissionController admission{1};
    bool draining = false;
    bool auto_load = true;
    int local_port = 7901;
    int master_port = 7901;
    int refresh_calls = 0;
    int readiness_calls = 0;
    int dispatch_calls = 0;
    int document_dispatch_calls = 0;
    std::string dispatched_model;
    std::string dispatched_method;
    HLIEInferenceOperation dispatched_operation =
        HLIEInferenceOperation::Responses;
    std::optional<std::string> embedded_model;
    std::optional<HLIELlmAutoLoadRequest> observed_load_request;
    HLIELlmReadinessOutcome readiness = HLIELlmReadinessOutcome::Ready;
    std::map<std::string, HLIELlmModelSnapshot> models;
    std::function<void()> on_refresh;
    std::function<void(const std::string&)> on_ready;
    std::function<std::optional<std::vector<int>>(const std::string&)>
        parse_devices = [](const std::string&) {
            return std::optional<std::vector<int>>{{1, 2}};
        };

    HLIELlmApplicationDependencies dependencies() {
        return {
            [this] { return draining; },
            [this](const std::string& requested)
                -> std::optional<HLIELlmModelSnapshot> {
                const auto it = models.find(requested);
                return it == models.end()
                    ? std::nullopt
                    : std::optional<HLIELlmModelSnapshot>{it->second};
            },
            [this] {
                ++refresh_calls;
                if (on_refresh) {
                    on_refresh();
                }
            },
            [this] { return auto_load; },
            [this](const std::string& devices) {
                return parse_devices ? parse_devices(devices) : std::nullopt;
            },
            [this](const std::string& model_key,
                   std::optional<HLIELlmAutoLoadRequest> request,
                   std::chrono::steady_clock::time_point,
                   const std::function<bool()>& should_stop) {
                ++readiness_calls;
                observed_load_request = std::move(request);
                (void)should_stop();
                if (on_ready) {
                    on_ready(model_key);
                }
                return readiness;
            },
            [this] { return local_port; },
            [this] { return master_port; },
            [this] { return embedded_model; },
            [this](const std::string& model_key,
                   const HLIEInferenceOperation operation,
                   const server_http_req&, const std::string& method) {
                ++dispatch_calls;
                dispatched_model = model_key;
                dispatched_operation = operation;
                dispatched_method = method;
                return response(200, R"({"route":"inference"})");
            },
            [this](const std::string& model_key,
                   const HLIEInferenceOperation operation,
                   const server_http_req&, const nlohmann::json&,
                   const std::string& method) {
                ++document_dispatch_calls;
                dispatched_model = model_key;
                dispatched_operation = operation;
                dispatched_method = method;
                return response(200, R"({"route":"document"})");
            },
        };
    }
};

HLIELlmModelSnapshot model(
    std::string key, const HLIELlmModelState state, const int port) {
    return {std::move(key), state, port};
}

} // namespace

TEST_CASE("LLM application service validates and refreshes model selection",
          "[p1][llm][application-service]") {
    LlmFixture fixture;
    HLIELlmApplicationService service(
        fixture.admission, fixture.dependencies());

    SECTION("missing model") {
        const auto result = service.handle(make_request());
        CHECK(result->status == 400);
        CHECK(response_body(result) == nlohmann::json{{"error", {
            {"code", 400},
            {"message", "model name is missing from the request"},
            {"type", "invalid_request_error"},
        }}});
    }

    SECTION("discovery retry publishes the requested model") {
        fixture.on_refresh = [&] {
            fixture.models["new-model"] = model(
                "canonical-model", HLIELlmModelState::Loaded, 7901);
        };
        const auto result = service.handle(
            make_request(R"({"model":"new-model"})"));
        CHECK(result->status == 200);
        CHECK(fixture.refresh_calls == 1);
        CHECK(fixture.document_dispatch_calls == 1);
        CHECK(fixture.dispatched_model == "canonical-model");
    }

    SECTION("unknown model remains a compatibility 400") {
        const auto result = service.handle(
            make_request(R"({"model":"unknown"})"));
        CHECK(result->status == 400);
        CHECK(fixture.refresh_calls == 1);
        CHECK(response_body(result)["error"]["message"] == "model not found");
    }
}

TEST_CASE("LLM application service preserves local and remote dispatch",
          "[p1][llm][application-service]") {
    LlmFixture fixture;
    fixture.models["local"] = model(
        "local-key", HLIELlmModelState::Loaded, 7901);
    fixture.models["remote"] = model(
        "remote-key", HLIELlmModelState::Loaded, 7902);
    HLIELlmApplicationService service(
        fixture.admission, fixture.dependencies());

    auto result = service.handle(make_request(R"({"model":"local"})"));
    CHECK(response_body(result)["route"] == "document");
    CHECK(fixture.document_dispatch_calls == 1);
    result.reset();

    result = service.handle(make_request(R"({"model":"remote"})"));
    CHECK(response_body(result)["route"] == "inference");
    CHECK(fixture.dispatch_calls == 1);
    CHECK(fixture.dispatched_model == "remote-key");
    result.reset();

    fixture.local_port = 7902;
    result = service.handle(make_request(R"({"model":"remote"})"));
    CHECK(response_body(result)["route"] == "document");
    CHECK(fixture.document_dispatch_calls == 2);
}

TEST_CASE("LLM application service owns automatic load request adaptation",
          "[p1][llm][application-service][auto-load]") {
    LlmFixture fixture;
    fixture.models["cold"] = model(
        "cold-key", HLIELlmModelState::Unloaded, 0);
    fixture.on_ready = [&](const std::string&) {
        fixture.models["cold"] = model(
            "cold-key", HLIELlmModelState::Loaded, fixture.local_port);
    };
    HLIELlmApplicationService service(
        fixture.admission, fixture.dependencies());

    SECTION("disabled automatic load") {
        fixture.auto_load = false;
        const auto result = service.handle(
            make_request(R"({"model":"cold"})"));
        CHECK(result->status == 400);
        CHECK(response_body(result)["error"]["message"] == "model not loaded");
        CHECK(fixture.readiness_calls == 0);
    }

    SECTION("GET cannot trigger automatic load") {
        const auto result = service.handle(
            make_request({}, {{"model", "cold"}}),
            HLIEInferenceOperation::Properties, "GET");
        CHECK(result->status == 400);
        CHECK(response_body(result)["error"]["message"] ==
              "model is not loaded; load it before using this GET endpoint");
        CHECK(fixture.readiness_calls == 0);
    }

    SECTION("device parser failure") {
        fixture.parse_devices = [](const std::string&) {
            return std::optional<std::vector<int>>{};
        };
        const auto result = service.handle(make_request(
            R"({"model":"cold","devices":"bad"})"));
        CHECK(result->status == 503);
        CHECK(response_body(result)["error"] == nlohmann::json{
            {"code", 503}, {"message", "The devices parsed failed."},
            {"type", "unavailable_error"}, {"ret", 80004}});
        CHECK(fixture.readiness_calls == 0);
    }

    SECTION("load request fields and document dispatch") {
        const auto result = service.handle(make_request(
            R"({"model":"cold","model_path":"model.gguf","devices":"1,2","params":{"mm_proj_path":"vision.gguf"}})"));
        CHECK(result->status == 200);
        CHECK(fixture.readiness_calls == 1);
        REQUIRE(fixture.observed_load_request);
        CHECK(fixture.observed_load_request->model == "cold");
        CHECK(fixture.observed_load_request->model_path == "model.gguf");
        CHECK(fixture.observed_load_request->mmproj_path == "vision.gguf");
        REQUIRE(fixture.observed_load_request->requested_device_ids);
        CHECK(*fixture.observed_load_request->requested_device_ids ==
              std::vector<int>{1, 2});
        CHECK(fixture.document_dispatch_calls == 1);
    }
}

TEST_CASE("LLM application service maps readiness failures compatibly",
          "[p1][llm][application-service][readiness]") {
    struct Expectation {
        HLIELlmReadinessOutcome outcome;
        int status;
        const char* message;
    };
    const std::vector<Expectation> expectations{
        {HLIELlmReadinessOutcome::Missing, 404,
         "model not found while loading"},
        {HLIELlmReadinessOutcome::Unavailable, 503,
         "model stopped becoming ready"},
        {HLIELlmReadinessOutcome::InvalidLoadRequest, 503,
         "model load request is unavailable"},
        {HLIELlmReadinessOutcome::ScheduleFailed, 503,
         "failed to schedule model loading"},
        {HLIELlmReadinessOutcome::TimedOut, 503,
         "timed out waiting for model to load"},
        {HLIELlmReadinessOutcome::Orphaned, 503,
         "model runtime ownership is uncertain"},
    };

    for (const auto& expectation : expectations) {
        CAPTURE(expectation.message);
        LlmFixture fixture;
        fixture.models["loading"] = model(
            "loading-key", HLIELlmModelState::Loading, 0);
        fixture.readiness = expectation.outcome;
        HLIELlmApplicationService service(
            fixture.admission, fixture.dependencies());
        const auto result = service.handle(
            make_request(R"({"model":"loading"})"));
        CHECK(result->status == expectation.status);
        CHECK(response_body(result)["error"]["message"] ==
              expectation.message);
    }

    for (const auto outcome : {HLIELlmReadinessOutcome::Failed,
                               HLIELlmReadinessOutcome::Unloaded}) {
        LlmFixture fixture;
        fixture.models["loading"] = model(
            "loading-key", HLIELlmModelState::Loading, 0);
        fixture.readiness = outcome;
        HLIELlmApplicationService service(
            fixture.admission, fixture.dependencies());
        const auto result = service.handle(
            make_request(R"({"model":"loading"})"));
        CHECK(result->status == 500);
        CHECK(response_body(result) ==
              nlohmann::json{{"error", "failed to load model: loading"}});
    }
}

TEST_CASE("LLM application service distinguishes cancellation from draining",
          "[p1][llm][application-service][cancellation]") {
    LlmFixture fixture;
    fixture.models["loading"] = model(
        "loading-key", HLIELlmModelState::Loading, 0);
    fixture.readiness = HLIELlmReadinessOutcome::Cancelled;
    bool stop_observed = false;
    const std::function<bool()> should_stop = [&] {
        stop_observed = true;
        return true;
    };
    HLIELlmApplicationService service(
        fixture.admission, fixture.dependencies());

    auto result = service.handle(
        make_request(R"({"model":"loading"})", {}, should_stop));
    CHECK(stop_observed);
    CHECK(result->status == 499);
    CHECK(response_body(result)["error"] == nlohmann::json{
        {"code", 499},
        {"message", "request cancelled while waiting for model to load"},
        {"type", "unavailable_error"},
    });
    result.reset();

    fixture.on_ready = [&](const std::string&) { fixture.draining = true; };
    result = service.handle(
        make_request(R"({"model":"loading"})", {}, should_stop));
    CHECK(result->status == 503);
    CHECK(response_body(result)["error"]["message"] ==
          "service started draining while waiting for model load");
}

TEST_CASE("LLM application service retains admission for response lifetime",
          "[p1][llm][application-service][admission]") {
    LlmFixture fixture;
    fixture.models["ready"] = model(
        "ready-key", HLIELlmModelState::Loaded, 7901);
    HLIELlmApplicationService service(
        fixture.admission, fixture.dependencies());
    auto request = make_request(R"({"model":"ready"})");

    auto result = service.handle(request);
    CHECK(fixture.admission.snapshot().in_flight == 1);
    CHECK_THROWS_AS(service.handle(request), InferenceQueueFullException);
    CHECK(fixture.admission.snapshot().rejected == 1);
    result.reset();
    CHECK(fixture.admission.snapshot().in_flight == 0);

    result = service.handle_admitted(request, HLIEInferenceOperation::Responses);
    CHECK(fixture.admission.snapshot().accepted == 1);
    CHECK(fixture.admission.snapshot().in_flight == 0);
}

TEST_CASE("LLM application service owns properties and slots compatibility",
          "[p1][llm][application-service][compatibility]") {
    LlmFixture fixture;
    HLIELlmApplicationService service(
        fixture.admission, fixture.dependencies());
    auto request = make_request();

    auto result = service.handle_properties(request);
    CHECK(result->status == 200);
    CHECK(response_body(result) == nlohmann::json{
        {"role", "router"},
        {"max_instances", 4},
        {"model_alias", "llama-server"},
        {"model_path", "none"},
        {"default_generation_settings", {
            {"params", nlohmann::json{}}, {"n_ctx", 0}}},
    });

    result = service.handle_slots(request);
    CHECK(result->status == 503);
    CHECK(response_body(result)["error"]["message"] ==
          "Model is not available for now.");

    fixture.embedded_model = "embedded-key";
    result = service.handle_properties(request);
    CHECK(result->status == 200);
    CHECK(fixture.dispatched_operation == HLIEInferenceOperation::Properties);
    CHECK(fixture.dispatched_method == "GET");

    result = service.handle_slots(request);
    CHECK(result->status == 200);
    CHECK(fixture.dispatched_model == "embedded-key");
    CHECK(fixture.dispatched_operation == HLIEInferenceOperation::Slots);
    CHECK(fixture.dispatched_method == "GET");
}

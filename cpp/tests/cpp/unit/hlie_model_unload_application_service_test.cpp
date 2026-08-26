#include <catch_amalgamated.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "hlie_app/models/hlie_model_unload_application_service.h"
#include "json.hpp"

namespace {

using namespace hm_HLIE;

server_http_req request_with_body(std::string body) {
    static const std::function<bool()> should_stop = [] { return false; };
    return {{}, {}, "/models/unload", {}, std::move(body), {}, should_stop};
}

nlohmann::json response_body(const server_http_res_ptr& response) {
    REQUIRE(response);
    return nlohmann::json::parse(response->data);
}

struct UnloadFixture {
    bool replicas_mode = false;
    bool eviction_succeeds = true;
    std::optional<std::string> loaded_model = std::string{"loaded-key"};
    std::string resolved_request;
    std::string requested_type;
    std::string evicted_model;
    std::string eviction_reason;
    std::vector<std::string> published_models;

    HLIEModelUnloadApplicationService service() {
        return HLIEModelUnloadApplicationService({
            [this] { return replicas_mode; },
            [this](const std::string& requested)
                -> std::optional<std::string> {
                resolved_request = requested;
                if (requested == "alias" || requested == "fallback") {
                    return "canonical-key";
                }
                return std::nullopt;
            },
            [this](const std::string& model_type)
                -> std::optional<std::string> {
                requested_type = model_type;
                return loaded_model;
            },
            [](const std::string& model_type) {
                return model_type == "llm" || model_type == "sd" ||
                    model_type == "whisper" || model_type == "cosyvoice" ||
                    model_type == "qwen3_tts";
            },
            [this](const std::string& model_key, const std::string& reason) {
                evicted_model = model_key;
                eviction_reason = reason;
                return eviction_succeeds;
            },
            [this](const std::string& model_key) {
                published_models.push_back(model_key);
            },
        });
    }
};

} // namespace

TEST_CASE("Model unload application prefers explicit model identity",
          "[p1][model-unload][application-service]") {
    UnloadFixture fixture;
    auto service = fixture.service();

    const auto response = service.handle(request_with_body(
        R"({"model_id":"alias","model":"ignored","model_type":"sd"})"));
    const auto body = response_body(response);

    CHECK(response->status == 200);
    CHECK(response->content_type == "application/json; charset=utf-8");
    CHECK(body["code"] == 0);
    CHECK(body["model"] == "canonical-key");
    CHECK(fixture.resolved_request == "alias");
    CHECK(fixture.requested_type.empty());
    CHECK(fixture.evicted_model == "canonical-key");
    CHECK(fixture.eviction_reason == "api_unload");
    CHECK(fixture.published_models ==
          std::vector<std::string>{"canonical-key"});
}

TEST_CASE("Model unload application keeps model fallback and failure DTOs",
          "[p1][model-unload][application-service]") {
    SECTION("model is the legacy identity fallback") {
        UnloadFixture fixture;
        auto service = fixture.service();
        const auto response = service.handle(
            request_with_body(R"({"model":"fallback"})"));
        CHECK(response->status == 200);
        CHECK(response_body(response)["model"] == "canonical-key");
        CHECK(fixture.resolved_request == "fallback");
    }

    SECTION("replica mode rejects external unload") {
        UnloadFixture fixture;
        fixture.replicas_mode = true;
        auto service = fixture.service();
        const auto response = service.handle(
            request_with_body(R"({"model_id":"alias"})"));
        const auto body = response_body(response);
        CHECK(response->status == 501);
        CHECK(body["error"]["ret"] == 80005);
        CHECK(fixture.resolved_request.empty());
    }

    SECTION("unknown model returns not found") {
        UnloadFixture fixture;
        auto service = fixture.service();
        const auto response = service.handle(
            request_with_body(R"({"model_id":"missing"})"));
        const auto body = response_body(response);
        CHECK(response->status == 404);
        CHECK(body["error"]["type"] == "not_found_error");
        CHECK_FALSE(body["error"].contains("ret"));
        CHECK(fixture.evicted_model.empty());
    }

    SECTION("busy model returns unavailable without publication") {
        UnloadFixture fixture;
        fixture.eviction_succeeds = false;
        auto service = fixture.service();
        const auto response = service.handle(
            request_with_body(R"({"model_id":"alias"})"));
        const auto body = response_body(response);
        CHECK(response->status == 503);
        CHECK(body["error"]["ret"] == 80004);
        CHECK(fixture.published_models.empty());
    }
}

TEST_CASE("Model unload application normalizes legacy type selectors",
          "[p1][model-unload][application-service]") {
    const auto verify_alias = [](const std::string& request_body,
                                 const std::string& expected_type) {
        UnloadFixture fixture;
        auto service = fixture.service();
        const auto response = service.handle(request_with_body(request_body));
        CHECK(response->status == 200);
        CHECK(response_body(response)["code"] == 0);
        CHECK(fixture.requested_type == expected_type);
        CHECK(fixture.evicted_model == "loaded-key");
        CHECK(fixture.eviction_reason == "api_unload_by_type");
        CHECK(fixture.published_models ==
              std::vector<std::string>{"loaded-key"});
    };

    SECTION("document shares the LLM lifecycle") {
        verify_alias(R"({"model_type":"document"})", "llm");
    }
    SECTION("tts preserves the CosyVoice compatibility alias") {
        verify_alias(R"({"model_type":"tts"})", "cosyvoice");
    }
    SECTION("missing type defaults to LLM") {
        verify_alias(R"({})", "llm");
    }
}

TEST_CASE("Model unload application rejects invalid or absent type targets",
          "[p1][model-unload][application-service]") {
    SECTION("invalid type") {
        UnloadFixture fixture;
        auto service = fixture.service();
        const auto response = service.handle(
            request_with_body(R"({"model_type":"unknown"})"));
        const auto body = response_body(response);
        CHECK(response->status == 400);
        CHECK(body["error"]["ret"] == 80006);
        CHECK(fixture.requested_type.empty());
    }

    SECTION("supported type without a loaded service") {
        UnloadFixture fixture;
        fixture.loaded_model.reset();
        auto service = fixture.service();
        const auto response = service.handle(
            request_with_body(R"({"model_type":"qwen3_tts"})"));
        const auto body = response_body(response);
        CHECK(response->status == 503);
        CHECK(body["error"]["ret"] == 80004);
        CHECK(fixture.requested_type == "qwen3_tts");
        CHECK(fixture.evicted_model.empty());
    }
}

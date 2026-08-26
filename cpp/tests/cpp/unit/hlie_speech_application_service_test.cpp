#include <catch_amalgamated.hpp>

#include <memory>
#include <optional>
#include <string>

#include "hlie_app/speech/hlie_speech_application_service.h"
#include "json.hpp"

namespace {

using namespace hm_HLIE;

HLIESpeechDispatchTarget speech_target(
    const std::string& model_key,
    const std::function<bool(const httplib::Request&, httplib::Response&,
                             std::shared_ptr<void>)>& dispatch) {
    HLIESpeechDispatchTarget target;
    target.identity.kind = HLIERuntimeKind::Embedded;
    target.identity.launch_ticket.model_key = model_key;
    target.identity.launch_ticket.generation = 7;
    target.identity.launch_ticket.token = "test-token";
    target.identity.endpoint_port = 7901;
    target.lifetime_guard = std::make_shared<int>(42);
    target.dispatch = dispatch;
    return target;
}

httplib::Request request_with_body(std::string body) {
    httplib::Request request;
    request.body = std::move(body);
    request.is_connection_closed = [] { return false; };
    return request;
}

} // namespace

TEST_CASE("Speech application service validates the JSON request",
          "[p1][speech][application-service]") {
    HLIEAdmissionController admission{1};
    HLIESpeechApplicationService service(admission, {});
    auto request = request_with_body("not-json");
    httplib::Response response;

    service.handle(request, response);

    CHECK(response.status == 400);
    const auto body = nlohmann::json::parse(response.body);
    CHECK(body["error"]["type"] == "invalid_request_error");
    CHECK(body["error"]["ret"] == 80004);
}

TEST_CASE("Speech application service preserves model-less CosyVoice routing",
          "[p1][speech][application-service]") {
    HLIEAdmissionController admission{1};
    bool dispatched = false;
    bool observed = false;
    HLIESpeechApplicationService service(
        admission,
        {
            {},
            [](const std::string& kind, HLIERuntimeKind) {
                return kind == "cosyvoice"
                    ? std::optional<std::string>{"cosy-key"}
                    : std::nullopt;
            },
            {},
            [&](const std::string& model_key) {
                HLIESpeechAcquireResult result;
                result.target = speech_target(
                    model_key,
                    [&](const httplib::Request&, httplib::Response& response,
                        std::shared_ptr<void> guard) {
                        CHECK(guard != nullptr);
                        dispatched = true;
                        response.set_content("audio", "audio/wav");
                        return true;
                    });
                return result;
            },
            [&](const bool success) { observed = success; },
        });
    auto request = request_with_body(R"({"input":"hello"})");
    httplib::Response response;

    service.handle(request, response);

    CHECK(dispatched);
    CHECK(observed);
    CHECK(response.get_header_value("X-HLIE-TTS-Backend") == "cosyvoice");
    CHECK(response.body == "audio");
}

TEST_CASE("Speech application service requires a model for Qwen3-TTS",
          "[p1][speech][application-service]") {
    HLIEAdmissionController admission{1};
    HLIESpeechApplicationService service(
        admission,
        {
            {},
            [](const std::string& kind, HLIERuntimeKind) {
                return kind == "qwen3_tts"
                    ? std::optional<std::string>{"qwen-key"}
                    : std::nullopt;
            },
        });
    auto request = request_with_body(R"({"input":"hello"})");
    httplib::Response response;

    service.handle(request, response);

    CHECK(response.status == 400);
    const auto body = nlohmann::json::parse(response.body);
    CHECK(body["error"]["message"] ==
          "model is required when using Qwen3-TTS");
}

TEST_CASE("Speech application service waits for Qwen and guards admission",
          "[p1][speech][application-service][readiness]") {
    HLIEAdmissionController admission{1};
    int resolves = 0;
    int readiness_calls = 0;
    HLIESpeechApplicationService service(
        admission,
        {
            [&](const std::string&) {
                ++resolves;
                return std::optional<HLIESpeechModelSnapshot>{{
                    "qwen-key", "qwen3_tts",
                    resolves == 1 ? HLIESpeechModelState::Loading
                                  : HLIESpeechModelState::Loaded,
                }};
            },
            {},
            [&](const std::string& model_key, const std::function<bool()>&) {
                ++readiness_calls;
                CHECK(model_key == "qwen-key");
                return HLIESpeechReadinessStatus::Ready;
            },
            [&](const std::string& model_key) {
                HLIESpeechAcquireResult result;
                result.target = speech_target(
                    model_key,
                    [&](const httplib::Request&, httplib::Response& response,
                        std::shared_ptr<void> guard) {
                        CHECK(guard != nullptr);
                        CHECK(admission.snapshot().in_flight == 1);
                        response.set_content("audio", "audio/wav");
                        return true;
                    });
                return result;
            },
        });
    auto request = request_with_body(
        R"({"model":"qwen-key","input":"hello"})");
    httplib::Response response;

    service.handle(request, response);

    CHECK(readiness_calls == 1);
    CHECK(resolves == 2);
    CHECK(response.get_header_value("X-HLIE-TTS-Backend") == "qwen3_tts");
    CHECK(admission.snapshot().in_flight == 0);
}

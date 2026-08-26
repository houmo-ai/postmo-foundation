#include <catch_amalgamated.hpp>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hlie_app/asr/hlie_asr_application_service.h"

namespace {

using namespace hm_HLIE;

struct FakeTransportState {
    int starts = 0;
    int stops = 0;
    int port = 0;
    HLIEAsrRealtimeHandler handler;
};

class FakeTransport final : public IHLIEAsrRealtimeTransport {
public:
    explicit FakeTransport(std::shared_ptr<FakeTransportState> state)
        : state_(std::move(state)) {}

    void start(HLIEAsrRealtimeHandler handler, const int port) override {
        ++state_->starts;
        state_->port = port;
        state_->handler = std::move(handler);
    }

    void stop() override { ++state_->stops; }

private:
    std::shared_ptr<FakeTransportState> state_;
};

server_http_req make_request(
    const std::function<bool()>& should_stop,
    std::string body,
    std::map<std::string, uploaded_file> files = {}) {
    return {{}, {}, "/v1/audio/transcriptions", {}, std::move(body),
            std::move(files), should_stop};
}

server_http_res_ptr response(int status, std::string data,
                             std::string content_type = "application/json") {
    auto result = std::make_unique<server_http_res>();
    result->status = status;
    result->data = std::move(data);
    result->content_type = std::move(content_type);
    return result;
}

} // namespace

TEST_CASE("ASR application service routes direct Whisper transcription",
          "[p1][asr][application-service]") {
    HLIEAdmissionController admission{2};
    HLIEInferenceOperation observed = HLIEInferenceOperation::Responses;
    HLIEAsrApplicationService service(
        admission,
        {
            [](const std::string&, HLIERuntimeKind) {
                return std::optional<std::string>{"whisper-key"};
            },
            {},
            {},
            [&](const std::string&, const HLIEInferenceOperation operation,
                const server_http_req&, const std::string&) {
                observed = operation;
                return response(200, R"({"text":"hello"})");
            },
            {},
            {},
        });

    const std::function<bool()> should_stop = [] { return false; };
    auto request = make_request(should_stop, R"({"model":"whisper-1"})");
    const auto result = service.handle_transcriptions(request);

    REQUIRE(result);
    CHECK(result->status == 200);
    CHECK(observed == HLIEInferenceOperation::WhisperTranscription);
}

TEST_CASE("ASR application service composes Qwen transcription and alignment",
          "[p1][asr][application-service][pipeline]") {
    HLIEAdmissionController admission{2};
    std::vector<std::string> routed_models;
    HLIEAsrApplicationService service(
        admission,
        {
            {},
            [](const std::string& requested)
                -> std::optional<HLIEAsrModelCandidate> {
                if (requested == "qwen3-asr") {
                    return HLIEAsrModelCandidate{"qwen3-asr", "qwen3-asr"};
                }
                return std::nullopt;
            },
            [] {
                return std::vector<HLIEAsrModelCandidate>{
                    {"qwen3-asr", "qwen3-asr"},
                    {"forcealigner", "ForceAligner"},
                };
            },
            {},
            [&](const server_http_req& request,
                const HLIEInferenceOperation operation) {
                REQUIRE(operation == HLIEInferenceOperation::LlmTranscription);
                const auto body = nlohmann::json::parse(request.body);
                routed_models.push_back(body.at("model").get<std::string>());
                if (routed_models.size() == 1) {
                    return response(200, R"({"text":"<asr_text>Hello</asr_text>"})");
                }
                return response(200, R"({"words":[{"word":"Hello"}]})");
            },
            {},
        });

    const std::function<bool()> should_stop = [] { return false; };
    uploaded_file audio;
    audio.filename = "audio.wav";
    audio.content_type = "audio/wav";
    audio.data = {0, 1, 2, 3};
    auto request = make_request(
        should_stop,
        R"({"model":"qwen3-asr","response_format":"verbose_json","timestamp_granularities":["word"]})",
        {{"file", std::move(audio)}});
    const auto result = service.handle_transcriptions(request);

    REQUIRE(result);
    REQUIRE(result->status == 200);
    const auto body = nlohmann::json::parse(result->data);
    CHECK(body.at("text") == "Hello");
    REQUIRE(routed_models.size() == 2);
    CHECK(routed_models[0] == "qwen3-asr");
    CHECK(routed_models[1] == "forcealigner");
}

TEST_CASE("ASR application service owns realtime transport lifecycle",
          "[p1][asr][application-service][realtime]") {
    HLIEAdmissionController admission{2};
    const auto state = std::make_shared<FakeTransportState>();
    bool listener_allowed = true;
    HLIEAsrApplicationService service(
        admission,
        {
            {}, {}, {}, {}, {}, {},
            [&] { return listener_allowed; },
            [] { return true; },
            [] { return 7901; },
            [state] {
                return std::make_unique<FakeTransport>(state);
            },
        });

    service.ensure_realtime();
    service.ensure_realtime();
    CHECK(state->starts == 1);
    CHECK(state->port == 7999);

    service.stop_realtime();
    CHECK(state->stops == 1);

    listener_allowed = false;
    service.ensure_realtime();
    CHECK(state->starts == 1);
}

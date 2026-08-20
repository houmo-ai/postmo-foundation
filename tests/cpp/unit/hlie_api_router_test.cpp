#include <catch_amalgamated.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "hlie_app/http/hlie_api_router.h"

namespace {

using namespace hm_HLIE;

server_http_res_ptr ok_response() {
    auto response = std::make_unique<server_http_res>();
    response->status = 200;
    return response;
}

const server_http_req& empty_request() {
    static const std::function<bool()> should_stop = [] { return false; };
    static const server_http_req request{{}, {}, "/", {}, "{}", {},
                                         should_stop};
    return request;
}

} // namespace

TEST_CASE("ApiRouter preserves the public route surface",
          "[p1][api-router]") {
    HLIEApiRouteHandlers handlers;
    HLIEApiRouter router(std::move(handlers));
    const auto routes = router.get_route_configs();
    REQUIRE(routes.size() == 30);

    const std::unordered_map<std::string, HttpType> expected_paths = {
        {"/live", GET},
        {"/livez", GET},
        {"/ready", GET},
        {"/readyz", GET},
        {"/metrics", GET},
        {"/hlie/v1/metrics", GET},
        {"/health", GET},
        {"/v1/health", GET},
        {"/v1/models", GET},
        {"/models", GET},
        {"/api/tags", GET},
        {"/hlie/v1/models/refresh", POST},
        {"/models/refresh", POST},
        {"/v1/chat/completions", POST},
        {"/chat/completions", POST},
        {"/api/chat", POST},
        {"/v1/images/generations", POST},
        {"/images/generations", POST},
        {"/v1/audio/transcriptions", POST},
        {"/v1/audio/speech", POST},
        {"/hlie/v1/devices", GET},
        {"/v1/completions", POST},
        {"/completion", POST},
        {"/completions", POST},
        {"/v1/responses", POST},
        {"/responses", POST},
        {"/embedding", POST},
        {"/embeddings", POST},
        {"/v1/embeddings", POST},
        {"/rerank", POST},
        {"/reranking", POST},
        {"/v1/rerank", POST},
        {"/v1/reranking", POST},
        {"/props", GET},
        {"/slots", GET},
        {"/hlie/v1/start", POST},
        {"/models/load", POST},
        {"/hlie/v1/stop", POST},
        {"/hlie/v1/status", GET},
        {"/models/status", POST},
        {"/internal/worker/shutdown", POST},
        {"/hlie/v1/qwen3_tts/prefill_debug", POST},
        {"/hlie/v1/qwen3_tts/codec_frames_debug", POST},
        {"/hlie/v1/qwen3_tts/reference_debug", POST},
        {"/hlie/v1/qwen3_tts/speech_debug", POST},
        {"/hlie/v1/qwen3_tts/stateful_decoder_debug", POST},
        {"/hlie/v1/unload_model", POST},
        {"/models/unload", POST},
    };
    std::unordered_set<std::string> paths;
    for (const auto& route : routes) {
        REQUIRE_FALSE(route.paths.empty());
        for (const auto& path : route.paths) {
            CHECK(paths.insert(path).second);
            const auto expected = expected_paths.find(path);
            REQUIRE(expected != expected_paths.end());
            CHECK(route.method == expected->second);
        }
    }
    CHECK(paths.size() == expected_paths.size());

    const auto speech = std::find_if(
        routes.begin(), routes.end(), [](const auto& route) {
            return std::find(route.paths.begin(), route.paths.end(),
                             "/v1/audio/speech") != route.paths.end();
        });
    REQUIRE(speech != routes.end());
    CHECK(speech->method == POST);
    CHECK_FALSE(speech->handler);
}

TEST_CASE("ApiRouter maps inference URLs to stable operations",
          "[p1][api-router]") {
    HLIEInferenceOperation observed = HLIEInferenceOperation::Slots;
    std::string observed_method;
    HLIEApiRouteHandlers handlers;
    handlers.llm = [&](const server_http_req&,
                       const HLIEInferenceOperation operation,
                       const std::string& method) {
        observed = operation;
        observed_method = method;
        return ok_response();
    };
    handlers.qwen3_tts_debug = [&](const server_http_req&,
                                   const HLIEInferenceOperation operation) {
        observed = operation;
        return ok_response();
    };
    HLIEApiRouter router(std::move(handlers));
    const auto routes = router.get_route_configs();

    const auto invoke = [&](const std::string& path) {
        const auto route = std::find_if(
            routes.begin(), routes.end(), [&](const auto& candidate) {
                return std::find(candidate.paths.begin(), candidate.paths.end(),
                                 path) != candidate.paths.end();
            });
        REQUIRE(route != routes.end());
        REQUIRE(route->handler);
        REQUIRE(route->handler(empty_request()));
    };

    const auto check_operation = [&](const std::string& path,
                                     const HLIEInferenceOperation expected,
                                     const std::string& method = "POST") {
        invoke(path);
        CHECK(observed == expected);
        if (!method.empty()) {
            CHECK(observed_method == method);
        }
    };
    check_operation("/v1/chat/completions",
                    HLIEInferenceOperation::ChatCompletions);
    check_operation("/v1/completions",
                    HLIEInferenceOperation::CompletionsOpenAI);
    check_operation("/completion", HLIEInferenceOperation::Completions);
    check_operation("/v1/responses", HLIEInferenceOperation::Responses);
    check_operation("/embedding", HLIEInferenceOperation::Embeddings);
    check_operation("/v1/embeddings",
                    HLIEInferenceOperation::EmbeddingsOpenAI);
    check_operation("/rerank", HLIEInferenceOperation::Rerank);
    check_operation("/hlie/v1/qwen3_tts/prefill_debug",
                    HLIEInferenceOperation::Qwen3TtsPrefillDebug, "");
    check_operation("/hlie/v1/qwen3_tts/codec_frames_debug",
                    HLIEInferenceOperation::Qwen3TtsCodecFramesDebug, "");
    check_operation("/hlie/v1/qwen3_tts/reference_debug",
                    HLIEInferenceOperation::Qwen3TtsReferenceDebug, "");
    check_operation("/hlie/v1/qwen3_tts/speech_debug",
                    HLIEInferenceOperation::Qwen3TtsSpeechDebug, "");
    check_operation("/hlie/v1/qwen3_tts/stateful_decoder_debug",
                    HLIEInferenceOperation::Qwen3TtsStatefulDecoderDebug, "");
}

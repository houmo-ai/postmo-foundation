#include "hlie_app/hlie_qwen_asr_stream_decoder.h"

#include <catch_amalgamated.hpp>
#include <nlohmann/json.hpp>

#include <string>
#include <vector>

using namespace hm_HLIE;

namespace {
    std::string join(const std::vector<std::string>& parts) {
        std::string result;
        for (const auto& part : parts) {
            result += part;
        }
        return result;
    }

    std::string sse_delta(const std::string& delta) {
        return "event: transcript.text.delta\n"
               "data: {\"type\":\"transcript.text.delta\",\"delta\":" +
            nlohmann::json(delta).dump() + "}\n\n";
    }
}

TEST_CASE("Qwen ASR uses the bundled SDK string form for streaming", "[asr][stream]") {
    CHECK(LLAMA_TRANSCRIPTION_SDK_STREAM_ENABLED == "true");
}

TEST_CASE("Qwen ASR stream removes the model prefix and end marker across chunks", "[asr][stream]") {
    std::vector<std::string> deltas;
    HLIEQwenAsrStreamDecoder decoder(true, [&](const std::string& delta) { deltas.push_back(delta); });

    const std::string stream =
        sse_delta("language Chin") +
        sse_delta("ese<asr_") +
        sse_delta("text>今天把两个人叫过来，") +
        sse_delta("主要。</asr_") +
        sse_delta("text>") +
        "data: [DONE]\n\n";

    for (size_t offset = 0; offset < stream.size(); offset += 7) {
        decoder.push(std::string_view(stream).substr(offset, 7));
    }

    const std::string transcript = decoder.finish();
    CHECK(transcript == "今天把两个人叫过来，主要。");
    CHECK(join(deltas) == transcript);
    CHECK(decoder.saw_stream_delta());
    CHECK(transcript.find("language Chinese") == std::string::npos);
    CHECK(transcript.find("asr_text") == std::string::npos);
}

TEST_CASE("Qwen ASR decoder uses final SSE text when no delta was produced", "[asr][stream]") {
    std::vector<std::string> deltas;
    HLIEQwenAsrStreamDecoder decoder(true, [&](const std::string& delta) { deltas.push_back(delta); });

    decoder.push(
        "event: transcript.text.done\n"
        "data: {\"type\":\"transcript.text.done\","
        "\"text\":\"language Chinese<asr_text>测试完成。</asr_text>\"}\n\n");

    const std::string transcript = decoder.finish();
    CHECK(transcript == "测试完成。");
    CHECK(deltas.empty());
    CHECK_FALSE(decoder.saw_stream_delta());
}

TEST_CASE("Qwen ASR decoder accepts non-SSE JSON without exposing protocol text", "[asr][stream]") {
    std::vector<std::string> deltas;
    HLIEQwenAsrStreamDecoder decoder(false, [&](const std::string& delta) { deltas.push_back(delta); });

    decoder.push(R"({"text":"language Chinese<asr_text>非流式回退。</asr_text>"})");

    const std::string transcript = decoder.finish();
    CHECK(transcript == "非流式回退。");
    CHECK(deltas.empty());
    CHECK_FALSE(decoder.saw_stream_delta());
}

TEST_CASE("Non-Qwen ASR streams language-prefixed customer text immediately", "[asr][stream]") {
    std::vector<std::string> deltas;
    HLIEQwenAsrStreamDecoder decoder(
        true,
        [&](const std::string& delta) { deltas.push_back(delta); },
        false);

    decoder.push(sse_delta("language models are useful"));
    CHECK(join(deltas) == "language models are useful");
    CHECK(decoder.saw_stream_delta());
    CHECK(decoder.finish() == "language models are useful");
}

TEST_CASE("Qwen ASR decoder preserves plain transcription text without model markers", "[asr][stream]") {
    std::vector<std::string> deltas;
    HLIEQwenAsrStreamDecoder decoder(false, [&](const std::string& delta) { deltas.push_back(delta); });

    decoder.push("普通转录文本");
    CHECK(join(deltas) == "普通转录文本");

    const std::string transcript = decoder.finish();
    CHECK(transcript == "普通转录文本");
    CHECK(join(deltas) == transcript);
}

TEST_CASE("Qwen ASR final text cleaning only removes the explicit ASR marker envelope", "[asr][stream]") {
    CHECK(HLIEQwenAsrStreamDecoder::clean_final_text(
              "language Chinese<asr_text>正文</asr_text>ignored") == "正文");
    CHECK(HLIEQwenAsrStreamDecoder::clean_final_text("language model is customer text") ==
          "language model is customer text");
}

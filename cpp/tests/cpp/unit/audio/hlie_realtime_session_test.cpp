#include "hlie_app/hlie_ws_service.h"

#include <catch_amalgamated.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "base64.hpp"

using namespace hm_HLIE;

namespace {
using json = nlohmann::json;

std::vector<float> makeAudio(const int sample_rate) {
    const size_t silence_before = static_cast<size_t>(sample_rate) * 600U / 1000U;
    const size_t speech_samples = static_cast<size_t>(sample_rate);
    const size_t silence_after = static_cast<size_t>(sample_rate) * 700U / 1000U;
    std::vector<float> samples(silence_before + speech_samples + silence_after, 0.0f);
    constexpr double pi = 3.14159265358979323846;
    for (size_t i = 0; i < speech_samples; ++i) {
        samples[silence_before + i] = 0.006f * static_cast<float>(
            std::sin(2.0 * pi * 220.0 * static_cast<double>(i) / sample_rate));
    }
    return samples;
}

std::string encodePcm16(const std::vector<float>& samples, const size_t offset, const size_t count) {
    std::string bytes(count * sizeof(int16_t), '\0');
    for (size_t i = 0; i < count; ++i) {
        const float value = std::clamp(samples[offset + i], -1.0f, 1.0f);
        const auto pcm = static_cast<int16_t>(value * 32767.0f);
        std::memcpy(bytes.data() + static_cast<std::ptrdiff_t>(i * sizeof(int16_t)), &pcm, sizeof(pcm));
    }
    return base64::encode(bytes);
}

void sendJson(Session& session, const json& event) {
    const std::string payload = event.dump();
    session.onMessage(payload, uWS::OpCode::TEXT);
}

std::vector<json> parseEvents(const std::vector<std::string>& events) {
    std::vector<json> parsed;
    for (const auto& event : events) {
        parsed.push_back(json::parse(event));
    }
    return parsed;
}

std::vector<json> eventsOfType(const std::vector<json>& events, const std::string& type) {
    std::vector<json> result;
    for (const auto& event : events) {
        if (event.value("type", "") == type) {
            result.push_back(event);
        }
    }
    return result;
}
}

TEST_CASE("Realtime Session preserves OpenAI VAD event ordering", "[audio][vad][realtime]") {
    constexpr int sample_rate = 16000;
    std::vector<std::string> sent_events;
    std::vector<WhisperStreamRequest> submitted;
    std::vector<std::string> submitted_ids;
    auto connection = std::make_shared<SessionConnectionState>();

    Session session;
    session.init(
        [&](WhisperStreamRequest request, std::string item_id) {
            submitted.push_back(std::move(request));
            submitted_ids.push_back(std::move(item_id));
            return true;
        },
        [&](std::string event) { sent_events.push_back(std::move(event)); },
        connection);

    sendJson(session, {
        {"type", "session.update"},
        {"session", {
            {"type", "transcription"},
            {"audio", {{"input", {
                {"format", {{"type", "audio/pcm"}, {"rate", sample_rate}}},
                {"turn_detection", {
                    {"type", "server_vad"},
                    {"threshold", 0.5},
                    {"prefix_padding_ms", 300},
                    {"silence_duration_ms", 500},
                    {"max_turn_duration_ms", 30000},
                }},
            }}}},
        }},
    });

    const auto audio = makeAudio(sample_rate);
    const size_t chunk_samples = sample_rate / 4U;
    for (size_t offset = 0; offset < audio.size(); offset += chunk_samples) {
        const size_t count = std::min(chunk_samples, audio.size() - offset);
        sendJson(session, {
            {"type", "input_audio_buffer.append"},
            {"audio", encodePcm16(audio, offset, count)},
        });
    }

    const auto parsed = parseEvents(sent_events);
    const auto starts = eventsOfType(parsed, "input_audio_buffer.speech_started");
    const auto stops = eventsOfType(parsed, "input_audio_buffer.speech_stopped");
    const auto commits = eventsOfType(parsed, "input_audio_buffer.committed");

    REQUIRE(starts.size() == 1);
    REQUIRE(stops.size() == 1);
    REQUIRE(commits.size() == 1);
    REQUIRE(submitted.size() == 1);
    REQUIRE(submitted_ids.size() == 1);

    const std::string item_id = starts.front().at("item_id").get<std::string>();
    CHECK(stops.front().at("item_id") == item_id);
    CHECK(commits.front().at("item_id") == item_id);
    CHECK(submitted_ids.front() == item_id);
    CHECK(starts.front().at("audio_start_ms").get<int64_t>() <= 350);
    CHECK(stops.front().at("audio_end_ms").get<int64_t>() >= 1500);
    CHECK(submitted.front().sample_rate == sample_rate);
    CHECK(submitted.front().pcmf32.size() >= static_cast<size_t>(sample_rate * 1.4));
    CHECK(connection->active.load());

    size_t start_index = 0;
    size_t stop_index = 0;
    size_t commit_index = 0;
    for (size_t i = 0; i < parsed.size(); ++i) {
        const std::string type = parsed[i].value("type", "");
        if (type == "input_audio_buffer.speech_started") start_index = i;
        if (type == "input_audio_buffer.speech_stopped") stop_index = i;
        if (type == "input_audio_buffer.committed") commit_index = i;
    }
    CHECK(start_index < stop_index);
    CHECK(stop_index < commit_index);
}

TEST_CASE("Realtime Session keeps manual commit behavior when VAD is disabled", "[audio][vad][realtime]") {
    constexpr int sample_rate = 16000;
    std::vector<std::string> sent_events;
    std::vector<WhisperStreamRequest> submitted;
    auto connection = std::make_shared<SessionConnectionState>();

    Session session;
    session.init(
        [&](WhisperStreamRequest request, std::string) {
            submitted.push_back(std::move(request));
            return true;
        },
        [&](std::string event) { sent_events.push_back(std::move(event)); },
        connection);

    sendJson(session, {
        {"type", "session.update"},
        {"session", {{"audio", {{"input", {
            {"format", {{"type", "audio/pcm"}, {"rate", sample_rate}}},
            {"turn_detection", nullptr},
        }}}}}},
    });

    const auto audio = makeAudio(sample_rate);
    sendJson(session, {
        {"type", "input_audio_buffer.append"},
        {"audio", encodePcm16(audio, 0, audio.size())},
    });
    sendJson(session, {{"type", "input_audio_buffer.commit"}});

    const auto parsed = parseEvents(sent_events);
    CHECK(eventsOfType(parsed, "input_audio_buffer.speech_started").empty());
    CHECK(eventsOfType(parsed, "input_audio_buffer.speech_stopped").empty());
    CHECK(eventsOfType(parsed, "input_audio_buffer.committed").size() == 1);
    REQUIRE(submitted.size() == 1);
    CHECK(submitted.front().pcmf32.size() == audio.size());
    CHECK(connection->active.load());
}

TEST_CASE("Realtime Session manual commit flushes a server VAD turn", "[audio][vad][realtime]") {
    constexpr int sample_rate = 16000;
    std::vector<std::string> sent_events;
    std::vector<WhisperStreamRequest> submitted;
    std::vector<std::string> submitted_ids;
    auto connection = std::make_shared<SessionConnectionState>();

    Session session;
    session.init(
        [&](WhisperStreamRequest request, std::string item_id) {
            submitted.push_back(std::move(request));
            submitted_ids.push_back(std::move(item_id));
            return true;
        },
        [&](std::string event) { sent_events.push_back(std::move(event)); },
        connection);

    sendJson(session, {
        {"type", "session.update"},
        {"session", {{"audio", {{"input", {
            {"format", {{"type", "audio/pcm"}, {"rate", sample_rate}}},
            {"turn_detection", {
                {"type", "server_vad"},
                {"threshold", 0.5},
                {"silence_duration_ms", 3000},
            }},
        }}}}}},
    });

    const auto audio = makeAudio(sample_rate);
    const size_t without_long_trailing_silence = static_cast<size_t>(sample_rate) * 18U / 10U;
    sendJson(session, {
        {"type", "input_audio_buffer.append"},
        {"audio", encodePcm16(audio, 0, without_long_trailing_silence)},
    });
    sendJson(session, {{"type", "input_audio_buffer.commit"}});

    const auto parsed = parseEvents(sent_events);
    const auto starts = eventsOfType(parsed, "input_audio_buffer.speech_started");
    const auto stops = eventsOfType(parsed, "input_audio_buffer.speech_stopped");
    const auto commits = eventsOfType(parsed, "input_audio_buffer.committed");

    REQUIRE(starts.size() == 1);
    REQUIRE(stops.size() == 1);
    REQUIRE(commits.size() == 1);
    REQUIRE(submitted.size() == 1);
    REQUIRE(submitted_ids.size() == 1);
    const auto item_id = starts.front().at("item_id").get<std::string>();
    CHECK(stops.front().at("item_id") == item_id);
    CHECK(commits.front().at("item_id") == item_id);
    CHECK(submitted_ids.front() == item_id);
    CHECK(submitted.front().pcmf32.size() >= static_cast<size_t>(sample_rate * 1.4));
    CHECK(submitted.front().pcmf32.size() <= without_long_trailing_silence);
    CHECK(connection->active.load());
}

TEST_CASE("Realtime Session reports invalid VAD configuration without closing", "[audio][vad][realtime]") {
    std::vector<std::string> sent_events;
    auto connection = std::make_shared<SessionConnectionState>();
    Session session;
    session.init(
        [](WhisperStreamRequest, std::string) { return true; },
        [&](std::string event) { sent_events.push_back(std::move(event)); },
        connection);

    sendJson(session, {
        {"type", "session.update"},
        {"session", {{"audio", {{"input", {{"turn_detection", {
            {"type", "server_vad"},
            {"threshold", 1.5},
        }}}}}}}},
    });
    sendJson(session, {{"type", "input_audio_buffer.clear"}});

    const auto parsed = parseEvents(sent_events);
    const auto errors = eventsOfType(parsed, "error");
    REQUIRE(errors.size() == 1);
    CHECK(errors.front().at("error").at("message").get<std::string>().find("threshold") != std::string::npos);
    CHECK(eventsOfType(parsed, "input_audio_buffer.cleared").size() == 1);
    CHECK(connection->active.load());
}

TEST_CASE("Realtime Session reports inference backpressure without closing", "[audio][vad][realtime]") {
    constexpr int sample_rate = 16000;
    std::vector<std::string> sent_events;
    auto connection = std::make_shared<SessionConnectionState>();
    Session session;
    session.init(
        [](WhisperStreamRequest, std::string) { return false; },
        [&](std::string event) { sent_events.push_back(std::move(event)); },
        connection);

    sendJson(session, {
        {"type", "session.update"},
        {"session", {{"audio", {{"input", {
            {"format", {{"type", "audio/pcm"}, {"rate", sample_rate}}},
            {"turn_detection", nullptr},
        }}}}}},
    });
    const auto audio = makeAudio(sample_rate);
    sendJson(session, {
        {"type", "input_audio_buffer.append"},
        {"audio", encodePcm16(audio, 0, audio.size())},
    });
    sendJson(session, {{"type", "input_audio_buffer.commit"}});

    const auto parsed = parseEvents(sent_events);
    const auto failures = eventsOfType(parsed, "conversation.item.input_audio_transcription.failed");
    REQUIRE(failures.size() == 1);
    CHECK(failures.front().at("error").at("code") == "server_busy");
    CHECK(connection->active.load());
}

TEST_CASE("Realtime Session can disable VAD after a completed turn", "[audio][vad][realtime]") {
    constexpr int sample_rate = 16000;
    std::vector<std::string> sent_events;
    auto connection = std::make_shared<SessionConnectionState>();
    Session session;
    session.init(
        [](WhisperStreamRequest, std::string) { return true; },
        [&](std::string event) { sent_events.push_back(std::move(event)); },
        connection);

    sendJson(session, {
        {"type", "session.update"},
        {"session", {{"audio", {{"input", {
            {"format", {{"type", "audio/pcm"}, {"rate", sample_rate}}},
            {"turn_detection", {
                {"type", "server_vad"},
                {"threshold", 0.5},
                {"silence_duration_ms", 500},
            }},
        }}}}}},
    });
    const auto audio = makeAudio(sample_rate);
    for (size_t offset = 0; offset < audio.size(); offset += sample_rate / 4U) {
        const size_t count = std::min<size_t>(sample_rate / 4U, audio.size() - offset);
        sendJson(session, {
            {"type", "input_audio_buffer.append"},
            {"audio", encodePcm16(audio, offset, count)},
        });
    }

    sendJson(session, {
        {"type", "session.update"},
        {"session", {{"audio", {{"input", {{"turn_detection", nullptr}}}}}}},
    });

    const auto parsed = parseEvents(sent_events);
    CHECK(eventsOfType(parsed, "error").empty());
    CHECK(eventsOfType(parsed, "session.updated").size() == 2);
    CHECK(connection->active.load());
}

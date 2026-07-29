#include "hlie_app/audio/hlie_realtime_vad.h"

#include <catch_amalgamated.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace hm_HLIE;

namespace {
std::vector<float> silence(const int sample_rate, const int duration_ms) {
    return std::vector<float>(static_cast<size_t>(sample_rate) * duration_ms / 1000U, 0.0f);
}

std::vector<float> tone(
    const int sample_rate,
    const int duration_ms,
    const float amplitude = 0.006f,
    const float frequency = 220.0f) {
    const size_t sample_count = static_cast<size_t>(sample_rate) * duration_ms / 1000U;
    std::vector<float> samples(sample_count);
    constexpr double pi = 3.14159265358979323846;
    for (size_t i = 0; i < sample_count; ++i) {
        samples[i] = amplitude * static_cast<float>(
            std::sin(2.0 * pi * frequency * static_cast<double>(i) / sample_rate));
    }
    return samples;
}

void append(std::vector<float>& target, const std::vector<float>& samples) {
    target.insert(target.end(), samples.begin(), samples.end());
}

std::vector<RealtimeVadEvent> runInChunks(
    HLIERealtimeVad& vad,
    const std::vector<float>& samples,
    const size_t chunk_samples) {
    std::vector<RealtimeVadEvent> events;
    for (size_t offset = 0; offset < samples.size(); offset += chunk_samples) {
        const size_t end = std::min(samples.size(), offset + chunk_samples);
        std::vector<float> chunk(
            samples.begin() + static_cast<std::ptrdiff_t>(offset),
            samples.begin() + static_cast<std::ptrdiff_t>(end));
        auto chunk_events = vad.append(chunk);
        events.insert(
            events.end(),
            std::make_move_iterator(chunk_events.begin()),
            std::make_move_iterator(chunk_events.end()));
    }
    return events;
}

std::vector<const RealtimeVadEvent*> eventsOfType(
    const std::vector<RealtimeVadEvent>& events,
    const RealtimeVadEventType type) {
    std::vector<const RealtimeVadEvent*> result;
    for (const auto& event : events) {
        if (event.type == type) {
            result.push_back(&event);
        }
    }
    return result;
}
}

TEST_CASE("Realtime VAD rejects invalid configuration", "[audio][vad]") {
    RealtimeVadConfig config;
    std::string error;

    CHECK(config.silence_duration_ms == 2000);

    config.threshold = 1.1f;
    CHECK_FALSE(validateRealtimeVadConfig(config, error));
    CHECK_FALSE(error.empty());

    config = {};
    config.max_turn_duration_ms = 2999;
    CHECK_FALSE(validateRealtimeVadConfig(config, error));
}

TEST_CASE("Realtime VAD detects quiet speech and preserves prefix audio", "[audio][vad]") {
    RealtimeVadConfig config;
    config.sample_rate = 16000;
    config.threshold = 0.5f;
    config.prefix_padding_ms = 300;
    config.silence_duration_ms = 500;

    std::vector<float> input;
    append(input, silence(config.sample_rate, 600));
    append(input, tone(config.sample_rate, 1000));
    append(input, silence(config.sample_rate, 700));

    HLIERealtimeVad vad(config);
    const auto events = runInChunks(vad, input, config.sample_rate / 4);
    const auto starts = eventsOfType(events, RealtimeVadEventType::SpeechStarted);
    const auto stops = eventsOfType(events, RealtimeVadEventType::SpeechStopped);
    const auto segments = eventsOfType(events, RealtimeVadEventType::SegmentReady);

    REQUIRE(starts.size() == 1);
    REQUIRE(stops.size() == 1);
    REQUIRE(segments.size() == 1);
    CHECK(starts.front()->start_sample <= static_cast<int64_t>(config.sample_rate * 0.35));
    CHECK(stops.front()->reason == RealtimeVadCommitReason::Silence);
    CHECK(segments.front()->audio.size() >= static_cast<size_t>(config.sample_rate * 1.4));
    CHECK(segments.front()->metrics.speech_frames > 0);
}

TEST_CASE("Realtime VAD does not submit silence", "[audio][vad]") {
    RealtimeVadConfig config;
    config.sample_rate = 24000;
    HLIERealtimeVad vad(config);

    const auto events = runInChunks(vad, silence(config.sample_rate, 5000), config.sample_rate / 10);
    CHECK(eventsOfType(events, RealtimeVadEventType::SpeechStarted).empty());
    CHECK(eventsOfType(events, RealtimeVadEventType::SegmentReady).empty());
}

TEST_CASE("Realtime VAD reacquires background after digital silence", "[audio][vad]") {
    RealtimeVadConfig config;
    config.sample_rate = 16000;
    config.threshold = 0.2f;
    HLIERealtimeVad vad(config);

    std::vector<float> input;
    append(input, silence(config.sample_rate, 2000));
    // A stable approximately -60 dBFS hum should be treated as background,
    // even after exact digital silence has lowered the noise estimate.
    append(input, tone(config.sample_rate, 2000, 0.001414f, 50.0f));
    append(input, silence(config.sample_rate, 700));

    const auto events = runInChunks(vad, input, config.sample_rate / 4);
    CHECK(eventsOfType(events, RealtimeVadEventType::SpeechStarted).empty());
    CHECK(eventsOfType(events, RealtimeVadEventType::SegmentReady).empty());
}

TEST_CASE("Realtime VAD ignores short transient audio", "[audio][vad]") {
    RealtimeVadConfig config;
    config.sample_rate = 16000;
    HLIERealtimeVad vad(config);

    std::vector<float> input;
    append(input, silence(config.sample_rate, 300));
    append(input, tone(config.sample_rate, 40, 0.2f));
    append(input, silence(config.sample_rate, 700));

    const auto events = runInChunks(vad, input, config.sample_rate / 50);
    CHECK(eventsOfType(events, RealtimeVadEventType::SpeechStarted).empty());
    CHECK(eventsOfType(events, RealtimeVadEventType::SegmentReady).empty());
}

TEST_CASE("Realtime VAD segmentation is independent of append chunk size", "[audio][vad]") {
    RealtimeVadConfig config;
    config.sample_rate = 16000;
    config.threshold = 0.4f;

    std::vector<float> input;
    append(input, silence(config.sample_rate, 400));
    append(input, tone(config.sample_rate, 1200));
    append(input, silence(config.sample_rate, 2200));

    HLIERealtimeVad small_chunks(config);
    HLIERealtimeVad large_chunks(config);
    const auto small_events = runInChunks(small_chunks, input, config.sample_rate / 100);
    const auto large_events = runInChunks(large_chunks, input, config.sample_rate);

    REQUIRE(small_events.size() == large_events.size());
    for (size_t i = 0; i < small_events.size(); ++i) {
        CHECK(small_events[i].type == large_events[i].type);
        CHECK(small_events[i].reason == large_events[i].reason);
        CHECK(small_events[i].start_sample == large_events[i].start_sample);
        CHECK(small_events[i].end_sample == large_events[i].end_sample);
        CHECK(small_events[i].audio.size() == large_events[i].audio.size());
    }
}

TEST_CASE("Realtime VAD default maximum does not split five second speech", "[audio][vad]") {
    RealtimeVadConfig config;
    config.sample_rate = 16000;

    std::vector<float> input;
    append(input, silence(config.sample_rate, 400));
    append(input, tone(config.sample_rate, 5000));
    append(input, silence(config.sample_rate, 2200));

    HLIERealtimeVad vad(config);
    const auto events = runInChunks(vad, input, config.sample_rate / 4);
    const auto segments = eventsOfType(events, RealtimeVadEventType::SegmentReady);
    REQUIRE(segments.size() == 1);
    CHECK(segments.front()->reason == RealtimeVadCommitReason::Silence);
    CHECK(segments.front()->audio.size() >= static_cast<size_t>(config.sample_rate * 5.3));
}

TEST_CASE("Realtime VAD reports an explicit maximum duration commit", "[audio][vad]") {
    RealtimeVadConfig config;
    config.sample_rate = 16000;
    config.max_turn_duration_ms = 3000;

    std::vector<float> input;
    append(input, silence(config.sample_rate, 300));
    append(input, tone(config.sample_rate, 4000));

    HLIERealtimeVad vad(config);
    const auto events = runInChunks(vad, input, config.sample_rate / 4);
    const auto segments = eventsOfType(events, RealtimeVadEventType::SegmentReady);
    REQUIRE_FALSE(segments.empty());
    CHECK(segments.front()->reason == RealtimeVadCommitReason::MaxDuration);
    CHECK(segments.front()->audio.size() >= static_cast<size_t>(config.sample_rate * 3.2));
}

TEST_CASE("Realtime VAD manual commit flushes an active segment", "[audio][vad]") {
    RealtimeVadConfig config;
    config.sample_rate = 24000;
    HLIERealtimeVad vad(config);

    std::vector<float> input;
    append(input, silence(config.sample_rate, 300));
    append(input, tone(config.sample_rate, 1000));
    auto events = runInChunks(vad, input, config.sample_rate / 4);
    auto committed = vad.commit();
    events.insert(
        events.end(),
        std::make_move_iterator(committed.begin()),
        std::make_move_iterator(committed.end()));

    const auto segments = eventsOfType(events, RealtimeVadEventType::SegmentReady);
    REQUIRE(segments.size() == 1);
    CHECK(segments.front()->reason == RealtimeVadCommitReason::Manual);
    CHECK_FALSE(segments.front()->audio.empty());
}

TEST_CASE("Realtime VAD preserves session time across sample rate changes", "[audio][vad]") {
    RealtimeVadConfig config;
    config.sample_rate = 16000;
    HLIERealtimeVad vad(config);

    runInChunks(vad, silence(config.sample_rate, 1000), config.sample_rate / 4);
    vad.clear();

    config.sample_rate = 24000;
    std::string error;
    REQUIRE(vad.configure(config, error));

    std::vector<float> input;
    append(input, silence(config.sample_rate, 600));
    append(input, tone(config.sample_rate, 1000));
    append(input, silence(config.sample_rate, 700));
    const auto events = runInChunks(vad, input, config.sample_rate / 4);
    const auto starts = eventsOfType(events, RealtimeVadEventType::SpeechStarted);

    REQUIRE(starts.size() == 1);
    const int64_t start_ms = starts.front()->start_sample * 1000 / config.sample_rate;
    CHECK(start_ms >= 1250);
    CHECK(start_ms <= 1350);
}

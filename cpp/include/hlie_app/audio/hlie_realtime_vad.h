#ifndef HLIE_REALTIME_VAD_H
#define HLIE_REALTIME_VAD_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

struct RealtimeVadConfig {
    int sample_rate = 24000;
    float threshold = 0.5f;
    int frame_duration_ms = 20;
    int prefix_padding_ms = 300;
    // Qwen3-ASR quality degrades on short meeting-speech fragments. Keep a
    // conservative default while still honoring an explicit OpenAI
    // turn_detection.silence_duration_ms value from the client.
    int silence_duration_ms = 2000;
    int min_speech_duration_ms = 120;
    int suffix_padding_ms = 200;
    int max_turn_duration_ms = 30000;
};

enum class RealtimeVadEventType {
    SpeechStarted,
    SpeechStopped,
    SegmentReady,
};

enum class RealtimeVadCommitReason {
    Silence,
    MaxDuration,
    Manual,
};

struct RealtimeVadMetrics {
    float average_level_dbfs = -100.0f;
    float noise_floor_dbfs = -55.0f;
    size_t analyzed_frames = 0;
    size_t speech_frames = 0;
};

struct RealtimeVadEvent {
    RealtimeVadEventType type = RealtimeVadEventType::SegmentReady;
    RealtimeVadCommitReason reason = RealtimeVadCommitReason::Silence;
    int64_t start_sample = 0;
    int64_t end_sample = 0;
    std::vector<float> audio;
    RealtimeVadMetrics metrics;
};

[[nodiscard]] HLIE_API bool validateRealtimeVadConfig(
    const RealtimeVadConfig& config,
    std::string& error);

class HLIE_API HLIERealtimeVad {
public:
    explicit HLIERealtimeVad(const RealtimeVadConfig& config = {});
    ~HLIERealtimeVad();

    HLIERealtimeVad(HLIERealtimeVad&&) noexcept;
    HLIERealtimeVad& operator=(HLIERealtimeVad&&) noexcept;

    HLIERealtimeVad(const HLIERealtimeVad&) = delete;
    HLIERealtimeVad& operator=(const HLIERealtimeVad&) = delete;

    bool configure(const RealtimeVadConfig& config, std::string& error);
    std::vector<RealtimeVadEvent> append(const std::vector<float>& samples);
    std::vector<RealtimeVadEvent> commit();
    void clear();

    [[nodiscard]] bool empty() const;
    [[nodiscard]] bool hasUncommittedAudio() const;
    [[nodiscard]] const RealtimeVadConfig& config() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

HLIE_NAMESPACE_END

#endif // HLIE_REALTIME_VAD_H

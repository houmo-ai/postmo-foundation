#include "hlie_app/audio/hlie_realtime_vad.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <utility>

HLIE_NAMESPACE_BEGIN

namespace {
constexpr float MIN_LEVEL_DBFS = -100.0f;
constexpr float INITIAL_NOISE_FLOOR_DBFS = -55.0f;
constexpr float MIN_NOISE_FLOOR_DBFS = -65.0f;
constexpr float MAX_NOISE_FLOOR_DBFS = -30.0f;

size_t millisecondsToSamples(const int sample_rate, const int milliseconds) {
    return static_cast<size_t>(sample_rate) * static_cast<size_t>(milliseconds) / 1000U;
}

size_t framesForDuration(const int duration_ms, const int frame_duration_ms) {
    return std::max<size_t>(1U, static_cast<size_t>(duration_ms + frame_duration_ms - 1) /
                                   static_cast<size_t>(frame_duration_ms));
}

struct FrameDecision {
    bool speech = false;
    float score = 0.0f;
    float level_dbfs = MIN_LEVEL_DBFS;
    float noise_floor_dbfs = INITIAL_NOISE_FLOOR_DBFS;
};

class AdaptiveEnergyVad {
public:
    explicit AdaptiveEnergyVad(const float threshold) : threshold_(threshold) {
    }

    void setThreshold(const float threshold) {
        threshold_ = threshold;
    }

    void reset() {
        noise_floor_dbfs_ = INITIAL_NOISE_FLOOR_DBFS;
    }

    [[nodiscard]] float noiseFloorDbfs() const {
        return noise_floor_dbfs_;
    }

    FrameDecision analyze(const float* samples, const size_t count, const bool speech_active) {
        FrameDecision result;
        if (!samples || count == 0) {
            return result;
        }

        const double mean = std::accumulate(samples, samples + count, 0.0) /
                            static_cast<double>(count);
        double energy = 0.0;
        for (size_t i = 0; i < count; ++i) {
            const double centered = static_cast<double>(samples[i]) - mean;
            energy += centered * centered;
        }
        const double rms = std::sqrt(energy / static_cast<double>(count));
        result.level_dbfs = rms > 1e-5 ? static_cast<float>(20.0 * std::log10(rms)) : MIN_LEVEL_DBFS;

        const float required_snr_db = 3.0f + std::clamp(threshold_, 0.0f, 1.0f) * 8.0f;
        const float hysteresis_db = speech_active ? 3.0f : 0.0f;
        const float absolute_floor_dbfs = -60.0f + std::clamp(threshold_, 0.0f, 1.0f) * 6.0f;
        const float gate_dbfs = std::max(
            absolute_floor_dbfs,
            noise_floor_dbfs_ + required_snr_db - hysteresis_db);
        result.speech = result.level_dbfs >= gate_dbfs;

        const float snr_db = result.level_dbfs - noise_floor_dbfs_;
        result.score = std::clamp((snr_db - 3.0f) / 12.0f, 0.0f, 1.0f);

        if (!speech_active && !result.speech) {
            const float alpha = result.level_dbfs < noise_floor_dbfs_ ? 0.15f : 0.01f;
            noise_floor_dbfs_ += alpha * (result.level_dbfs - noise_floor_dbfs_);
            noise_floor_dbfs_ = std::clamp(
                noise_floor_dbfs_, MIN_NOISE_FLOOR_DBFS, MAX_NOISE_FLOOR_DBFS);
        }

        result.noise_floor_dbfs = noise_floor_dbfs_;
        return result;
    }

private:
    float threshold_ = 0.5f;
    float noise_floor_dbfs_ = INITIAL_NOISE_FLOOR_DBFS;
};
}

bool validateRealtimeVadConfig(const RealtimeVadConfig& config, std::string& error) {
    if (config.sample_rate < 8000 || config.sample_rate > 192000) {
        error = "VAD sample rate must be between 8000 and 192000 Hz";
        return false;
    }
    if (!std::isfinite(config.threshold) || config.threshold < 0.0f || config.threshold > 1.0f) {
        error = "VAD threshold must be between 0 and 1";
        return false;
    }
    if (config.frame_duration_ms < 10 || config.frame_duration_ms > 30) {
        error = "VAD frame duration must be between 10 and 30 ms";
        return false;
    }
    if (config.prefix_padding_ms < 0 || config.prefix_padding_ms > 5000) {
        error = "VAD prefix padding must be between 0 and 5000 ms";
        return false;
    }
    if (config.silence_duration_ms < 100 || config.silence_duration_ms > 10000) {
        error = "VAD silence duration must be between 100 and 10000 ms";
        return false;
    }
    if (config.min_speech_duration_ms < 40 || config.min_speech_duration_ms > 2000) {
        error = "VAD minimum speech duration must be between 40 and 2000 ms";
        return false;
    }
    if (config.suffix_padding_ms < 0 || config.suffix_padding_ms > config.silence_duration_ms) {
        error = "VAD suffix padding must be between 0 and silence_duration_ms";
        return false;
    }
    if (config.max_turn_duration_ms < 3000 || config.max_turn_duration_ms > 120000) {
        error = "VAD maximum turn duration must be between 3000 and 120000 ms";
        return false;
    }
    error.clear();
    return true;
}

class HLIERealtimeVad::Impl {
public:
    explicit Impl(const RealtimeVadConfig& config)
        : config_(config), detector_(config.threshold) {
        refreshDerivedValues();
    }

    bool configure(const RealtimeVadConfig& config, std::string& error) {
        if (!validateRealtimeVadConfig(config, error)) {
            return false;
        }

        const int previous_sample_rate = config_.sample_rate;
        const bool sample_geometry_changed =
            config.sample_rate != config_.sample_rate ||
            config.frame_duration_ms != config_.frame_duration_ms;
        config_ = config;
        detector_.setThreshold(config.threshold);
        refreshDerivedValues();

        if (sample_geometry_changed) {
            const int64_t converted_processed_samples = previous_sample_rate > 0
                ? static_cast<int64_t>(std::llround(
                    static_cast<double>(processed_samples_) * config_.sample_rate /
                    previous_sample_rate))
                : 0;
            resetState();
            processed_samples_ = converted_processed_samples;
        } else {
            trimPrefix();
        }
        return true;
    }

    std::vector<RealtimeVadEvent> append(const std::vector<float>& samples) {
        std::vector<RealtimeVadEvent> events;
        if (samples.empty()) {
            return events;
        }

        size_t offset = 0;
        while (offset < samples.size()) {
            const size_t copy_count = std::min(frame_samples_ - frame_buffer_.size(), samples.size() - offset);
            frame_buffer_.insert(
                frame_buffer_.end(),
                samples.begin() + static_cast<std::ptrdiff_t>(offset),
                samples.begin() + static_cast<std::ptrdiff_t>(offset + copy_count));
            offset += copy_count;

            if (frame_buffer_.size() == frame_samples_) {
                processFrame(frame_buffer_, events);
                processed_samples_ += static_cast<int64_t>(frame_buffer_.size());
                frame_buffer_.clear();
            }
        }
        return events;
    }

    std::vector<RealtimeVadEvent> commit() {
        std::vector<RealtimeVadEvent> events;

        if (!frame_buffer_.empty()) {
            if (active_) {
                active_audio_.insert(active_audio_.end(), frame_buffer_.begin(), frame_buffer_.end());
                last_voice_end_sample_ = processed_samples_ + static_cast<int64_t>(frame_buffer_.size());
            } else {
                appendPrefix(frame_buffer_);
            }
            processed_samples_ += static_cast<int64_t>(frame_buffer_.size());
            frame_buffer_.clear();
        }

        if (active_) {
            // Client endpointing commits after it has already observed trailing
            // silence. Reuse the normal silence trimming so that the ASR model
            // receives only suffix padding instead of the full endpoint window.
            finalizeSegment(RealtimeVadCommitReason::Manual, events, true);
        } else if (!prefix_audio_.empty()) {
            RealtimeVadEvent segment;
            segment.type = RealtimeVadEventType::SegmentReady;
            segment.reason = RealtimeVadCommitReason::Manual;
            segment.end_sample = processed_samples_;
            segment.start_sample = segment.end_sample - static_cast<int64_t>(prefix_audio_.size());
            segment.audio.assign(prefix_audio_.begin(), prefix_audio_.end());
            events.push_back(std::move(segment));
            prefix_audio_.clear();
            candidate_speech_frames_ = 0;
        }
        return events;
    }

    void clear() {
        resetState();
    }

    bool empty() const {
        return frame_buffer_.empty() && prefix_audio_.empty() && active_audio_.empty();
    }

    bool hasUncommittedAudio() const {
        return active_ || candidate_speech_frames_ > 0 || !frame_buffer_.empty();
    }

    const RealtimeVadConfig& config() const {
        return config_;
    }

private:
    void refreshDerivedValues() {
        frame_samples_ = std::max<size_t>(
            1U, millisecondsToSamples(config_.sample_rate, config_.frame_duration_ms));
        prefix_samples_ = millisecondsToSamples(config_.sample_rate, config_.prefix_padding_ms);
        suffix_samples_ = millisecondsToSamples(config_.sample_rate, config_.suffix_padding_ms);
        start_trigger_frames_ = framesForDuration(
            config_.min_speech_duration_ms, config_.frame_duration_ms);
        prefix_capacity_samples_ = prefix_samples_ + start_trigger_frames_ * frame_samples_;
        silence_trigger_frames_ = framesForDuration(
            config_.silence_duration_ms, config_.frame_duration_ms);
        max_turn_samples_ = millisecondsToSamples(config_.sample_rate, config_.max_turn_duration_ms);
    }

    void resetState() {
        frame_buffer_.clear();
        prefix_audio_.clear();
        active_audio_.clear();
        active_ = false;
        candidate_speech_frames_ = 0;
        silence_frames_ = 0;
        segment_start_sample_ = 0;
        speech_start_sample_ = 0;
        last_voice_end_sample_ = 0;
        level_sum_dbfs_ = 0.0;
        analyzed_frames_ = 0;
        speech_frames_ = 0;
        detector_.reset();
    }

    void trimPrefix() {
        while (prefix_audio_.size() > prefix_capacity_samples_) {
            prefix_audio_.pop_front();
        }
    }

    void appendPrefix(const std::vector<float>& samples) {
        prefix_audio_.insert(prefix_audio_.end(), samples.begin(), samples.end());
        trimPrefix();
    }

    void appendPrefix(const std::vector<float>& samples, const size_t offset) {
        if (offset >= samples.size()) {
            return;
        }
        prefix_audio_.insert(
            prefix_audio_.end(),
            samples.begin() + static_cast<std::ptrdiff_t>(offset),
            samples.end());
        trimPrefix();
    }

    void processFrame(const std::vector<float>& frame, std::vector<RealtimeVadEvent>& events) {
        const FrameDecision decision = detector_.analyze(frame.data(), frame.size(), active_);
        const int64_t frame_end_sample = processed_samples_ + static_cast<int64_t>(frame.size());

        if (!active_) {
            appendPrefix(frame);
            candidate_speech_frames_ = decision.speech ? candidate_speech_frames_ + 1U : 0U;
            if (candidate_speech_frames_ < start_trigger_frames_) {
                return;
            }

            active_ = true;
            silence_frames_ = 0;
            speech_frames_ = candidate_speech_frames_;
            analyzed_frames_ = candidate_speech_frames_;
            level_sum_dbfs_ = static_cast<double>(decision.level_dbfs) * candidate_speech_frames_;
            speech_start_sample_ = frame_end_sample -
                static_cast<int64_t>(candidate_speech_frames_ * frame_samples_);
            segment_start_sample_ = frame_end_sample - static_cast<int64_t>(prefix_audio_.size());
            last_voice_end_sample_ = frame_end_sample;
            active_audio_.assign(prefix_audio_.begin(), prefix_audio_.end());
            prefix_audio_.clear();
            candidate_speech_frames_ = 0;

            RealtimeVadEvent started;
            started.type = RealtimeVadEventType::SpeechStarted;
            started.start_sample = segment_start_sample_;
            started.end_sample = frame_end_sample;
            events.push_back(std::move(started));
            return;
        }

        active_audio_.insert(active_audio_.end(), frame.begin(), frame.end());
        ++analyzed_frames_;
        level_sum_dbfs_ += decision.level_dbfs;

        if (decision.speech) {
            ++speech_frames_;
            silence_frames_ = 0;
            last_voice_end_sample_ = frame_end_sample;
        } else {
            ++silence_frames_;
        }

        if (silence_frames_ >= silence_trigger_frames_) {
            finalizeSegment(RealtimeVadCommitReason::Silence, events, true);
            return;
        }

        if (frame_end_sample - speech_start_sample_ >= static_cast<int64_t>(max_turn_samples_)) {
            finalizeSegment(RealtimeVadCommitReason::MaxDuration, events, false);
        }
    }

    void finalizeSegment(
        const RealtimeVadCommitReason reason,
        std::vector<RealtimeVadEvent>& events,
        const bool trim_silence) {
        if (!active_) {
            return;
        }

        if (trim_silence && silence_frames_ > 0) {
            const size_t silence_samples = std::min(
                active_audio_.size(), silence_frames_ * frame_samples_);
            const size_t retained_silence = std::min(silence_samples, suffix_samples_);
            const size_t trim_count = silence_samples - retained_silence;
            if (trim_count > 0) {
                const size_t trim_offset = active_audio_.size() - trim_count;
                appendPrefix(active_audio_, trim_offset);
                active_audio_.resize(trim_offset);
            }
        }

        RealtimeVadMetrics metrics;
        metrics.average_level_dbfs = analyzed_frames_ > 0
            ? static_cast<float>(level_sum_dbfs_ / static_cast<double>(analyzed_frames_))
            : MIN_LEVEL_DBFS;
        metrics.noise_floor_dbfs = detector_.noiseFloorDbfs();
        metrics.analyzed_frames = analyzed_frames_;
        metrics.speech_frames = speech_frames_;

        RealtimeVadEvent stopped;
        stopped.type = RealtimeVadEventType::SpeechStopped;
        stopped.reason = reason;
        stopped.start_sample = segment_start_sample_;
        stopped.end_sample = last_voice_end_sample_;
        stopped.metrics = metrics;
        events.push_back(std::move(stopped));

        RealtimeVadEvent segment;
        segment.type = RealtimeVadEventType::SegmentReady;
        segment.reason = reason;
        segment.start_sample = segment_start_sample_;
        segment.end_sample = segment_start_sample_ + static_cast<int64_t>(active_audio_.size());
        segment.audio = std::move(active_audio_);
        segment.metrics = metrics;
        events.push_back(std::move(segment));

        active_audio_.clear();
        active_ = false;
        candidate_speech_frames_ = 0;
        silence_frames_ = 0;
        analyzed_frames_ = 0;
        speech_frames_ = 0;
        level_sum_dbfs_ = 0.0;
    }

    RealtimeVadConfig config_;
    AdaptiveEnergyVad detector_;
    size_t frame_samples_ = 0;
    size_t prefix_samples_ = 0;
    size_t prefix_capacity_samples_ = 0;
    size_t suffix_samples_ = 0;
    size_t start_trigger_frames_ = 0;
    size_t silence_trigger_frames_ = 0;
    size_t max_turn_samples_ = 0;

    int64_t processed_samples_ = 0;
    int64_t segment_start_sample_ = 0;
    int64_t speech_start_sample_ = 0;
    int64_t last_voice_end_sample_ = 0;
    std::vector<float> frame_buffer_;
    std::deque<float> prefix_audio_;
    std::vector<float> active_audio_;
    bool active_ = false;
    size_t candidate_speech_frames_ = 0;
    size_t silence_frames_ = 0;
    double level_sum_dbfs_ = 0.0;
    size_t analyzed_frames_ = 0;
    size_t speech_frames_ = 0;
};

HLIERealtimeVad::HLIERealtimeVad(const RealtimeVadConfig& config)
    : impl_(std::make_unique<Impl>(config)) {
}

HLIERealtimeVad::~HLIERealtimeVad() = default;

HLIERealtimeVad::HLIERealtimeVad(HLIERealtimeVad&&) noexcept = default;

HLIERealtimeVad& HLIERealtimeVad::operator=(HLIERealtimeVad&&) noexcept = default;

bool HLIERealtimeVad::configure(const RealtimeVadConfig& config, std::string& error) {
    return impl_->configure(config, error);
}

std::vector<RealtimeVadEvent> HLIERealtimeVad::append(const std::vector<float>& samples) {
    return impl_->append(samples);
}

std::vector<RealtimeVadEvent> HLIERealtimeVad::commit() {
    return impl_->commit();
}

void HLIERealtimeVad::clear() {
    impl_->clear();
}

bool HLIERealtimeVad::empty() const {
    return impl_->empty();
}

bool HLIERealtimeVad::hasUncommittedAudio() const {
    return impl_->hasUncommittedAudio();
}

const RealtimeVadConfig& HLIERealtimeVad::config() const {
    return impl_->config();
}

HLIE_NAMESPACE_END

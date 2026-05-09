#include "hlie_base/hlie_audio.h"

#define _USE_MATH_DEFINES // for M_PI
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "ggml.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/librosa.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

HLIE_NAMESPACE_BEGIN


bool HLIEAudio::LoadAudio(const std::string& audio_path, int target_sr) {

    return true;
}

float HLIEAudio::GetDuration() const {
    if (sample_rate_ <= 0)
        return 0.0f;
    return static_cast<float>(total_samples_) / sample_rate_;
}

MelFeatures80 HLIEAudio::ComputeMelSpectrogram80(const std::vector<float>& pcm_data) {
    MelFeatures80 mel;
    mel.n_mels = kNMels80;

    int pad_len = (kNFFT24k - kHopSize24k) / 2;
    std::vector<float> librosa_input = pcm_data.empty() ? std::vector<float>(kNFFT24k, 0.0f) : ApplyReflectionPadding(pcm_data, pad_len);
    auto mel_frames = librosa::Feature::melspectrogram(librosa_input, kSampleRate24k, kNFFT24k, kHopSize24k, "hann", false, "reflect", 1.0f,
                                                       kNMels80, 0, kSampleRate24k / 2);

    mel.n_frames = std::max(1, static_cast<int>(mel_frames.size()));
    mel.data.resize(mel.n_mels * mel.n_frames);

    if (mel_frames.empty()) {
        std::fill(mel.data.begin(), mel.data.end(), static_cast<TensorFP16>(std::log(1e-5f)));
        return mel;
    }

    for (int t = 0; t < mel.n_frames; ++t) {
        const auto& frame = mel_frames[t];
        int mel_bins = std::min(mel.n_mels, static_cast<int>(frame.size()));
        for (int m = 0; m < mel_bins; ++m) {
            float log_spec = std::log(std::max(frame[m], 1e-5f));
            mel.data[t * mel.n_mels + m] = log_spec;
        }
        for (int m = mel_bins; m < mel.n_mels; ++m) {
            mel.data[t * mel.n_mels + m] = std::log(1e-5f);
        }
    }

    return mel;
}

MelFeatures128 HLIEAudio::ComputeMelSpectrogram128(const std::vector<float>& pcm_data) {
    MelFeatures128 mel;
    mel.n_mels = kNMels128;

    std::vector<float> librosa_input = pcm_data.empty() ? std::vector(kNFFT16k, 0.0f) : std::vector(pcm_data.begin(), pcm_data.end());
    const auto mel_frames = librosa::Feature::melspectrogram(librosa_input, kSampleRate16k, kNFFT16k, kHopSize16k, "hann", true, "reflect",
                                                             2.0f, kNMels128, 0, kSampleRate16k / 2);

    mel.n_frames = std::max(1, static_cast<int>(mel_frames.size()));
    std::vector<float> tmp_mel_data(mel.n_mels * mel.n_frames, 0.0f);
    float max_log_spec = -1e20f;

    for (int t = 0; t < mel.n_frames; ++t) {
        const auto& frame = mel_frames[t];
        const int mel_bins = std::min(mel.n_mels, static_cast<int>(frame.size()));
        for (int m = 0; m < mel_bins; ++m) {
            const float log_spec = std::log10(std::max(frame[m], 1e-10f));
            tmp_mel_data[m * mel.n_frames + t] = log_spec;
            if (log_spec > max_log_spec) {
                max_log_spec = log_spec;
            }
        }
        for (int m = mel_bins; m < mel.n_mels; ++m) {
            float log_spec = std::log10(1e-10f);
            tmp_mel_data[m * mel.n_frames + t] = log_spec;
            if (log_spec > max_log_spec) {
                max_log_spec = log_spec;
            }
        }
    }

    // Whisper-style normalization
    mel.data.resize(mel.n_mels * mel.n_frames);
    for (int i = 0; i < mel.n_mels * mel.n_frames; i++) {
        float val = tmp_mel_data[i];
        val = std::max(val, max_log_spec - 8.0f);
        mel.data[i] = static_cast<TensorFP16>((val + 4.0f) / 4.0f);
    }

    return mel;
}

FbankFeatures HLIEAudio::ComputeFbank(const std::vector<float>& pcm_data) {
    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = kSampleRate16k; // sample_frequency=16000
    opts.frame_opts.dither = 0.0f; // dither=0
    opts.mel_opts.num_bins = kFbankNMels;
    // torchaudio defaults snip_edges=True, kaldi-native-fbank also defaults to True
    opts.frame_opts.snip_edges = true;

    // 2. Initialize Fbank calculator
    knf::OnlineFbank fbank(opts);

    fbank.AcceptWaveform(kSampleRate16k, pcm_data.data(), pcm_data.size());
    fbank.InputFinished();

    // 4. Get feature matrix
    const int num_frames = fbank.NumFramesReady();
    const int feat_dim = opts.mel_opts.num_bins; // 80

    FbankFeatures features;
    features.n_frames = num_frames;
    features.n_mels = feat_dim;
    features.data.reserve(num_frames * feat_dim);

    // Put features of all frames continuously into vector
    for (int i = 0; i < num_frames; ++i) {
        const float* frame = fbank.GetFrame(i); // Returns float pointer of single frame (size is n_mels)
        features.data.insert(features.data.end(), frame, frame + feat_dim);
    }
    return features;
}

MelFeatures80 HLIEAudio::GetMelFeatures80() const {
    if (!audio_loaded_ || sample_rate_ != kSampleRate24k) {
        LOG_ERROR("Audio not loaded at 24kHz, returning empty features");
        return MelFeatures80();
    }

    return ComputeMelSpectrogram80(pcm_data_);
}

MelFeatures128 HLIEAudio::GetMelFeatures128() const {
    if (!audio_loaded_ || sample_rate_ != kSampleRate16k) {
        LOG_ERROR("Audio not loaded at 16kHz, returning empty features");
        return MelFeatures128();
    }

    return ComputeMelSpectrogram128(pcm_data_);
}

FbankFeatures HLIEAudio::GetFbankFeatures() const {
    if (!audio_loaded_ || sample_rate_ != kSampleRate16k) {
        LOG_ERROR("Audio not loaded at 16kHz, returning empty features");
        return FbankFeatures();
    }

    return ComputeFbank(pcm_data_);
}

std::vector<float> HLIEAudio::ComputeHannWindow(int win_size) {
    std::vector<float> window(win_size);
    for (int i = 0; i < win_size; i++) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (win_size - 1)));
    }
    return window;
}

std::vector<float> HLIEAudio::ApplyReflectionPadding(const std::vector<float>& audio, const int pad_len) {
    if (audio.empty())
        return audio;

    std::vector<float> padded(audio.size() + 2 * pad_len);

    // Reflection padding for start
    for (int i = 0; i < pad_len; i++) {
        int src_idx = std::min(i + 1, static_cast<int>(audio.size()) - 1);
        padded[pad_len - 1 - i] = audio[src_idx];
    }

    // Original audio
    for (size_t i = 0; i < audio.size(); i++) {
        padded[pad_len + i] = audio[i];
    }

    // Reflection padding for end
    const int audio_size = static_cast<int>(audio.size());
    for (int i = 0; i < pad_len; i++) {
        const int src_idx = std::max(audio_size - 2 - i, 0);
        padded[pad_len + audio_size + i] = audio[src_idx];
    }

    return padded;
}

float HLIEAudio::HzToMel(const float hz) {
    // HTK-style mel scale: mel = 2595 * log10(1 + hz/700)
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float HLIEAudio::MelToHz(const float mel) {
    // HTK-style inverse: hz = 700 * (10^(mel/2595) - 1)
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

HLIE_NAMESPACE_END

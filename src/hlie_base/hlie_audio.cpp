#include "hlie_base/hlie_audio.h"

#define _USE_MATH_DEFINES // for M_PI
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ggml.h"
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#include "hlie_base/hlie_logger.h"
#include "hlie_base/librosa.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

HLIE_NAMESPACE_BEGIN

namespace {
std::vector<float> resample_torchaudio_sinc(const std::vector<float>& input,
                                            const int source_rate,
                                            const int target_rate) {
    if (source_rate == target_rate) {
        return input;
    }
    if (input.empty() || source_rate <= 0 || target_rate <= 0) {
        return {};
    }

    constexpr int kLowpassFilterWidth = 6;
    constexpr double kRolloff = 0.99;
    const int divisor = std::gcd(source_rate, target_rate);
    const int source = source_rate / divisor;
    const int target = target_rate / divisor;
    const double base_frequency =
        static_cast<double>(std::min(source, target)) * kRolloff;
    const int width = static_cast<int>(std::ceil(
        static_cast<double>(kLowpassFilterWidth * source) / base_frequency));
    const int kernel_length = 2 * width + source;

    std::vector<float> kernels(
        static_cast<size_t>(target) * static_cast<size_t>(kernel_length));
    for (int phase = 0; phase < target; ++phase) {
        for (int kernel_index = 0; kernel_index < kernel_length; ++kernel_index) {
            const int sample_offset = kernel_index - width;
            double value = -static_cast<double>(phase) / target +
                           static_cast<double>(sample_offset) / source;
            value *= base_frequency;
            value = std::clamp(value, -static_cast<double>(kLowpassFilterWidth),
                               static_cast<double>(kLowpassFilterWidth));
            const double window = std::pow(
                std::cos(value * M_PI / kLowpassFilterWidth / 2.0), 2.0);
            const double radians = value * M_PI;
            const double sinc = radians == 0.0 ? 1.0 : std::sin(radians) / radians;
            kernels[static_cast<size_t>(phase) * kernel_length + kernel_index] =
                static_cast<float>(sinc * window * base_frequency / source);
        }
    }

    const size_t target_length = static_cast<size_t>(std::ceil(
        static_cast<double>(target) * input.size() / source));
    std::vector<float> output;
    output.reserve(target_length);
    const size_t convolution_steps = input.size() / static_cast<size_t>(source) + 1;
    for (size_t step = 0; step < convolution_steps && output.size() < target_length;
         ++step) {
        const int64_t padded_start =
            static_cast<int64_t>(step * static_cast<size_t>(source)) - width;
        for (int phase = 0; phase < target && output.size() < target_length; ++phase) {
            float sum = 0.0F;
            const float* kernel = kernels.data() +
                static_cast<size_t>(phase) * static_cast<size_t>(kernel_length);
            for (int index = 0; index < kernel_length; ++index) {
                const int64_t source_index = padded_start + index;
                if (source_index >= 0 &&
                    source_index < static_cast<int64_t>(input.size())) {
                    sum += input[static_cast<size_t>(source_index)] * kernel[index];
                }
            }
            output.push_back(sum);
        }
    }
    return output;
}
}


bool HLIEAudio::LoadAudio(const std::string& audio_path, int target_sr) {
    if (audio_path.empty() || target_sr <= 0) {
        LOG_ERROR("Invalid audio path or target sample rate: path={}, sample_rate={}",
                  audio_path, target_sr);
        return false;
    }

    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, 1, 0);
    ma_decoder decoder{};
    const ma_result init_result =
        ma_decoder_init_file(audio_path.c_str(), &decoder_config, &decoder);
    if (init_result != MA_SUCCESS) {
        LOG_ERROR("Failed to decode audio {}: {}", audio_path,
                  ma_result_description(init_result));
        return false;
    }

    ma_format output_format = ma_format_unknown;
    ma_uint32 output_channels = 0;
    ma_uint32 source_rate = 0;
    if (ma_decoder_get_data_format(&decoder, &output_format, &output_channels,
                                   &source_rate, nullptr, 0) != MA_SUCCESS ||
        output_format != ma_format_f32 || output_channels != 1 || source_rate == 0) {
        LOG_ERROR("Failed to determine decoded audio format: {}", audio_path);
        ma_decoder_uninit(&decoder);
        return false;
    }

    ma_uint64 frame_count = 0;
    ma_uint64 frames_read = 0;
    const ma_result length_result =
        ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (length_result != MA_SUCCESS || frame_count == 0 ||
        frame_count > static_cast<ma_uint64>(std::numeric_limits<size_t>::max())) {
        LOG_ERROR("Failed to determine audio length {}: {}", audio_path,
                  ma_result_description(length_result));
        ma_decoder_uninit(&decoder);
        return false;
    }

    std::vector<float> samples(static_cast<size_t>(frame_count));
    const ma_result read_result = ma_decoder_read_pcm_frames(
        &decoder, samples.data(), frame_count, &frames_read);
    ma_decoder_uninit(&decoder);
    if (read_result != MA_SUCCESS || frames_read == 0) {
        LOG_ERROR("Failed to read audio {}: {}", audio_path,
                  ma_result_description(read_result));
        return false;
    }
    samples.resize(static_cast<size_t>(frames_read));
    pcm_data_ = resample_torchaudio_sinc(
        samples, static_cast<int>(source_rate), target_sr);
    if (pcm_data_.empty()) {
        LOG_ERROR("Failed to resample audio {} from {} Hz to {} Hz", audio_path,
                  source_rate, target_sr);
        return false;
    }
    sample_rate_ = target_sr;
    total_samples_ = static_cast<int>(pcm_data_.size());
    audio_loaded_ = true;
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

std::vector<TensorFP16> HLIEAudio::ComputeQwen3TTSSpeakerMel(
    const std::vector<float>& pcm_data, const int target_frames,
    const int target_mels) {
    if (target_frames <= 0 || target_mels <= 0) {
        throw std::invalid_argument("Qwen3-TTS speaker mel shape must be positive");
    }

    constexpr int kSampleRate = 24000;
    constexpr int kNfft = 1024;
    constexpr int kHop = 256;
    constexpr int kWin = 1024;
    constexpr int kFmin = 0;
    constexpr int kFmax = 12000;
    constexpr int kPadding = (kNfft - kHop) / 2;

    std::vector<float> input = pcm_data;
    if (input.empty()) {
        input.assign(kNfft, 0.0F);
    }
    input = ApplyReflectionPadding(input, kPadding);
    const auto mel_frames = librosa::Feature::melspectrogram(
        input, kSampleRate, kNfft, kHop, "hann", false, "reflect", 1.0F,
        target_mels, kFmin, kFmax);

    std::vector<TensorFP16> output(
        static_cast<size_t>(target_frames) * static_cast<size_t>(target_mels),
        TensorFP16(0.0F));
    const int valid_frames = std::min(target_frames,
                                      static_cast<int>(mel_frames.size()));
    for (int frame = 0; frame < valid_frames; ++frame) {
        const int valid_mels = std::min(target_mels,
                                        static_cast<int>(mel_frames[frame].size()));
        for (int mel = 0; mel < valid_mels; ++mel) {
            const float value = std::log(std::max(mel_frames[frame][mel], 1.0e-5F));
            output[static_cast<size_t>(frame) * static_cast<size_t>(target_mels) +
                   static_cast<size_t>(mel)] = TensorFP16(value);
        }
    }
    return output;
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

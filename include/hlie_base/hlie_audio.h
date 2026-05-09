#ifndef HLIE_AUDIO_H
#define HLIE_AUDIO_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "half.hpp"
#include "hlie_tcim_util.h"

#define hlie_INTERNAL_HEADER

HLIE_NAMESPACE_BEGIN

/**
 * @brief Mel spectrogram features for Flow/Vocoder (80-bin at 24kHz)
 */
struct MelFeatures80 {
    std::vector<float> data; ///< Mel features [n_mels * n_frames] float16
    int n_mels; ///< Number of mel bins (80)
    int n_frames; ///< Number of time frames
};

/**
 * @brief Mel spectrogram features for Speech Tokenizer (128-bin at 16kHz)
 */
struct MelFeatures128 {
    std::vector<TensorFP16> data; ///< Mel features [n_mels * n_frames] float16
    int n_mels; ///< Number of mel bins (128)
    int n_frames; ///< Number of time frames
};

/**
 * @brief FBANK features for CampPlus (80-bin at 16kHz)
 */
struct FbankFeatures {
    std::vector<float> data; ///< FBANK features [n_frames * n_mels] float32
    int n_mels; ///< Number of mel bins (80)
    int n_frames; ///< Number of time frames
};

/**
 * @brief Audio preprocessing class for CosyVoice3 TTS
 *
 * Handles:
 * - Audio file loading (any format via miniaudio)
 * - Automatic resampling to target sample rate (16kHz or 24kHz)
 * - Mono conversion
 * - Mel spectrogram computation (80-bin for Flow/Vocoder at 24kHz)
 * - Mel spectrogram computation (128-bin for Speech Tokenizer at 16kHz)
 * - FBANK feature computation (80-bin for CampPlus at 16kHz)
 */
class HLIEAudio {
public:
    // CosyVoice3 audio processing constants

    // Sample rates
    static constexpr int kSampleRate16k = 16000; ///< Sample rate for tokenizer/campplus
    static constexpr int kSampleRate24k = 24000; ///< Sample rate for vocoder

    // Mel params for 24kHz (Flow/Vocoder)
    static constexpr int kNFFT24k = 1920; ///< FFT size for 24kHz mel
    static constexpr int kHopSize24k = 480; ///< Hop size for 24kHz mel (20ms stride)
    static constexpr int kWinSize24k = 1920; ///< Window size for 24kHz mel
    static constexpr int kNMels80 = 80; ///< Mel bins for Flow/Vocoder
    static constexpr int kNFFTBins24k = 961; ///< FFT output bins (n_fft/2 + 1)

    // Mel params for 16kHz (Speech Tokenizer - Whisper style)
    static constexpr int kNFFT16k = 400; ///< FFT size for 16kHz mel
    static constexpr int kHopSize16k = 160; ///< Hop size for 16kHz mel (10ms stride)
    static constexpr int kNMels128 = 128; ///< Mel bins for Speech Tokenizer
    static constexpr int kNFFTBins16k = 201; ///< FFT output bins (n_fft/2 + 1)

    // FBANK params for 16kHz (CampPlus)
    static constexpr int kFbankNMels = 80; ///< FBANK mel bins
    static constexpr int kFbankFixedT = 1000; ///< Fixed time frames for CampPlus

    /**
     * @brief Default constructor
     */
    HLIEAudio() : sample_rate_(0), total_samples_(0), audio_loaded_(false) {
    }

    // Non-copyable
    HLIEAudio(const HLIEAudio&) = delete;
    HLIEAudio& operator=(const HLIEAudio&) = delete;

    // Moveable
    HLIEAudio(HLIEAudio&&) noexcept = default;
    HLIEAudio& operator=(HLIEAudio&&) noexcept = default;

    ~HLIEAudio() = default;

    /**
     * @brief Load audio file, convert to mono, resample to target sample rate
     * @param audio_path Path to audio file (WAV/MP3/FLAC etc.)
     * @param target_sr Target sample rate (16000 or 24000)
     * @return true if successful
     */
    bool LoadAudio(const std::string& audio_path, int target_sr);

    /**
     * @brief Get loaded PCM data
     * @return PCM samples as float32 vector
     */
    [[nodiscard]] const std::vector<float>& GetPcmData() const {
        return pcm_data_;
    }

    /**
     * @brief Get sample rate of loaded audio
     */
    [[nodiscard]] int GetSampleRate() const {
        return sample_rate_;
    }

    /**
     * @brief Get number of samples
     */
    [[nodiscard]] int GetNumSamples() const {
        return total_samples_;
    }

    /**
     * @brief Get audio duration in seconds
     */
    [[nodiscard]] float GetDuration() const;

    /**
     * @brief Compute 80-bin mel spectrogram for Flow/Vocoder (24kHz)
     * @param pcm_data Input PCM samples (24kHz mono)
     * @return MelFeatures80 with mel spectrogram [n_mels, n_frames]
     */
    [[nodiscard]] static MelFeatures80 ComputeMelSpectrogram80(const std::vector<float>& pcm_data);

    /**
     * @brief Compute 128-bin mel spectrogram for Speech Tokenizer (16kHz)
     *        Uses Whisper-style parameters
     * @param pcm_data Input PCM samples (16kHz mono)
     * @return MelFeatures128 with mel spectrogram [n_mels, n_frames]
     */
    [[nodiscard]] static MelFeatures128 ComputeMelSpectrogram128(const std::vector<float>& pcm_data);

    /**
     * @brief Compute FBANK features for CampPlus (16kHz)
     * @param pcm_data Input PCM samples (16kHz mono)
     * @return FbankFeatures with FBANK features [n_frames, n_mels]
     */
    [[nodiscard]] static FbankFeatures ComputeFbank(const std::vector<float>& pcm_data);

    /**
     * @brief Get mel features for loaded audio (80-bin, 24kHz)
     */
    [[nodiscard]] MelFeatures80 GetMelFeatures80() const;

    /**
     * @brief Get mel features for loaded audio (128-bin, 16kHz)
     */
    [[nodiscard]] MelFeatures128 GetMelFeatures128() const;

    /**
     * @brief Get FBANK features for loaded audio (16kHz)
     */
    [[nodiscard]] FbankFeatures GetFbankFeatures() const;

private:
    std::vector<float> pcm_data_;
    int sample_rate_;
    int total_samples_;
    bool audio_loaded_;

    // Helper functions
    [[nodiscard]] static std::vector<float> ComputeHannWindow(int win_size);
    [[nodiscard]] static std::vector<float> ApplyReflectionPadding(const std::vector<float>& audio, int pad_len);

    // Hz to mel conversion
    [[nodiscard]] static float HzToMel(float hz);
    [[nodiscard]] static float MelToHz(float mel);
};

HLIE_NAMESPACE_END

#endif

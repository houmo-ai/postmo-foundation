#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

#define hlie_INTERNAL_HEADER


struct audio_hparams {
    // audio
    int32_t n_mel_bins = 0; // whisper preprocessor

    // audio-to-mel preprocessor params
    int32_t audio_chunk_len   = -1; // in seconds
    int32_t audio_sample_rate = -1;
    int32_t audio_n_fft       = -1;
    int32_t audio_window_len  = -1;
    int32_t audio_hop_len     = -1;
};

struct hlie_audio_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct hlie_audio_mel_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

// cache for audio processing, each processor instance owns its own cache
struct hlie_audio_cache {
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;

    std::vector<float> hann_window;

    hlie_audio_mel_filters filters;

    void fill_sin_cos_table(int n);

    void fill_hann_window(int length, bool periodic);

    // Build mel filterbank matrix [n_mel × n_fft_bins] at runtime.
    // n_fft_bins must be (N_fft / 2 + 1). Example: if N_fft=512 -> n_fft_bins=257.
    void fill_mel_filterbank_matrix(int   n_mel,
                                    int   n_fft,
                                    int   sample_rate,               // e.g. 16000
                                    float fmin             = 0.0f,   // e.g. 0.0
                                    float fmax             = -1.0f,  // e.g. sr/2; pass -1 for auto
                                    bool  slaney_area_norm = true,
                                    float scale = 1.0f  // optional extra scaling
    );
};

struct hlie_audio_preprocessor {
    const audio_hparams & hparams;

    hlie_audio_preprocessor(const audio_hparams& ctx): hparams(ctx) {}

    virtual ~hlie_audio_preprocessor() = default;
    virtual void initialize() = 0; // NOT thread-safe
    virtual bool preprocess(const float * samples, size_t n_samples, std::vector<hlie_audio_mel> & output) = 0;
};

struct hlie_audio_preprocessor_whisper : hlie_audio_preprocessor {
    hlie_audio_preprocessor_whisper(const audio_hparams& ctx) : hlie_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<hlie_audio_mel> & output) override;

  private:
    hlie_audio_cache cache;
};

// NEW: Match librosa/speech specific mel_spectrogram implementation
// Key differences from whisper:
// - Uses magnitude (sqrt) instead of power
// - Symmetric reflect padding: pad = (n_fft - hop_size) / 2
// - Librosa-style mel filterbank with Slaney normalization
struct hlie_audio_preprocessor_librosa : hlie_audio_preprocessor {
    hlie_audio_preprocessor_librosa(const audio_hparams& ctx) : hlie_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<hlie_audio_mel> & output) override;

private:
    hlie_audio_cache cache;
};

// struct hlie_audio_preprocessor_conformer : hlie_audio_preprocessor {
//     hlie_audio_preprocessor_conformer(const clip_ctx * ctx) : hlie_audio_preprocessor(ctx) {}
//     void initialize() override;
//     bool preprocess(const float * samples, size_t n_samples, std::vector<hlie_audio_mel> & output) override;
//
//   private:
//     hlie_audio_cache cache;
// };

//
// streaming ISTFT - converts spectrogram frames back to audio one frame at a time
//
struct hlie_audio_streaming_istft {
    hlie_audio_streaming_istft(int n_fft, int hop_length);

    // reset streaming state
    void reset();

    // process a single STFT frame (streaming)
    // frame_spectrum: [n_fft_bins x 2] interleaved real/imag
    // returns: up to hop_length samples
    std::vector<float> process_frame(const float * frame_spectrum);

    // flush remaining samples at end of stream
    std::vector<float> flush();

  private:
    int n_fft;
    int hop_length;
    int n_fft_bins;

    // Own cache for output processing
    hlie_audio_cache cache;

    // Streaming state
    std::vector<float> overlap_buffer;
    std::vector<float> window_sum_buffer;
    int                padding_to_remove;

    // Working buffers for IFFT
    std::vector<float> ifft_in;
    std::vector<float> ifft_out;
};


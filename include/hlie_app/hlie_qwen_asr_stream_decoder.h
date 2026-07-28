#ifndef HLIE_QWEN_ASR_STREAM_DECODER_H
#define HLIE_QWEN_ASR_STREAM_DECODER_H

#include <functional>
#include <string>
#include <string_view>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

// The bundled llama.cpp transcription adapter currently parses multipart-style
// options as strings even when the public OpenAI-compatible value is boolean.
inline constexpr std::string_view LLAMA_TRANSCRIPTION_SDK_STREAM_ENABLED = "true";

class HLIEQwenAsrStreamDecoder {
public:
    using DeltaCallback = std::function<void(const std::string&)>;

    explicit HLIEQwenAsrStreamDecoder(
        bool sse_transport,
        DeltaCallback callback = {},
        bool filter_qwen_markers = true);

    void push(std::string_view chunk);
    std::string finish();

    [[nodiscard]] const std::string& transcript() const;
    [[nodiscard]] bool saw_stream_delta() const;
    [[nodiscard]] bool final_text_mismatch() const;

    static std::string clean_final_text(std::string_view text);

private:
    void consume_sse_events(bool flush);
    void consume_sse_event(std::string_view event);
    void consume_payload(std::string_view payload);
    void push_model_text(std::string_view text);
    void push_transcript_text(std::string_view text);
    void emit(std::string_view text);
    void flush_model_text();

    bool sse_transport_ = false;
    bool filter_qwen_markers_ = true;
    bool prefix_resolved_ = false;
    bool end_marker_seen_ = false;
    bool saw_stream_delta_ = false;
    bool finished_ = false;
    bool final_text_mismatch_ = false;
    DeltaCallback callback_;
    std::string transport_buffer_;
    std::string prefix_buffer_;
    std::string transcript_buffer_;
    std::string transcript_;
    std::string final_text_;
};

HLIE_NAMESPACE_END

#endif

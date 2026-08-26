#include "hlie_app/hlie_qwen_asr_stream_decoder.h"

#include <algorithm>
#include <cctype>
#include <utility>

#include <nlohmann/json.hpp>

HLIE_NAMESPACE_BEGIN

namespace {
    using json = nlohmann::json;

    constexpr std::string_view ASR_TEXT_BEGIN = "<asr_text>";
    constexpr std::string_view ASR_TEXT_END = "</asr_text>";
    constexpr std::string_view LANGUAGE_PREFIX = "language ";
    constexpr size_t MAX_PREFIX_BUFFER = 256;

    std::string_view trim_view(std::string_view value) {
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
            value.remove_prefix(1);
        }
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
            value.remove_suffix(1);
        }
        return value;
    }

    std::string json_string(const json& value, const char* key) {
        if (value.contains(key) && value.at(key).is_string()) {
            return value.at(key).get<std::string>();
        }
        return {};
    }

    std::string extract_chat_delta(const json& value) {
        if (!value.contains("choices") || !value.at("choices").is_array() || value.at("choices").empty()) {
            return {};
        }
        const auto& choice = value.at("choices").front();
        if (!choice.is_object() || !choice.contains("delta") || !choice.at("delta").is_object()) {
            return {};
        }
        return json_string(choice.at("delta"), "content");
    }

    bool could_be_qwen_prefix(std::string_view value) {
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
            value.remove_prefix(1);
        }
        if (value.empty()) {
            return true;
        }
        if (value.size() <= LANGUAGE_PREFIX.size()) {
            return LANGUAGE_PREFIX.compare(0, value.size(), value) == 0;
        }
        return value.substr(0, LANGUAGE_PREFIX.size()) == LANGUAGE_PREFIX;
    }

    size_t marker_prefix_suffix_size(const std::string_view value, const std::string_view marker) {
        for (size_t size = std::min(value.size(), marker.size() - 1); size > 0; --size) {
            if (value.substr(value.size() - size) == marker.substr(0, size)) {
                return size;
            }
        }
        return 0;
    }
}

HLIEQwenAsrStreamDecoder::HLIEQwenAsrStreamDecoder(
    const bool sse_transport,
    DeltaCallback callback,
    const bool filter_qwen_markers)
    : sse_transport_(sse_transport),
      filter_qwen_markers_(filter_qwen_markers),
      prefix_resolved_(!filter_qwen_markers),
      callback_(std::move(callback)) {
}

void HLIEQwenAsrStreamDecoder::push(const std::string_view chunk) {
    if (finished_ || chunk.empty()) {
        return;
    }

    if (!sse_transport_) {
        consume_payload(chunk);
        return;
    }

    transport_buffer_.append(chunk);
    consume_sse_events(false);
}

std::string HLIEQwenAsrStreamDecoder::finish() {
    if (finished_) {
        return transcript_;
    }
    finished_ = true;

    if (sse_transport_) {
        consume_sse_events(true);
    }
    flush_model_text();

    if (!final_text_.empty()) {
        const std::string canonical_final = filter_qwen_markers_
            ? clean_final_text(final_text_)
            : final_text_;
        if (transcript_.empty()) {
            // Preserve the authoritative final text, but do not manufacture a
            // Realtime delta after the upstream stream has already ended.
            transcript_ = canonical_final;
        } else if (canonical_final.size() >= transcript_.size() &&
                   canonical_final.compare(0, transcript_.size(), transcript_) == 0) {
            emit(std::string_view(canonical_final).substr(transcript_.size()));
        } else if (canonical_final != transcript_) {
            final_text_mismatch_ = true;
        }
    }

    return transcript_;
}

const std::string& HLIEQwenAsrStreamDecoder::transcript() const {
    return transcript_;
}

bool HLIEQwenAsrStreamDecoder::saw_stream_delta() const {
    return saw_stream_delta_;
}

bool HLIEQwenAsrStreamDecoder::final_text_mismatch() const {
    return final_text_mismatch_;
}

std::string HLIEQwenAsrStreamDecoder::clean_final_text(const std::string_view text) {
    std::string_view result = text;
    if (const size_t begin = result.find(ASR_TEXT_BEGIN); begin != std::string_view::npos) {
        result.remove_prefix(begin + ASR_TEXT_BEGIN.size());
    }
    if (const size_t end = result.find(ASR_TEXT_END); end != std::string_view::npos) {
        result = result.substr(0, end);
    }
    return std::string(result);
}

void HLIEQwenAsrStreamDecoder::consume_sse_events(const bool flush) {
    while (!transport_buffer_.empty()) {
        const size_t lf_delimiter = transport_buffer_.find("\n\n");
        const size_t crlf_delimiter = transport_buffer_.find("\r\n\r\n");
        const size_t delimiter = std::min(lf_delimiter, crlf_delimiter);
        if (delimiter == std::string::npos) {
            break;
        }

        const size_t delimiter_size = delimiter == crlf_delimiter ? 4 : 2;
        const std::string event = transport_buffer_.substr(0, delimiter);
        transport_buffer_.erase(0, delimiter + delimiter_size);
        consume_sse_event(event);
    }

    if (flush && !transport_buffer_.empty()) {
        const std::string remaining = std::move(transport_buffer_);
        transport_buffer_.clear();
        consume_sse_event(remaining);
    }
}

void HLIEQwenAsrStreamDecoder::consume_sse_event(const std::string_view event) {
    std::string data;
    size_t offset = 0;
    while (offset <= event.size()) {
        const size_t line_end = event.find('\n', offset);
        std::string_view line = event.substr(
            offset, line_end == std::string_view::npos ? event.size() - offset : line_end - offset);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        if (line.rfind("data:", 0) == 0) {
            line.remove_prefix(5);
            if (!line.empty() && line.front() == ' ') {
                line.remove_prefix(1);
            }
            if (!data.empty()) {
                data.push_back('\n');
            }
            data.append(line);
        }
        if (line_end == std::string_view::npos) {
            break;
        }
        offset = line_end + 1;
    }

    if (!data.empty()) {
        consume_payload(data);
    }
}

void HLIEQwenAsrStreamDecoder::consume_payload(const std::string_view payload) {
    const std::string_view trimmed = trim_view(payload);
    if (trimmed.empty() || trimmed == "[DONE]") {
        return;
    }

    const json value = json::parse(trimmed, nullptr, false);
    if (value.is_discarded() || !value.is_object()) {
        if (!sse_transport_) {
            saw_stream_delta_ = true;
            push_model_text(payload);
        }
        return;
    }

    const std::string type = json_string(value, "type");
    if (type == "transcript.text.delta") {
        const std::string delta = json_string(value, "delta");
        if (!delta.empty()) {
            saw_stream_delta_ = true;
            push_model_text(delta);
        }
        return;
    }
    if (type == "transcript.text.done" || type == "transcription.completed") {
        final_text_ = json_string(value, "text");
        if (final_text_.empty()) {
            final_text_ = json_string(value, "transcript");
        }
        return;
    }

    if (const std::string delta = json_string(value, "delta"); !delta.empty()) {
        saw_stream_delta_ = true;
        push_model_text(delta);
        return;
    }
    if (const std::string delta = extract_chat_delta(value); !delta.empty()) {
        saw_stream_delta_ = true;
        push_model_text(delta);
        return;
    }

    const std::string text = json_string(value, "text");
    if (!text.empty()) {
        final_text_ = text;
    }
}

void HLIEQwenAsrStreamDecoder::push_model_text(const std::string_view text) {
    if (text.empty() || end_marker_seen_) {
        return;
    }

    if (!filter_qwen_markers_) {
        emit(text);
        return;
    }

    if (!prefix_resolved_) {
        prefix_buffer_.append(text);
        if (const size_t marker = prefix_buffer_.find(ASR_TEXT_BEGIN); marker != std::string::npos) {
            std::string content = prefix_buffer_.substr(marker + ASR_TEXT_BEGIN.size());
            prefix_buffer_.clear();
            prefix_resolved_ = true;
            push_transcript_text(content);
        } else if (!could_be_qwen_prefix(prefix_buffer_) || prefix_buffer_.size() >= MAX_PREFIX_BUFFER) {
            std::string content = std::move(prefix_buffer_);
            prefix_buffer_.clear();
            prefix_resolved_ = true;
            push_transcript_text(content);
        }
        return;
    }

    push_transcript_text(text);
}

void HLIEQwenAsrStreamDecoder::push_transcript_text(const std::string_view text) {
    if (text.empty() || end_marker_seen_) {
        return;
    }

    transcript_buffer_.append(text);
    if (const size_t marker = transcript_buffer_.find(ASR_TEXT_END); marker != std::string::npos) {
        emit(std::string_view(transcript_buffer_).substr(0, marker));
        transcript_buffer_.clear();
        end_marker_seen_ = true;
        return;
    }

    const size_t protected_tail = marker_prefix_suffix_size(transcript_buffer_, ASR_TEXT_END);
    if (transcript_buffer_.size() > protected_tail) {
        const size_t emit_size = transcript_buffer_.size() - protected_tail;
        emit(std::string_view(transcript_buffer_).substr(0, emit_size));
        transcript_buffer_.erase(0, emit_size);
    }
}

void HLIEQwenAsrStreamDecoder::emit(const std::string_view text) {
    if (text.empty()) {
        return;
    }
    transcript_.append(text);
    if (callback_) {
        callback_(std::string(text));
    }
}

void HLIEQwenAsrStreamDecoder::flush_model_text() {
    if (!filter_qwen_markers_) {
        return;
    }

    if (!prefix_resolved_) {
        std::string content = std::move(prefix_buffer_);
        prefix_buffer_.clear();
        prefix_resolved_ = true;
        push_transcript_text(content);
    }

    if (end_marker_seen_ || transcript_buffer_.empty()) {
        return;
    }

    const size_t emit_size = transcript_buffer_.size() - marker_prefix_suffix_size(transcript_buffer_, ASR_TEXT_END);
    emit(std::string_view(transcript_buffer_).substr(0, emit_size));
    transcript_buffer_.clear();
}

HLIE_NAMESPACE_END

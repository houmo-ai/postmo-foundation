#include "hlie_app/hlie_ws_service.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include "base64.hpp"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_singleton.h"
#include "hlie_app/hlie_ws_limits.h"
#include "json.hpp"
#include "uwebsockets/App.h"

HLIE_NAMESPACE_BEGIN

namespace {
using json = nlohmann::ordered_json;

std::vector<float> resample(const std::vector<float>& input, const int source_rate, const int target_rate) {
    if (input.empty() || source_rate <= 0 || source_rate == target_rate) {
        return input;
    }

    const auto output_size = static_cast<size_t>(
        std::ceil(static_cast<double>(input.size()) * target_rate / source_rate));
    std::vector<float> output(output_size);
    const double step = static_cast<double>(source_rate) / target_rate;

    for (size_t i = 0; i < output_size; ++i) {
        const double source_pos = i * step;
        const size_t left = std::min(static_cast<size_t>(source_pos), input.size() - 1);
        const size_t right = std::min(left + 1, input.size() - 1);
        const float fraction = static_cast<float>(source_pos - left);
        output[i] = input[left] + (input[right] - input[left]) * fraction;
    }
    return output;
}

std::string eventId(const char* prefix, const uint64_t sequence) {
    return std::string(prefix) + "_" + std::to_string(sequence);
}
}

void AudioBuffer::append(const std::vector<float>& chunk) {
    data_.insert(data_.end(), chunk.begin(), chunk.end());
}

void AudioBuffer::reserve(const size_t sample_count) {
    data_.reserve(sample_count);
}

std::vector<float> AudioBuffer::consume() {
    auto result = std::move(data_);
    data_.clear();
    return result;
}

void AudioBuffer::clear() {
    data_.clear();
}

void AudioBuffer::retainLast(const size_t sample_count) {
    if (data_.size() <= sample_count) {
        return;
    }
    data_.erase(data_.begin(), data_.end() - static_cast<std::ptrdiff_t>(sample_count));
}

bool AudioBuffer::empty() const {
    return data_.empty();
}

size_t AudioBuffer::size() const {
    return data_.size();
}

void Session::init(
    AudioSubmitter submitter,
    EventSender sender,
    std::shared_ptr<SessionConnectionState> connection) {
    submitter_ = std::move(submitter);
    sender_ = std::move(sender);
    connection_ = std::move(connection);
}

void Session::setInitialContext(std::string model, std::string path) {
    model_ = std::move(model);
    path_ = std::move(path);
}

void Session::onMessage(const std::string_view msg, const uWS::OpCode op_code) {
    if (realtimeWsMessageExceedsLimit(msg.size())) {
        LOG_WARN("Websocket message rejected: payload_bytes={}, limit_bytes={}",
                 msg.size(), REALTIME_WS_MESSAGE_SIZE_LIMIT);
        sendError("WebSocket message size limit exceeded: maximum is 16 MiB", "payload_too_large");
        return;
    }

    if (op_code == uWS::OpCode::TEXT) {
        if (msg == "EOS") {
            commit();
            return;
        }
        handleTextMessage(msg);
        return;
    }

    // Legacy protocol: binary little-endian float32 PCM at 16 kHz followed by EOS.
    if (msg.size() % sizeof(float) != 0) {
        sendError("Binary audio payload must contain float32 samples");
        return;
    }

    std::vector<float> chunk(msg.size() / sizeof(float));
    std::memcpy(chunk.data(), msg.data(), msg.size());
    input_sample_rate_ = 16000;
    server_vad_enabled_ = false;
    buffer_.append(chunk);
}

void Session::handleTextMessage(const std::string_view msg) {
    const auto request = json::parse(msg, nullptr, false);
    if (request.is_discarded() || !request.is_object()) {
        sendError("Invalid JSON event");
        return;
    }

    openai_protocol_ = true;
    const std::string type = request.value("type", "");

    try {
        if (type == "session.update" || type == "transcription_session.update") {
            int proposed_sample_rate = input_sample_rate_;
            bool proposed_vad_enabled = server_vad_enabled_;
            float proposed_vad_threshold = vad_threshold_;
            int proposed_prefix_padding_ms = prefix_padding_ms_;
            int proposed_silence_duration_ms = silence_duration_ms_;
            int proposed_max_turn_duration_ms = max_turn_duration_ms_;
            const auto& session = request.contains("session") ? request["session"] : json::object();
            if (session.contains("audio") && session["audio"].contains("input")) {
                const auto& input = session["audio"]["input"];
                if (input.contains("format") && input["format"].is_object()) {
                    proposed_sample_rate = input["format"].value("rate", proposed_sample_rate);
                }
                if (input.contains("transcription") && input["transcription"].is_object()) {
                    const auto& transcription = input["transcription"];
                    model_ = transcription.value("model", model_);
                    prompt_ = transcription.value("prompt", prompt_);
                    language_ = transcription.value("language", language_);
                    response_format_ = transcription.value("response_format", response_format_);
                }
                if (input.contains("turn_detection")) {
                    const auto& turn_detection = input["turn_detection"];
                    proposed_vad_enabled = !turn_detection.is_null();
                    if (turn_detection.is_object()) {
                        const std::string vad_type = turn_detection.value("type", "server_vad");
                        if (vad_type != "server_vad") {
                            throw std::invalid_argument("Only server_vad turn detection is supported");
                        }
                        proposed_vad_threshold = turn_detection.value("threshold", proposed_vad_threshold);
                        proposed_prefix_padding_ms =
                            turn_detection.value("prefix_padding_ms", proposed_prefix_padding_ms);
                        proposed_silence_duration_ms =
                            turn_detection.value("silence_duration_ms", proposed_silence_duration_ms);
                        proposed_max_turn_duration_ms =
                            turn_detection.value("max_turn_duration_ms", proposed_max_turn_duration_ms);
                    }
                }
            } else if (session.contains("input_audio_format") && session["input_audio_format"].is_object()) {
                proposed_sample_rate = session["input_audio_format"].value("rate", proposed_sample_rate);
                if (session.contains("input_audio_transcription") && session["input_audio_transcription"].is_object()) {
                    const auto& transcription = session["input_audio_transcription"];
                    model_ = transcription.value("model", model_);
                    prompt_ = transcription.value("prompt", prompt_);
                    language_ = transcription.value("language", language_);
                    response_format_ = transcription.value("response_format", response_format_);
                }
                if (session.contains("turn_detection")) {
                    const auto& turn_detection = session["turn_detection"];
                    proposed_vad_enabled = !turn_detection.is_null();
                    if (turn_detection.is_object()) {
                        proposed_vad_threshold = turn_detection.value("threshold", proposed_vad_threshold);
                        proposed_prefix_padding_ms =
                            turn_detection.value("prefix_padding_ms", proposed_prefix_padding_ms);
                        proposed_silence_duration_ms =
                            turn_detection.value("silence_duration_ms", proposed_silence_duration_ms);
                        proposed_max_turn_duration_ms =
                            turn_detection.value("max_turn_duration_ms", proposed_max_turn_duration_ms);
                    }
                }
            }

            const bool input_mode_changed =
                proposed_sample_rate != input_sample_rate_ ||
                proposed_vad_enabled != server_vad_enabled_;
            if (input_mode_changed && (!buffer_.empty() || vad_.hasUncommittedAudio())) {
                throw std::invalid_argument(
                    "Cannot change input audio format or turn detection while the input audio buffer contains data");
            }
            if (input_mode_changed) {
                // Prefix lookback retained after a completed turn is internal
                // VAD state, not uncommitted user audio. It can be discarded
                // safely when the input mode changes between turns.
                vad_.clear();
            }

            RealtimeVadConfig vad_config = vad_.config();
            vad_config.sample_rate = proposed_sample_rate;
            vad_config.threshold = proposed_vad_threshold;
            vad_config.prefix_padding_ms = proposed_prefix_padding_ms;
            vad_config.silence_duration_ms = proposed_silence_duration_ms;
            vad_config.suffix_padding_ms = std::min(
                vad_config.suffix_padding_ms, proposed_silence_duration_ms);
            vad_config.max_turn_duration_ms = proposed_max_turn_duration_ms;
            std::string vad_error;
            if (!vad_.configure(vad_config, vad_error)) {
                throw std::invalid_argument(vad_error);
            }

            input_sample_rate_ = proposed_sample_rate;
            server_vad_enabled_ = proposed_vad_enabled;
            vad_threshold_ = proposed_vad_threshold;
            prefix_padding_ms_ = proposed_prefix_padding_ms;
            silence_duration_ms_ = proposed_silence_duration_ms;
            max_turn_duration_ms_ = proposed_max_turn_duration_ms;
            const size_t target_reserve =
                static_cast<size_t>(std::max(input_sample_rate_, 1)) *
                static_cast<size_t>(std::max(max_turn_duration_ms_, 1000)) / 1000 + 8192;
            if (target_reserve > reserved_buffer_samples_) {
                buffer_.reserve(target_reserve);
                reserved_buffer_samples_ = target_reserve;
            }
            LOG_INFO("Websocket session updated: path={}, model={}, sample_rate={}, vad={}, max_turn_duration_ms={}",
                     path_, model_.empty() ? "<default>" : model_, input_sample_rate_, server_vad_enabled_, max_turn_duration_ms_);
            sendSessionEvent("session.updated");
        } else if (type == "input_audio_buffer.append") {
            appendPcm16(request.value("audio", ""));
        } else if (type == "input_audio_buffer.commit") {
            commit();
        } else if (type == "input_audio_buffer.clear") {
            buffer_.clear();
            vad_.clear();
            pending_item_id_.clear();
            sender_(json{{"event_id", eventId("event", ++item_sequence_)},
                         {"type", "input_audio_buffer.cleared"}}
                        .dump());
        } else {
            sendError("Unsupported event type: " + type, "invalid_event");
        }
    } catch (const std::exception& ex) {
        sendError(ex.what());
    }
}

void Session::appendPcm16(const std::string& encoded_audio) {
    if (encoded_audio.empty()) {
        throw std::invalid_argument("input_audio_buffer.append requires audio");
    }

    const std::string bytes = base64::decode(encoded_audio);
    if (bytes.size() % sizeof(int16_t) != 0) {
        throw std::invalid_argument("PCM16 audio must contain complete 16-bit samples");
    }

    std::vector<float> chunk(bytes.size() / sizeof(int16_t));
    for (size_t i = 0; i < chunk.size(); ++i) {
        const auto lo = static_cast<uint8_t>(bytes[i * 2]);
        const auto hi = static_cast<uint8_t>(bytes[i * 2 + 1]);
        const auto sample = static_cast<int16_t>(lo | (static_cast<uint16_t>(hi) << 8));
        chunk[i] = static_cast<float>(sample) / 32768.0f;
    }
    if (server_vad_enabled_) {
        handleVadEvents(vad_.append(chunk));
    } else {
        buffer_.append(chunk);
    }
}

void Session::handleVadEvents(std::vector<RealtimeVadEvent> events) {
    for (auto& event : events) {
        if (event.type == RealtimeVadEventType::SpeechStarted) {
            pending_item_id_ = eventId("item", ++item_sequence_);
            sender_(json{{"event_id", eventId("event", item_sequence_)},
                         {"type", "input_audio_buffer.speech_started"},
                         {"audio_start_ms", event.start_sample * 1000 / std::max(input_sample_rate_, 1)},
                         {"item_id", pending_item_id_}}
                        .dump());
            continue;
        }

        if (event.type == RealtimeVadEventType::SpeechStopped) {
            if (pending_item_id_.empty()) {
                pending_item_id_ = eventId("item", ++item_sequence_);
            }
            sender_(json{{"event_id", eventId("event", item_sequence_)},
                         {"type", "input_audio_buffer.speech_stopped"},
                         {"audio_end_ms", event.end_sample * 1000 / std::max(input_sample_rate_, 1)},
                         {"item_id", pending_item_id_}}
                        .dump());
            continue;
        }

        const std::string item_id = pending_item_id_.empty()
            ? eventId("item", ++item_sequence_)
            : pending_item_id_;
        if (openai_protocol_) {
            sender_(json{{"event_id", eventId("event", item_sequence_)},
                         {"type", "input_audio_buffer.committed"},
                         {"previous_item_id", nullptr},
                         {"item_id", item_id}}
                        .dump());
        }
        submitAudio(std::move(event.audio), item_id, event.reason, event.metrics);
        pending_item_id_.clear();
    }
}

void Session::commit() {
    if (server_vad_enabled_) {
        auto events = vad_.commit();
        const bool has_segment = std::any_of(
            events.begin(), events.end(), [](const RealtimeVadEvent& event) {
                return event.type == RealtimeVadEventType::SegmentReady && !event.audio.empty();
            });
        if (!has_segment) {
            sendError("Cannot commit an empty audio buffer", "input_audio_buffer_empty");
            return;
        }
        handleVadEvents(std::move(events));
        return;
    }

    if (buffer_.empty()) {
        sendError("Cannot commit an empty audio buffer", "input_audio_buffer_empty");
        return;
    }

    const std::string item_id = eventId("item", ++item_sequence_);
    auto audio = buffer_.consume();
    if (openai_protocol_) {
        sender_(json{{"event_id", eventId("event", item_sequence_)},
                     {"type", "input_audio_buffer.committed"},
                     {"previous_item_id", nullptr},
                     {"item_id", item_id}}
                    .dump());
    }
    submitAudio(std::move(audio), item_id, RealtimeVadCommitReason::Manual, {});
}

void Session::submitAudio(
    std::vector<float> audio,
    const std::string& item_id,
    const RealtimeVadCommitReason reason,
    const RealtimeVadMetrics& metrics) {
    WhisperStreamRequest request;
    request.pcmf32 = std::move(audio);
    request.sample_rate = input_sample_rate_;
    request.model = model_;
    request.prompt = prompt_;
    request.language = language_;
    request.response_format = response_format_;
    request.path = path_;
    const char* reason_name = reason == RealtimeVadCommitReason::Silence
        ? "silence"
        : reason == RealtimeVadCommitReason::MaxDuration ? "max_duration" : "manual";
    LOG_INFO("Websocket audio committed: item_id={}, reason={}, model={}, sample_rate={}, samples={}, "
             "duration_ms={}, vad_frames={}, vad_speech_frames={}, vad_avg_dbfs={}, vad_noise_dbfs={}",
             item_id, reason_name, request.model.empty() ? "<default>" : request.model,
             request.sample_rate, request.pcmf32.size(),
             request.sample_rate > 0 ? request.pcmf32.size() * 1000 / request.sample_rate : 0,
             metrics.analyzed_frames, metrics.speech_frames,
             metrics.average_level_dbfs, metrics.noise_floor_dbfs);
    if (!submitter_(std::move(request), item_id)) {
        sender_(json{{"event_id", eventId("event", item_sequence_)},
                     {"type", "conversation.item.input_audio_transcription.failed"},
                     {"item_id", item_id},
                     {"content_index", 0},
                     {"error",
                      {{"type", "server_error"},
                       {"code", "server_busy"},
                       {"message", "Realtime ASR inference queue is full"},
                       {"param", nullptr}}}}
                    .dump());
    }
}

void Session::sendError(const std::string& message, const std::string& code) const {
    if (!sender_) {
        return;
    }
    sender_(json{{"event_id", eventId("event", item_sequence_)},
                 {"type", "error"},
                 {"error",
                  {{"type", "invalid_request_error"},
                   {"code", code},
                   {"message", message},
                   {"param", nullptr},
                   {"event_id", nullptr}}}}
                .dump());
}

void Session::sendSessionEvent(const std::string& type) const {
    sender_(json{{"event_id", eventId("event", item_sequence_)},
                 {"type", type},
                 {"session",
                  {{"id", eventId("sess", item_sequence_)},
                   {"object", "realtime.transcription_session"},
                   {"type", "transcription"},
                   {"audio",
                    {{"input",
                      {{"format", {{"type", "audio/pcm"}, {"rate", input_sample_rate_}}},
                       {"transcription",
                        {{"model", model_.empty() ? "whisper-1" : model_},
                         {"prompt", prompt_},
                         {"language", language_},
                         {"response_format", response_format_}}},
                       {"turn_detection",
                        server_vad_enabled_
                            ? json{{"type", "server_vad"},
                                   {"threshold", vad_threshold_},
                                   {"prefix_padding_ms", prefix_padding_ms_},
                                   {"silence_duration_ms", silence_duration_ms_},
                                   {"max_turn_duration_ms", max_turn_duration_ms_}}
                            : json(nullptr)}}}}}}}}
                .dump());
}

void Session::close() {
    if (connection_) {
        connection_->active = false;
    }
    buffer_.clear();
    vad_.clear();
    submitter_ = {};
    sender_ = {};
    connection_.reset();
}

HLIEWebSocketService::HLIEWebSocketService() = default;

HLIEWebSocketService::~HLIEWebSocketService() {
    stop();
}

void HLIEWebSocketService::init() {
    const auto config = Singleton<HLIEConfigService>::instance();
    max_pending_inference_tasks_ = static_cast<size_t>(std::clamp(
        config->get_int("realtime", "max_pending_segments", 8, true), 1, 64));
    LOG_INFO("Realtime websocket configuration: max_pending_segments={}", max_pending_inference_tasks_);
}

void HLIEWebSocketService::run(AudioHandler handler, const int port) {
    handler_ = std::move(handler);
    stopping_ = false;
    inference_thread_ = std::make_unique<std::thread>(&HLIEWebSocketService::processInferenceTasks, this);

    worker_thread_ = std::make_unique<std::thread>([this, port] {
        loop_ = uWS::Loop::get();

        uWS::App()
            .ws<Session>(
                "/*",
                {.maxPayloadLength = REALTIME_WS_TRANSPORT_PAYLOAD_LIMIT,
                 .idleTimeout = 60,
                 .maxBackpressure = 4 * 1024 * 1024,
                 .closeOnBackpressureLimit = false,
                 .resetIdleTimeoutOnSend = true,
                 .sendPingsAutomatically = true,
                 .upgrade =
                     [](auto* res, auto* req, auto* context) {
                         Session session;
                         session.setInitialContext(std::string(req->getQuery("model")), std::string(req->getUrl()));
                         std::string_view sec_websocket_key = req->getHeader("sec-websocket-key");
                         std::string_view sec_websocket_protocol = req->getHeader("sec-websocket-protocol");
                         std::string_view sec_websocket_extensions = req->getHeader("sec-websocket-extensions");
                         res->template upgrade<Session>(std::move(session), sec_websocket_key, sec_websocket_protocol,
                                                        sec_websocket_extensions, context);
                     },
                 .open =
                     [this](auto* ws) {
                         auto connection = std::make_shared<SessionConnectionState>();
                         auto* session = ws->getUserData();
                         session->init(
                             [this, ws, connection](WhisperStreamRequest request, std::string item_id) {
                                 return enqueue({ws, connection, std::move(request), std::move(item_id)});
                             },
                             [this, ws, connection](std::string event) {
                                 InferenceTask target{ws, connection};
                                 sendEvent(target, std::move(event));
                             },
                             connection);

                         {
                             std::lock_guard lock(clients_mtx_);
                             clients_.insert(ws);
                         }
                         session->sendSessionEvent("session.created");
                         LOG_INFO("New websocket connection established");
                     },
                 .message =
                     [](auto* ws, std::string_view msg, const uWS::OpCode op_code) {
                         ws->getUserData()->onMessage(msg, op_code);
                     },
                 .close =
                     [this](auto* ws, int, std::string_view) {
                         ws->getUserData()->close();
                         std::lock_guard lock(clients_mtx_);
                         clients_.erase(ws);
                         LOG_INFO("Websocket connection closed");
                     }})
            .listen(port,
                    [port, this](auto* token) {
                        if (token) {
                            listen_token_ = token;
                            LOG_INFO("Websocket server listening on port {}", port);
                        } else {
                            LOG_ERROR("Failed to listen on port {}", port);
                        }
                    })
            .run();
    });

    LOG_INFO("Websocket server is listening, hostname: 0.0.0.0, port: {}", port);
}

bool HLIEWebSocketService::enqueue(InferenceTask task) {
    {
        std::lock_guard lock(queue_mtx_);
        if (stopping_) {
            LOG_WARN("Realtime ASR inference rejected during shutdown: item_id={}", task.item_id);
            return false;
        }
        if (queue_.size() >= max_pending_inference_tasks_) {
            LOG_WARN("Realtime ASR inference queue full: pending={}, limit={}, item_id={}",
                     queue_.size(), max_pending_inference_tasks_, task.item_id);
            return false;
        }
        queue_.push_back(std::move(task));
    }
    queue_cv_.notify_one();
    return true;
}

void HLIEWebSocketService::processInferenceTasks() {
    while (true) {
        InferenceTask task;
        {
            std::unique_lock lock(queue_mtx_);
            queue_cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_ && queue_.empty()) {
                return;
            }
            task = std::move(queue_.front());
            queue_.pop_front();
        }

        if (!task.connection->active || !handler_) {
            continue;
        }

        WhisperStreamRequest request = std::move(task.request);
        request.pcmf32 = resample(request.pcmf32, request.sample_rate, 16000);
        request.sample_rate = 16000;
        std::string streamed_text;

        const auto on_delta = [this, &task, &streamed_text](const std::string& delta) {
            if (delta.empty()) {
                return;
            }
            streamed_text += delta;
            sendEvent(task,
                      json{{"event_id", eventId("event", streamed_text.size())},
                           {"type", "conversation.item.input_audio_transcription.delta"},
                           {"item_id", task.item_id},
                           {"content_index", 0},
                           {"delta", delta}}
                          .dump());
        };

        const std::string transcript = handler_(request, on_delta);
        if (!task.connection->active) {
            continue;
        }

        if (transcript == "HLIECpp-FAILED-888") {
            sendEvent(task,
                      json{{"event_id", eventId("event", task.item_id.size())},
                           {"type", "conversation.item.input_audio_transcription.failed"},
                           {"item_id", task.item_id},
                           {"content_index", 0},
                           {"error",
                            {{"type", "transcription_error"},
                             {"code", "transcription_failed"},
                             {"message", "Audio transcription failed"},
                             {"param", nullptr}}}}
                          .dump());
            continue;
        }

        sendEvent(task,
                  json{{"event_id", eventId("event", transcript.size())},
                       {"type", "conversation.item.input_audio_transcription.completed"},
                       {"item_id", task.item_id},
                       {"content_index", 0},
                       {"transcript", transcript}}
                      .dump());
    }
}

void HLIEWebSocketService::sendEvent(const InferenceTask& task, std::string event) {
    if (!loop_) {
        return;
    }
    loop_->defer([task, event = std::move(event)] {
        if (task.connection && task.connection->active && task.ws) {
            if (!task.ws->send(event, uWS::OpCode::TEXT)) {
                LOG_WARN("Websocket send backpressure: item_id={}, buffered={}",
                         task.item_id, task.ws->getBufferedAmount());
            }
        }
    });
}

void HLIEWebSocketService::stop() {
    if (!worker_thread_ && !inference_thread_) {
        return;
    }

    stopping_ = true;
    {
        std::lock_guard lock(queue_mtx_);
        queue_.clear();
    }
    queue_cv_.notify_all();

    // Keep the uWebSockets loop alive until the active inference has stopped
    // scheduling callbacks onto it.
    if (inference_thread_ && inference_thread_->joinable()) {
        inference_thread_->join();
    }
    inference_thread_.reset();

    if (loop_) {
        loop_->defer([this] {
            if (listen_token_) {
                us_listen_socket_close(0, listen_token_);
                listen_token_ = nullptr;
            }

            std::lock_guard lock(clients_mtx_);
            for (auto* ws : clients_) {
                ws->end(1001, "Server Shutting Down");
            }
            clients_.clear();
        });
    }

    if (worker_thread_ && worker_thread_->joinable()) {
        worker_thread_->join();
    }
    worker_thread_.reset();
    loop_ = nullptr;
}

HLIE_NAMESPACE_END

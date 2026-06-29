#include "hlie_app/hlie_ws_service.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include "base64.hpp"
#include "hlie_base/hlie_logger.h"
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

void Session::onMessage(auto* ws, const std::string_view msg, const uWS::OpCode op_code) {
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
            const auto& session = request.contains("session") ? request["session"] : json::object();
            if (session.contains("audio") && session["audio"].contains("input")) {
                const auto& input = session["audio"]["input"];
                if (input.contains("format") && input["format"].is_object()) {
                    input_sample_rate_ = input["format"].value("rate", input_sample_rate_);
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
                    server_vad_enabled_ = !turn_detection.is_null();
                    if (turn_detection.is_object()) {
                        const std::string vad_type = turn_detection.value("type", "server_vad");
                        if (vad_type != "server_vad") {
                            throw std::invalid_argument("Only server_vad turn detection is supported");
                        }
                        vad_threshold_ = turn_detection.value("threshold", vad_threshold_);
                        prefix_padding_ms_ =
                            turn_detection.value("prefix_padding_ms", prefix_padding_ms_);
                        silence_duration_ms_ =
                            turn_detection.value("silence_duration_ms", silence_duration_ms_);
                        max_turn_duration_ms_ =
                            turn_detection.value("max_turn_duration_ms", max_turn_duration_ms_);
                    }
                    resetTurnState();
                }
            } else if (session.contains("input_audio_format") && session["input_audio_format"].is_object()) {
                input_sample_rate_ = session["input_audio_format"].value("rate", input_sample_rate_);
                if (session.contains("input_audio_transcription") && session["input_audio_transcription"].is_object()) {
                    const auto& transcription = session["input_audio_transcription"];
                    model_ = transcription.value("model", model_);
                    prompt_ = transcription.value("prompt", prompt_);
                    language_ = transcription.value("language", language_);
                    response_format_ = transcription.value("response_format", response_format_);
                }
                if (session.contains("turn_detection")) {
                    const auto& turn_detection = session["turn_detection"];
                    server_vad_enabled_ = !turn_detection.is_null();
                    if (turn_detection.is_object()) {
                        vad_threshold_ = turn_detection.value("threshold", vad_threshold_);
                        prefix_padding_ms_ =
                            turn_detection.value("prefix_padding_ms", prefix_padding_ms_);
                        silence_duration_ms_ =
                            turn_detection.value("silence_duration_ms", silence_duration_ms_);
                        max_turn_duration_ms_ =
                            turn_detection.value("max_turn_duration_ms", max_turn_duration_ms_);
                    }
                    resetTurnState();
                }
            }
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
            resetTurnState();
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
    buffer_.append(chunk);
    total_audio_samples_ += static_cast<int64_t>(chunk.size());
    processTurnDetection(chunk);
}

void Session::processTurnDetection(const std::vector<float>& chunk) {
    if (!server_vad_enabled_ || chunk.empty()) {
        return;
    }

    double energy = 0.0;
    for (const float sample : chunk) {
        energy += static_cast<double>(sample) * sample;
    }
    const float rms = static_cast<float>(std::sqrt(energy / chunk.size()));
    // OpenAI exposes threshold as a normalized 0..1 setting. Map it to a
    // practical PCM RMS range for this lightweight local VAD.
    const float rms_threshold = 0.005f + std::clamp(vad_threshold_, 0.0f, 1.0f) * 0.03f;
    const int chunk_duration_ms =
        std::max(1, static_cast<int>(chunk.size() * 1000 / input_sample_rate_));
    const bool voice_detected = rms >= rms_threshold;

    if (voice_detected) {
        silence_ms_ = 0;
        speech_duration_ms_ += chunk_duration_ms;
        if (!speech_active_) {
            speech_active_ = true;
            const size_t prefix_samples =
                static_cast<size_t>(input_sample_rate_) * prefix_padding_ms_ / 1000;
            buffer_.retainLast(prefix_samples + chunk.size());
            pending_item_id_ = eventId("item", ++item_sequence_);
            sender_(json{{"event_id", eventId("event", item_sequence_)},
                         {"type", "input_audio_buffer.speech_started"},
                         {"audio_start_ms", std::max<int64_t>(0, audioTimeMs() - chunk_duration_ms)},
                         {"item_id", pending_item_id_}}
                        .dump());
        }
    } else if (speech_active_) {
        silence_ms_ += chunk_duration_ms;
        speech_duration_ms_ += chunk_duration_ms;
    }

    if (speech_active_ &&
        (silence_ms_ >= silence_duration_ms_ || speech_duration_ms_ >= max_turn_duration_ms_)) {
        commit(true);
    } else if (!speech_active_ &&
               buffer_.size() >= static_cast<size_t>(input_sample_rate_ * max_turn_duration_ms_ / 1000)) {
        // Do not send silence-only turns to Whisper.
        buffer_.clear();
        resetTurnState();
    }
}

void Session::commit(const bool automatic) {
    if (buffer_.empty()) {
        if (!automatic) {
            sendError("Cannot commit an empty audio buffer", "input_audio_buffer_empty");
        }
        return;
    }

    const std::string item_id =
        pending_item_id_.empty() ? eventId("item", ++item_sequence_) : pending_item_id_;
    auto audio = buffer_.consume();

    if (automatic && speech_active_) {
        sender_(json{{"event_id", eventId("event", item_sequence_)},
                     {"type", "input_audio_buffer.speech_stopped"},
                     {"audio_end_ms", audioTimeMs()},
                     {"item_id", item_id}}
                    .dump());
    }
    if (openai_protocol_) {
        sender_(json{{"event_id", eventId("event", item_sequence_)},
                     {"type", "input_audio_buffer.committed"},
                     {"previous_item_id", nullptr},
                     {"item_id", item_id}}
                    .dump());
    }
    WhisperStreamRequest request;
    request.pcmf32 = std::move(audio);
    request.sample_rate = input_sample_rate_;
    request.model = model_;
    request.prompt = prompt_;
    request.language = language_;
    request.response_format = response_format_;
    request.path = path_;
    LOG_INFO("Websocket audio committed: item_id={}, model={}, sample_rate={}, samples={}, duration_ms={}",
             item_id, request.model.empty() ? "<default>" : request.model, request.sample_rate, request.pcmf32.size(),
             request.sample_rate > 0 ? request.pcmf32.size() * 1000 / request.sample_rate : 0);
    submitter_(std::move(request), item_id);
    resetTurnState();
}

void Session::resetTurnState() {
    speech_active_ = false;
    speech_duration_ms_ = 0;
    silence_ms_ = 0;
    pending_item_id_.clear();
}

int64_t Session::audioTimeMs() const {
    return input_sample_rate_ > 0 ? total_audio_samples_ * 1000 / input_sample_rate_ : 0;
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
    submitter_ = {};
    sender_ = {};
    connection_.reset();
}

HLIEWebSocketService::HLIEWebSocketService() = default;

HLIEWebSocketService::~HLIEWebSocketService() {
    stop();
}

void HLIEWebSocketService::init() {
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
                {.maxPayloadLength = 16 * 1024 * 1024,
                 .idleTimeout = 60,
                 .maxBackpressure = 4 * 1024 * 1024,
                 .closeOnBackpressureLimit = false,
                 .resetIdleTimeoutOnSend = true,
                 .sendPingsAutomatically = false,
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
                                 enqueue({ws, connection, std::move(request), std::move(item_id)});
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
                         ws->getUserData()->onMessage(ws, msg, op_code);
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

void HLIEWebSocketService::enqueue(InferenceTask task) {
    {
        std::lock_guard lock(queue_mtx_);
        queue_.push_back(std::move(task));
    }
    queue_cv_.notify_one();
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
                             {"message", "Whisper transcription failed"},
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

#ifndef HLIE_WEBSOCKET_SERVICE_H
#define HLIE_WEBSOCKET_SERVICE_H

#define UWS_NO_ZLIB

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "hlie_base/hlie_base.h"
#include "hlie_app/audio/hlie_realtime_vad.h"
#include "uwebsockets/WebSocket.h"

struct us_listen_socket_t;

HLIE_NAMESPACE_BEGIN

struct WhisperStreamRequest {
    std::vector<float> pcmf32; // mono-channel F32 PCM at 16 kHz
    std::vector<std::vector<float>> pcmf32s;
    int sample_rate = 16000;
    std::string model;
    std::string prompt;
    std::string language;
    std::string response_format = "json";
    std::string path;
};

using TranscriptCallback = std::function<void(const std::string&)>;
using AudioHandler = std::function<std::string(const WhisperStreamRequest&, const TranscriptCallback&)>;
using AudioSubmitter = std::function<bool(WhisperStreamRequest, std::string)>;
using EventSender = std::function<void(std::string)>;

class AudioBuffer {
public:
    void append(const std::vector<float>& chunk);
    void reserve(size_t sample_count);
    std::vector<float> consume();
    void clear();
    void retainLast(size_t sample_count);
    [[nodiscard]] bool empty() const;
    [[nodiscard]] size_t size() const;

private:
    std::vector<float> data_;
};

struct SessionConnectionState {
    std::atomic<bool> active{true};
};

class Session {
public:
    void init(
        AudioSubmitter submitter,
        EventSender sender,
        std::shared_ptr<SessionConnectionState> connection);
    void setInitialContext(std::string model, std::string path);
    void onMessage(std::string_view msg, uWS::OpCode op_code);
    void close();
    void sendSessionEvent(const std::string& type) const;

private:
    void handleTextMessage(std::string_view msg);
    void appendPcm16(const std::string& encoded_audio);
    void handleVadEvents(std::vector<RealtimeVadEvent> events);
    void submitAudio(std::vector<float> audio, const std::string& item_id,
                     RealtimeVadCommitReason reason, const RealtimeVadMetrics& metrics);
    void commit();
    void sendError(const std::string& message, const std::string& code = "invalid_request_error") const;
    AudioBuffer buffer_;
    HLIERealtimeVad vad_;
    AudioSubmitter submitter_;
    EventSender sender_;
    std::shared_ptr<SessionConnectionState> connection_;
    int input_sample_rate_ = 24000;
    uint64_t item_sequence_ = 0;
    bool openai_protocol_ = false;
    bool server_vad_enabled_ = true;
    float vad_threshold_ = 0.5f;
    int silence_duration_ms_ = 2000;
    int prefix_padding_ms_ = 300;
    int max_turn_duration_ms_ = 30000;
    std::string pending_item_id_;
    std::string model_;
    std::string prompt_;
    std::string language_;
    std::string response_format_ = "json";
    std::string path_;
    size_t reserved_buffer_samples_ = 0;
};

class HLIE_API HLIEWebSocketService {
public:
    HLIEWebSocketService();
    ~HLIEWebSocketService();

    void init();
    void run(AudioHandler handler, int port = 7999);
    void stop();

private:
    using WebSocketPtr = uWS::WebSocket<false, true, Session>*;

    struct InferenceTask {
        WebSocketPtr ws = nullptr;
        std::shared_ptr<SessionConnectionState> connection;
        WhisperStreamRequest request;
        std::string item_id;
    };

    bool enqueue(InferenceTask task);
    void processInferenceTasks();
    void sendEvent(const InferenceTask& task, std::string event);

    std::unique_ptr<std::thread> worker_thread_;
    std::unique_ptr<std::thread> inference_thread_;
    AudioHandler handler_;
    std::atomic<bool> stopping_{false};

    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    std::deque<InferenceTask> queue_;
    size_t max_pending_inference_tasks_ = 8;

    struct us_listen_socket_t* listen_token_ = nullptr;
    uWS::Loop* loop_ = nullptr;

    std::set<WebSocketPtr> clients_;
    std::mutex clients_mtx_;
};

HLIE_NAMESPACE_END

#endif

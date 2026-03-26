#ifndef HLIE_WEBSOCKET_SERVICE_H
#define HLIE_WEBSOCKET_SERVICE_H
#define UWS_NO_ZLIB
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "hlie_base/hlie_base.h"
#include "uwebsockets/WebSocket.h"


struct us_listen_socket_t;

HLIE_NAMESPACE_BEGIN

class AudioBuffer {
public:
    void append(const std::vector<float>& chunk) {
        data_.insert(data_.end(), chunk.begin(), chunk.end());
    }

    std::vector<float> consume() {
        auto tmp = data_;
        data_.clear();
        return tmp;
    }

private:
    std::vector<float> data_;
};

class VAD {
public:
    static bool is_voice(const std::vector<float>& audio) {
        if (audio.empty())
            return false;

        float energy = 0;
        for (const auto v : audio) {
            energy += v * v;
        }
        energy /= audio.size();
        return energy > 0.001;
    }
};

struct WhisperStreamRequest {
    std::vector<float> pcmf32; // mono-channel F32 PCM
    std::vector<std::vector<float>> pcmf32s; // stereo-channel F32 PCM
};

using AudioHandler = std::function<std::string(WhisperStreamRequest)>;

class Session {
public:
    void init(const AudioHandler& handler);
    void onMessage(auto* ws, std::string_view msg);

private:
    AudioBuffer buffer_;
    VAD vad_;

    AudioHandler handler_;

    std::chrono::steady_clock::time_point last_audio_time_;

    bool speaking_ = false;

    void processAudio(auto* ws, const std::vector<float>& chunk);
};

class HLIE_API HLIEWebSocketService {
public:
    explicit HLIEWebSocketService();
    ~HLIEWebSocketService() = default;

    void init();

    void run(AudioHandler handler, int port = 7999);

    void stop();

private:
    std::unique_ptr<std::thread> worker_thread_; // 工作线程
    std::atomic<bool> is_ready = false;


    struct us_listen_socket_t* listen_token_ = nullptr;

    // 跨线程通信的 loop 指针
    uWS::Loop* loop_ = nullptr;

    // 追踪所有活跃连接，以便在 stop 时强制断开
    using WebSocketPtr = uWS::WebSocket<false, true, Session>*;
    std::set<WebSocketPtr> clients_;
    std::mutex clients_mtx_;
};

HLIE_NAMESPACE_END

#endif

#include "hlie_app/hlie_ws_service.h"
#include "uwebsockets/App.h"

#include "hlie_app/hlie_model_info.h"
#include "hlie_app/hlie_workflow.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN


void Session::init(const AudioHandler& handler) {
    handler_ = handler;
}

void Session::onMessage(auto* ws, const std::string_view msg) {

    // 1. 判断是否是结束指令
    if (msg == "EOS") {
        LOG_INFO("Received EOS signal, flushing buffer...");
        auto remaining_audio = buffer_.consume();
        if (!remaining_audio.empty()) {
            processAudio(ws, remaining_audio);
        }
        return;
    }


    // 2. 校验数据长度是否是 float 的整数倍
    if (msg.size() % sizeof(float) != 0) {
        LOG_WARN("Received invalid data size: {}", msg.size());
        return;
    }

    // 3. 将二进制流转为 float
    const size_t n = msg.size() / sizeof(float);
    const float* data = reinterpret_cast<const float*>(msg.data());

    // 放入缓冲区
    std::vector<float> chunk(data, data + n);
    buffer_.append(chunk);

    /*

    // 简单 VAD
    if (VAD::is_voice(chunk)) {
        LOG_INFO("voice is enabled");
        speaking_ = true;
    } else {
        LOG_INFO("voice is not");
        if (speaking_) {
            auto audio = buffer_.consume();
            processAudio(ws, audio);
            speaking_ = false;
        }
    }*/
}

void Session::processAudio(auto* ws, const std::vector<float>& chunk) {
    if (chunk.empty()) {
        return;
    }

    const WhisperStreamRequest whisper_stream_request = {chunk, {}};

    if (!handler_) {
        return;
    }

    const auto& text = handler_(whisper_stream_request);

    if (text.empty()) {

        return;
    }

    ws->send(text, uWS::OpCode::TEXT);
}


HLIEWebSocketService::HLIEWebSocketService() = default;

void HLIEWebSocketService::init() {
}

void HLIEWebSocketService::run(AudioHandler handler, const int port) {
    worker_thread_ = std::make_unique<std::thread>([this, port, handler] {
        this->loop_ = uWS::Loop::get();

        uWS::App()
            .ws<Session>("/*",
                         {.maxPayloadLength = 16 * 1024 * 1024,
                          .idleTimeout = 60,
                          .open =
                              [this, handler](auto* ws) {
                                  auto* session = ws->getUserData();
                                  session->init(handler);

                                  std::lock_guard lock(this->clients_mtx_);
                                  this->clients_.insert(ws);
                                  LOG_INFO("New ws  connection established");
                              },

                          .message =
                              [](auto* ws, std::string_view msg, uWS::OpCode) {
                                  auto* session = ws->getUserData();
                                  session->onMessage(ws, msg);
                              },

                          .close =
                              [this](auto* ws, int, std::string_view) {
                                  std::lock_guard lock(this->clients_mtx_);
                                  this->clients_.erase(ws);
                                  LOG_INFO("Ws connection closed");
                              }

                         })
            .listen(port,
                    [port, this](auto* token) {
                        if (token) {
                            this->listen_token_ = token;
                            LOG_INFO("Websocket server listening on port {}", port);
                        } else {
                            LOG_ERROR("Failed to listen on port {}", port);
                        }
                    })
            .run();
    });

    LOG_INFO("websocket server is listening, hostname: 0.0.0.0, port: {}", port);
}

void HLIEWebSocketService::stop() {
    if (!worker_thread_) {
        return;
    }

    LOG_INFO("websocket server is stopping");

    loop_->defer([this] {
        // 停止接受新请求
        if (this->listen_token_) {
            us_listen_socket_close(0, this->listen_token_);
            this->listen_token_ = nullptr;
        }

        // 强制断开所有现有连接
        std::lock_guard<std::mutex> lock(this->clients_mtx_);
        LOG_INFO("Closing {} active connections", this->clients_.size());
        for (auto* ws : this->clients_) {
            // 1001 表示服务器关闭，true 表示立即强制切断
            ws->end(1001, "Server Shutting Down");
        }
        this->clients_.clear();
    });

    LOG_INFO("websocket loop server close all sessions");

    if (worker_thread_->joinable()) {
        worker_thread_->join();
    }

    LOG_INFO("websocket server is stopped");
    worker_thread_.reset();

    LOG_INFO("websocket server is reset");
}

HLIE_NAMESPACE_END

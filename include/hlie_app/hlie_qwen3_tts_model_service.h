#ifndef HLIE_QWEN3_TTS_MODEL_SERVICE_H
#define HLIE_QWEN3_TTS_MODEL_SERVICE_H

#include <memory>
#include <mutex>
#include <string>

#include "hlie_app/hlie_model_service.h"
#include "httplib.h"
#include "server/server-http.h"

HLIE_NAMESPACE_BEGIN

class HLIEQwen3TTSModel;

/**
 * Lifecycle owner for a Qwen3-TTS model.
 * Request execution is added independently from model loading so incomplete
 * synthesis code is never exposed through the HTTP routing table.
 */
class Qwen3TTSModelService final : public IHLIEModelService {
public:
    Qwen3TTSModelService() = default;
    ~Qwen3TTSModelService() override;

    int init(const std::string& model_dir, const HLIEModelInfo& model_info,
             const std::shared_ptr<HLIEDeviceService>& device_service) override;
    void run() override;
    void process() override;
    void stop() override;
    int get_state() override;
    server_http_res_ptr handle_prefill_debug(const server_http_req& req);
    server_http_res_ptr handle_codec_frames_debug(const server_http_req& req);
    server_http_res_ptr handle_speech_debug(const server_http_req& req);
    server_http_res_ptr handle_stateful_decoder_debug(const server_http_req& req);
    void handle_speech(const httplib::Request& req, httplib::Response& res);

    [[nodiscard]] const std::string& get_model_key() const { return model_key_; }
    [[nodiscard]] const std::shared_ptr<HLIEQwen3TTSModel>& model() const { return model_; }

private:
    mutable std::mutex model_mutex_;
    std::shared_ptr<HLIEQwen3TTSModel> model_;
    std::string model_key_;
    bool stopped_ = false;
};

HLIE_NAMESPACE_END

#endif

#ifndef HLIE_MODEL_SERVICE_H
#define HLIE_MODEL_SERVICE_H

#include <atomic>
#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class HLIEDeviceService;

struct HLIEModelType {
    static constexpr auto LLM = "llm";
    static constexpr auto SD = "sd";
    static constexpr auto WHISPER = "whisper";
    static constexpr auto COSYVOICE = "cosyvoice";
    static constexpr auto TTS = "tts";
    static constexpr auto QWEN3_TTS = "qwen3_tts";
    static constexpr auto DOCUMENT = "document";

    // model real instance type
    static constexpr auto Qwen25 = "qwen2.5";
    static constexpr auto Qwen3 = "qwen3";
    static constexpr auto DeepSeek = "deepseek";
    static constexpr auto SD3 = "sd3";
};

// Backward-compatible aliases. Keep HLIEModelType as the single source of string literals.
inline const std::string LLM = HLIEModelType::LLM;
inline const std::string SD = HLIEModelType::SD;
inline const std::string WHISPER = HLIEModelType::WHISPER;

class HLIEModelInfo;

class IHLIEModelService {
public:
    IHLIEModelService() = default;
    virtual ~IHLIEModelService() = default;

    virtual int init(const std::string& model_dir, const HLIEModelInfo& model_info, const std::shared_ptr<HLIEDeviceService>& device_service) = 0;

    virtual void run() = 0;

    virtual void process() = 0;

    virtual void stop() = 0;

    virtual int get_state() = 0;

protected:
    std::string model_path_;
    std::atomic<int> state_;
    std::vector<int> devices_;
    int acquired_mem_mb_ = 0;
    bool is_multi_instance_ {false};
    std::shared_ptr<HLIEDeviceService> device_service_;
};

HLIE_NAMESPACE_END


#endif // HLIE_MODEL_SERVICE_H

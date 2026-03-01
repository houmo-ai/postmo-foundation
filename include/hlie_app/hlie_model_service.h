#ifndef HLIE_MODEL_SERVICE_H
#define HLIE_MODEL_SERVICE_H

#include <atomic>
#include <string>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class HLIEDeviceService;


const std::string LLM = "llm";
const std::string SD = "sd";
const std::string WHISPER = "whisper";

struct HLIEModelType {
    static constexpr auto LLM = "llm";
    static constexpr auto SD = "sd";
    static constexpr auto WHISPER = "whisper";

    // model real instance type
    static constexpr auto Qwen25 = "qwen2.5";
    static constexpr auto Qwen3 = "qwen3";
    static constexpr auto DeepSeek = "deepseek";
    static constexpr auto SD3 = "sd3";
};

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
    bool is_multi_instance_ {false};
    std::shared_ptr<HLIEDeviceService> device_service_;
};

HLIE_NAMESPACE_END


#endif // HLIE_MODEL_SERVICE_H

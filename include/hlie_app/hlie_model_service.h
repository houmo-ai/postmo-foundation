#ifndef HLIE_MODEL_SERVICE_H
#define HLIE_MODEL_SERVICE_H

#include <atomic>
#include <string>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN


const std::string LLM = "LLM";
const std::string SD = "SD";

struct HLIEModelType {
    static constexpr auto LLM = "LLM";
    static constexpr auto SD = "SD";

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

    virtual int init(const std::string& model_path, const HLIEModelInfo& model_info) = 0;

    virtual void run() = 0;

    virtual void process() = 0;

    virtual void stop() = 0;

    virtual int get_state() = 0;

protected:
    std::string model_path_;
    std::atomic<int> state_;
};

HLIE_NAMESPACE_END


#endif // HLIE_MODEL_SERVICE_H

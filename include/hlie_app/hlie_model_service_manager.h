#ifndef HLIE_MODEL_SERVICE_MANAGER_H
#define HLIE_MODEL_SERVICE_MANAGER_H

#include <memory>
#include <string>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class IHLIEModelService;
class LLModelService;
class SDModelService;
class WhisperModelService;
class CosyVoiceModelService;
class HLIEDeviceService;
class HLIEModelInfo;

// Owns live model service instances. Workflow keeps orchestration; this class keeps service storage.
class HLIEModelServiceManager {
public:
    std::shared_ptr<LLModelService>& llm() { return llm_model_service_; }
    [[nodiscard]] const std::shared_ptr<LLModelService>& llm() const { return llm_model_service_; }

    std::shared_ptr<SDModelService>& sd() { return sd_model_service_; }
    [[nodiscard]] const std::shared_ptr<SDModelService>& sd() const { return sd_model_service_; }

    std::shared_ptr<WhisperModelService>& whisper() { return whisper_model_service_; }
    [[nodiscard]] const std::shared_ptr<WhisperModelService>& whisper() const { return whisper_model_service_; }

    std::shared_ptr<CosyVoiceModelService>& cosyvoice() { return cosy_voice_model_service_; }
    [[nodiscard]] const std::shared_ptr<CosyVoiceModelService>& cosyvoice() const { return cosy_voice_model_service_; }

    [[nodiscard]] std::shared_ptr<IHLIEModelService> get_by_type(const std::string& type) const;

    bool load_llm(const std::string& model_dir, const HLIEModelInfo& model_info,
                  const std::shared_ptr<HLIEDeviceService>& device_service);
    bool load_sd(const std::string& model_dir, const HLIEModelInfo& model_info,
                 const std::shared_ptr<HLIEDeviceService>& device_service);
    bool load_whisper(const std::string& model_dir, const HLIEModelInfo& model_info,
                      const std::shared_ptr<HLIEDeviceService>& device_service);
    bool load_cosyvoice(const std::string& model_dir, const HLIEModelInfo& model_info,
                        const std::shared_ptr<HLIEDeviceService>& device_service);

    bool start_llm(const std::string& model_dir, const HLIEModelInfo& model_info,
                   const std::shared_ptr<HLIEDeviceService>& device_service);
    bool start_sd(const std::string& model_dir, const HLIEModelInfo& model_info,
                  const std::shared_ptr<HLIEDeviceService>& device_service);
    bool start_whisper(const std::string& model_dir, const HLIEModelInfo& model_info,
                       const std::shared_ptr<HLIEDeviceService>& device_service);
    bool start_cosyvoice(const std::string& model_dir, const HLIEModelInfo& model_info,
                         const std::shared_ptr<HLIEDeviceService>& device_service);

    std::shared_ptr<LLModelService> detach_llm();
    std::shared_ptr<SDModelService> detach_sd();
    std::shared_ptr<WhisperModelService> detach_whisper();
    std::shared_ptr<CosyVoiceModelService> detach_cosyvoice();

    void reset_llm();
    void reset_sd();
    void reset_whisper();
    void reset_cosyvoice();

    void stop_and_reset_llm();
    void stop_and_reset_sd();
    void stop_and_reset_whisper();
    void stop_and_reset_cosyvoice();

private:
    std::shared_ptr<LLModelService> llm_model_service_;
    std::shared_ptr<SDModelService> sd_model_service_;
    std::shared_ptr<WhisperModelService> whisper_model_service_;
    std::shared_ptr<CosyVoiceModelService> cosy_voice_model_service_;
};

HLIE_NAMESPACE_END

#endif // HLIE_MODEL_SERVICE_MANAGER_H

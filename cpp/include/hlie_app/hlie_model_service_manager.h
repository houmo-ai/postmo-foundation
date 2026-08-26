#ifndef HLIE_MODEL_SERVICE_MANAGER_H
#define HLIE_MODEL_SERVICE_MANAGER_H

#include <memory>
#include <optional>
#include <string>

#include "hlie_core/runtime/hlie_service_slot.h"
#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class IHLIEModelService;
class LLModelService;
class SDModelService;
class WhisperModelService;
class CosyVoiceModelService;
class Qwen3TTSModelService;
class DocumentPipelineService;
class HLIEDeviceService;
class HLIEModelInfo;

// Owns live model service instances. Workflow keeps orchestration; this class keeps service storage.
class HLIEModelServiceManager {
public:
    [[nodiscard]] std::shared_ptr<LLModelService> llm() const { return llm_model_service_.snapshot(); }

    [[nodiscard]] std::shared_ptr<SDModelService> sd() const { return sd_model_service_.snapshot(); }

    [[nodiscard]] std::shared_ptr<WhisperModelService> whisper() const { return whisper_model_service_.snapshot(); }

    [[nodiscard]] std::shared_ptr<CosyVoiceModelService> cosyvoice() const { return cosy_voice_model_service_.snapshot(); }

    [[nodiscard]] std::shared_ptr<Qwen3TTSModelService> qwen3_tts() const { return qwen3_tts_model_service_.snapshot(); }

    [[nodiscard]] std::shared_ptr<DocumentPipelineService> document_pipeline() const {
        return document_pipeline_service_.snapshot();
    }

    [[nodiscard]] std::shared_ptr<IHLIEModelService> get_by_type(const std::string& type) const;
    // Compatibility projection for initialized slots that have not yet been
    // published as Controller Runtime handles.
    [[nodiscard]] std::optional<std::string> model_key_by_type(
        const std::string& type) const;

    // Compatibility façade used while Workflow is strangled. Backend-specific
    // slot ownership remains here until P1-4 introduces IHLIEModelRuntime.
    bool load_by_model_info(const std::string& model_dir,
                            const HLIEModelInfo& model_info,
                            const std::shared_ptr<HLIEDeviceService>& device_service);
    bool start_by_model_info(const std::string& model_dir,
                             const HLIEModelInfo& model_info,
                             const std::shared_ptr<HLIEDeviceService>& device_service);

    bool load_llm(const std::string& model_dir, const HLIEModelInfo& model_info,
                  const std::shared_ptr<HLIEDeviceService>& device_service);
    bool load_sd(const std::string& model_dir, const HLIEModelInfo& model_info,
                 const std::shared_ptr<HLIEDeviceService>& device_service);
    bool load_whisper(const std::string& model_dir, const HLIEModelInfo& model_info,
                      const std::shared_ptr<HLIEDeviceService>& device_service);
    bool load_cosyvoice(const std::string& model_dir, const HLIEModelInfo& model_info,
                        const std::shared_ptr<HLIEDeviceService>& device_service);
    bool load_qwen3_tts(const std::string& model_dir, const HLIEModelInfo& model_info,
                        const std::shared_ptr<HLIEDeviceService>& device_service);
    bool load_document_pipeline(const std::string& model_dir, const HLIEModelInfo& model_info,
                                const std::shared_ptr<HLIEDeviceService>& device_service);

    bool start_llm(const std::string& model_dir, const HLIEModelInfo& model_info,
                   const std::shared_ptr<HLIEDeviceService>& device_service);
    bool start_sd(const std::string& model_dir, const HLIEModelInfo& model_info,
                  const std::shared_ptr<HLIEDeviceService>& device_service);
    bool start_whisper(const std::string& model_dir, const HLIEModelInfo& model_info,
                       const std::shared_ptr<HLIEDeviceService>& device_service);
    bool start_cosyvoice(const std::string& model_dir, const HLIEModelInfo& model_info,
                         const std::shared_ptr<HLIEDeviceService>& device_service);
    bool start_qwen3_tts(const std::string& model_dir, const HLIEModelInfo& model_info,
                         const std::shared_ptr<HLIEDeviceService>& device_service);
    bool start_document_pipeline(const std::string& model_dir, const HLIEModelInfo& model_info,
                                 const std::shared_ptr<HLIEDeviceService>& device_service);

    std::shared_ptr<LLModelService> detach_llm();
    std::shared_ptr<SDModelService> detach_sd();
    std::shared_ptr<WhisperModelService> detach_whisper();
    std::shared_ptr<CosyVoiceModelService> detach_cosyvoice();
    std::shared_ptr<Qwen3TTSModelService> detach_qwen3_tts();
    std::shared_ptr<DocumentPipelineService> detach_document_pipeline();

    void reset_llm();
    void reset_sd();
    void reset_whisper();
    void reset_cosyvoice();
    void reset_qwen3_tts();
    void reset_document_pipeline();

    [[nodiscard]] bool stop_and_reset_llm();
    [[nodiscard]] bool stop_and_reset_sd();
    [[nodiscard]] bool stop_and_reset_whisper();
    [[nodiscard]] bool stop_and_reset_cosyvoice();
    [[nodiscard]] bool stop_and_reset_qwen3_tts();
    [[nodiscard]] bool stop_and_reset_document_pipeline();
    [[nodiscard]] bool stop_and_reset_by_type(
        const std::string& type, bool owns_document_companion = false);

private:
    HLIEServiceSlot<LLModelService> llm_model_service_;
    HLIEServiceSlot<SDModelService> sd_model_service_;
    HLIEServiceSlot<WhisperModelService> whisper_model_service_;
    HLIEServiceSlot<CosyVoiceModelService> cosy_voice_model_service_;
    HLIEServiceSlot<Qwen3TTSModelService> qwen3_tts_model_service_;
    HLIEServiceSlot<DocumentPipelineService> document_pipeline_service_;
};

HLIE_NAMESPACE_END

#endif // HLIE_MODEL_SERVICE_MANAGER_H

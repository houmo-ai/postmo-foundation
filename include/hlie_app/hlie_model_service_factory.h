#ifndef HLIE_MODEL_SERVICE_FACTORY_H
#define HLIE_MODEL_SERVICE_FACTORY_H

#include <memory>
#include <string>

#include "hlie_base/hlie_base.h"
#include "hlie_core/backend/hlie_backend_registry.h"

HLIE_NAMESPACE_BEGIN

class HLIEModelInfo;
class IHLIEModelService;
class LLModelService;
class SDModelService;
class WhisperModelService;
class CosyVoiceModelService;
class Qwen3TTSModelService;
class DocumentPipelineService;

// Migration alias retained for Workflow call sites. Backend classification is
// owned by hlie-core; concrete service construction remains in hlie-app.
using ModelServiceKind = HLIEBackendKind;

// Factory for concrete model services. Keeps Workflow from knowing constructor details.
// Centralizes model type classification and service creation.
class HLIEModelServiceFactory {
public:
    // Canonical immutable backend table for the current HOUMO runtime.
    // Composition services should source their backend policy from here.
    [[nodiscard]] static const HLIEBackendRegistry& default_registry();

    static ModelServiceKind service_kind(const HLIEModelInfo& model_info);
    static bool uses_llm_primary_service(const HLIEModelInfo& model_info);
    static bool is_supported_model_type(const std::string& model_type);
    static std::shared_ptr<IHLIEModelService> create_by_model_info(const HLIEModelInfo& model_info);
    static std::shared_ptr<IHLIEModelService> create_by_model_type(const std::string& model_type);

    template <typename ServiceT>
    static std::shared_ptr<ServiceT> create_typed_by_model_info(const HLIEModelInfo& model_info) {
        return std::dynamic_pointer_cast<ServiceT>(create_by_model_info(model_info));
    }

    template <typename ServiceT>
    static std::shared_ptr<ServiceT> create_typed_by_model_type(const std::string& model_type) {
        return std::dynamic_pointer_cast<ServiceT>(create_by_model_type(model_type));
    }

    static std::shared_ptr<LLModelService> create_llm();
    static std::shared_ptr<SDModelService> create_sd();
    static std::shared_ptr<WhisperModelService> create_whisper();
    static std::shared_ptr<CosyVoiceModelService> create_cosyvoice();
    static std::shared_ptr<Qwen3TTSModelService> create_qwen3_tts();
    static std::shared_ptr<DocumentPipelineService> create_document_pipeline();
};

HLIE_NAMESPACE_END

#endif // HLIE_MODEL_SERVICE_FACTORY_H

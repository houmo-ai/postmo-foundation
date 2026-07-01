#include "hlie_app/hlie_model_service_factory.h"

#include "hlie_app/hlie_cosyvoice_model_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_sd_model_service.h"
#include "hlie_app/hlie_whisper_model_service.h"
#include "hlie_base/hlie_util.h"

HLIE_NAMESPACE_BEGIN

ModelServiceKind HLIEModelServiceFactory::service_kind(const HLIEModelInfo& model_info) {
    const auto& model_name = model_info.get_model_name();
    if (HlieUtil::contains_ignore_case(model_name, WHISPER)) {
        return ModelServiceKind::Whisper;
    }
    if (HlieUtil::contains_ignore_case(model_name, HLIEModelType::COSYVOICE)) {
        return ModelServiceKind::CosyVoice;
    }
    if (model_info.is_llm_) {
        return ModelServiceKind::LLM;
    }
    return ModelServiceKind::SD;
}

std::shared_ptr<IHLIEModelService> HLIEModelServiceFactory::create_by_model_info(const HLIEModelInfo& model_info) {
    switch (service_kind(model_info)) {
    case ModelServiceKind::Whisper:
        return create_whisper();
    case ModelServiceKind::CosyVoice:
        return create_cosyvoice();
    case ModelServiceKind::SD:
        return create_sd();
    case ModelServiceKind::LLM:
    default:
        return create_llm();
    }
}

bool HLIEModelServiceFactory::is_supported_model_type(const std::string& model_type) {
    return model_type == HLIEModelType::LLM || model_type == HLIEModelType::SD ||
           model_type == HLIEModelType::WHISPER || model_type == HLIEModelType::COSYVOICE;
}

std::shared_ptr<IHLIEModelService> HLIEModelServiceFactory::create_by_model_type(const std::string& model_type) {
    if (model_type == HLIEModelType::LLM) {
        return create_llm();
    }
    if (model_type == HLIEModelType::SD) {
        return create_sd();
    }
    if (model_type == HLIEModelType::WHISPER) {
        return create_whisper();
    }
    if (model_type == HLIEModelType::COSYVOICE) {
        return create_cosyvoice();
    }
    return nullptr;
}

std::shared_ptr<LLModelService> HLIEModelServiceFactory::create_llm() {
    return std::make_shared<LLModelService>();
}

std::shared_ptr<SDModelService> HLIEModelServiceFactory::create_sd() {
    return std::make_shared<SDModelService>();
}

std::shared_ptr<WhisperModelService> HLIEModelServiceFactory::create_whisper() {
    return std::make_shared<WhisperModelService>();
}

std::shared_ptr<CosyVoiceModelService> HLIEModelServiceFactory::create_cosyvoice() {
    return std::make_shared<CosyVoiceModelService>();
}

HLIE_NAMESPACE_END

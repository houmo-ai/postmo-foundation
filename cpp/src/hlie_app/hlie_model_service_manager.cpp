#include "hlie_app/hlie_model_service_manager.h"

#include "hlie_app/hlie_cosyvoice_model_service.h"
#include "hlie_app/document/hlie_document_pipeline_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_service_factory.h"
#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_qwen3_tts_model_service.h"
#include "hlie_app/hlie_sd_model_service.h"
#include "hlie_app/hlie_whisper_model_service.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

namespace {
    template <typename ServiceT>
    bool stop_service_noexcept(const std::shared_ptr<ServiceT>& service) noexcept {
        if (!service) {
            return true;
        }

        try {
            service->stop();
            return true;
        } catch (const std::exception& error) {
            LOG_ERROR("Model service stop threw: {}", error.what());
        } catch (...) {
            LOG_ERROR("Model service stop threw an unknown exception");
        }
        return false;
    }

    template <typename ServiceT, typename CreateFn>
    bool load_service(HLIEServiceSlot<ServiceT>& slot,
                      CreateFn create_service,
                      const std::string& model_dir,
                      const HLIEModelInfo& model_info,
                      const std::shared_ptr<HLIEDeviceService>& device_service,
                      const bool start_after_init) {
        const auto ticket = slot.begin_load();
        if (!ticket) {
            return false;
        }

        std::shared_ptr<ServiceT> candidate;
        try {
            candidate = create_service();
            if (!candidate || candidate->init(model_dir, model_info, device_service) < 0) {
                (void)slot.fail_load(*ticket);
                stop_service_noexcept(candidate);
                return false;
            }

            if (start_after_init) {
                candidate->run();
            }

            if (!slot.publish(*ticket, candidate)) {
                stop_service_noexcept(candidate);
                return false;
            }
            return true;
        } catch (...) {
            (void)slot.fail_load(*ticket);
            stop_service_noexcept(candidate);
            throw;
        }
    }
} // namespace

std::shared_ptr<IHLIEModelService> HLIEModelServiceManager::get_by_type(const std::string& type) const {
    if (type == HLIEModelType::LLM) {
        return llm_model_service_.snapshot();
    }
    if (type == HLIEModelType::SD) {
        return sd_model_service_.snapshot();
    }
    if (type == HLIEModelType::WHISPER) {
        return whisper_model_service_.snapshot();
    }
    if (type == HLIEModelType::COSYVOICE || type == HLIEModelType::TTS) {
        return cosy_voice_model_service_.snapshot();
    }
    if (type == HLIEModelType::QWEN3_TTS) {
        return qwen3_tts_model_service_.snapshot();
    }
    if (type == HLIEModelType::DOCUMENT) {
        return document_pipeline_service_.snapshot();
    }
    return nullptr;
}

std::optional<std::string> HLIEModelServiceManager::model_key_by_type(
    const std::string& type) const {
    if (type == HLIEModelType::LLM) {
        const auto service = llm();
        return service ? std::optional<std::string>{service->get_model_key()}
                       : std::nullopt;
    }
    if (type == HLIEModelType::SD) {
        return sd() ? std::optional<std::string>{HLIEModelType::SD3}
                    : std::nullopt;
    }
    if (type == HLIEModelType::WHISPER) {
        const auto service = whisper();
        return service ? std::optional<std::string>{service->get_model_key()}
                       : std::nullopt;
    }
    if (type == HLIEModelType::COSYVOICE || type == HLIEModelType::TTS) {
        const auto service = cosyvoice();
        return service ? std::optional<std::string>{service->get_model_key()}
                       : std::nullopt;
    }
    if (type == HLIEModelType::QWEN3_TTS) {
        const auto service = qwen3_tts();
        return service ? std::optional<std::string>{service->get_model_key()}
                       : std::nullopt;
    }
    return std::nullopt;
}

bool HLIEModelServiceManager::load_by_model_info(
    const std::string& model_dir,
    const HLIEModelInfo& model_info,
    const std::shared_ptr<HLIEDeviceService>& device_service) {
    switch (HLIEModelServiceFactory::service_kind(model_info)) {
    case ModelServiceKind::Whisper:
        return load_whisper(model_dir, model_info, device_service);
    case ModelServiceKind::CosyVoice:
        return load_cosyvoice(model_dir, model_info, device_service);
    case ModelServiceKind::Qwen3TTS:
        return load_qwen3_tts(model_dir, model_info, device_service);
    case ModelServiceKind::SD:
        return load_sd(model_dir, model_info, device_service);
    case ModelServiceKind::Document:
    case ModelServiceKind::LLM:
    default:
        if (!load_llm(model_dir, model_info, device_service)) {
            return false;
        }
        if (DocumentPipelineService::supports_document_layout(model_dir, model_info) &&
            !load_document_pipeline(model_dir, model_info, device_service)) {
            (void)stop_and_reset_llm();
            return false;
        }
        return true;
    }
}

bool HLIEModelServiceManager::start_by_model_info(
    const std::string& model_dir,
    const HLIEModelInfo& model_info,
    const std::shared_ptr<HLIEDeviceService>& device_service) {
    switch (HLIEModelServiceFactory::service_kind(model_info)) {
    case ModelServiceKind::Whisper:
        return start_whisper(model_dir, model_info, device_service);
    case ModelServiceKind::CosyVoice:
        return start_cosyvoice(model_dir, model_info, device_service);
    case ModelServiceKind::Qwen3TTS:
        return start_qwen3_tts(model_dir, model_info, device_service);
    case ModelServiceKind::SD:
        return start_sd(model_dir, model_info, device_service);
    case ModelServiceKind::Document:
    case ModelServiceKind::LLM:
    default:
        if (!start_llm(model_dir, model_info, device_service)) {
            return false;
        }
        if (DocumentPipelineService::supports_document_layout(model_dir, model_info) &&
            !start_document_pipeline(model_dir, model_info, device_service)) {
            (void)stop_and_reset_llm();
            return false;
        }
        return true;
    }
}

bool HLIEModelServiceManager::load_llm(const std::string& model_dir, const HLIEModelInfo& model_info,
                                       const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(llm_model_service_, HLIEModelServiceFactory::create_llm, model_dir, model_info, device_service, false);
}

bool HLIEModelServiceManager::load_sd(const std::string& model_dir, const HLIEModelInfo& model_info,
                                      const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(sd_model_service_, HLIEModelServiceFactory::create_sd, model_dir, model_info, device_service, false);
}

bool HLIEModelServiceManager::load_whisper(const std::string& model_dir, const HLIEModelInfo& model_info,
                                           const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(whisper_model_service_, HLIEModelServiceFactory::create_whisper, model_dir, model_info, device_service, false);
}

bool HLIEModelServiceManager::load_cosyvoice(const std::string& model_dir, const HLIEModelInfo& model_info,
                                             const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(cosy_voice_model_service_, HLIEModelServiceFactory::create_cosyvoice,
                        model_dir, model_info, device_service, false);
}

bool HLIEModelServiceManager::load_qwen3_tts(const std::string& model_dir, const HLIEModelInfo& model_info,
                                             const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(qwen3_tts_model_service_, HLIEModelServiceFactory::create_qwen3_tts,
                        model_dir, model_info, device_service, false);
}

bool HLIEModelServiceManager::load_document_pipeline(const std::string& model_dir, const HLIEModelInfo& model_info,
                                                     const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(document_pipeline_service_, HLIEModelServiceFactory::create_document_pipeline,
                        model_dir, model_info, device_service, false);
}

bool HLIEModelServiceManager::start_llm(const std::string& model_dir, const HLIEModelInfo& model_info,
                                        const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(llm_model_service_, HLIEModelServiceFactory::create_llm,
                        model_dir, model_info, device_service, true);
}

bool HLIEModelServiceManager::start_sd(const std::string& model_dir, const HLIEModelInfo& model_info,
                                       const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(sd_model_service_, HLIEModelServiceFactory::create_sd,
                        model_dir, model_info, device_service, true);
}

bool HLIEModelServiceManager::start_whisper(const std::string& model_dir, const HLIEModelInfo& model_info,
                                            const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(whisper_model_service_, HLIEModelServiceFactory::create_whisper,
                        model_dir, model_info, device_service, true);
}

bool HLIEModelServiceManager::start_cosyvoice(const std::string& model_dir, const HLIEModelInfo& model_info,
                                              const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(cosy_voice_model_service_, HLIEModelServiceFactory::create_cosyvoice,
                        model_dir, model_info, device_service, true);
}

bool HLIEModelServiceManager::start_qwen3_tts(const std::string& model_dir, const HLIEModelInfo& model_info,
                                              const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(qwen3_tts_model_service_, HLIEModelServiceFactory::create_qwen3_tts,
                        model_dir, model_info, device_service, true);
}

bool HLIEModelServiceManager::start_document_pipeline(const std::string& model_dir, const HLIEModelInfo& model_info,
                                                      const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(document_pipeline_service_, HLIEModelServiceFactory::create_document_pipeline,
                        model_dir, model_info, device_service, true);
}

std::shared_ptr<LLModelService> HLIEModelServiceManager::detach_llm() {
    return llm_model_service_.detach();
}

std::shared_ptr<SDModelService> HLIEModelServiceManager::detach_sd() {
    return sd_model_service_.detach();
}

std::shared_ptr<WhisperModelService> HLIEModelServiceManager::detach_whisper() {
    return whisper_model_service_.detach();
}

std::shared_ptr<CosyVoiceModelService> HLIEModelServiceManager::detach_cosyvoice() {
    return cosy_voice_model_service_.detach();
}

std::shared_ptr<Qwen3TTSModelService> HLIEModelServiceManager::detach_qwen3_tts() {
    return qwen3_tts_model_service_.detach();
}

std::shared_ptr<DocumentPipelineService> HLIEModelServiceManager::detach_document_pipeline() {
    return document_pipeline_service_.detach();
}

void HLIEModelServiceManager::reset_llm() {
    (void)llm_model_service_.detach();
}

void HLIEModelServiceManager::reset_sd() {
    (void)sd_model_service_.detach();
}

void HLIEModelServiceManager::reset_whisper() {
    (void)whisper_model_service_.detach();
}

void HLIEModelServiceManager::reset_cosyvoice() {
    (void)cosy_voice_model_service_.detach();
}

void HLIEModelServiceManager::reset_qwen3_tts() {
    (void)qwen3_tts_model_service_.detach();
}

void HLIEModelServiceManager::reset_document_pipeline() {
    (void)document_pipeline_service_.detach();
}

bool HLIEModelServiceManager::stop_and_reset_llm() {
    return stop_service_noexcept(llm_model_service_.detach());
}

bool HLIEModelServiceManager::stop_and_reset_sd() {
    return stop_service_noexcept(sd_model_service_.detach());
}

bool HLIEModelServiceManager::stop_and_reset_whisper() {
    return stop_service_noexcept(whisper_model_service_.detach());
}

bool HLIEModelServiceManager::stop_and_reset_cosyvoice() {
    return stop_service_noexcept(cosy_voice_model_service_.detach());
}

bool HLIEModelServiceManager::stop_and_reset_qwen3_tts() {
    return stop_service_noexcept(qwen3_tts_model_service_.detach());
}

bool HLIEModelServiceManager::stop_and_reset_document_pipeline() {
    return stop_service_noexcept(document_pipeline_service_.detach());
}

bool HLIEModelServiceManager::stop_and_reset_by_type(
    const std::string& type, const bool owns_document_companion) {
    bool confirmed = true;
    if (type == HLIEModelType::LLM) {
        if (owns_document_companion) {
            confirmed = stop_and_reset_document_pipeline() && confirmed;
        }
        confirmed = stop_and_reset_llm() && confirmed;
    } else if (type == HLIEModelType::SD) {
        confirmed = stop_and_reset_sd();
    } else if (type == HLIEModelType::WHISPER) {
        confirmed = stop_and_reset_whisper();
    } else if (type == HLIEModelType::COSYVOICE || type == HLIEModelType::TTS) {
        confirmed = stop_and_reset_cosyvoice();
    } else if (type == HLIEModelType::QWEN3_TTS) {
        confirmed = stop_and_reset_qwen3_tts();
    } else if (type == HLIEModelType::DOCUMENT) {
        confirmed = stop_and_reset_document_pipeline();
    } else {
        return false;
    }
    return confirmed;
}

HLIE_NAMESPACE_END

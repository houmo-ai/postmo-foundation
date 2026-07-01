#include "hlie_app/hlie_model_service_manager.h"

#include "hlie_app/hlie_cosyvoice_model_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_service_factory.h"
#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_sd_model_service.h"
#include "hlie_app/hlie_whisper_model_service.h"

HLIE_NAMESPACE_BEGIN

namespace {
    template <typename ServiceT>
    void stop_and_reset(std::shared_ptr<ServiceT>& service) {
        if (service) {
            service->stop();
        }
        service.reset();
    }

    template <typename ServiceT, typename CreateFn>
    bool load_service(std::shared_ptr<ServiceT>& service,
                      CreateFn create_service,
                      const std::string& model_dir,
                      const HLIEModelInfo& model_info,
                      const std::shared_ptr<HLIEDeviceService>& device_service) {
        // Loading must not replace a live service silently. Call detach/stop/reset first.
        if (service) {
            return false;
        }

        service = create_service();
        if (!service || service->init(model_dir, model_info, device_service) < 0) {
            stop_and_reset(service);
            return false;
        }
        return true;
    }

    template <typename ServiceT, typename LoadFn>
    bool start_service(std::shared_ptr<ServiceT>& service,
                       LoadFn load,
                       const std::string& model_dir,
                       const HLIEModelInfo& model_info,
                       const std::shared_ptr<HLIEDeviceService>& device_service) {
        if (!load(model_dir, model_info, device_service)) {
            return false;
        }
        service->run();
        return true;
    }
} // namespace

std::shared_ptr<IHLIEModelService> HLIEModelServiceManager::get_by_type(const std::string& type) const {
    if (type == HLIEModelType::LLM) {
        return llm_model_service_;
    }
    if (type == HLIEModelType::SD) {
        return sd_model_service_;
    }
    if (type == HLIEModelType::WHISPER) {
        return whisper_model_service_;
    }
    if (type == HLIEModelType::COSYVOICE) {
        return cosy_voice_model_service_;
    }
    return nullptr;
}

bool HLIEModelServiceManager::load_llm(const std::string& model_dir, const HLIEModelInfo& model_info,
                                       const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(llm_model_service_, HLIEModelServiceFactory::create_llm, model_dir, model_info, device_service);
}

bool HLIEModelServiceManager::load_sd(const std::string& model_dir, const HLIEModelInfo& model_info,
                                      const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(sd_model_service_, HLIEModelServiceFactory::create_sd, model_dir, model_info, device_service);
}

bool HLIEModelServiceManager::load_whisper(const std::string& model_dir, const HLIEModelInfo& model_info,
                                           const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(whisper_model_service_, HLIEModelServiceFactory::create_whisper, model_dir, model_info, device_service);
}

bool HLIEModelServiceManager::load_cosyvoice(const std::string& model_dir, const HLIEModelInfo& model_info,
                                             const std::shared_ptr<HLIEDeviceService>& device_service) {
    return load_service(cosy_voice_model_service_, HLIEModelServiceFactory::create_cosyvoice, model_dir, model_info, device_service);
}

bool HLIEModelServiceManager::start_llm(const std::string& model_dir, const HLIEModelInfo& model_info,
                                        const std::shared_ptr<HLIEDeviceService>& device_service) {
    return start_service(llm_model_service_,
                         [this](const std::string& dir, const HLIEModelInfo& info,
                                const std::shared_ptr<HLIEDeviceService>& service) {
                             return load_llm(dir, info, service);
                         },
                         model_dir, model_info, device_service);
}

bool HLIEModelServiceManager::start_sd(const std::string& model_dir, const HLIEModelInfo& model_info,
                                       const std::shared_ptr<HLIEDeviceService>& device_service) {
    return start_service(sd_model_service_,
                         [this](const std::string& dir, const HLIEModelInfo& info,
                                const std::shared_ptr<HLIEDeviceService>& service) {
                             return load_sd(dir, info, service);
                         },
                         model_dir, model_info, device_service);
}

bool HLIEModelServiceManager::start_whisper(const std::string& model_dir, const HLIEModelInfo& model_info,
                                            const std::shared_ptr<HLIEDeviceService>& device_service) {
    return start_service(whisper_model_service_,
                         [this](const std::string& dir, const HLIEModelInfo& info,
                                const std::shared_ptr<HLIEDeviceService>& service) {
                             return load_whisper(dir, info, service);
                         },
                         model_dir, model_info, device_service);
}

bool HLIEModelServiceManager::start_cosyvoice(const std::string& model_dir, const HLIEModelInfo& model_info,
                                              const std::shared_ptr<HLIEDeviceService>& device_service) {
    return start_service(cosy_voice_model_service_,
                         [this](const std::string& dir, const HLIEModelInfo& info,
                                const std::shared_ptr<HLIEDeviceService>& service) {
                             return load_cosyvoice(dir, info, service);
                         },
                         model_dir, model_info, device_service);
}

std::shared_ptr<LLModelService> HLIEModelServiceManager::detach_llm() {
    auto service = llm_model_service_;
    llm_model_service_.reset();
    return service;
}

std::shared_ptr<SDModelService> HLIEModelServiceManager::detach_sd() {
    auto service = sd_model_service_;
    sd_model_service_.reset();
    return service;
}

std::shared_ptr<WhisperModelService> HLIEModelServiceManager::detach_whisper() {
    auto service = whisper_model_service_;
    whisper_model_service_.reset();
    return service;
}

std::shared_ptr<CosyVoiceModelService> HLIEModelServiceManager::detach_cosyvoice() {
    auto service = cosy_voice_model_service_;
    cosy_voice_model_service_.reset();
    return service;
}

void HLIEModelServiceManager::reset_llm() {
    llm_model_service_.reset();
}

void HLIEModelServiceManager::reset_sd() {
    sd_model_service_.reset();
}

void HLIEModelServiceManager::reset_whisper() {
    whisper_model_service_.reset();
}

void HLIEModelServiceManager::reset_cosyvoice() {
    cosy_voice_model_service_.reset();
}

void HLIEModelServiceManager::stop_and_reset_llm() {
    stop_and_reset(llm_model_service_);
}

void HLIEModelServiceManager::stop_and_reset_sd() {
    stop_and_reset(sd_model_service_);
}

void HLIEModelServiceManager::stop_and_reset_whisper() {
    stop_and_reset(whisper_model_service_);
}

void HLIEModelServiceManager::stop_and_reset_cosyvoice() {
    stop_and_reset(cosy_voice_model_service_);
}

HLIE_NAMESPACE_END

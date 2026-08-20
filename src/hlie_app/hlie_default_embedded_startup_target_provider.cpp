#include "hlie_app/startup/hlie_default_embedded_startup_target_provider.h"

#include <utility>

#include "hlie_app/document/hlie_document_pipeline_service.h"
#include "hlie_app/hlie_cosyvoice_model_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_service_manager.h"
#include "hlie_app/hlie_qwen3_tts_model_service.h"
#include "hlie_app/hlie_sd_model_service.h"
#include "hlie_app/hlie_whisper_model_service.h"

HLIE_NAMESPACE_BEGIN

HLIEDefaultEmbeddedStartupTargetProvider::
HLIEDefaultEmbeddedStartupTargetProvider(
    HLIEModelServiceManager& service_manager,
    LocalPortProvider local_port_provider)
    : service_manager_(service_manager),
      local_port_provider_(std::move(local_port_provider)) {}

std::vector<HLIEEmbeddedStartupTarget>
HLIEDefaultEmbeddedStartupTargetProvider::targets(
    const HLIEEmbeddedStartupGroup group) const {
    std::vector<HLIEEmbeddedStartupTarget> result;
    const int local_port = local_port_provider_ ? local_port_provider_() : 0;

    if (group == HLIEEmbeddedStartupGroup::Primary) {
        if (const auto llm = service_manager_.llm()) {
            result.push_back({
                group,
                llm->get_model_key(),
                HLIEModelType::LLM,
                llm,
                service_manager_.document_pipeline(),
                std::nullopt,
                true,
            });
        }
        return result;
    }

    if (const auto sd = service_manager_.sd()) {
        result.push_back({group, HLIEModelType::SD3, HLIEModelType::SD,
                          sd, {}, local_port, false});
    }
    if (const auto whisper = service_manager_.whisper()) {
        result.push_back({group, whisper->get_model_key(),
                          HLIEModelType::WHISPER, whisper, {}, local_port,
                          false});
    }
    if (const auto cosyvoice = service_manager_.cosyvoice()) {
        result.push_back({group, cosyvoice->get_model_key(),
                          HLIEModelType::COSYVOICE, cosyvoice, {}, local_port,
                          false});
    }
    if (const auto qwen3_tts = service_manager_.qwen3_tts()) {
        result.push_back({group, qwen3_tts->get_model_key(),
                          HLIEModelType::QWEN3_TTS, qwen3_tts, {}, local_port,
                          false});
    }
    return result;
}

HLIE_NAMESPACE_END

#include "hlie_app/hlie_model_service_factory.h"

#include <functional>
#include <stdexcept>
#include <utility>

#include "hlie_app/hlie_cosyvoice_model_service.h"
#include "hlie_app/runtime/hlie_backend_adapter_registry.h"
#include "hlie_app/document/hlie_document_pipeline_service.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_qwen3_tts_model_service.h"
#include "hlie_app/hlie_sd_model_service.h"
#include "hlie_app/hlie_whisper_model_service.h"

HLIE_NAMESPACE_BEGIN

namespace {

class FunctionBackendAdapter final : public IHLIEBackendAdapter {
public:
    using Factory = std::function<std::shared_ptr<IHLIEModelService>()>;

    FunctionBackendAdapter(std::string backend_id, Factory factory)
        : backend_id_(std::move(backend_id)), factory_(std::move(factory)) {}

    [[nodiscard]] std::string backend_id() const override { return backend_id_; }

    [[nodiscard]] std::shared_ptr<IHLIEModelService>
    create_embedded_service() const override {
        return factory_ ? factory_() : nullptr;
    }

private:
    std::string backend_id_;
    Factory factory_;
};

const HLIEBackendRegistry& default_backend_registry() {
    return houmo_default_registry();
}

const HLIEBackendAdapterRegistry& adapter_registry() {
    static const HLIEBackendAdapterRegistry registry = [] {
        HLIEBackendAdapterRegistry result;
        const auto add = [&](std::shared_ptr<const IHLIEBackendAdapter> adapter) {
            if (!result.register_adapter(std::move(adapter))) {
                throw std::logic_error("duplicate or invalid HOUMO backend adapter");
            }
        };
        add(std::make_shared<FunctionBackendAdapter>(
            "houmo.llm", [] { return std::make_shared<LLModelService>(); }));
        add(std::make_shared<FunctionBackendAdapter>(
            "houmo.sd", [] { return std::make_shared<SDModelService>(); }));
        add(std::make_shared<FunctionBackendAdapter>(
            "houmo.whisper", [] { return std::make_shared<WhisperModelService>(); }));
        add(std::make_shared<FunctionBackendAdapter>(
            "houmo.cosyvoice", [] { return std::make_shared<CosyVoiceModelService>(); }));
        add(std::make_shared<FunctionBackendAdapter>(
            "houmo.qwen3_tts", [] { return std::make_shared<Qwen3TTSModelService>(); }));
        add(std::make_shared<FunctionBackendAdapter>(
            "houmo.document", [] { return std::make_shared<DocumentPipelineService>(); }));

        for (const auto& descriptor : default_backend_registry().descriptors()) {
            if (!result.find(descriptor.backend_id)) {
                throw std::logic_error("HOUMO backend descriptor has no app adapter");
            }
        }
        return result;
    }();
    return registry;
}

std::shared_ptr<IHLIEModelService> create_for_descriptor(
    const HLIEBackendDescriptor& descriptor,
    const bool model_record_semantics) {
    // A document-capable model record is still an LLM primary service. Explicit
    // create_by_model_type("document") creates the companion service itself.
    const std::string backend_id = model_record_semantics &&
        descriptor.kind == HLIEBackendKind::Document
        ? std::string{"houmo.llm"}
        : descriptor.backend_id;
    return adapter_registry().create_embedded_service(backend_id);
}

} // namespace

const HLIEBackendRegistry& HLIEModelServiceFactory::default_registry() {
    return default_backend_registry();
}

ModelServiceKind HLIEModelServiceFactory::service_kind(const HLIEModelInfo& model_info) {
    const auto descriptor = default_backend_registry().resolve(model_info.to_model_spec());
    return descriptor ? descriptor->kind : ModelServiceKind::LLM;
}

std::shared_ptr<IHLIEModelService> HLIEModelServiceFactory::create_by_model_info(const HLIEModelInfo& model_info) {
    const auto descriptor = default_backend_registry().resolve(model_info.to_model_spec());
    return descriptor ? create_for_descriptor(*descriptor, true) : nullptr;
}

bool HLIEModelServiceFactory::is_supported_model_type(const std::string& model_type) {
    return default_backend_registry().find_by_alias(model_type).has_value();
}

bool HLIEModelServiceFactory::uses_llm_primary_service(const HLIEModelInfo& model_info) {
    const auto kind = service_kind(model_info);
    return kind == ModelServiceKind::LLM || kind == ModelServiceKind::Document;
}

std::shared_ptr<IHLIEModelService> HLIEModelServiceFactory::create_by_model_type(const std::string& model_type) {
    const auto descriptor = default_backend_registry().find_by_alias(model_type);
    return descriptor ? create_for_descriptor(*descriptor, false) : nullptr;
}

std::shared_ptr<LLModelService> HLIEModelServiceFactory::create_llm() {
    return std::dynamic_pointer_cast<LLModelService>(
        adapter_registry().create_embedded_service("houmo.llm"));
}

std::shared_ptr<SDModelService> HLIEModelServiceFactory::create_sd() {
    return std::dynamic_pointer_cast<SDModelService>(
        adapter_registry().create_embedded_service("houmo.sd"));
}

std::shared_ptr<WhisperModelService> HLIEModelServiceFactory::create_whisper() {
    return std::dynamic_pointer_cast<WhisperModelService>(
        adapter_registry().create_embedded_service("houmo.whisper"));
}

std::shared_ptr<CosyVoiceModelService> HLIEModelServiceFactory::create_cosyvoice() {
    return std::dynamic_pointer_cast<CosyVoiceModelService>(
        adapter_registry().create_embedded_service("houmo.cosyvoice"));
}

std::shared_ptr<Qwen3TTSModelService> HLIEModelServiceFactory::create_qwen3_tts() {
    return std::dynamic_pointer_cast<Qwen3TTSModelService>(
        adapter_registry().create_embedded_service("houmo.qwen3_tts"));
}

std::shared_ptr<DocumentPipelineService> HLIEModelServiceFactory::create_document_pipeline() {
    return std::dynamic_pointer_cast<DocumentPipelineService>(
        adapter_registry().create_embedded_service("houmo.document"));
}

HLIE_NAMESPACE_END

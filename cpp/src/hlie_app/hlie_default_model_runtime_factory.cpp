#include "hlie_app/runtime/hlie_default_model_runtime_factory.h"

#include <exception>
#include <utility>

#include "hlie_app/hlie_model_service_manager.h"
#include "hlie_app/runtime/hlie_embedded_model_runtime.h"
#include "hlie_app/runtime/hlie_process_model_runtime.h"
#include "hlie_app/runtime/hlie_worker_supervisor.h"
#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

HLIEDefaultModelRuntimeFactory::HLIEDefaultModelRuntimeFactory(
    HLIEModelServiceManager& service_manager,
    SupervisorProvider supervisor_provider,
    EmbeddedIdentityAllocator embedded_identity_allocator)
    : service_manager_(service_manager),
      supervisor_provider_(std::move(supervisor_provider)),
      embedded_identity_allocator_(std::move(embedded_identity_allocator)) {}

std::shared_ptr<IHLIEModelRuntime>
HLIEDefaultModelRuntimeFactory::create_embedded(
    const HLIEEmbeddedRuntimeCreateRequest& request) {
    if (!request.launch_ticket.valid() || request.service_kind.empty() ||
        !embedded_identity_allocator_) {
        return {};
    }
    const auto identity = embedded_identity_allocator_(request.launch_ticket);
    if (!identity.valid() || identity.kind != HLIERuntimeKind::Embedded) {
        return {};
    }
    try {
        return std::make_shared<HLIEEmbeddedModelRuntime>(
            service_manager_, identity, request.capabilities,
            request.service_kind, request.owns_document_companion);
    } catch (const std::exception& error) {
        LOG_ERROR("Embedded RuntimeFactory creation failed: model={}, generation={}, error={}",
                  request.launch_ticket.model_key,
                  request.launch_ticket.generation, error.what());
        return {};
    }
}

std::shared_ptr<IHLIEModelRuntime>
HLIEDefaultModelRuntimeFactory::create_process(
    const HLIEProcessRuntimeCreateRequest& request) {
    auto* supervisor = supervisor_provider_ ? supervisor_provider_() : nullptr;
    if (!supervisor || !request.identity.valid() ||
        request.identity.kind != HLIERuntimeKind::Process) {
        return {};
    }
    HLIEWorkerRuntime worker{
        request.identity.runtime_id,
        request.identity.instance_id,
        request.identity.endpoint_port,
        request.model_names,
        request.identity.launch_ticket,
        request.ready,
    };
    try {
        return std::make_shared<HLIEProcessModelRuntime>(
            *supervisor, std::move(worker), request.capabilities);
    } catch (const std::exception& error) {
        LOG_ERROR("Process RuntimeFactory creation failed: model={}, generation={}, runtime_id={}, error={}",
                  request.identity.launch_ticket.model_key,
                  request.identity.launch_ticket.generation,
                  request.identity.runtime_id, error.what());
        return {};
    }
}

HLIE_NAMESPACE_END

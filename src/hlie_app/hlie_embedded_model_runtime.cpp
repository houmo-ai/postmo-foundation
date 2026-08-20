#include "hlie_app/runtime/hlie_embedded_model_runtime.h"

#include <mutex>
#include <stdexcept>
#include <utility>

#include "hlie_app/hlie_model_service.h"
#include "hlie_app/hlie_model_service_manager.h"
#include "hlie_base/hlie_common_define.h"

HLIE_NAMESPACE_BEGIN

class HLIEEmbeddedModelRuntime::Impl {
public:
    Impl(HLIEModelServiceManager& service_manager,
         HLIERuntimeIdentity identity,
         HLIEModelCapabilities capabilities,
         std::string service_kind,
         const bool owns_document_companion)
        : service_manager_(service_manager), identity_(std::move(identity)),
          capabilities_(capabilities), service_kind_(std::move(service_kind)),
          owns_document_companion_(owns_document_companion) {
        const bool supported_kind = service_kind_ == HLIEModelType::LLM ||
            service_kind_ == HLIEModelType::SD ||
            service_kind_ == HLIEModelType::WHISPER ||
            service_kind_ == HLIEModelType::COSYVOICE ||
            service_kind_ == HLIEModelType::TTS ||
            service_kind_ == HLIEModelType::QWEN3_TTS ||
            service_kind_ == HLIEModelType::DOCUMENT;
        if (!identity_.valid() || identity_.kind != HLIERuntimeKind::Embedded ||
            !supported_kind) {
            throw std::invalid_argument("invalid embedded runtime identity or service kind");
        }
    }

    [[nodiscard]] HLIERuntimeIdentity identity() const { return identity_; }
    [[nodiscard]] HLIEModelCapabilities capabilities() const { return capabilities_; }

    [[nodiscard]] HLIERuntimeHealth health() const {
        std::lock_guard lock(mutex_);
        if (stopped_) {
            return {identity_, HLIERuntimeHealthState::Stopped, 0};
        }
        if (draining_) {
            return {identity_, HLIERuntimeHealthState::Draining, 0};
        }
        const auto service = service_manager_.get_by_type(service_kind_);
        if (!service) {
            return {identity_, HLIERuntimeHealthState::Starting, 0};
        }
        if (service->get_state() != SERVER_STATE_READY) {
            return {identity_, HLIERuntimeHealthState::Starting, 0};
        }
        if (owns_document_companion_) {
            const auto document = service_manager_.get_by_type(HLIEModelType::DOCUMENT);
            if (!document || document->get_state() != SERVER_STATE_READY) {
                return {identity_, HLIERuntimeHealthState::Starting, 0};
            }
        }
        return {identity_, HLIERuntimeHealthState::Ready, 0};
    }

    void begin_draining() {
        std::lock_guard lock(mutex_);
        if (!stopped_) {
            draining_ = true;
        }
    }

    [[nodiscard]] HLIERuntimeStopResult stop(const HLIERuntimeStopRequest& request) {
        HLIERuntimeStopResult result;
        result.identity = identity_;
        if (!request.valid()) {
            result.native_code = -1;
            return result;
        }

        std::lock_guard lock(mutex_);
        if (stopped_) {
            result.confirmed = true;
            result.graceful = true;
            result.already_stopped = true;
            result.deadline_exceeded = std::chrono::steady_clock::now() > request.deadline;
            return result;
        }
        draining_ = true;
        const bool existed = service_manager_.get_by_type(service_kind_) != nullptr ||
            (owns_document_companion_ &&
             service_manager_.get_by_type(HLIEModelType::DOCUMENT) != nullptr);
        const bool native_stop_confirmed = stop_bundle();
        const bool primary_gone = !service_manager_.get_by_type(service_kind_);
        const bool companion_gone = !owns_document_companion_ ||
            !service_manager_.get_by_type(HLIEModelType::DOCUMENT);
        result.confirmed = native_stop_confirmed &&
            primary_gone && companion_gone;
        result.already_stopped = !existed;
        result.deadline_exceeded = std::chrono::steady_clock::now() > request.deadline;
        result.graceful = result.confirmed && !result.deadline_exceeded;
        if (result.confirmed) {
            stopped_ = true;
        } else {
            result.native_code = -2;
        }
        return result;
    }

private:
    [[nodiscard]] bool stop_bundle() {
        if (service_kind_ == HLIEModelType::LLM) {
            bool confirmed = true;
            if (owns_document_companion_) {
                confirmed = service_manager_.stop_and_reset_document_pipeline();
            }
            return service_manager_.stop_and_reset_llm() && confirmed;
        } else if (service_kind_ == HLIEModelType::SD) {
            return service_manager_.stop_and_reset_sd();
        } else if (service_kind_ == HLIEModelType::WHISPER) {
            return service_manager_.stop_and_reset_whisper();
        } else if (service_kind_ == HLIEModelType::COSYVOICE ||
                   service_kind_ == HLIEModelType::TTS) {
            return service_manager_.stop_and_reset_cosyvoice();
        } else if (service_kind_ == HLIEModelType::QWEN3_TTS) {
            return service_manager_.stop_and_reset_qwen3_tts();
        } else if (service_kind_ == HLIEModelType::DOCUMENT) {
            return service_manager_.stop_and_reset_document_pipeline();
        }
        return false;
    }

    HLIEModelServiceManager& service_manager_;
    HLIERuntimeIdentity identity_;
    HLIEModelCapabilities capabilities_;
    std::string service_kind_;
    bool owns_document_companion_ = false;
    mutable std::mutex mutex_;
    bool draining_ = false;
    bool stopped_ = false;
};

HLIEEmbeddedModelRuntime::HLIEEmbeddedModelRuntime(
    HLIEModelServiceManager& service_manager,
    HLIERuntimeIdentity identity,
    HLIEModelCapabilities capabilities,
    std::string service_kind,
    const bool owns_document_companion)
    : impl_(std::make_unique<Impl>(service_manager, std::move(identity), capabilities,
                                   std::move(service_kind), owns_document_companion)) {}

HLIEEmbeddedModelRuntime::~HLIEEmbeddedModelRuntime() = default;

HLIERuntimeIdentity HLIEEmbeddedModelRuntime::identity() const { return impl_->identity(); }
HLIEModelCapabilities HLIEEmbeddedModelRuntime::capabilities() const { return impl_->capabilities(); }
HLIERuntimeHealth HLIEEmbeddedModelRuntime::health() const { return impl_->health(); }
void HLIEEmbeddedModelRuntime::begin_draining() { impl_->begin_draining(); }
HLIERuntimeStopResult HLIEEmbeddedModelRuntime::stop(
    const HLIERuntimeStopRequest& request) {
    return impl_->stop(request);
}

HLIE_NAMESPACE_END

#ifndef HLIE_SPEECH_APPLICATION_SERVICE_H
#define HLIE_SPEECH_APPLICATION_SERVICE_H

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "hlie_core/control/hlie_admission_controller.h"
#include "hlie_core/runtime/hlie_model_runtime.h"
#include "httplib.h"

HLIE_NAMESPACE_BEGIN

enum class HLIESpeechModelState {
    Other,
    Loading,
    Loaded,
};

enum class HLIESpeechReadinessStatus {
    Ready,
    Cancelled,
    TimedOut,
    Missing,
    Orphaned,
    Unavailable,
};

struct HLIESpeechModelSnapshot {
    std::string key;
    std::string service_kind;
    HLIESpeechModelState state = HLIESpeechModelState::Other;
};

struct HLIESpeechDispatchTarget {
    HLIERuntimeIdentity identity;
    std::shared_ptr<void> lifetime_guard;
    std::function<bool(const httplib::Request&, httplib::Response&,
                       std::shared_ptr<void>)>
        dispatch;
};

struct HLIESpeechAcquireResult {
    std::optional<HLIESpeechDispatchTarget> target;
    std::string detail;
};

struct HLIESpeechApplicationDependencies {
    std::function<std::optional<HLIESpeechModelSnapshot>(const std::string&)>
        resolve_model;
    std::function<std::optional<std::string>(
        const std::string&, HLIERuntimeKind)> find_ready_model;
    std::function<HLIESpeechReadinessStatus(
        const std::string&, const std::function<bool()>&)> wait_until_ready;
    std::function<HLIESpeechAcquireResult(const std::string&)> acquire_target;
    std::function<void(bool)> observe_dispatch;
};

// Owns OpenAI-compatible speech model selection and request-lifetime
// orchestration. Concrete TTS services and Worker proxy behavior stay behind
// the Dispatcher endpoint adapter.
class HLIESpeechApplicationService final {
public:
    HLIESpeechApplicationService(
        HLIEAdmissionController& qwen_admission,
        HLIESpeechApplicationDependencies dependencies);

    void handle(const httplib::Request& request,
                httplib::Response& response) const;

private:
    HLIEAdmissionController& qwen_admission_;
    HLIESpeechApplicationDependencies dependencies_;
};

HLIE_NAMESPACE_END

#endif // HLIE_SPEECH_APPLICATION_SERVICE_H

#ifndef HLIE_APP_STARTUP_EMBEDDED_STARTUP_SERVICE_H
#define HLIE_APP_STARTUP_EMBEDDED_STARTUP_SERVICE_H

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "hlie_app/hlie_model_service.h"

HLIE_NAMESPACE_BEGIN

class HLIEModelController;

enum class HLIEEmbeddedStartupGroup {
    Primary,
    Secondary,
};

struct HLIEEmbeddedStartupTarget {
    HLIEEmbeddedStartupGroup group = HLIEEmbeddedStartupGroup::Secondary;
    std::string model_key;
    std::string service_kind;
    std::shared_ptr<IHLIEModelService> primary_service;
    std::shared_ptr<IHLIEModelService> companion_service;
    std::optional<int> ready_port;
    bool abort_if_model_missing = false;
};

class IHLIEEmbeddedStartupTargetProvider {
public:
    virtual ~IHLIEEmbeddedStartupTargetProvider() = default;

    [[nodiscard]] virtual std::vector<HLIEEmbeddedStartupTarget> targets(
        HLIEEmbeddedStartupGroup group) const = 0;
};

struct HLIEEmbeddedStartupSummary {
    std::size_t attempted = 0;
    std::size_t started = 0;
    std::size_t failed = 0;
    bool abort_remaining = false;

    [[nodiscard]] bool complete() const noexcept {
        return failed == 0 && !abort_remaining;
    }
};

// Publishes already-initialized in-process Services as generation-fenced
// Runtime handles. Model selection and Service construction happen earlier;
// this service owns only startup sequencing and Controller publication.
class HLIEEmbeddedStartupService {
public:
    using ReadyCallback = std::function<void(const std::string&)>;
    using UnpublishedCleanupCallback =
        std::function<bool(const HLIEEmbeddedStartupTarget&)>;

    HLIEEmbeddedStartupService(
        HLIEModelController& controller,
        IHLIEEmbeddedStartupTargetProvider& target_provider,
        ReadyCallback ready_callback = {},
        UnpublishedCleanupCallback unpublished_cleanup = {});

    [[nodiscard]] HLIEEmbeddedStartupSummary start_group(
        HLIEEmbeddedStartupGroup group);

private:
    [[nodiscard]] bool publish_started_target(
        const HLIEEmbeddedStartupTarget& target,
        bool& model_missing);
    [[nodiscard]] bool cleanup_unpublished_target(
        const HLIEEmbeddedStartupTarget& target,
        const char* reason) const;

    HLIEModelController& controller_;
    IHLIEEmbeddedStartupTargetProvider& target_provider_;
    ReadyCallback ready_callback_;
    UnpublishedCleanupCallback unpublished_cleanup_;
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_STARTUP_EMBEDDED_STARTUP_SERVICE_H

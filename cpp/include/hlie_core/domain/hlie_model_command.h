#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <utility>

#include "hlie_core/domain/hlie_control_result.h"
#include "hlie_core/domain/hlie_model_spec.h"
#include "hlie_core/resource/hlie_placement.h"

HLIE_NAMESPACE_BEGIN

using HLIEDeadline = std::chrono::steady_clock::time_point;

enum class HLIEModelSelectorKind {
    ModelKey,
    ModelName,
};

struct HLIEModelSelector {
    HLIEModelSelectorKind kind = HLIEModelSelectorKind::ModelKey;
    std::string value;

    [[nodiscard]] static HLIEModelSelector by_key(std::string model_key) {
        return {HLIEModelSelectorKind::ModelKey, std::move(model_key)};
    }

    [[nodiscard]] static HLIEModelSelector by_name(std::string model_name) {
        return {HLIEModelSelectorKind::ModelName, std::move(model_name)};
    }

    [[nodiscard]] bool valid() const noexcept {
        return !value.empty();
    }
};

struct HLIEModelLoadCommand {
    HLIEModelSpec spec;
    HLIEPlacementIntent placement;
    std::optional<HLIERuntimeKind> preferred_runtime;
    bool pinned = false;

    [[nodiscard]] bool valid() const noexcept {
        const auto& identity = spec.identity;
        const bool has_identity = !identity.model_key.empty() ||
                                  !identity.model_name.empty() ||
                                  !identity.model_path.empty();
        return has_identity && spec.batch_size > 0 && spec.context_size >= 0 &&
               spec.required_device_count > 0 && placement.valid() &&
               placement.required_device_count == spec.required_device_count;
    }
};

struct HLIEModelUnloadCommand {
    HLIEModelSelector selector;
    HLIEDeadline deadline{};
    bool force = false;

    [[nodiscard]] bool valid() const noexcept {
        return selector.valid() && deadline != HLIEDeadline{};
    }
};

struct HLIEModelOperationReceipt {
    HLIEModelIdentity identity;
    HLIEModelGeneration generation = 0;
    HLIEModelLifecycleState state = HLIEModelLifecycleState::Unloaded;
    bool coalesced = false;
};

using HLIEModelOperationResult = HLIEControlResult<HLIEModelOperationReceipt>;

HLIE_NAMESPACE_END

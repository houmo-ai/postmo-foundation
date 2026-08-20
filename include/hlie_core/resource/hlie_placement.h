#pragma once

#include <cstddef>
#include <vector>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

// Caller placement constraints. This is intent only: no HAL observation,
// reservation ownership, or assumption that a listed device is available.
struct HLIEPlacementIntent {
    std::vector<int> requested_device_ids;
    int required_device_count = 1;
    bool allow_auto_selection = true;
    bool allow_eviction = true;

    [[nodiscard]] bool valid() const noexcept {
        if (required_device_count <= 0) {
            return false;
        }
        if (!allow_auto_selection && requested_device_ids.empty()) {
            return false;
        }
        if (!requested_device_ids.empty() &&
            static_cast<int>(requested_device_ids.size()) != required_device_count) {
            return false;
        }
        for (std::size_t i = 0; i < requested_device_ids.size(); ++i) {
            if (requested_device_ids[i] < 0) {
                return false;
            }
            for (std::size_t j = i + 1; j < requested_device_ids.size(); ++j) {
                if (requested_device_ids[i] == requested_device_ids[j]) {
                    return false;
                }
            }
        }
        return true;
    }
};

// Backend capability limits. A max count of zero means no descriptor-level
// upper bound; the scheduler may still reject based on observed hardware.
struct HLIEPlacementConstraints {
    int min_device_count = 1;
    int max_device_count = 0;
    bool supports_auto_selection = true;
    bool supports_explicit_selection = true;
};

// Validated plan passed to the future ResourceScheduler. It is not a grant and
// does not authorize model startup before a reservation is committed.
struct HLIEPlacementPlan {
    std::vector<int> requested_device_ids;
    int required_device_count = 1;
    bool auto_select = true;
    bool allow_eviction = true;
};

HLIE_NAMESPACE_END

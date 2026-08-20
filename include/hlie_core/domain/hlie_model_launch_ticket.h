#ifndef HLIE_CORE_MODEL_LAUNCH_TICKET_H
#define HLIE_CORE_MODEL_LAUNCH_TICKET_H

#include <cstdint>
#include <string>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

using HLIEModelGeneration = std::uint64_t;

struct HLIEModelLaunchTicket {
    std::string model_key;
    HLIEModelGeneration generation = 0;
    std::string token;

    [[nodiscard]] bool valid() const {
        return !model_key.empty() && generation != 0 && !token.empty();
    }

    [[nodiscard]] bool matches(const HLIEModelLaunchTicket& other) const {
        return valid() && other.valid() && model_key == other.model_key &&
               generation == other.generation && token == other.token;
    }
};

HLIE_NAMESPACE_END

#endif // HLIE_CORE_MODEL_LAUNCH_TICKET_H

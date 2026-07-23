#ifndef HLIE_MODEL_FAMILY_REGISTRY_H
#define HLIE_MODEL_FAMILY_REGISTRY_H

#include <string>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

// Canonical model-family classification and projector pairing rules.
// Keep filename conventions here instead of duplicating regexes across scanners.
class HLIE_API HLIEModelFamilyRegistry {
public:
    static constexpr auto GEMMA4 = "gemma4";

    [[nodiscard]] static std::string detect_projector_family(const std::string& name);
    [[nodiscard]] static std::string detect_metadata_family(const std::string& architecture,
                                                            const std::string& projector_type = {},
                                                            const std::string& vision_projector_type = {},
                                                            const std::string& audio_projector_type = {});
    [[nodiscard]] static bool supports_standalone(const std::string& name);
    [[nodiscard]] static bool is_projector_main_model_name(const std::string& name);
    [[nodiscard]] static std::string build_projector_match_key(const std::string& name);
};

HLIE_NAMESPACE_END

#endif // HLIE_MODEL_FAMILY_REGISTRY_H

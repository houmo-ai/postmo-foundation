#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"
#include "hlie_core/domain/hlie_control_result.h"
#include "hlie_core/domain/hlie_model_spec.h"
#include "hlie_core/resource/hlie_placement.h"

HLIE_NAMESPACE_BEGIN

enum class HLIEBackendKind {
    LLM,
    SD,
    Whisper,
    CosyVoice,
    Qwen3TTS,
    Document,
};

struct HLIEModelMemoryObservation {
    int empirical_mb = 0;
    int configured_mb = 0;
    int artifact_mb = 0;
    int companion_mb = 0;
};

enum class HLIEModelMemoryEstimateSource {
    None,
    Empirical,
    Configured,
    Artifact,
    Heuristic,
};

struct HLIEModelMemoryEstimate {
    int base_mb = 0;
    int companion_mb = 0;
    int total_mb = 0;
    HLIEModelMemoryEstimateSource source = HLIEModelMemoryEstimateSource::None;
};

struct HLIEBackendMemoryPolicy {
    int fallback_mb = 4096;
    bool use_legacy_name_heuristics = true;
};

struct HLIEBackendDescriptor {
    HLIEBackendKind kind = HLIEBackendKind::LLM;
    std::string backend_id;
    std::uint32_t contract_version = 1;
    int match_priority = 0;
    std::vector<std::string> aliases;
    std::vector<std::string> model_name_markers;
    HLIEModelCapabilities default_capabilities;
    std::vector<HLIERuntimeKind> supported_runtimes;
    bool owns_document_companion = false;
    HLIEBackendMemoryPolicy memory_policy;
    HLIEPlacementConstraints placement_constraints;

    [[nodiscard]] bool supports(HLIERuntimeKind runtime_kind) const;
    [[nodiscard]] HLIEModelMemoryEstimate estimate_memory(
        const HLIEModelSpec& spec,
        const HLIEModelMemoryObservation& observation) const;
    [[nodiscard]] HLIEControlResult<HLIEPlacementPlan> plan_placement(
        const HLIEModelSpec& spec,
        const HLIEPlacementIntent& intent) const;
};

// Registry of the model backends supported by the current HOUMO TCIM product.
// This is a typed classification table, not a dynamic SDK/plugin system.
class HLIEBackendRegistry {
public:
    [[nodiscard]] bool register_backend(HLIEBackendDescriptor descriptor);
    [[nodiscard]] std::optional<HLIEBackendDescriptor> find_by_alias(
        const std::string& alias) const;
    [[nodiscard]] std::optional<HLIEBackendDescriptor> resolve(
        const HLIEModelSpec& spec) const;
    [[nodiscard]] const std::vector<HLIEBackendDescriptor>& descriptors() const noexcept {
        return descriptors_;
    }

    [[nodiscard]] static HLIEBackendRegistry houmo_defaults();

private:
    static std::string normalize(std::string value);
    static bool contains_ignore_case(const std::string& value,
                                     const std::string& marker);
    [[nodiscard]] const HLIEBackendDescriptor* find_by_alias_ptr(
        const std::string& alias) const;
    [[nodiscard]] const HLIEBackendDescriptor* find_by_kind_ptr(
        HLIEBackendKind kind) const;

    std::vector<HLIEBackendDescriptor> descriptors_;
};

// Canonical immutable table for the current single-SDK HOUMO product.
// Callers that need policy identity should use this accessor instead of
// constructing a second default table.
[[nodiscard]] const HLIEBackendRegistry& houmo_default_registry();

HLIE_NAMESPACE_END

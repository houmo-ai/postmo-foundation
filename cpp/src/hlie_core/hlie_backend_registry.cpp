#include "hlie_core/backend/hlie_backend_registry.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <utility>

HLIE_NAMESPACE_BEGIN

namespace {

std::string normalize_value(std::string value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(), [](const unsigned char c) { return std::isspace(c); });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(), [](const unsigned char c) { return std::isspace(c); }).base();
    value = first < last ? std::string(first, last) : std::string{};
    std::transform(value.begin(), value.end(), value.begin(),
                   [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

} // namespace

bool HLIEBackendDescriptor::supports(const HLIERuntimeKind runtime_kind) const {
    return std::find(supported_runtimes.begin(), supported_runtimes.end(), runtime_kind) !=
           supported_runtimes.end();
}

HLIEModelMemoryEstimate HLIEBackendDescriptor::estimate_memory(
    const HLIEModelSpec& spec,
    const HLIEModelMemoryObservation& observation) const {
    HLIEModelMemoryEstimate estimate;
    const auto consider = [&](const int value, const HLIEModelMemoryEstimateSource source) {
        if (value > estimate.base_mb) {
            estimate.base_mb = value;
            estimate.source = source;
        }
    };
    consider(observation.empirical_mb, HLIEModelMemoryEstimateSource::Empirical);
    consider(observation.configured_mb, HLIEModelMemoryEstimateSource::Configured);
    consider(observation.artifact_mb, HLIEModelMemoryEstimateSource::Artifact);

    if (estimate.base_mb <= 0) {
        const std::string name = normalize_value(spec.identity.model_name);
        int fallback_mb = name.empty() ? 0 : std::max(0, memory_policy.fallback_mb);
        if (memory_policy.use_legacy_name_heuristics) {
            if (name.find("qwen2.5") != std::string::npos) {
                fallback_mb = 8192;
            } else if (name.find("qwen3") != std::string::npos) {
                fallback_mb = 10240;
            } else if (name.find("gte") != std::string::npos ||
                       name.find("bge") != std::string::npos) {
                fallback_mb = 1536;
            } else if (name.find("whisper") != std::string::npos) {
                fallback_mb = 2048;
            }
        }
        estimate.base_mb = fallback_mb;
        estimate.source = fallback_mb > 0
            ? HLIEModelMemoryEstimateSource::Heuristic
            : HLIEModelMemoryEstimateSource::None;
    }

    estimate.companion_mb = std::max(0, observation.companion_mb);
    const auto total = static_cast<long long>(estimate.base_mb) + estimate.companion_mb;
    estimate.total_mb = total > std::numeric_limits<int>::max()
        ? std::numeric_limits<int>::max()
        : static_cast<int>(total);
    return estimate;
}

HLIEControlResult<HLIEPlacementPlan> HLIEBackendDescriptor::plan_placement(
    const HLIEModelSpec& spec,
    const HLIEPlacementIntent& intent) const {
    const auto reject = [](const HLIEControlErrorCode code, std::string message) {
        return HLIEControlResult<HLIEPlacementPlan>::failure({
            code,
            HLIEControlErrorStage::Reservation,
            std::move(message),
            0,
            false,
        });
    };

    if (!intent.valid()) {
        return reject(HLIEControlErrorCode::InvalidArgument,
                      "invalid placement intent");
    }
    if (spec.required_device_count <= 0 ||
        spec.required_device_count != intent.required_device_count) {
        return reject(HLIEControlErrorCode::InvalidArgument,
                      "model and placement device counts do not match");
    }
    if (placement_constraints.min_device_count <= 0 ||
        placement_constraints.max_device_count < 0 ||
        (placement_constraints.max_device_count > 0 &&
         placement_constraints.max_device_count < placement_constraints.min_device_count)) {
        return reject(HLIEControlErrorCode::Internal,
                      "backend has invalid placement constraints");
    }
    if (intent.required_device_count < placement_constraints.min_device_count ||
        (placement_constraints.max_device_count > 0 &&
         intent.required_device_count > placement_constraints.max_device_count)) {
        return reject(HLIEControlErrorCode::Unsupported,
                      "requested device count is unsupported by backend");
    }
    if (intent.requested_device_ids.empty() &&
        !placement_constraints.supports_auto_selection) {
        return reject(HLIEControlErrorCode::Unsupported,
                      "backend requires explicit device selection");
    }
    if (!intent.requested_device_ids.empty() &&
        !placement_constraints.supports_explicit_selection) {
        return reject(HLIEControlErrorCode::Unsupported,
                      "backend does not support explicit device selection");
    }

    return HLIEControlResult<HLIEPlacementPlan>::success({
        intent.requested_device_ids,
        intent.required_device_count,
        intent.requested_device_ids.empty(),
        intent.allow_eviction,
    });
}

std::string HLIEBackendRegistry::normalize(std::string value) {
    return normalize_value(std::move(value));
}

bool HLIEBackendRegistry::contains_ignore_case(const std::string& value,
                                               const std::string& marker) {
    return normalize(value).find(normalize(marker)) != std::string::npos;
}

const HLIEBackendDescriptor* HLIEBackendRegistry::find_by_alias_ptr(
    const std::string& alias) const {
    const std::string normalized_alias = normalize(alias);
    if (normalized_alias.empty()) {
        return nullptr;
    }
    for (const auto& descriptor : descriptors_) {
        if (std::find(descriptor.aliases.begin(), descriptor.aliases.end(),
                      normalized_alias) != descriptor.aliases.end()) {
            return &descriptor;
        }
    }
    return nullptr;
}

const HLIEBackendDescriptor* HLIEBackendRegistry::find_by_kind_ptr(
    const HLIEBackendKind kind) const {
    const auto it = std::find_if(
        descriptors_.begin(), descriptors_.end(),
        [kind](const HLIEBackendDescriptor& descriptor) { return descriptor.kind == kind; });
    return it == descriptors_.end() ? nullptr : &*it;
}

bool HLIEBackendRegistry::register_backend(HLIEBackendDescriptor descriptor) {
    descriptor.backend_id = normalize(std::move(descriptor.backend_id));
    if (descriptor.backend_id.empty() || descriptor.aliases.empty() ||
        find_by_kind_ptr(descriptor.kind)) {
        return false;
    }
    if (std::any_of(descriptors_.begin(), descriptors_.end(),
                    [&](const HLIEBackendDescriptor& existing) {
                        return existing.backend_id == descriptor.backend_id;
                    })) {
        return false;
    }

    for (auto& alias : descriptor.aliases) {
        alias = normalize(std::move(alias));
        if (alias.empty() || find_by_alias_ptr(alias)) {
            return false;
        }
    }
    std::sort(descriptor.aliases.begin(), descriptor.aliases.end());
    if (std::adjacent_find(descriptor.aliases.begin(), descriptor.aliases.end()) !=
        descriptor.aliases.end()) {
        return false;
    }

    for (auto& marker : descriptor.model_name_markers) {
        marker = normalize(std::move(marker));
    }
    descriptor.model_name_markers.erase(
        std::remove(descriptor.model_name_markers.begin(),
                    descriptor.model_name_markers.end(), std::string{}),
        descriptor.model_name_markers.end());
    descriptors_.push_back(std::move(descriptor));
    return true;
}

std::optional<HLIEBackendDescriptor> HLIEBackendRegistry::find_by_alias(
    const std::string& alias) const {
    const auto* descriptor = find_by_alias_ptr(alias);
    return descriptor ? std::optional<HLIEBackendDescriptor>{*descriptor} : std::nullopt;
}

std::optional<HLIEBackendDescriptor> HLIEBackendRegistry::resolve(
    const HLIEModelSpec& spec) const {
    const HLIEBackendDescriptor* matched = nullptr;
    const auto consider = [&](const HLIEBackendDescriptor* candidate) {
        if (candidate && (!matched || candidate->match_priority > matched->match_priority)) {
            matched = candidate;
        }
    };
    consider(find_by_alias_ptr(spec.backend_alias));
    consider(find_by_alias_ptr(spec.legacy_model_type));

    for (const auto& descriptor : descriptors_) {
        const bool marker_matched = std::any_of(
            descriptor.model_name_markers.begin(), descriptor.model_name_markers.end(),
            [&](const std::string& marker) {
                return contains_ignore_case(spec.identity.model_name, marker);
            });
        if (marker_matched) {
            consider(&descriptor);
        }
    }
    if (matched) {
        return *matched;
    }

    const auto fallback_kind = spec.is_llm ? HLIEBackendKind::LLM : HLIEBackendKind::SD;
    if (const auto* descriptor = find_by_kind_ptr(fallback_kind)) {
        return *descriptor;
    }
    return std::nullopt;
}

HLIEBackendRegistry HLIEBackendRegistry::houmo_defaults() {
    HLIEBackendRegistry registry;
    const std::vector<HLIERuntimeKind> both{
        HLIERuntimeKind::Embedded, HLIERuntimeKind::Process};

    (void)registry.register_backend(HLIEBackendDescriptor{
        HLIEBackendKind::LLM, "houmo.llm", 1, 10, {"llm"}, {},
        {true, false, false}, both, true, {4096, true}, {1, 0, true, true}});
    (void)registry.register_backend(HLIEBackendDescriptor{
        HLIEBackendKind::SD, "houmo.sd", 1, 20, {"sd"}, {},
        {true, false, false}, both, false, {4096, true}, {1, 0, true, true}});
    (void)registry.register_backend(HLIEBackendDescriptor{
        HLIEBackendKind::Whisper, "houmo.whisper", 1, 80, {"whisper"}, {"whisper"},
        {true, false, true}, both, false, {4096, true}, {1, 0, true, true}});
    (void)registry.register_backend(HLIEBackendDescriptor{
        HLIEBackendKind::CosyVoice, "houmo.cosyvoice", 1, 70,
        {"cosyvoice", "tts"}, {"cosyvoice"},
        {true, false, true}, both, false, {4096, true}, {1, 0, true, true}});
    (void)registry.register_backend(HLIEBackendDescriptor{
        HLIEBackendKind::Qwen3TTS, "houmo.qwen3_tts", 1, 100,
        {"qwen3_tts", "qwen3-tts"}, {"qwen3_tts", "qwen3-tts"},
        {true, false, true}, both, false, {4096, true}, {1, 0, true, true}});
    (void)registry.register_backend(HLIEBackendDescriptor{
        HLIEBackendKind::Document, "houmo.document", 1, 30, {"document"}, {},
        {true, true, false}, {HLIERuntimeKind::Embedded}, false, {4096, true},
        {1, 0, true, true}});
    return registry;
}

const HLIEBackendRegistry& houmo_default_registry() {
    static const HLIEBackendRegistry registry = HLIEBackendRegistry::houmo_defaults();
    return registry;
}

HLIE_NAMESPACE_END

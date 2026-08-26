#include "hlie_core/backend/hlie_backend_registry.h"

#include <catch_amalgamated.hpp>

#include <string>
#include <utility>

using namespace hm_HLIE;

namespace {

HLIEModelSpec spec_for(std::string backend_alias,
                       std::string legacy_model_type,
                       std::string model_name,
                       const bool is_llm) {
    HLIEModelSpec spec;
    spec.backend_alias = std::move(backend_alias);
    spec.legacy_model_type = std::move(legacy_model_type);
    spec.identity.model_name = std::move(model_name);
    spec.is_llm = is_llm;
    return spec;
}

} // namespace

TEST_CASE("HOUMO backend descriptors preserve legacy classification precedence",
          "[backend-registry][p1][core]") {
    const auto registry = HLIEBackendRegistry::houmo_defaults();
    REQUIRE(registry.descriptors().size() == 6);

    const auto qwen_tts = registry.resolve(
        spec_for("llm", {}, "HiModel-Qwen3-TTS-1.7B.gguf", true));
    REQUIRE(qwen_tts);
    CHECK(qwen_tts->kind == HLIEBackendKind::Qwen3TTS);
    CHECK(qwen_tts->backend_id == "houmo.qwen3_tts");
    CHECK(qwen_tts->supports(HLIERuntimeKind::Embedded));
    CHECK(qwen_tts->supports(HLIERuntimeKind::Process));

    const auto whisper = registry.resolve(
        spec_for("cosyvoice", {}, "Whisper-medium.gguf", true));
    REQUIRE(whisper);
    CHECK(whisper->kind == HLIEBackendKind::Whisper);

    const auto legacy_tts = registry.find_by_alias("  TTS  ");
    REQUIRE(legacy_tts);
    CHECK(legacy_tts->kind == HLIEBackendKind::CosyVoice);

    const auto fallback = registry.resolve(spec_for({}, {}, "unknown", false));
    REQUIRE(fallback);
    CHECK(fallback->kind == HLIEBackendKind::SD);

    const auto explicit_llm = registry.resolve(spec_for("llm", {}, "sd3", true));
    REQUIRE(explicit_llm);
    CHECK(explicit_llm->kind == HLIEBackendKind::LLM);
}

TEST_CASE("Backend memory policy separates observations from pure estimation",
          "[backend-registry][memory][p1][core]") {
    const auto registry = HLIEBackendRegistry::houmo_defaults();
    const auto llm = registry.find_by_alias("llm");
    REQUIRE(llm);

    auto spec = spec_for("llm", {}, "Qwen3-8B", true);
    auto estimate = llm->estimate_memory(
        spec, HLIEModelMemoryObservation{9000, 8000, 8500, 512});
    CHECK(estimate.base_mb == 9000);
    CHECK(estimate.companion_mb == 512);
    CHECK(estimate.total_mb == 9512);
    CHECK(estimate.source == HLIEModelMemoryEstimateSource::Empirical);

    estimate = llm->estimate_memory(spec, {});
    CHECK(estimate.base_mb == 10240);
    CHECK(estimate.total_mb == 10240);
    CHECK(estimate.source == HLIEModelMemoryEstimateSource::Heuristic);

    spec.identity.model_name = "bge-large";
    estimate = llm->estimate_memory(spec, {});
    CHECK(estimate.total_mb == 1536);

    spec.identity.model_name.clear();
    estimate = llm->estimate_memory(spec, {});
    CHECK(estimate.total_mb == 0);
    CHECK(estimate.source == HLIEModelMemoryEstimateSource::None);
}

TEST_CASE("Document backend is an embedded LLM companion descriptor",
          "[backend-registry][p1][core]") {
    const auto registry = HLIEBackendRegistry::houmo_defaults();
    const auto llm = registry.find_by_alias("llm");
    REQUIRE(llm);
    CHECK(llm->owns_document_companion);

    const auto document = registry.find_by_alias("document");
    REQUIRE(document);
    CHECK(document->kind == HLIEBackendKind::Document);
    CHECK(document->supports(HLIERuntimeKind::Embedded));
    CHECK_FALSE(document->supports(HLIERuntimeKind::Process));
}

TEST_CASE("Backend placement policy validates intent without observing hardware",
          "[backend-registry][placement][p1][core]") {
    const auto registry = HLIEBackendRegistry::houmo_defaults();
    auto llm = registry.find_by_alias("llm");
    REQUIRE(llm);

    auto spec = spec_for("llm", {}, "Qwen3-8B", true);
    spec.required_device_count = 2;
    HLIEPlacementIntent intent;
    intent.required_device_count = 2;
    intent.requested_device_ids = {0, 1};
    intent.allow_auto_selection = false;
    intent.allow_eviction = false;

    auto decision = llm->plan_placement(spec, intent);
    REQUIRE(decision);
    REQUIRE(decision.value_if() != nullptr);
    const std::vector<int> requested_devices{0, 1};
    CHECK(decision.value_if()->requested_device_ids == requested_devices);
    CHECK_FALSE(decision.value_if()->auto_select);
    CHECK_FALSE(decision.value_if()->allow_eviction);

    intent.required_device_count = 1;
    intent.requested_device_ids = {0};
    decision = llm->plan_placement(spec, intent);
    REQUIRE_FALSE(decision);
    REQUIRE(decision.error_if() != nullptr);
    CHECK(decision.error_if()->code == HLIEControlErrorCode::InvalidArgument);

    intent.required_device_count = 2;
    intent.requested_device_ids = {0, 1};
    llm->placement_constraints.max_device_count = 1;
    decision = llm->plan_placement(spec, intent);
    REQUIRE_FALSE(decision);
    REQUIRE(decision.error_if() != nullptr);
    CHECK(decision.error_if()->code == HLIEControlErrorCode::Unsupported);

    llm->placement_constraints = {1, 0, true, false};
    decision = llm->plan_placement(spec, intent);
    REQUIRE_FALSE(decision);
    REQUIRE(decision.error_if() != nullptr);
    CHECK(decision.error_if()->stage == HLIEControlErrorStage::Reservation);
}

TEST_CASE("Backend registry rejects ambiguous contracts",
          "[backend-registry][p1][core]") {
    HLIEBackendRegistry registry;
    REQUIRE(registry.register_backend(HLIEBackendDescriptor{
        HLIEBackendKind::LLM, "backend-a", 1, 1, {"a"}, {}, {},
        {HLIERuntimeKind::Embedded}, false}));
    CHECK_FALSE(registry.register_backend(HLIEBackendDescriptor{
        HLIEBackendKind::SD, "backend-b", 1, 1, {"a"}, {}, {},
        {HLIERuntimeKind::Embedded}, false}));
    CHECK_FALSE(registry.register_backend(HLIEBackendDescriptor{
        HLIEBackendKind::SD, "backend-a", 1, 1, {"b"}, {}, {},
        {HLIERuntimeKind::Embedded}, false}));
    CHECK_FALSE(registry.register_backend(HLIEBackendDescriptor{
        HLIEBackendKind::LLM, "backend-c", 1, 1, {"c"}, {}, {},
        {HLIERuntimeKind::Embedded}, false}));
}

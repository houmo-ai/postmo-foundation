#include "hlie_app/hlie_llm_preset_resolver.h"

#include <catch_amalgamated.hpp>

using namespace hm_HLIE;

TEST_CASE("Llama param layers support switches repeats and deletion", "[llm-preset][params]") {
    HLIELlmParamSet base;
    std::string error;
    REQUIRE(HLIELlmParamSet::from_json({
        {"--ctx-size", 32768},
        {"--no-warmup", true},
        {"--parallel", 1},
    }, base, &error));

    HLIELlmParamSet layer;
    REQUIRE(HLIELlmParamSet::from_json({
        {"--ctx-size", 65536},
        {"--no-warmup", false},
        {"--lora", {"a.gguf", "b.gguf"}},
    }, layer, &error));
    base.merge_layer(layer);

    CHECK(base.to_json() == nlohmann::json{
        {"--ctx-size", 65536},
        {"--lora", {"a.gguf", "b.gguf"}},
        {"--parallel", 1},
    });
    CHECK(base.to_argv() == std::vector<std::string>{
        "hlie-llm", "--ctx-size", "65536", "--lora", "a.gguf", "--lora", "b.gguf",
        "--parallel", "1",
    });
    CHECK(base.stable_hash() == base.stable_hash());
}

TEST_CASE("Llama param catalog rejects framework owned arguments", "[llm-preset][validation]") {
    HLIELlmPresetCatalog catalog;
    std::string error;
    CHECK_FALSE(catalog.load_json({
        {"schema_version", 1},
        {"presets", {{
            {"id", "bad"},
            {"match", {{"model_family", "qwen3"}}},
            {"params", {{"--model", "other.gguf"}}},
        }}},
    }, "official", &error));
    CHECK(error.find("framework-owned") != std::string::npos);

    CHECK_FALSE(catalog.load_json({
        {"schema_version", 1},
        {"presets", {{{"id", "unknown-match"}, {"match", {{"unknown_field", "x"}}}}}},
    }, "official", &error));
    CHECK(error.find("unknown preset match field") != std::string::npos);

    CHECK_FALSE(catalog.load_json({
        {"schema_version", 1},
        {"runtime", {{"hlie_min_version", "v99.0.0"}}},
        {"presets", nlohmann::json::array()},
    }, "official", &error));
    CHECK(error.find("not compatible") != std::string::npos);

    CHECK_FALSE(catalog.load_json({
        {"schema_version", 1},
        {"presets", {
            {{"id", "a"}, {"priority", 10}, {"match", {{"role", "chat"}}},
             {"params", {{"--ctx-size", 8192}}}},
            {{"id", "b"}, {"priority", 10}, {"match", {{"role", "chat"}}},
             {"params", {{"--ctx-size", 16384}}}},
        }},
    }, "official", &error));
    CHECK(error.find("conflicting preset rules") != std::string::npos);
}

TEST_CASE("Preset resolver layers broad exact and local rules deterministically", "[llm-preset][resolver]") {
    HLIEModelInfo info;
    REQUIRE(info.parse_identity_from_gguf_filename(
        "HiModel_xh2_qwen3.5_9b_256_32k_b1_1chip_2cores_v1.2.0_20260401.gguf") == 0);
    info.model_key_ = info.get_model_name();

    HLIELlmPresetCatalog official;
    HLIELlmPresetCatalog local;
    std::string error;
    REQUIRE(official.load_json({
        {"schema_version", 1},
        {"catalog_version", "test"},
        {"presets", {
            {
                {"id", "xh2-default"}, {"priority", 10},
                {"match", {{"service_kind", "llm"}, {"xh_version", 2}}},
                {"params", {{"--parallel", 1}, {"--ctx-size", 8192}}},
            },
            {
                {"id", "qwen35-9b"}, {"priority", 200},
                {"match", {{"model_family", "qwen3.5"}, {"params", "9b"}, {"device_count", 1}}},
                {"params", {{"--ctx-size", 32768}, {"--batch-size", 1024}}},
            },
        }},
    }, "official", &error));
    REQUIRE(local.load_json({
        {"schema_version", 1},
        {"models", {{info.model_key_, {
            {"params", {{"--ctx-size", 65536}}},
        }}}},
    }, "local", &error));

    HLIELlmParamSet base;
    REQUIRE(HLIELlmParamSet::from_json({{"--no-warmup", true}}, base, &error));
    const auto resolved = HLIELlmPresetResolver::resolve(info, base, {&official, &local});

    CHECK(resolved.matched_presets == std::vector<std::string>{
        "xh2-default", "qwen35-9b", "model:" + info.model_key_,
    });
    CHECK(resolved.sources == std::vector<std::string>{"official", "local"});
    CHECK(resolved.params.to_json()["--ctx-size"] == 65536);
    CHECK(resolved.params.to_json()["--parallel"] == 1);
    CHECK(resolved.params.to_json()["--batch-size"] == 1024);
    CHECK(resolved.params.to_json()["--no-warmup"] == true);
    CHECK_FALSE(resolved.params_hash.empty());
}

TEST_CASE("Preset resolver distinguishes rerank ASR and visual variants", "[llm-preset][roles]") {
    HLIELlmPresetCatalog catalog;
    std::string error;
    REQUIRE(catalog.load_json({
        {"schema_version", 1},
        {"presets", {
            {{"id", "rerank"}, {"priority", 100}, {"match", {{"role", "rerank"}}},
             {"params", {{"--embedding", true}}}},
            {{"id", "asr"}, {"priority", 100}, {"match", {{"role", "asr"}}},
             {"params", {{"--ctx-size", 8192}}}},
            {{"id", "vit-448"}, {"priority", 100}, {"match", {{"vl_spec", "vit_448_448"}}},
             {"params", {{"--ctx-size", 32768}}}},
        }},
    }, "official", &error));

    HLIEModelInfo rerank;
    REQUIRE(rerank.parse_identity_from_gguf_filename("HiModel_xh2_qwen3-reranker_8b_v1.2.0.gguf") == 0);
    CHECK(HLIELlmPresetResolver::resolve(rerank, {}, {&catalog}).matched_presets ==
          std::vector<std::string>{"rerank"});

    HLIEModelInfo asr;
    REQUIRE(asr.parse_identity_from_gguf_filename("HiModel_xh2_qwen3-asr_0.6b_v1.3.0.gguf") == 0);
    CHECK(HLIELlmPresetResolver::resolve(asr, {}, {&catalog}).matched_presets ==
          std::vector<std::string>{"asr"});

    HLIEModelInfo vision;
    REQUIRE(vision.parse_identity_from_gguf_filename("HiModel_xh2_qwen3-vl_8b_v1.2.0.gguf") == 0);
    vision.vl_spec_ = "vit_448_448";
    CHECK(HLIELlmPresetResolver::resolve(vision, {}, {&catalog}).matched_presets ==
          std::vector<std::string>{"vit-448"});
}

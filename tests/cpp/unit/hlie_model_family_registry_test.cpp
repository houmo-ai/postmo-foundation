#include "hlie_app/hlie_model_family_registry.h"

#include <catch_amalgamated.hpp>

#include <string>
#include <tuple>

namespace {

using namespace hm_HLIE;

using FamilyCase = std::tuple<std::string, std::string>;

} // namespace

TEST_CASE("Projector family names are normalized", "[model-family]") {
    const auto [input, expected] = GENERATE(table<std::string, std::string>({
        FamilyCase{"HiModel_xh2_gemma4-26b-a4b_320_256k.gguf", HLIEModelFamilyRegistry::GEMMA4},
        FamilyCase{"Gemma-4 26B A4B", HLIEModelFamilyRegistry::GEMMA4},
        FamilyCase{"HiModel_xh2_glm-asr_1.9b.gguf", "glm-asr"},
        FamilyCase{"HiModel_xh2_qwen3-forcealigner-0.6b.gguf", "qwen3-forcealigner"},
        FamilyCase{"unknown-model.gguf", ""},
    }));

    CAPTURE(input);
    CHECK(HLIEModelFamilyRegistry::detect_projector_family(input) == expected);
}

TEST_CASE("Standalone and projector-main policies remain explicit", "[model-family]") {
    const std::string main_filename =
        "HiModel_xh2_gemma4-26b-a4b_320_256k_b1_1chip_2cores_v1.4.0_20260709.gguf";

    CHECK(HLIEModelFamilyRegistry::supports_standalone(main_filename));
    CHECK(HLIEModelFamilyRegistry::is_projector_main_model_name(main_filename));
    CHECK_FALSE(HLIEModelFamilyRegistry::is_projector_main_model_name("MMPROJ_Gemma4_26B.gguf"));
    CHECK_FALSE(HLIEModelFamilyRegistry::supports_standalone("qwen2.5-vl_7b_v1.2.0.gguf"));
    CHECK_FALSE(HLIEModelFamilyRegistry::supports_standalone("unknown-model.gguf"));
}

TEST_CASE("Repository and legacy projector names produce the same key", "[model-family][projector]") {
    const std::string main_filename =
        "HiModel_xh2_gemma4-26b-a4b_320_256k_b1_1chip_2cores_v1.4.0_20260709.gguf";

    const std::string mmproj_filename =
        "mmproj_xh2_gemma4-26b-a4b_2cores_vit_768x2520_v1.4.0_20260709.gguf";
    CHECK(HLIEModelFamilyRegistry::build_projector_match_key(main_filename) ==
          HLIEModelFamilyRegistry::build_projector_match_key(mmproj_filename));

    const std::string old_style_main =
        "HiModel-Gemma4-26b-a4b-p256-128K-b1-2core-1d-xh2-v1.4.0-20260623163431.gguf";
    CHECK(HLIEModelFamilyRegistry::build_projector_match_key(old_style_main) ==
          HLIEModelFamilyRegistry::build_projector_match_key(mmproj_filename));
}

TEST_CASE("Projector pairing keeps behavior-changing variants distinct", "[model-family][projector]") {
    const std::string main_filename =
        "HiModel_xh2_gemma4-26b-a4b_320_256k_b1_1chip_2cores_v1.4.0_20260709.gguf";
    const std::string mtp_main =
        "HiModel_xh2_gemma4-26b-a4b-mtp_320_256k_b1_1chip_2cores_v1.4.0_20260713.gguf";
    const std::string mtp_mmproj =
        "mmproj_xh2_gemma4-26b-a4b-mtp_320_2cores_vit_768x2520_v1.4.0_20260713.gguf";
    CHECK(HLIEModelFamilyRegistry::build_projector_match_key(mtp_main) ==
          HLIEModelFamilyRegistry::build_projector_match_key(mtp_mmproj));
    CHECK(HLIEModelFamilyRegistry::build_projector_match_key(main_filename) !=
          HLIEModelFamilyRegistry::build_projector_match_key(mtp_mmproj));

    const std::string qwen_w4_main =
        "HiModel_xh2_qwen3.6-35b-a3b_w4a8_256_256k_b1_1chip_2cores_v1.4.0_20260716.gguf";
    const std::string qwen_w5_mmproj =
        "mmproj_xh2_qwen3.6-35b-a3b_w5a8_2cores_vit_448x448_v1.4.0_20260716.gguf";
    CHECK(HLIEModelFamilyRegistry::build_projector_match_key(qwen_w4_main) !=
          HLIEModelFamilyRegistry::build_projector_match_key(qwen_w5_mmproj));
}

TEST_CASE("Incomplete projector names do not produce ambiguous keys", "[model-family][projector]") {
    CHECK(HLIEModelFamilyRegistry::build_projector_match_key("gemma4-26b-v1.4.0.gguf").empty());
    CHECK(HLIEModelFamilyRegistry::build_projector_match_key("xh2-gemma4-v1.4.0.gguf").empty());
    CHECK(HLIEModelFamilyRegistry::build_projector_match_key("xh2-gemma4-26b.gguf").empty());
    CHECK(HLIEModelFamilyRegistry::build_projector_match_key("xh2-unknown-26b-v1.4.0.gguf").empty());
}

TEST_CASE("Metadata architectures are normalized independently of filenames", "[model-family][metadata]") {
    CHECK(HLIEModelFamilyRegistry::detect_metadata_family("gemma4") == HLIEModelFamilyRegistry::GEMMA4);
    CHECK(HLIEModelFamilyRegistry::detect_metadata_family("clip", {}, "gemma4v", "gemma4a") ==
          HLIEModelFamilyRegistry::GEMMA4);
    CHECK(HLIEModelFamilyRegistry::detect_metadata_family("qwen35moe") == "qwen3.5");
    CHECK(HLIEModelFamilyRegistry::detect_metadata_family("qwen3vlmoe") == "qwen3-vl");
    CHECK(HLIEModelFamilyRegistry::detect_metadata_family("clip", "glma") == "glm-asr");
    CHECK(HLIEModelFamilyRegistry::detect_metadata_family("clip", "glm") == "glm-ocr");
}

#include "hlie_app/hlie_model_family_registry.h"

#include <iostream>
#include <string>

namespace {

using namespace hm_HLIE;

bool expect(const bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
    }
    return condition;
}

} // namespace

int main() {
    const std::string main_filename =
        "HiModel_xh2_gemma4-26b-a4b_320_256k_b1_1chip_2cores_v1.4.0_20260709.gguf";
    bool ok = expect(HLIEModelFamilyRegistry::detect_projector_family(main_filename) ==
                         HLIEModelFamilyRegistry::GEMMA4,
                     "Gemma 4 should use the canonical family name");
    ok &= expect(HLIEModelFamilyRegistry::supports_standalone(main_filename),
                 "Gemma 4 should register without a companion projector");
    ok &= expect(HLIEModelFamilyRegistry::is_projector_main_model_name(main_filename),
                 "Gemma 4 base GGUF should be classified as a projector-aware main model");
    ok &= expect(HLIEModelFamilyRegistry::detect_projector_family("Gemma-4 26B A4B") ==
                     HLIEModelFamilyRegistry::GEMMA4,
                 "Human-readable Gemma 4 names should normalize to gemma4");

    const std::string mmproj_filename =
        "mmproj_xh2_gemma4-26b-a4b_2cores_vit_768x2520_v1.4.0_20260709.gguf";
    ok &= expect(HLIEModelFamilyRegistry::build_projector_match_key(main_filename) ==
                     HLIEModelFamilyRegistry::build_projector_match_key(mmproj_filename),
                 "Gemma 4 should pair with its repository projector name");

    const std::string old_style_main =
        "HiModel-Gemma4-26b-a4b-p256-128K-b1-2core-1d-xh2-v1.4.0-20260623163431.gguf";
    ok &= expect(HLIEModelFamilyRegistry::build_projector_match_key(old_style_main) ==
                     HLIEModelFamilyRegistry::build_projector_match_key(mmproj_filename),
                 "Legacy Gemma 4 names should pair across different build dates");

    const std::string mtp_main =
        "HiModel_xh2_gemma4-26b-a4b-mtp_320_256k_b1_1chip_2cores_v1.4.0_20260713.gguf";
    const std::string mtp_mmproj =
        "mmproj_xh2_gemma4-26b-a4b-mtp_320_2cores_vit_768x2520_v1.4.0_20260713.gguf";
    ok &= expect(HLIEModelFamilyRegistry::build_projector_match_key(mtp_main) ==
                     HLIEModelFamilyRegistry::build_projector_match_key(mtp_mmproj),
                 "Gemma 4 MTP should pair with its MTP projector");
    ok &= expect(HLIEModelFamilyRegistry::build_projector_match_key(main_filename) !=
                     HLIEModelFamilyRegistry::build_projector_match_key(mtp_mmproj),
                 "Gemma 4 base and MTP projectors must not collide");

    const std::string qwen_w4_main =
        "HiModel_xh2_qwen3.6-35b-a3b_w4a8_256_256k_b1_1chip_2cores_v1.4.0_20260716.gguf";
    const std::string qwen_w5_mmproj =
        "mmproj_xh2_qwen3.6-35b-a3b_w5a8_2cores_vit_448x448_v1.4.0_20260716.gguf";
    ok &= expect(HLIEModelFamilyRegistry::build_projector_match_key(qwen_w4_main) !=
                     HLIEModelFamilyRegistry::build_projector_match_key(qwen_w5_mmproj),
                 "Different projector quantization variants must not collide");

    ok &= expect(!HLIEModelFamilyRegistry::is_projector_main_model_name("MMPROJ_Gemma4_26B.gguf"),
                 "Projector file detection should be case-insensitive");
    ok &= expect(!HLIEModelFamilyRegistry::supports_standalone("qwen2.5-vl_7b_v1.2.0.gguf"),
                 "Existing required-projector families should keep their behavior");
    ok &= expect(HLIEModelFamilyRegistry::detect_projector_family("HiModel_xh2_glm-asr_1.9b.gguf") == "glm-asr",
                 "The 1.4.0 GLM-ASR family should be projector-aware");
    ok &= expect(HLIEModelFamilyRegistry::detect_projector_family("HiModel_xh2_qwen3-forcealigner-0.6b.gguf") ==
                     "qwen3-forcealigner",
                 "The repository forcealigner spelling should be recognized");
    ok &= expect(HLIEModelFamilyRegistry::detect_metadata_family("gemma4") == HLIEModelFamilyRegistry::GEMMA4,
                 "Gemma 4 should be identified from general.architecture");
    ok &= expect(HLIEModelFamilyRegistry::detect_metadata_family("clip", {}, "gemma4v", "gemma4a") ==
                     HLIEModelFamilyRegistry::GEMMA4,
                 "Gemma 4 should be identified from mmproj metadata");
    ok &= expect(HLIEModelFamilyRegistry::detect_metadata_family("qwen35moe") == "qwen3.5",
                 "Underlying Qwen35 metadata should be normalized");
    return ok ? 0 : 1;
}

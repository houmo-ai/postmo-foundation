#include "hlie_app/hlie_model_info.h"

#include <catch_amalgamated.hpp>

using namespace hm_HLIE;

TEST_CASE("ModelInfo parses projector vision specifications", "[model-info]") {
    CHECK(HLIEModelInfo::parse_mmproj_vl_spec("mmproj_vit_448x448.gguf") == "vit_448_448");
    CHECK(HLIEModelInfo::parse_mmproj_vl_spec("mmproj_vit_768_2520.gguf") == "vit_768_2520");
    CHECK(HLIEModelInfo::parse_mmproj_vl_spec("mmproj_vit_644.gguf") == "vit_644");
    CHECK(HLIEModelInfo::parse_mmproj_vl_spec("mmproj_without_resolution.gguf") == "unknown");
}

TEST_CASE("ModelInfo estimates device count from deployment naming", "[model-info]") {
    CHECK(HLIEModelInfo::estimate_device_num("model_1chip.gguf") == 1);
    CHECK(HLIEModelInfo::estimate_device_num("model_4chips.gguf") == 4);
    CHECK(HLIEModelInfo::estimate_device_num("model-2d.gguf") == 2);
    CHECK(HLIEModelInfo::estimate_device_num("model-0d.gguf") == 1);
    CHECK(HLIEModelInfo::estimate_device_num("model.gguf") == 1);
}

TEST_CASE("ModelInfo parses routing identity without opening a GGUF", "[model-info][identity]") {
    HLIEModelInfo info;
    REQUIRE(info.parse_identity_from_gguf_filename(
                "Qwen3-8B-v1.2.3-32K-b4-6core-2chips-20260101.gguf") == 0);

    CHECK(info.get_model_name() == "Qwen3-8B-v1.2.3-32K-b4-6core-2chips-20260101");
    CHECK(info.get_model_family() == "qwen3");
    CHECK(info.get_model_version() == "v1.2.3");
    CHECK(info.get_params() == "8b");
    CHECK(info.get_context_size() == 32 * 1024);
    CHECK(info.get_batch_num() == 4);
    CHECK(info.get_cores() == 6);
    CHECK(info.get_device_num() == 2);
    CHECK(info.get_xh_version() == 2);
    CHECK(info.get_service_kind() == "llm");
    CHECK(info.get_role() == HLIEModelRole::Chat);
}

TEST_CASE("ModelInfo projects an SDK-free core model spec", "[model-info][p1][core-spec]") {
    HLIEModelInfo info;
    REQUIRE(info.parse_identity_from_gguf_filename(
                "HiModel_xh2_qwen3-asr_0.6b_32k_b2_2chips_v1.3.0.gguf") == 0);

    const auto spec = info.to_model_spec();
    CHECK(spec.identity.model_key == info.model_key_);
    CHECK(spec.identity.model_name == info.get_model_name());
    CHECK(spec.identity.model_path == info.get_model_path());
    CHECK(spec.backend_alias == "llm");
    CHECK(spec.model_family == "qwen3-asr");
    CHECK(spec.role == HLIEModelRole::Asr);
    CHECK(spec.capabilities.audio);
    CHECK(spec.context_size == 32 * 1024);
    CHECK(spec.batch_size == 2);
    CHECK(spec.required_device_count == 2);
}

TEST_CASE("ModelInfo memory estimation delegates pure policy without changing fallbacks",
          "[model-info][p1][memory]") {
    HLIEModelInfo empty;
    CHECK(empty.get_estimated_mem() == 0);

    HLIEModelInfo qwen;
    REQUIRE(qwen.parse_identity_from_gguf_filename(
                "HiModel_xh2_qwen3_8b_v1.2.0.gguf") == 0);
    CHECK(qwen.get_estimated_mem() == 10240);
}

TEST_CASE("ModelInfo infers model roles from deployment identities", "[model-info][role]") {
    struct Case {
        const char* filename;
        HLIEModelRole role;
        bool audio;
        bool vision;
    };
    const Case cases[] = {
        {"HiModel_xh2_qwen3-reranker_8b_v1.2.0.gguf", HLIEModelRole::Rerank, false, false},
        {"HiModel_xh2_qwen3-asr_0.6b_v1.3.0.gguf", HLIEModelRole::Asr, true, false},
        {"HiModel_xh2_qwen3-ForcedAligner_0.6b_v1.3.0.gguf", HLIEModelRole::ForcedAligner, true, false},
        {"HiModel_xh2_glm-ocr_0.9b_v1.4.0.gguf", HLIEModelRole::Ocr, false, true},
        {"HiModel-Whisper-medium-v1.2.0.gguf", HLIEModelRole::Asr, true, false},
        {"HiModel_xh2_qwen3-tts-1.7b-v1.4.0.gguf", HLIEModelRole::Tts, true, false},
    };

    for (const auto& test : cases) {
        HLIEModelInfo info;
        REQUIRE(info.parse_identity_from_gguf_filename(test.filename) == 0);
        CHECK(info.get_role() == test.role);
        CHECK(info.get_capabilities().audio == test.audio);
        CHECK(info.get_capabilities().vision == test.vision);
    }
}

TEST_CASE("ModelInfo normalizes explicit service kinds", "[model-info]") {
    HLIEModelInfo info;
    info.set_service_kind("WhIsPeR");
    CHECK(info.get_service_kind() == "whisper");
    CHECK(info.get_role() == HLIEModelRole::Asr);
    CHECK(info.is_llm_);

    info.set_service_kind("SD");
    CHECK(info.get_service_kind() == "sd");
    CHECK(info.get_role() == HLIEModelRole::ImageGeneration);
    CHECK_FALSE(info.is_llm_);

    info.set_service_kind("");
    CHECK(info.get_service_kind() == "llm");
    CHECK(info.get_role() == HLIEModelRole::Chat);
    CHECK(info.is_llm_);
}

TEST_CASE("ModelInfo builds a complete SD descriptor", "[model-info]") {
    HLIEModelInfo info;
    info.build_sd("SD", "vae.gguf", "t5.gguf", "clip.gguf", "clip-l.gguf", "mmdit.gguf", "0,1");

    CHECK(info.get_model_name() == "sd3");
    CHECK(info.model_key_ == "sd3");
    CHECK(info.get_model_type() == "sd");
    CHECK(info.get_service_kind() == "sd");
    CHECK(info.get_model_family() == "sd3");
    CHECK(info.get_role() == HLIEModelRole::ImageGeneration);
    CHECK(info.get_vae_model() == "vae.gguf");
    CHECK(info.get_t5_model() == "t5.gguf");
    CHECK(info.get_clip_model() == "clip.gguf");
    CHECK(info.get_clip_l_model() == "clip-l.gguf");
    CHECK(info.get_mmdit_model() == "mmdit.gguf");
    CHECK(info.get_devices() == "0,1");
    CHECK_FALSE(info.is_llm_);
}

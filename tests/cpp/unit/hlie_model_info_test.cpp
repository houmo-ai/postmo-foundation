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
    CHECK(info.get_service_kind() == "llm");
}

TEST_CASE("ModelInfo normalizes explicit service kinds", "[model-info]") {
    HLIEModelInfo info;
    info.set_service_kind("WhIsPeR");
    CHECK(info.get_service_kind() == "whisper");
    CHECK(info.is_llm_);

    info.set_service_kind("SD");
    CHECK(info.get_service_kind() == "sd");
    CHECK_FALSE(info.is_llm_);

    info.set_service_kind("");
    CHECK(info.get_service_kind() == "llm");
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
    CHECK(info.get_vae_model() == "vae.gguf");
    CHECK(info.get_t5_model() == "t5.gguf");
    CHECK(info.get_clip_model() == "clip.gguf");
    CHECK(info.get_clip_l_model() == "clip-l.gguf");
    CHECK(info.get_mmdit_model() == "mmdit.gguf");
    CHECK(info.get_devices() == "0,1");
    CHECK_FALSE(info.is_llm_);
}

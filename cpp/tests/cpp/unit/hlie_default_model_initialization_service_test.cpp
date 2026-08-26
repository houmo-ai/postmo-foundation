#include <catch_amalgamated.hpp>

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "hlie_app/startup/hlie_default_model_initialization_service.h"

namespace {

using namespace hm_HLIE;

HLIEModelInfo llm_model(const std::string& filename) {
    HLIEModelInfo model;
    REQUIRE(model.parse_identity_from_gguf_filename(filename) == 0);
    return model;
}

HLIEModelInfo sd3_model() {
    HLIEModelInfo model;
    model.build_sd("sd", "vae", "t5", "clip", "clip-l", "mmdit", "0");
    return model;
}

} // namespace

TEST_CASE("Default model initialization preserves zero-load startup",
          "[p1][default-initialization]") {
    HLIEDefaultModelInitializationService service({});
    HLIEDefaultModelInitializationOptions master_options;
    master_options.zero_load_allowed = true;
    HLIEDefaultModelInitializationOptions worker_options;
    worker_options.zero_load_allowed = false;

    const auto master = service.initialize({}, master_options);
    const auto worker = service.initialize({}, worker_options);

    CHECK(master.status == HLIEDefaultModelInitializationStatus::ZeroLoad);
    CHECK(worker.status == HLIEDefaultModelInitializationStatus::Failed);
}

TEST_CASE("Default model initialization prepares and loads LLM primary",
          "[p1][default-initialization]") {
    auto model = llm_model("default-model.gguf");
    bool mmproj_applied = false;
    bool registered = false;
    bool resolved = false;
    bool loaded = false;
    const auto expected_mmproj =
        (std::filesystem::path("models") / "vision.gguf")
            .lexically_normal().string();
    HLIEDefaultModelInitializationService service({
        [](const HLIEModelInfo&) { return true; },
        [&](HLIEModelInfo& selected, const std::string& path) {
            CHECK(path == expected_mmproj);
            selected.mm_proj_path_ = path;
            mmproj_applied = true;
        },
        [&](HLIEModelInfo&) {
            resolved = true;
            return true;
        },
        [](const std::string&) { return false; },
        [&](HLIEModelInfo&) { registered = true; },
        [&](const HLIEModelInfo& selected) {
            CHECK(selected.get_mm_proj_path() == expected_mmproj);
            loaded = true;
            return true;
        },
        {},
        [](const std::string&) { return true; },
    });

    const auto result = service.initialize(
        {model},
        {"models", model.model_key_, "vision", false, true});

    CHECK(result.initialized());
    CHECK(result.primary_initialized);
    CHECK_FALSE(result.sd3_initialized);
    CHECK(mmproj_applied);
    CHECK(registered);
    CHECK(resolved);
    CHECK(loaded);
}

TEST_CASE("Default model initialization keeps optional SD3 fallback",
          "[p1][default-initialization]") {
    bool sd3_loaded = false;
    HLIEDefaultModelInitializationService service({
        {}, {}, {}, {}, {}, {},
        [&](const HLIEModelInfo& selected) {
            CHECK(selected.get_model_name() == "sd3");
            sd3_loaded = true;
            return true;
        },
    });

    const auto result = service.initialize(
        {sd3_model()}, {"models", "missing-default", {}, true, true});

    CHECK(result.initialized());
    CHECK_FALSE(result.primary_initialized);
    CHECK(result.sd3_initialized);
    CHECK(sd3_loaded);
}

TEST_CASE("Default model initialization rejects missing configured SD3",
          "[p1][default-initialization]") {
    HLIEDefaultModelInitializationService service({});

    const auto result = service.initialize(
        {llm_model("other.gguf")}, {"models", {}, {}, true, true});

    CHECK(result.status == HLIEDefaultModelInitializationStatus::Failed);
}

#include "hlie_app/hlie_config_bundle_validator.h"

#include <catch_amalgamated.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"

using namespace hm_HLIE;

namespace {
struct TemporaryBundle {
    std::filesystem::path root;

    TemporaryBundle() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() / ("hlie-config-bundle-test-" + std::to_string(suffix));
        std::filesystem::create_directories(root);
    }

    ~TemporaryBundle() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    void write(const std::string& name, const std::string& content) const {
        std::ofstream(root / name, std::ios::binary) << content;
    }

    void write_valid_files(const bool presets_enabled = true, const bool preset_strict = true) const {
        write("sys.cfg", "[model]\nllm_presets_enabled = " + std::string(presets_enabled ? "true" : "false") +
              "\nllm_preset_strict = " + std::string(preset_strict ? "true" : "false") + "\n");
        write("config_schema.json", R"({"version":1,"sections":{}})");
        write("memory_estimates.json",
              R"({"rules":[{"model_family":"qwen3.6","params":"27b","chips":1,"memory_per_device_mb":1024.5}]})");
        write("model_presets.json", R"({"schema_version":1,"catalog_version":"test","presets":[]})");
    }
};

void set_config_arg(HLIEConfigService& service, const std::filesystem::path& directory) {
    std::string program = "HLIECpp";
    std::string option = "--config";
    std::string value = directory.string();
    char* argv[] = {program.data(), option.data(), value.data()};
    service.set_args(3, argv);
}

void prepare_and_parse(HLIEConfigService& service, const TemporaryBundle& bundle) {
    HLIELogger::create_console_logger(HLIE_BOOT, "off");
    set_config_arg(service, bundle.root);
    std::string error;
    REQUIRE(service.prepare_config_bundle("unused/sys.cfg", &error));
    REQUIRE(error.empty());
    REQUIRE(service.parse((bundle.root / "sys.cfg").string()) == 0);
    service.finalize_config_bundle(
        service.get_string("model", "llm_preset_catalog", "model_presets.json"),
        service.get_string("model", "llm_override_catalog", "model_overrides.json"));
}
} // namespace

TEST_CASE("Explicit config bundle validates required assets and permits missing overrides", "[config][bundle]") {
    TemporaryBundle bundle;
    bundle.write_valid_files();
    HLIEConfigService service;
    prepare_and_parse(service, bundle);

    const auto result = HLIEConfigBundleValidator::validate(service);
    CHECK(result.valid);
    CHECK(result.error.empty());
    CHECK_FALSE(result.notes.empty());
}

TEST_CASE("Explicit config bundle requires schema and memory estimate files", "[config][bundle]") {
    TemporaryBundle bundle;
    bundle.write_valid_files();
    std::filesystem::remove(bundle.root / "config_schema.json");
    HLIEConfigService service;
    prepare_and_parse(service, bundle);

    auto result = HLIEConfigBundleValidator::validate(service);
    CHECK_FALSE(result.valid);
    CHECK(result.error.find("config_schema.json") != std::string::npos);

    bundle.write("config_schema.json", R"({"version":1,"sections":{}})");
    std::filesystem::remove(bundle.root / "memory_estimates.json");
    result = HLIEConfigBundleValidator::validate(service);
    CHECK_FALSE(result.valid);
    CHECK(result.error.find("memory_estimates.json") != std::string::npos);
}

TEST_CASE("Explicit config bundle rejects malformed memory rules", "[config][bundle]") {
    TemporaryBundle bundle;
    bundle.write_valid_files();
    bundle.write("memory_estimates.json", R"({"rules":[{"model_family":"qwen3.6","chips":0}]})");
    HLIEConfigService service;
    prepare_and_parse(service, bundle);

    const auto result = HLIEConfigBundleValidator::validate(service);
    CHECK_FALSE(result.valid);
    CHECK(result.error.find("invalid rule") != std::string::npos);
}

TEST_CASE("Preset catalog is conditionally required", "[config][bundle]") {
    TemporaryBundle bundle;
    bundle.write_valid_files(false, true);
    std::filesystem::remove(bundle.root / "model_presets.json");
    HLIEConfigService service;
    prepare_and_parse(service, bundle);

    const auto result = HLIEConfigBundleValidator::validate(service);
    CHECK(result.valid);
}

TEST_CASE("Invalid optional overrides follow preset strictness", "[config][bundle]") {
    TemporaryBundle bundle;
    bundle.write_valid_files(true, false);
    bundle.write("model_overrides.json", "not-json");
    HLIEConfigService service;
    prepare_and_parse(service, bundle);

    const auto result = HLIEConfigBundleValidator::validate(service);
    CHECK(result.valid);
    REQUIRE(result.warnings.size() == 1);
    CHECK(result.warnings.front().find("override") != std::string::npos);
}

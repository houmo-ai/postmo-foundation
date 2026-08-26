#include "hlie_base/hlie_config_service.h"

#include <catch_amalgamated.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace hm_HLIE;

namespace {
struct TemporaryConfigDirectory {
    std::filesystem::path path;

    TemporaryConfigDirectory() {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / ("hlie-config-path-test-" + std::to_string(suffix));
        std::filesystem::create_directories(path);
    }

    ~TemporaryConfigDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void set_args(HLIEConfigService& service, std::vector<std::string> values) {
    std::vector<char*> argv;
    argv.reserve(values.size());
    for (auto& value : values) {
        argv.push_back(value.data());
    }
    service.set_args(static_cast<int>(argv.size()), argv.data());
}
} // namespace

TEST_CASE("Config directory override resolves a sibling sys.cfg", "[config][cli]") {
    TemporaryConfigDirectory config;
    std::ofstream(config.path / "sys.cfg") << "[server]\nport = 7901\n";

    HLIEConfigService service;
    set_args(service, {"HLIECpp", "--config", config.path.string()});

    std::string error;
    REQUIRE(service.prepare_config_bundle("default/config/sys.cfg", &error));
    const auto bundle = service.get_config_bundle();
    CHECK(error.empty());
    CHECK(bundle.explicit_override);
    CHECK(bundle.root == std::filesystem::absolute(config.path).lexically_normal());
    CHECK(bundle.sys_config == std::filesystem::absolute(config.path / "sys.cfg").lexically_normal());
    CHECK(bundle.config_schema == bundle.root / "config_schema.json");
    CHECK(bundle.memory_estimates == bundle.root / "memory_estimates.json");
    CHECK(service.get_args().at("--config") == std::filesystem::absolute(config.path).lexically_normal().string());

    const auto absolute_override = config.path.parent_path() / "shared-overrides.json";
    service.finalize_config_bundle("catalogs/presets.json", absolute_override.string());
    const auto finalized = service.get_config_bundle();
    CHECK(finalized.model_presets == bundle.root / "catalogs/presets.json");
    CHECK(finalized.model_overrides == absolute_override.lexically_normal());
}

TEST_CASE("Config directory accepts standard equals syntax", "[config][cli]") {
    TemporaryConfigDirectory config;
    std::ofstream(config.path / "sys.cfg") << "[server]\nport = 7901\n";

    HLIEConfigService service;
    set_args(service, {"HLIECpp", "--config=" + config.path.string()});

    REQUIRE(service.prepare_config_bundle("unused/sys.cfg"));
    CHECK(service.get_config_bundle().sys_config ==
          std::filesystem::absolute(config.path / "sys.cfg").lexically_normal());
}

TEST_CASE("Config directory override rejects missing values and files", "[config][cli]") {
    HLIEConfigService service;
    set_args(service, {"HLIECpp", "--config"});

    std::string error;
    CHECK_FALSE(service.prepare_config_bundle("default/sys.cfg", &error));
    CHECK(error.find("requires") != std::string::npos);

    TemporaryConfigDirectory config;
    set_args(service, {"HLIECpp", "--config", config.path.string()});
    CHECK_FALSE(service.prepare_config_bundle("default/sys.cfg", &error));
    CHECK(error.find("sys.cfg") != std::string::npos);
}

TEST_CASE("Config path keeps platform default when override is absent", "[config][cli]") {
    HLIEConfigService service;
    set_args(service, {"HLIECpp", "--port", "9000"});

    std::string error = "stale";
    REQUIRE(service.prepare_config_bundle("platform/config/sys.cfg", &error));
    const auto bundle = service.get_config_bundle();
    CHECK(bundle.sys_config == std::filesystem::path("platform/config/sys.cfg"));
    CHECK(bundle.root == std::filesystem::path("platform/config"));
    CHECK_FALSE(bundle.explicit_override);
    CHECK(error.empty());
}

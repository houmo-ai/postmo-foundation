#include "hlie_app/util/hlie_hal_path_resolver.h"

#include <catch_amalgamated.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>

using namespace hm_HLIE;

namespace {
    struct TemporaryDirectory {
        std::filesystem::path path;

        TemporaryDirectory() {
            const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
            path = std::filesystem::temp_directory_path() / ("hlie-hal-path-test-" + std::to_string(suffix));
            std::filesystem::create_directories(path);
        }

        ~TemporaryDirectory() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    };
}

TEST_CASE("HAL SDK environment accepts roots and direct hal library paths", "[hal][paths]") {
    std::vector<std::filesystem::path> paths;
    const std::filesystem::path sdk_root = "/opt/vendor/customer-sdk";

    hal_paths::append_sdk_environment_paths(paths, sdk_root.string().c_str());
    REQUIRE(paths.size() == 2);
    CHECK(paths[0] == sdk_root);
    CHECK(paths[1] == sdk_root / "hal" / "lib");

    paths.clear();
    const auto direct_hal_path = sdk_root / "hal" / "lib";
    hal_paths::append_sdk_environment_paths(paths, (direct_hal_path.string() + "/").c_str());
    REQUIRE(paths.size() == 1);
    CHECK(paths[0] == direct_hal_path);
}

TEST_CASE("HAL discovery finds every directory ending in sdk", "[hal][paths]") {
    TemporaryDirectory root;
    std::filesystem::create_directories(root.path / "sensenext-sdk" / "hal" / "lib");
    std::filesystem::create_directories(root.path / "customer-sdk" / "hal" / "lib");
    std::filesystem::create_directories(root.path / "not-an-sdk-installation" / "hal" / "lib");
    std::ofstream(root.path / "file-sdk").put('\n');

    std::vector<std::filesystem::path> paths = {root.path / "customer-sdk" / "hal" / "lib"};
    hal_paths::append_installed_sdk_paths(paths, root.path);

    REQUIRE(paths.size() == 2);
    CHECK(paths[0] == root.path / "customer-sdk" / "hal" / "lib");
    CHECK(paths[1] == root.path / "sensenext-sdk" / "hal" / "lib");
}

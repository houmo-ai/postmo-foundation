#include "hlie_app/util/hlie_hal_path_resolver.h"

#include <algorithm>
#include <string>

namespace hm_HLIE::hal_paths {
    void append_unique(std::vector<std::filesystem::path>& paths, const std::filesystem::path& path) {
        if (std::find(paths.begin(), paths.end(), path) == paths.end()) {
            paths.push_back(path);
        }
    }

    void append_sdk_environment_paths(std::vector<std::filesystem::path>& paths, const char* sdk_path_value) {
        if (!sdk_path_value || !*sdk_path_value) {
            return;
        }

        auto sdk_path = std::filesystem::path(sdk_path_value).lexically_normal();
        if (sdk_path.filename().empty()) {
            sdk_path = sdk_path.parent_path();
        }
        append_unique(paths, sdk_path);

        const bool already_hal_lib = sdk_path.filename() == "lib" && sdk_path.parent_path().filename() == "hal";
        if (!already_hal_lib) {
            append_unique(paths, sdk_path / "hal" / "lib");
        }
    }

    void append_installed_sdk_paths(std::vector<std::filesystem::path>& paths,
                                    const std::filesystem::path& install_root) {
        constexpr auto SDK_SUFFIX = "-sdk";
        constexpr auto SDK_SUFFIX_LENGTH = std::char_traits<char>::length(SDK_SUFFIX);
        std::vector<std::filesystem::path> discovered;
        std::error_code error;

        for (const auto& entry : std::filesystem::directory_iterator(install_root, error)) {
            if (error) {
                break;
            }
            if (!entry.is_directory(error) || error) {
                error.clear();
                continue;
            }

            const auto directory_name = entry.path().filename().string();
            if (directory_name.size() >= SDK_SUFFIX_LENGTH &&
                directory_name.compare(directory_name.size() - SDK_SUFFIX_LENGTH, SDK_SUFFIX_LENGTH, SDK_SUFFIX) == 0) {
                discovered.push_back(entry.path() / "hal" / "lib");
            }
        }

        std::sort(discovered.begin(), discovered.end());
        for (const auto& path : discovered) {
            append_unique(paths, path);
        }
    }
}

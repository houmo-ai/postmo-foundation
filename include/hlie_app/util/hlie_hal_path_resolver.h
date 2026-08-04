#pragma once

#include <filesystem>
#include <vector>

namespace hm_HLIE::hal_paths {
    void append_unique(std::vector<std::filesystem::path>& paths, const std::filesystem::path& path);

    void append_sdk_environment_paths(std::vector<std::filesystem::path>& paths, const char* sdk_path_value);

    void append_installed_sdk_paths(std::vector<std::filesystem::path>& paths,
                                    const std::filesystem::path& install_root = "/usr/local");
}

#include <catch_amalgamated.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <zlib.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "hlie_base/hlie_logger.h"

namespace {
    namespace fs = std::filesystem;

    fs::path unique_log_root(const std::string& name) {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        return fs::temp_directory_path() / ("hlie-logger-test-" + name + "-" + std::to_string(suffix));
    }

    std::uint64_t current_process_id() {
#ifdef _WIN32
        return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
        return static_cast<std::uint64_t>(getpid());
#endif
    }

    std::vector<fs::path> regular_files(const fs::path& directory) {
        std::vector<fs::path> files;
        std::error_code error;
        for (fs::recursive_directory_iterator it(directory, error), end; !error && it != end; it.increment(error)) {
            if (it->is_regular_file(error) && !error && it->path().filename() != ".hlie-log-directory") {
                files.push_back(it->path());
            }
        }
        return files;
    }

    std::string decompress_gzip(const fs::path& path) {
        const auto name = path.string();
        gzFile file = gzopen(name.c_str(), "rb");
        REQUIRE(file != nullptr);
        std::string output;
        std::array<char, 4096> buffer{};
        int count = 0;
        while ((count = gzread(file, buffer.data(), static_cast<unsigned>(buffer.size()))) > 0) {
            output.append(buffer.data(), static_cast<std::size_t>(count));
        }
        REQUIRE(count == 0);
        REQUIRE(gzclose(file) == Z_OK);
        return output;
    }

    void write_rotating_logs(const std::shared_ptr<spdlog::logger>& logger) {
        const std::string payload(180, 'A');
        for (int i = 0; i < 80; ++i) {
            logger->info("record={:04d} payload={}", i, payload);
        }
        logger->flush();
    }
}

TEST_CASE("managed logger keeps date directories and limits daily parts", "[logger]") {
    const auto root = unique_log_root("date");
    hm_HLIE::HLIEManagedLogConfig config;
    config.max_file_size_bytes = 512;
    config.retention_days = 7;
    config.max_files = 3;
    config.compress_rotated_files = true;
    config.compression_level = 1;

    auto logger = hm_HLIE::HLIELogger::create_managed_logger(root.string(), "hlie-llm-0.log",
                                                             "HLIE-LOGGER-DATE-TEST", "info", config);
    write_rotating_logs(logger);
    spdlog::drop("HLIE-LOGGER-DATE-TEST");
    logger.reset();
    hm_HLIE::HLIELogger::wait_for_compression();

    const auto files = regular_files(root);
    REQUIRE(files.size() <= 3);
    REQUIRE_FALSE(files.empty());
    bool found_gzip = false;
    for (const auto& file : files) {
        REQUIRE(file.parent_path() != root);
        if (file.extension() == ".gz") {
            found_gzip = true;
            REQUIRE(decompress_gzip(file).find("record=") != std::string::npos);
        }
    }
    REQUIRE(found_gzip);
    if (std::getenv("HLIE_KEEP_TEST_LOGS") == nullptr) {
        std::error_code error;
        fs::remove_all(root, error);
    }
}

TEST_CASE("managed log file sizes require an integer binary unit", "[logger]") {
    std::uintmax_t bytes = 0;
    REQUIRE(hm_HLIE::HLIELogger::parse_file_size("512K", bytes));
    REQUIRE(bytes == 512U * 1024U);
    REQUIRE(hm_HLIE::HLIELogger::parse_file_size(" 10m ", bytes));
    REQUIRE(bytes == 10U * 1024U * 1024U);
    REQUIRE(hm_HLIE::HLIELogger::parse_file_size("1G", bytes));
    REQUIRE(bytes == 1024ULL * 1024ULL * 1024ULL);
    REQUIRE_FALSE(hm_HLIE::HLIELogger::parse_file_size("10", bytes));
    REQUIRE_FALSE(hm_HLIE::HLIELogger::parse_file_size("1.5M", bytes));
    REQUIRE_FALSE(hm_HLIE::HLIELogger::parse_file_size("0M", bytes));
    REQUIRE_FALSE(hm_HLIE::HLIELogger::parse_file_size("abc", bytes));
}

TEST_CASE("zero retention uses flat globally count-limited files", "[logger]") {
    const auto root = unique_log_root("count");
    hm_HLIE::HLIEManagedLogConfig config;
    config.max_file_size_bytes = 512;
    config.retention_days = 0;
    config.max_files = 3;
    config.compress_rotated_files = true;
    config.compression_level = 1;

    auto logger = hm_HLIE::HLIELogger::create_managed_logger(root.string(), "hlie-llm-0.log",
                                                             "HLIE-LOGGER-COUNT-TEST", "info", config);
    write_rotating_logs(logger);
    spdlog::drop("HLIE-LOGGER-COUNT-TEST");
    logger.reset();
    hm_HLIE::HLIELogger::wait_for_compression();

    const auto files = regular_files(root);
    REQUIRE(files.size() <= 3);
    REQUIRE_FALSE(files.empty());
    for (const auto& file : files) {
        REQUIRE(file.parent_path() == root);
        REQUIRE(file.filename().string().find("hlie-llm-0.") == 0);
        REQUIRE(file.filename().string().find(".log") != std::string::npos);
    }
    if (std::getenv("HLIE_KEEP_TEST_LOGS") == nullptr) {
        std::error_code error;
        fs::remove_all(root, error);
    }
}

TEST_CASE("zero max files leaves the part count unlimited", "[logger]") {
    const auto root = unique_log_root("unlimited");
    hm_HLIE::HLIEManagedLogConfig config;
    config.max_file_size_bytes = 512;
    config.retention_days = 0;
    config.max_files = 0;
    config.compress_rotated_files = false;

    auto logger = hm_HLIE::HLIELogger::create_managed_logger(root.string(), "access-0.log",
                                                             "HLIE-LOGGER-UNLIMITED-TEST", "info", config);
    write_rotating_logs(logger);
    spdlog::drop("HLIE-LOGGER-UNLIMITED-TEST");
    logger.reset();

    const auto files = regular_files(root);
    REQUIRE(files.size() > 3);
    for (const auto& file : files) {
        REQUIRE(file.parent_path() == root);
        REQUIRE(file.extension() == ".log");
    }
    if (std::getenv("HLIE_KEEP_TEST_LOGS") == nullptr) {
        std::error_code error;
        fs::remove_all(root, error);
    }
}

TEST_CASE("retention never resolves log and gzip conflicts by deleting data", "[logger]") {
    const auto root = unique_log_root("conflict");
    const auto date_directory = hm_HLIE::HLIELogger::current_date_directory(root.string());
    const auto source = date_directory / "hlie-llm-0.0001.log";
    auto gzip = source;
    gzip += ".gz";
    {
        std::ofstream source_stream(source, std::ios::binary);
        source_stream << "source must survive";
        std::ofstream gzip_stream(gzip, std::ios::binary);
        gzip_stream << "possibly corrupt gzip must not authorize source deletion";
    }

    hm_HLIE::HLIEManagedLogConfig config;
    config.max_file_size_bytes = 1024;
    config.retention_days = 7;
    config.max_files = 1;
    config.compress_rotated_files = false;
    auto logger = hm_HLIE::HLIELogger::create_managed_logger(root.string(), "hlie-llm-0.log",
                                                             "HLIE-LOGGER-CONFLICT-TEST", "info", config);
    logger->info("new active part");
    logger->flush();
    spdlog::drop("HLIE-LOGGER-CONFLICT-TEST");
    logger.reset();

    REQUIRE(fs::exists(source));
    REQUIRE(fs::exists(gzip));
    if (std::getenv("HLIE_KEEP_TEST_LOGS") == nullptr) {
        std::error_code error;
        fs::remove_all(root, error);
    }
}

TEST_CASE("gzip never overwrites a pre-existing destination", "[logger]") {
    const auto root = unique_log_root("gzip-conflict");
    const auto date_directory = hm_HLIE::HLIELogger::current_date_directory(root.string());
    const auto source = date_directory / "hlie-llm-0.0001.log";
    auto gzip = source;
    gzip += ".gz";
    {
        std::ofstream source_stream(source, std::ios::binary);
        source_stream << "source must survive";
        std::ofstream gzip_stream(gzip, std::ios::binary);
        gzip_stream << "existing destination must survive";
    }

    hm_HLIE::HLIEManagedLogConfig config;
    config.max_file_size_bytes = 1024;
    config.retention_days = 7;
    config.max_files = 0;
    config.compress_rotated_files = true;
    auto logger = hm_HLIE::HLIELogger::create_managed_logger(root.string(), "hlie-llm-0.log",
                                                             "HLIE-LOGGER-GZIP-CONFLICT-TEST", "info", config);
    spdlog::drop("HLIE-LOGGER-GZIP-CONFLICT-TEST");
    logger.reset();
    hm_HLIE::HLIELogger::wait_for_compression();

    REQUIRE(fs::exists(source));
    REQUIRE(fs::exists(gzip));
    std::ifstream gzip_stream(gzip, std::ios::binary);
    REQUIRE(std::string(std::istreambuf_iterator<char>(gzip_stream), {}) == "existing destination must survive");
    if (std::getenv("HLIE_KEEP_TEST_LOGS") == nullptr) {
        std::error_code error;
        fs::remove_all(root, error);
    }
}

TEST_CASE("abandoned compression claims are restored before reopening a log", "[logger]") {
    const auto root = unique_log_root("claim-recovery");
    const auto date_directory = hm_HLIE::HLIELogger::current_date_directory(root.string());
    const auto claimed = date_directory / "hlie-llm-0.0001.log.compressing.999999999.1";
    {
        std::ofstream stream(claimed, std::ios::binary);
        stream << "recovered-prefix\n";
    }

    hm_HLIE::HLIEManagedLogConfig config;
    config.max_file_size_bytes = 1024;
    config.retention_days = 7;
    config.max_files = 0;
    config.compress_rotated_files = false;
    auto logger = hm_HLIE::HLIELogger::create_managed_logger(root.string(), "hlie-llm-0.log",
                                                             "HLIE-LOGGER-CLAIM-RECOVERY-TEST", "info", config);
    logger->info("after recovery");
    logger->flush();
    spdlog::drop("HLIE-LOGGER-CLAIM-RECOVERY-TEST");
    logger.reset();

    const auto source = date_directory / "hlie-llm-0.0001.log";
    REQUIRE(fs::exists(source));
    REQUIRE_FALSE(fs::exists(claimed));
    std::ifstream source_stream(source, std::ios::binary);
    const std::string contents(std::istreambuf_iterator<char>(source_stream), {});
    REQUIRE(contents.find("recovered-prefix") != std::string::npos);
    REQUIRE(contents.find("after recovery") != std::string::npos);
    if (std::getenv("HLIE_KEEP_TEST_LOGS") == nullptr) {
        std::error_code error;
        fs::remove_all(root, error);
    }
}

TEST_CASE("recovery does not delete a live compressor temporary file", "[logger]") {
    const auto root = unique_log_root("live-temporary");
    const auto date_directory = hm_HLIE::HLIELogger::current_date_directory(root.string());
    const auto temporary = date_directory /
        ("hlie-llm-0.0001.log.gz.tmp." + std::to_string(current_process_id()) + ".1");
    const auto abandoned = date_directory / "hlie-llm-0.0002.log.gz.tmp.999999999.1";
    {
        std::ofstream stream(temporary, std::ios::binary);
        stream << "active temporary data";
        std::ofstream abandoned_stream(abandoned, std::ios::binary);
        abandoned_stream << "abandoned temporary data";
    }

    hm_HLIE::HLIEManagedLogConfig config;
    config.max_file_size_bytes = 1024;
    config.retention_days = 7;
    config.max_files = 0;
    config.compress_rotated_files = false;
    auto logger = hm_HLIE::HLIELogger::create_managed_logger(root.string(), "hlie-llm-0.log",
                                                             "HLIE-LOGGER-LIVE-TEMPORARY-TEST", "info", config);
    spdlog::drop("HLIE-LOGGER-LIVE-TEMPORARY-TEST");
    logger.reset();

    REQUIRE(fs::exists(temporary));
    REQUIRE_FALSE(fs::exists(abandoned));
    if (std::getenv("HLIE_KEEP_TEST_LOGS") == nullptr) {
        std::error_code error;
        fs::remove_all(root, error);
    }
}

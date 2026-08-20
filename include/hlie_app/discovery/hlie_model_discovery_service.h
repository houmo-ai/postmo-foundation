#ifndef HLIE_APP_MODEL_DISCOVERY_SERVICE_H
#define HLIE_APP_MODEL_DISCOVERY_SERVICE_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

struct HLIEModelDiscoveryOptions {
    std::chrono::milliseconds scan_interval{std::chrono::seconds(30)};
    std::chrono::milliseconds stability_window{std::chrono::seconds(10)};
    bool scan_on_request = false;
    std::chrono::milliseconds request_cooldown{std::chrono::seconds(5)};
    std::chrono::milliseconds request_wait_timeout{std::chrono::seconds(15)};
};

struct HLIEModelDiscoveryScanResult {
    bool successful = false;
    bool stable = false;
    bool parsed = false;
    int published_models = 0;
};

struct HLIEModelDiscoveryMetrics {
    std::uint64_t scan_attempts = 0;
    std::uint64_t parse_attempts = 0;
    std::uint64_t scan_failures = 0;
    std::uint64_t published_models = 0;
    std::uint64_t requested_generation = 0;
    std::uint64_t completed_generation = 0;
    bool running = false;
    bool pending_stability = false;
};

// Filesystem discovery coordinator. Parsing returns a deferred catalog update;
// the update is published only after the directory snapshot is verified again.
class HLIEModelDiscoveryService {
public:
    using OptionsProvider = std::function<HLIEModelDiscoveryOptions()>;
    using DirectorySnapshotProvider =
        std::function<std::optional<std::string>()>;
    using PreparedCatalogUpdate = std::function<int()>;
    using PrepareCatalogUpdate =
        std::function<std::optional<PreparedCatalogUpdate>()>;

    HLIEModelDiscoveryService(
        std::string model_directory,
        OptionsProvider options_provider,
        PrepareCatalogUpdate prepare_catalog_update,
        DirectorySnapshotProvider snapshot_provider = {});
    ~HLIEModelDiscoveryService();

    HLIEModelDiscoveryService(const HLIEModelDiscoveryService&) = delete;
    HLIEModelDiscoveryService& operator=(
        const HLIEModelDiscoveryService&) = delete;

    [[nodiscard]] bool initialize_baseline();
    void start();
    void stop();

    // explicit_request bypasses scan_on_request but still honors cooldown and
    // the file stability window.
    [[nodiscard]] std::uint64_t request_scan(bool explicit_request = false);
    [[nodiscard]] bool wait_for(
        std::uint64_t generation,
        std::optional<std::chrono::milliseconds> timeout = std::nullopt);
    [[nodiscard]] HLIEModelDiscoveryScanResult scan_once();
    [[nodiscard]] HLIEModelDiscoveryMetrics metrics() const;

private:
    [[nodiscard]] HLIEModelDiscoveryOptions options() const;
    [[nodiscard]] bool pending_stability() const;
    void complete_requested_generation();
    void run_loop();

    std::string model_directory_;
    OptionsProvider options_provider_;
    PrepareCatalogUpdate prepare_catalog_update_;
    DirectorySnapshotProvider snapshot_provider_;

    mutable std::mutex lifecycle_mutex_;
    mutable std::mutex state_mutex_;
    std::mutex scan_mutex_;
    std::condition_variable wake_cv_;
    std::condition_variable completion_cv_;
    std::thread worker_;
    bool stop_requested_ = false;
    bool request_pending_ = false;
    bool baseline_initialized_ = false;
    bool stability_pending_ = false;
    std::string observed_snapshot_;
    std::chrono::steady_clock::time_point stability_pending_since_{};
    std::chrono::steady_clock::time_point last_request_{};
    std::uint64_t requested_generation_ = 0;
    std::uint64_t completed_generation_ = 0;

    std::atomic<bool> running_{false};
    std::atomic<std::uint64_t> scan_attempts_{0};
    std::atomic<std::uint64_t> parse_attempts_{0};
    std::atomic<std::uint64_t> scan_failures_{0};
    std::atomic<std::uint64_t> published_models_{0};
};

HLIE_NAMESPACE_END

#endif // HLIE_APP_MODEL_DISCOVERY_SERVICE_H

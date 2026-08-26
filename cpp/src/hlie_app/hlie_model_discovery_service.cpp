#include "hlie_app/discovery/hlie_model_discovery_service.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

namespace {
constexpr auto maximum_discovery_duration = std::chrono::hours(1);

HLIEModelDiscoveryOptions normalize_options(HLIEModelDiscoveryOptions options) {
    options.scan_interval = std::clamp(
        options.scan_interval, std::chrono::milliseconds(0),
        std::chrono::duration_cast<std::chrono::milliseconds>(
            maximum_discovery_duration));
    options.stability_window = std::clamp(
        options.stability_window, std::chrono::milliseconds(0),
        std::chrono::duration_cast<std::chrono::milliseconds>(
            maximum_discovery_duration));
    options.request_cooldown = std::clamp(
        options.request_cooldown, std::chrono::milliseconds(0),
        std::chrono::duration_cast<std::chrono::milliseconds>(
            maximum_discovery_duration));
    options.request_wait_timeout = std::clamp(
        options.request_wait_timeout, std::chrono::milliseconds(0),
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::minutes(2)));
    return options;
}

std::optional<std::string> snapshot_directory(
    const std::string& directory_path) {
    std::error_code ec;
    if (!std::filesystem::is_directory(directory_path, ec) || ec) {
        LOG_WARN("Cannot snapshot model directory: path={}, error={}",
                 directory_path, ec.message());
        return std::nullopt;
    }

    using SnapshotEntry = std::tuple<std::string, std::uintmax_t, std::int64_t>;
    std::vector<SnapshotEntry> entries;
    for (std::filesystem::directory_iterator it(directory_path, ec), end;
         it != end && !ec; it.increment(ec)) {
        std::error_code entry_ec;
        if (!it->is_regular_file(entry_ec) || entry_ec) {
            continue;
        }

        std::string extension = it->path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](const unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
        if (extension != ".gguf" && extension != ".json") {
            continue;
        }

        const auto size = it->file_size(entry_ec);
        if (entry_ec) {
            LOG_DEBUG("Skipping model discovery snapshot entry {}: {}",
                      it->path().string(), entry_ec.message());
            continue;
        }
        const auto write_time = it->last_write_time(entry_ec);
        if (entry_ec) {
            LOG_DEBUG("Skipping model discovery snapshot entry {}: {}",
                      it->path().string(), entry_ec.message());
            continue;
        }
        entries.emplace_back(
            it->path().filename().generic_string(), size,
            static_cast<std::int64_t>(
                write_time.time_since_epoch().count()));
    }
    if (ec) {
        LOG_WARN("Model directory snapshot stopped early: path={}, error={}",
                 directory_path, ec.message());
        return std::nullopt;
    }

    std::sort(entries.begin(), entries.end());
    std::ostringstream snapshot;
    for (const auto& [name, size, write_time] : entries) {
        snapshot << name.size() << ':' << name << ':' << size << ':'
                 << write_time << '\n';
    }
    return snapshot.str();
}
} // namespace

HLIEModelDiscoveryService::HLIEModelDiscoveryService(
    std::string model_directory,
    OptionsProvider options_provider,
    PrepareCatalogUpdate prepare_catalog_update,
    DirectorySnapshotProvider snapshot_provider)
    : model_directory_(std::move(model_directory)),
      options_provider_(std::move(options_provider)),
      prepare_catalog_update_(std::move(prepare_catalog_update)),
      snapshot_provider_(std::move(snapshot_provider)) {
    if (model_directory_.empty()) {
        throw std::invalid_argument(
            "ModelDiscoveryService requires a model directory");
    }
    if (!options_provider_ || !prepare_catalog_update_) {
        throw std::invalid_argument(
            "ModelDiscoveryService requires options and catalog callbacks");
    }
    if (!snapshot_provider_) {
        snapshot_provider_ = [directory = model_directory_] {
            return snapshot_directory(directory);
        };
    }
}

HLIEModelDiscoveryService::~HLIEModelDiscoveryService() {
    stop();
}

bool HLIEModelDiscoveryService::initialize_baseline() {
    std::lock_guard scan_lock(scan_mutex_);
    const auto snapshot = snapshot_provider_();
    if (!snapshot) {
        scan_failures_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    {
        std::lock_guard state_lock(state_mutex_);
        observed_snapshot_ = *snapshot;
        baseline_initialized_ = true;
        stability_pending_ = false;
    }
    LOG_INFO("Model discovery baseline initialized: directory={}",
             model_directory_);
    return true;
}

void HLIEModelDiscoveryService::start() {
    std::lock_guard lifecycle_lock(lifecycle_mutex_);
    if (worker_.joinable()) {
        return;
    }

    const auto current_options = options();
    if (current_options.scan_interval.count() <= 0 &&
        !current_options.scan_on_request) {
        LOG_INFO("Model discovery service is disabled");
        return;
    }
    {
        std::lock_guard state_lock(state_mutex_);
        stop_requested_ = false;
        request_pending_ = false;
        last_request_ = {};
    }
    running_.store(true, std::memory_order_release);
    worker_ = std::thread(&HLIEModelDiscoveryService::run_loop, this);
    LOG_INFO("Model discovery service started: directory={}, interval_ms={}, scan_on_request={}",
             model_directory_, current_options.scan_interval.count(),
             current_options.scan_on_request);
}

void HLIEModelDiscoveryService::stop() {
    std::lock_guard lifecycle_lock(lifecycle_mutex_);
    {
        std::lock_guard state_lock(state_mutex_);
        stop_requested_ = true;
    }
    wake_cv_.notify_all();
    completion_cv_.notify_all();
    if (!worker_.joinable()) {
        running_.store(false, std::memory_order_release);
        return;
    }
    worker_.join();
    running_.store(false, std::memory_order_release);
    LOG_INFO("Model discovery service stopped");
}

std::uint64_t HLIEModelDiscoveryService::request_scan(
    const bool explicit_request) {
    const auto current_options = options();
    if (!explicit_request && !current_options.scan_on_request) {
        return 0;
    }

    std::uint64_t generation = 0;
    {
        std::lock_guard state_lock(state_mutex_);
        if (!running_.load(std::memory_order_acquire) || stop_requested_) {
            return 0;
        }
        if (request_pending_) {
            return requested_generation_;
        }
        const auto now = std::chrono::steady_clock::now();
        if (current_options.request_cooldown.count() > 0 &&
            last_request_.time_since_epoch().count() > 0 &&
            now - last_request_ < current_options.request_cooldown) {
            return 0;
        }
        generation = ++requested_generation_;
        last_request_ = now;
        request_pending_ = true;
    }
    wake_cv_.notify_all();
    return generation;
}

bool HLIEModelDiscoveryService::wait_for(
    const std::uint64_t generation,
    const std::optional<std::chrono::milliseconds> timeout) {
    if (generation == 0) {
        return false;
    }
    const auto wait_timeout = timeout.value_or(options().request_wait_timeout);
    if (wait_timeout.count() <= 0) {
        return false;
    }

    std::unique_lock state_lock(state_mutex_);
    completion_cv_.wait_for(state_lock, wait_timeout, [this, generation] {
        return stop_requested_ || completed_generation_ >= generation;
    });
    const bool completed = completed_generation_ >= generation;
    state_lock.unlock();
    if (!completed) {
        LOG_WARN("Timed out waiting for model discovery: generation={}, timeout_ms={}",
                 generation, wait_timeout.count());
    }
    return completed;
}

HLIEModelDiscoveryScanResult HLIEModelDiscoveryService::scan_once() {
    std::lock_guard scan_lock(scan_mutex_);
    scan_attempts_.fetch_add(1, std::memory_order_relaxed);
    try {
        const auto current_options = options();
        const auto observed = snapshot_provider_();
        if (!observed) {
            scan_failures_.fetch_add(1, std::memory_order_relaxed);
            return {};
        }

        const auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard state_lock(state_mutex_);
            if (!baseline_initialized_) {
                observed_snapshot_ = *observed;
                baseline_initialized_ = true;
                stability_pending_ = false;
                return {true, true, false, 0};
            }
            if (*observed != observed_snapshot_) {
                observed_snapshot_ = *observed;
                stability_pending_ = true;
                stability_pending_since_ = now;
                LOG_INFO("Model directory change detected; waiting for stable files: directory={}, stability_ms={}",
                         model_directory_,
                         current_options.stability_window.count());
                if (current_options.stability_window.count() > 0) {
                    return {true, false, false, 0};
                }
            } else if (!stability_pending_) {
                return {true, true, false, 0};
            } else if (now - stability_pending_since_ <
                       current_options.stability_window) {
                return {true, false, false, 0};
            }
        }

        parse_attempts_.fetch_add(1, std::memory_order_relaxed);
        auto prepared = prepare_catalog_update_();
        if (!prepared || !*prepared) {
            scan_failures_.fetch_add(1, std::memory_order_relaxed);
            return {};
        }

        const auto verified = snapshot_provider_();
        if (!verified || *verified != *observed) {
            std::lock_guard state_lock(state_mutex_);
            if (verified) {
                observed_snapshot_ = *verified;
            }
            stability_pending_ = true;
            stability_pending_since_ = std::chrono::steady_clock::now();
            LOG_INFO("Model directory changed during discovery; deferring catalog publication: directory={}",
                     model_directory_);
            return {verified.has_value(), false, true, 0};
        }

        const int published = std::max(0, (*prepared)());
        {
            std::lock_guard state_lock(state_mutex_);
            observed_snapshot_ = *verified;
            stability_pending_ = false;
        }
        published_models_.fetch_add(
            static_cast<std::uint64_t>(published),
            std::memory_order_relaxed);
        if (published > 0) {
            LOG_INFO("Model discovery published {} new model(s) from {}",
                     published, model_directory_);
        } else {
            LOG_DEBUG("Model discovery completed without new catalog entries: directory={}",
                      model_directory_);
        }
        return {true, true, true, published};
    } catch (const std::exception& ex) {
        scan_failures_.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR("Model discovery failed: {}", ex.what());
    } catch (...) {
        scan_failures_.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR("Model discovery failed with an unknown exception");
    }
    return {};
}

HLIEModelDiscoveryMetrics HLIEModelDiscoveryService::metrics() const {
    std::lock_guard state_lock(state_mutex_);
    return {
        scan_attempts_.load(std::memory_order_relaxed),
        parse_attempts_.load(std::memory_order_relaxed),
        scan_failures_.load(std::memory_order_relaxed),
        published_models_.load(std::memory_order_relaxed),
        requested_generation_,
        completed_generation_,
        running_.load(std::memory_order_acquire),
        stability_pending_,
    };
}

HLIEModelDiscoveryOptions HLIEModelDiscoveryService::options() const {
    return normalize_options(options_provider_());
}

bool HLIEModelDiscoveryService::pending_stability() const {
    std::lock_guard state_lock(state_mutex_);
    return stability_pending_;
}

void HLIEModelDiscoveryService::complete_requested_generation() {
    {
        std::lock_guard state_lock(state_mutex_);
        completed_generation_ = std::max(
            completed_generation_, requested_generation_);
        request_pending_ = false;
    }
    completion_cv_.notify_all();
}

void HLIEModelDiscoveryService::run_loop() {
    while (true) {
        HLIEModelDiscoveryOptions current_options;
        try {
            current_options = options();
        } catch (const std::exception& ex) {
            scan_failures_.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR("Model discovery options refresh failed: {}", ex.what());
            current_options = HLIEModelDiscoveryOptions{};
        }

        auto wait_duration = current_options.scan_interval.count() > 0
            ? current_options.scan_interval
            : std::chrono::duration_cast<std::chrono::milliseconds>(
                  maximum_discovery_duration);
        bool was_pending = false;
        {
            std::lock_guard state_lock(state_mutex_);
            was_pending = stability_pending_;
            if (stability_pending_) {
                const auto elapsed = std::chrono::steady_clock::now() -
                    stability_pending_since_;
                const auto remaining = current_options.stability_window -
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        elapsed);
                wait_duration = std::min(
                    wait_duration,
                    std::max(remaining, std::chrono::milliseconds(10)));
            }
        }

        std::unique_lock state_lock(state_mutex_);
        wake_cv_.wait_for(state_lock, wait_duration, [this] {
            return stop_requested_ || request_pending_;
        });
        if (stop_requested_) {
            break;
        }
        const bool requested = request_pending_;
        request_pending_ = false;
        state_lock.unlock();

        if (current_options.scan_interval.count() > 0 || requested ||
            was_pending || pending_stability()) {
            const auto result = scan_once();
            if (result.successful && result.stable) {
                complete_requested_generation();
            }
        }
    }
    running_.store(false, std::memory_order_release);
}

HLIE_NAMESPACE_END

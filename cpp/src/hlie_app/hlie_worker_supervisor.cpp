#include "hlie_app/runtime/hlie_worker_supervisor.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "hlie_app/hlie_process_manager.h"
#include "hlie_app/runtime/hlie_worker_port_allocator.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_singleton.h"

HLIE_NAMESPACE_BEGIN

namespace {
    using json = nlohmann::json;

    bool valid_process_id(const ProcessID process_id) {
        return process_id != 0 && process_id != static_cast<ProcessID>(-1);
    }

    HLIEWorkerSupervisorOptions clamp_options(HLIEWorkerSupervisorOptions options) {
        options.master_port = std::clamp(options.master_port, 1, 65535);
        options.port_probe_attempts = std::clamp(options.port_probe_attempts, 1, 4096);
        options.poll_interval = std::clamp(
            options.poll_interval, std::chrono::milliseconds(10),
            std::chrono::milliseconds(10000));
        options.ready_timeout = std::clamp(
            options.ready_timeout, std::chrono::seconds(1),
            std::chrono::seconds(3600));
        options.graceful_shutdown_timeout = std::clamp(
            options.graceful_shutdown_timeout,
            std::chrono::milliseconds(100), std::chrono::milliseconds(60000));
        return options;
    }
}

class HLIEWorkerSupervisor::Impl {
public:
    struct RuntimeRecord {
        HLIEWorkerRuntime runtime;
        std::string launch_manifest_path;
        std::chrono::steady_clock::time_point started_at;
        bool ready_observed = false;
    };

    Impl(HLIEProcessManager& process_manager,
         HLIEWorkerSupervisorOptions options,
         IHLIERuntimeEventSink& event_sink)
        : process_manager_(process_manager),
          options_(clamp_options(options)),
          event_sink_(event_sink) {}

    ~Impl() {
        stop();
        std::vector<std::string> manifests;
        {
            std::lock_guard lock(state_mutex_);
            manifests.reserve(runtimes_.size());
            for (const auto& [runtime_id, record] : runtimes_) {
                (void)runtime_id;
                manifests.push_back(record.launch_manifest_path);
            }
            runtimes_.clear();
            leases_.clear();
        }
        for (const auto& manifest : manifests) {
            HLIEWorkerSupervisor::discard_launch_manifest(
                manifest, "supervisor_destroyed");
        }
    }

    void start() {
        std::lock_guard lock(thread_mutex_);
        if (thread_) {
            return;
        }
        stop_requested_.store(false);
        thread_ = std::make_unique<std::thread>([this] { loop(); });
        LOG_INFO("Worker supervisor started");
    }

    void stop() {
        stop_requested_.store(true);
        wait_cv_.notify_all();
        std::unique_ptr<std::thread> thread;
        {
            std::lock_guard lock(thread_mutex_);
            thread = std::move(thread_);
        }
        if (thread && thread->joinable()) {
            thread->join();
            LOG_INFO("Worker supervisor stopped");
        }
    }

    [[nodiscard]] bool running() const {
        std::lock_guard lock(thread_mutex_);
        return thread_ != nullptr && !stop_requested_.load();
    }

    [[nodiscard]] std::optional<HLIEWorkerPortLease> reserve_port() {
        std::lock_guard lock(state_mutex_);
        const auto assignment = HLIEWorkerPortAllocator::find_available(
            options_.master_port, next_worker_instance_id_, options_.port_probe_attempts,
            [this](const int port) {
                if (!HLIEWorkerPortAllocator::is_tcp_port_available(port)) {
                    return false;
                }
                return std::none_of(leases_.begin(), leases_.end(), [port](const auto& entry) {
                    return entry.second.port == port;
                });
            });
        if (!assignment) {
            LOG_WARN("No available dynamic worker port: master_port={}, first_instance_id={}, attempts={}",
                     options_.master_port, next_worker_instance_id_,
                     options_.port_probe_attempts);
            return std::nullopt;
        }

        if (assignment->skipped_candidates > 0) {
            port_conflict_count_.fetch_add(
                static_cast<std::uint64_t>(assignment->skipped_candidates));
            LOG_WARN("Skipped occupied worker ports: master_port={}, selected_instance_id={}, selected_port={}, skipped={}",
                     options_.master_port, assignment->instance_id, assignment->port,
                     assignment->skipped_candidates);
        }

        const HLIEWorkerPortLease lease{
            next_lease_id_++, assignment->instance_id, assignment->port};
        leases_.emplace(lease.lease_id, lease);
        next_worker_instance_id_ = assignment->instance_id + 1;
        return lease;
    }

    void release_port_lease(const HLIEWorkerPortLease& lease) {
        if (!lease.valid()) {
            return;
        }
        std::lock_guard lock(state_mutex_);
        const auto it = leases_.find(lease.lease_id);
        if (it != leases_.end() && it->second.instance_id == lease.instance_id &&
            it->second.port == lease.port) {
            leases_.erase(it);
        }
    }

    [[nodiscard]] std::optional<HLIEWorkerRuntime> spawn(
        const HLIEWorkerLaunchRequest& request) {
        if (!request.port_lease.valid() || !request.launch_ticket.valid() ||
            request.executable.empty()) {
            HLIEWorkerSupervisor::discard_launch_manifest(
                request.launch_manifest_path, "invalid_spawn_request");
            release_port_lease(request.port_lease);
            return std::nullopt;
        }

        {
            std::lock_guard lock(state_mutex_);
            const auto lease = leases_.find(request.port_lease.lease_id);
            if (lease == leases_.end() ||
                lease->second.instance_id != request.port_lease.instance_id ||
                lease->second.port != request.port_lease.port) {
                LOG_ERROR("Worker spawn rejected due to stale port lease: instance_id={}, port={}, lease_id={}",
                          request.port_lease.instance_id, request.port_lease.port,
                          request.port_lease.lease_id);
                HLIEWorkerSupervisor::discard_launch_manifest(
                    request.launch_manifest_path, "stale_port_lease");
                return std::nullopt;
            }
        }

        std::lock_guard lifecycle_lock(lifecycle_mutex_);
        const ProcessID process_id = process_manager_.create_process(
            request.executable, request.arguments, request.model_names,
            request.port_lease.instance_id, request.launch_ticket);
        if (!valid_process_id(process_id)) {
            HLIEWorkerSupervisor::discard_launch_manifest(
                request.launch_manifest_path, "worker_spawn_failed");
            release_port_lease(request.port_lease);
            return std::nullopt;
        }

        HLIEWorkerRuntime runtime{
            static_cast<std::uint64_t>(process_id),
            request.port_lease.instance_id,
            request.port_lease.port,
            request.model_names,
            request.launch_ticket,
        };
        {
            std::lock_guard lock(state_mutex_);
            runtimes_.insert_or_assign(runtime.runtime_id, RuntimeRecord{
                runtime, request.launch_manifest_path,
                std::chrono::steady_clock::now(), false});
        }
        LOG_INFO("Started supervised Worker: pid={}, instance_id={}, port={}, model={}, generation={}",
                 runtime.runtime_id, runtime.instance_id, runtime.port,
                 json(runtime.launch_ticket.model_key).dump(),
                 runtime.launch_ticket.generation);
        return runtime;
    }

    [[nodiscard]] std::optional<HLIEWorkerRuntime> find_runtime(
        const std::uint64_t runtime_id) const {
        std::lock_guard lock(state_mutex_);
        const auto it = runtimes_.find(runtime_id);
        return it == runtimes_.end()
            ? std::nullopt
            : std::optional<HLIEWorkerRuntime>{it->second.runtime};
    }

    [[nodiscard]] std::vector<HLIEWorkerRuntime> running_runtimes() const {
        std::vector<HLIEWorkerRuntime> result;
        std::lock_guard lock(state_mutex_);
        result.reserve(runtimes_.size());
        for (const auto& [runtime_id, record] : runtimes_) {
            (void)runtime_id;
            result.push_back(record.runtime);
        }
        return result;
    }

    [[nodiscard]] HLIEWorkerStopResult request_stop(
        const HLIEWorkerRuntime& runtime, const std::string& reason) {
        return request_stop(runtime, HLIEWorkerStopRequest{
            reason, options_.graceful_shutdown_timeout, false});
    }

    [[nodiscard]] HLIEWorkerStopResult request_stop(
        const HLIEWorkerRuntime& runtime, const HLIEWorkerStopRequest& request) {
        HLIEWorkerStopResult result{runtime, false, false};
        if (!runtime.valid()) {
            return result;
        }

        std::lock_guard lifecycle_lock(lifecycle_mutex_);
        const ProcessID process_id = static_cast<ProcessID>(runtime.runtime_id);
        const auto process = process_manager_.get_process(process_id);
        if (!process) {
            result.confirmed = true;
            cleanup_runtime(runtime.runtime_id, "worker_already_exited");
            return result;
        }
        if (!runtime.launch_ticket.valid()) {
            result.confirmed = process_manager_.terminate_process(process_id) ||
                process_manager_.wait_for_exit(
                    process_id, std::chrono::milliseconds::zero());
            if (result.confirmed) {
                forced_shutdown_count_.fetch_add(1);
                cleanup_runtime(runtime.runtime_id, "legacy_forced_shutdown");
            }
            return result;
        }
        if (!process->launch_ticket.matches(runtime.launch_ticket) ||
            process->instance_id != runtime.instance_id) {
            LOG_WARN("Refused to stop Worker with stale runtime identity: pid={}, instance_id={}, generation={}",
                     runtime.runtime_id, runtime.instance_id,
                     runtime.launch_ticket.generation);
            return result;
        }

        if (!process_manager_.mark_expected_exit(process_id)) {
            result.confirmed = process_manager_.wait_for_exit(
                process_id, std::chrono::milliseconds::zero());
            if (result.confirmed) {
                cleanup_runtime(runtime.runtime_id, "worker_exit_race");
            }
            return result;
        }

        const auto graceful_timeout = std::max(
            std::chrono::milliseconds::zero(), request.graceful_timeout);
        bool accepted = false;
        int response_status = 0;
        std::string response_error = "not_attempted";
        if (!request.force && graceful_timeout > std::chrono::milliseconds::zero()) {
            httplib::Client client("127.0.0.1", runtime.port);
            client.set_connection_timeout(0, 500000);
            client.set_read_timeout(2, 0);
            client.set_write_timeout(2, 0);
            const json request_body{
                {"generation", runtime.launch_ticket.generation},
                {"launch_token", runtime.launch_ticket.token},
                {"reason", request.reason},
            };
            const auto response = client.Post(
                "/internal/worker/shutdown", request_body.dump(), "application/json");
            accepted = response && response->status >= 200 && response->status < 300;
            response_status = response ? response->status : 0;
            response_error = response ? "none" : httplib::to_string(response.error());
        }
        if (accepted && process_manager_.wait_for_exit(process_id, graceful_timeout)) {
            result.confirmed = true;
            result.graceful = true;
            graceful_shutdown_count_.fetch_add(1);
            cleanup_runtime(runtime.runtime_id, "graceful_shutdown");
            LOG_INFO("Worker exited gracefully: pid={}, model={}, generation={}, reason={}",
                     runtime.runtime_id,
                     json(runtime.launch_ticket.model_key).dump(),
                     runtime.launch_ticket.generation, request.reason);
            return result;
        }

        if (request.force || graceful_timeout == std::chrono::milliseconds::zero()) {
            LOG_WARN("Worker forced shutdown requested: pid={}, generation={}, reason={}",
                     runtime.runtime_id, runtime.launch_ticket.generation,
                     request.reason);
        } else if (!accepted) {
            LOG_WARN("Graceful worker shutdown request failed: pid={}, port={}, generation={}, http_status={}, error={}",
                     runtime.runtime_id, runtime.port,
                     runtime.launch_ticket.generation,
                     response_status, response_error);
        } else {
            LOG_WARN("Graceful worker shutdown timed out: pid={}, generation={}, timeout_ms={}",
                     runtime.runtime_id, runtime.launch_ticket.generation,
                     graceful_timeout.count());
        }

        if (process_manager_.terminate_process(process_id)) {
            result.confirmed = true;
            forced_shutdown_count_.fetch_add(1);
            cleanup_runtime(runtime.runtime_id, "forced_shutdown");
            LOG_WARN("Worker force-terminated after graceful shutdown failure: pid={}, generation={}",
                     runtime.runtime_id, runtime.launch_ticket.generation);
            return result;
        }

        result.confirmed = process_manager_.wait_for_exit(
            process_id, std::chrono::milliseconds::zero());
        if (result.confirmed) {
            cleanup_runtime(runtime.runtime_id, "confirmed_after_stop_failure");
        } else {
            (void)process_manager_.mark_expected_exit(process_id, false);
        }
        return result;
    }

    [[nodiscard]] std::vector<HLIEWorkerStopResult> shutdown_all(
        const std::string& reason) {
        const auto runtimes = running_runtimes();
        std::vector<HLIEWorkerStopResult> results;
        results.reserve(runtimes.size());
        for (const auto& runtime : runtimes) {
            results.push_back(request_stop(runtime, reason));
        }
        return results;
    }

    [[nodiscard]] HLIEWorkerSupervisorMetrics metrics() const noexcept {
        return {
            unexpected_exit_count_.load(),
            ready_timeout_count_.load(),
            graceful_shutdown_count_.load(),
            forced_shutdown_count_.load(),
            port_conflict_count_.load(),
        };
    }

private:
    void emit(const HLIERuntimeEvent& event) const {
        try {
            event_sink_.on_runtime_event(event);
        } catch (const std::exception& exception) {
            LOG_ERROR("Worker runtime event sink threw: kind={}, pid={}, generation={}, error={}",
                      static_cast<int>(event.kind), event.runtime_id,
                      event.launch_ticket.generation, exception.what());
        } catch (...) {
            LOG_ERROR("Worker runtime event sink threw an unknown exception: kind={}, pid={}, generation={}",
                      static_cast<int>(event.kind), event.runtime_id,
                      event.launch_ticket.generation);
        }
    }

    void cleanup_runtime(const std::uint64_t runtime_id, const char* reason) {
        std::string manifest;
        {
            std::lock_guard lock(state_mutex_);
            const auto it = runtimes_.find(runtime_id);
            if (it == runtimes_.end()) {
                return;
            }
            manifest = it->second.launch_manifest_path;
            for (auto lease = leases_.begin(); lease != leases_.end(); ++lease) {
                if (lease->second.instance_id == it->second.runtime.instance_id) {
                    leases_.erase(lease);
                    break;
                }
            }
            runtimes_.erase(it);
        }
        HLIEWorkerSupervisor::discard_launch_manifest(manifest, reason);
    }

    void loop() {
        LOG_DEBUG("Worker supervisor configured: poll_interval_ms={}, ready_timeout_sec={}, graceful_shutdown_timeout_ms={}",
                  options_.poll_interval.count(), options_.ready_timeout.count(),
                  options_.graceful_shutdown_timeout.count());
        while (!stop_requested_.load()) {
            std::unique_lock wait_lock(wait_mutex_);
            wait_cv_.wait_for(wait_lock, options_.poll_interval, [this] {
                return stop_requested_.load();
            });
            wait_lock.unlock();
            if (stop_requested_.load()) {
                break;
            }

            drain_process_exits();
            probe_starting_workers();
        }
    }

    void drain_process_exits() {
        std::vector<HLIEProcessManager::ProcessExitEvent> events;
        {
            std::lock_guard lifecycle_lock(lifecycle_mutex_);
            process_manager_.monitor_processes();
            events = process_manager_.drain_exit_events();
        }
        for (const auto& process_event : events) {
            const auto& process = process_event.process;
            HLIEWorkerRuntime runtime{
                static_cast<std::uint64_t>(process.process_id),
                process.instance_id,
                options_.master_port + process.instance_id,
                process.model_names,
                process.launch_ticket,
            };
            if (const auto tracked = find_runtime(runtime.runtime_id)) {
                runtime = *tracked;
            }
            cleanup_runtime(runtime.runtime_id, "process_exit");

            if (process.expected_exit) {
                LOG_DEBUG("Worker completed expected exit: pid={}, instance_id={}, exit_code={}, generation={}",
                          runtime.runtime_id, runtime.instance_id,
                          process_event.exit_code,
                          runtime.launch_ticket.generation);
            } else {
                unexpected_exit_count_.fetch_add(1);
                LOG_ERROR("Worker exited: pid={}, instance_id={}, exit_code={}, models={}, generation={}",
                          runtime.runtime_id, runtime.instance_id,
                          process_event.exit_code, json(runtime.model_names).dump(),
                          runtime.launch_ticket.generation);
            }
            emit(HLIERuntimeEvent{
                HLIERuntimeEventKind::Exited,
                runtime.launch_ticket,
                runtime.runtime_id,
                runtime.instance_id,
                runtime.port,
                process_event.exit_code,
                process.expected_exit,
                true,
            });
        }
    }

    void probe_starting_workers() {
        std::vector<RuntimeRecord> records;
        {
            std::lock_guard lock(state_mutex_);
            records.reserve(runtimes_.size());
            for (const auto& [runtime_id, record] : runtimes_) {
                (void)runtime_id;
                if (!record.ready_observed) {
                    records.push_back(record);
                }
            }
        }

        const auto now = std::chrono::steady_clock::now();
        for (const auto& record : records) {
            if (stop_requested_.load()) {
                return;
            }
            const auto& runtime = record.runtime;
            httplib::Client ready_client("127.0.0.1", runtime.port);
            ready_client.set_connection_timeout(0, 100000);
            ready_client.set_read_timeout(0, 200000);
            ready_client.set_write_timeout(0, 200000);
            if (const auto ready = ready_client.Get("/ready");
                ready && ready->status >= 200 && ready->status < 300) {
                bool current = false;
                {
                    std::lock_guard lock(state_mutex_);
                    const auto it = runtimes_.find(runtime.runtime_id);
                    if (it != runtimes_.end() &&
                        it->second.runtime.launch_ticket.matches(runtime.launch_ticket) &&
                        !it->second.ready_observed) {
                        it->second.ready_observed = true;
                        it->second.runtime.ready = true;
                        current = true;
                    }
                }
                if (current) {
                    LOG_INFO("Worker readiness confirmed by supervisor: pid={}, model={}, generation={}, port={}",
                             runtime.runtime_id,
                             json(runtime.launch_ticket.model_key).dump(),
                             runtime.launch_ticket.generation, runtime.port);
                    emit(HLIERuntimeEvent{
                        HLIERuntimeEventKind::Ready,
                        runtime.launch_ticket,
                        runtime.runtime_id,
                        runtime.instance_id,
                        runtime.port,
                        0,
                        false,
                        false,
                    });
                }
                continue;
            }

            if (now - record.started_at < options_.ready_timeout) {
                continue;
            }

            bool current = false;
            {
                std::lock_guard lock(state_mutex_);
                const auto it = runtimes_.find(runtime.runtime_id);
                current = it != runtimes_.end() &&
                    it->second.runtime.launch_ticket.matches(runtime.launch_ticket) &&
                    !it->second.ready_observed;
            }
            if (!current) {
                continue;
            }

            LOG_ERROR("Worker READY timeout: pid={}, instance_id={}, timeout_sec={}, model={}, generation={}",
                      runtime.runtime_id, runtime.instance_id,
                      options_.ready_timeout.count(),
                      json(runtime.launch_ticket.model_key).dump(),
                      runtime.launch_ticket.generation);
            ready_timeout_count_.fetch_add(1);
            bool terminated = false;
            {
                std::lock_guard lifecycle_lock(lifecycle_mutex_);
                terminated = process_manager_.terminate_process(
                    static_cast<ProcessID>(runtime.runtime_id)) ||
                    process_manager_.wait_for_exit(
                        static_cast<ProcessID>(runtime.runtime_id),
                        std::chrono::milliseconds::zero());
            }
            if (terminated) {
                cleanup_runtime(runtime.runtime_id, "ready_timeout");
            } else {
                LOG_ERROR("Worker READY timeout termination was not confirmed: pid={}, generation={}",
                          runtime.runtime_id, runtime.launch_ticket.generation);
            }
            emit(HLIERuntimeEvent{
                HLIERuntimeEventKind::ReadyTimeout,
                runtime.launch_ticket,
                runtime.runtime_id,
                runtime.instance_id,
                runtime.port,
                -2,
                false,
                terminated,
            });
        }
    }

    HLIEProcessManager& process_manager_;
    const HLIEWorkerSupervisorOptions options_;
    IHLIERuntimeEventSink& event_sink_;

    mutable std::mutex state_mutex_;
    std::unordered_map<std::uint64_t, RuntimeRecord> runtimes_;
    std::unordered_map<std::uint64_t, HLIEWorkerPortLease> leases_;
    std::uint64_t next_lease_id_ = 1;
    int next_worker_instance_id_ = 1;

    mutable std::mutex thread_mutex_;
    std::unique_ptr<std::thread> thread_;
    std::atomic<bool> stop_requested_{false};
    std::condition_variable wait_cv_;
    std::mutex wait_mutex_;
    std::mutex lifecycle_mutex_;

    std::atomic<std::uint64_t> unexpected_exit_count_{0};
    std::atomic<std::uint64_t> ready_timeout_count_{0};
    std::atomic<std::uint64_t> graceful_shutdown_count_{0};
    std::atomic<std::uint64_t> forced_shutdown_count_{0};
    std::atomic<std::uint64_t> port_conflict_count_{0};
};

HLIEWorkerSupervisor::HLIEWorkerSupervisor(
    HLIEProcessManager& process_manager,
    HLIEWorkerSupervisorOptions options,
    IHLIERuntimeEventSink& event_sink)
    : impl_(std::make_unique<Impl>(process_manager, options, event_sink)) {}

HLIEWorkerSupervisor::~HLIEWorkerSupervisor() = default;

HLIEWorkerSupervisorOptions HLIEWorkerSupervisor::configured_options(
    const int master_port) {
    const auto& config = Singleton<HLIEConfigService>::instance();
    HLIEWorkerSupervisorOptions options;
    options.master_port = master_port;
    options.port_probe_attempts = std::clamp(config->get_int(
        "worker_supervisor", "port_probe_attempts", 128, true), 1, 4096);
    options.poll_interval = std::chrono::milliseconds(std::clamp(config->get_int(
        "worker_supervisor", "poll_interval_ms", 500, true), 100, 10000));
    options.ready_timeout = std::chrono::seconds(std::clamp(config->get_int(
        "worker_supervisor", "ready_timeout_sec", 120, true), 5, 3600));
    options.graceful_shutdown_timeout = std::chrono::milliseconds(std::clamp(config->get_int(
        "worker_supervisor", "graceful_shutdown_timeout_ms", 5000, true), 100, 60000));
    return clamp_options(options);
}

void HLIEWorkerSupervisor::start() {
    impl_->start();
}

void HLIEWorkerSupervisor::stop() {
    impl_->stop();
}

bool HLIEWorkerSupervisor::running() const {
    return impl_->running();
}

std::optional<HLIEWorkerPortLease> HLIEWorkerSupervisor::reserve_port() {
    return impl_->reserve_port();
}

void HLIEWorkerSupervisor::release_port_lease(const HLIEWorkerPortLease& lease) {
    impl_->release_port_lease(lease);
}

std::optional<HLIEWorkerRuntime> HLIEWorkerSupervisor::spawn(
    const HLIEWorkerLaunchRequest& request) {
    return impl_->spawn(request);
}

std::optional<HLIEWorkerRuntime> HLIEWorkerSupervisor::find_runtime(
    const std::uint64_t runtime_id) const {
    return impl_->find_runtime(runtime_id);
}

std::vector<HLIEWorkerRuntime> HLIEWorkerSupervisor::running_runtimes() const {
    return impl_->running_runtimes();
}

HLIEWorkerStopResult HLIEWorkerSupervisor::request_stop(
    const HLIEWorkerRuntime& runtime, const std::string& reason) {
    return impl_->request_stop(runtime, reason);
}

HLIEWorkerStopResult HLIEWorkerSupervisor::request_stop(
    const HLIEWorkerRuntime& runtime, const HLIEWorkerStopRequest& request) {
    return impl_->request_stop(runtime, request);
}

std::vector<HLIEWorkerStopResult> HLIEWorkerSupervisor::shutdown_all(
    const std::string& reason) {
    return impl_->shutdown_all(reason);
}

HLIEWorkerSupervisorMetrics HLIEWorkerSupervisor::metrics() const noexcept {
    return impl_->metrics();
}

void HLIEWorkerSupervisor::discard_launch_manifest(
    const std::string& path, const char* reason) {
    if (path.empty()) {
        return;
    }
    std::error_code error;
    const bool removed = std::filesystem::remove(path, error);
    if (error) {
        LOG_WARN("Failed to remove Worker launch manifest: path={}, reason={}, error={}",
                 path, reason, error.message());
    } else if (removed) {
        LOG_DEBUG("Removed Worker launch manifest: path={}, reason={}", path, reason);
    }
}

HLIE_NAMESPACE_END

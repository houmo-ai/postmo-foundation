#include <catch_amalgamated.hpp>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "hlie_app/hlie_process_manager.h"
#include "hlie_app/runtime/hlie_process_model_runtime.h"
#include "hlie_app/runtime/hlie_worker_supervisor.h"

using namespace hm_HLIE;
using namespace std::chrono_literals;

namespace {

using json = nlohmann::json;

std::string fake_worker_executable() {
    const char* configured = std::getenv("HLIE_FAKE_WORKER_EXECUTABLE");
    std::string executable = configured == nullptr ? std::string{} : std::string{configured};
#ifdef HLIE_FAKE_WORKER_EXECUTABLE_DEFAULT
    if (executable.empty()) {
        executable = HLIE_FAKE_WORKER_EXECUTABLE_DEFAULT;
    }
#endif
    if (executable.empty() || !std::filesystem::is_regular_file(executable)) {
        throw std::runtime_error("HLIE fake worker executable is unavailable: " + executable);
    }
    return executable;
}

HLIEWorkerSupervisorOptions test_options() {
    static std::atomic<int> next_master_port{24000};
    HLIEWorkerSupervisorOptions options;
    options.master_port = next_master_port.fetch_add(256);
    options.port_probe_attempts = 192;
    options.poll_interval = 25ms;
    options.ready_timeout = 2s;
    options.graceful_shutdown_timeout = 1s;
    return options;
}

class RecordingRuntimeEventSink final : public IHLIERuntimeEventSink {
public:
    void on_runtime_event(const HLIERuntimeEvent& event) override {
        {
            std::lock_guard lock(mutex_);
            events_.push_back(event);
        }
        cv_.notify_all();
    }

    std::optional<HLIERuntimeEvent> wait_for(
        const HLIERuntimeEventKind kind,
        const HLIEModelLaunchTicket& ticket,
        const std::chrono::milliseconds timeout = 5s) {
        std::unique_lock lock(mutex_);
        const auto matches = [&] {
            return std::find_if(events_.begin(), events_.end(), [&](const auto& event) {
                return event.kind == kind && event.launch_ticket.matches(ticket);
            });
        };
        if (!cv_.wait_for(lock, timeout, [&] { return matches() != events_.end(); })) {
            return std::nullopt;
        }
        return *matches();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<HLIERuntimeEvent> events_;
};

struct TemporaryManifest {
    std::filesystem::path path;

    TemporaryManifest() {
        path = std::filesystem::temp_directory_path() /
            ("hlie-worker-manifest-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) + ".hlif");
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output << "{}";
    }

    ~TemporaryManifest() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

std::optional<HLIEWorkerRuntime> spawn_fake_worker(
    HLIEWorkerSupervisor& supervisor,
    const HLIEModelLaunchTicket& ticket,
    std::vector<std::string> arguments,
    const std::string& manifest_path = {}) {
    const auto lease = supervisor.reserve_port();
    if (!lease) {
        return std::nullopt;
    }
    if (std::find(arguments.begin(), arguments.end(), "--port") == arguments.end()) {
        arguments.insert(arguments.end(), {"--port", std::to_string(lease->port)});
    }
    arguments.insert(arguments.end(), {
        "--generation", std::to_string(ticket.generation),
        "--launch-token", ticket.token,
    });
    if (!manifest_path.empty()) {
        arguments.insert(arguments.end(), {"--launch_manifest", manifest_path});
    }
    return supervisor.spawn(HLIEWorkerLaunchRequest{
        *lease,
        fake_worker_executable(),
        std::move(arguments),
        {ticket.model_key},
        manifest_path,
        ticket,
    });
}

} // namespace

TEST_CASE("WorkerSupervisor completes ready infer drain and expected exit",
          "[component][fake-worker][worker-supervisor][graceful-shutdown]") {
    HLIEProcessManager manager;
    RecordingRuntimeEventSink sink;
    HLIEWorkerSupervisor supervisor(manager, test_options(), sink);
    supervisor.start();

    const HLIEModelLaunchTicket ticket{"model-graceful", 11, "graceful-token"};
    const auto runtime = spawn_fake_worker(supervisor, ticket, {"--mode", "serve"});
    REQUIRE(runtime);
    const auto ready = sink.wait_for(HLIERuntimeEventKind::Ready, ticket);
    REQUIRE(ready);
    CHECK(ready->runtime_id == runtime->runtime_id);
    CHECK(ready->port == runtime->port);

    httplib::Client client("127.0.0.1", runtime->port);
    client.set_connection_timeout(1, 0);
    client.set_read_timeout(2, 0);
    const auto inference = client.Post("/v1/fake/infer", R"({"prompt":"hello"})",
                                       "application/json");
    REQUIRE(inference);
    CHECK(inference->status == 200);
    CHECK(inference->body.find("fake.inference") != std::string::npos);

    const auto rejected = client.Post(
        "/internal/worker/shutdown",
        json{{"generation", ticket.generation}, {"launch_token", "wrong-token"}}.dump(),
        "application/json");
    REQUIRE(rejected);
    CHECK(rejected->status == 403);

    auto stale_runtime = *runtime;
    stale_runtime.launch_ticket.generation += 1;
    stale_runtime.launch_ticket.token = "stale-token";
    CHECK_FALSE(supervisor.request_stop(stale_runtime, "stale_stop").confirmed);
    CHECK(manager.get_process(static_cast<ProcessID>(runtime->runtime_id)).has_value());

    const auto stopped = supervisor.request_stop(*runtime, "component_test");
    REQUIRE(stopped.confirmed);
    CHECK(stopped.graceful);
    const auto exited = sink.wait_for(HLIERuntimeEventKind::Exited, ticket);
    REQUIRE(exited);
    CHECK(exited->expected_exit);
    CHECK(exited->exit_confirmed);
    CHECK_FALSE(supervisor.find_runtime(runtime->runtime_id));
    CHECK(supervisor.metrics().graceful_shutdowns == 1);
}

TEST_CASE("WorkerSupervisor reports crash before ready and owns manifest cleanup",
          "[component][fake-worker][worker-supervisor][crash-before-ready]") {
    HLIEProcessManager manager;
    RecordingRuntimeEventSink sink;
    HLIEWorkerSupervisor supervisor(manager, test_options(), sink);
    supervisor.start();

    TemporaryManifest manifest;
    const HLIEModelLaunchTicket ticket{"model-startup-crash", 12, "startup-token"};
    const auto runtime = spawn_fake_worker(
        supervisor, ticket,
        {"--mode", "crash-before-ready", "--crash-after-ms", "250", "--exit-code", "23"},
        manifest.path.string());
    REQUIRE(runtime);

    const auto exited = sink.wait_for(HLIERuntimeEventKind::Exited, ticket);
    REQUIRE(exited);
    CHECK(exited->exit_code == 23);
    CHECK_FALSE(exited->expected_exit);
    CHECK(exited->exit_confirmed);
    CHECK_FALSE(std::filesystem::exists(manifest.path));
    CHECK(supervisor.metrics().unexpected_exits == 1);
}

TEST_CASE("WorkerSupervisor reports unexpected crash after ready",
          "[component][fake-worker][worker-supervisor][crash-after-ready]") {
    HLIEProcessManager manager;
    RecordingRuntimeEventSink sink;
    HLIEWorkerSupervisor supervisor(manager, test_options(), sink);
    supervisor.start();

    const HLIEModelLaunchTicket ticket{"model-runtime-crash", 13, "runtime-token"};
    const auto runtime = spawn_fake_worker(
        supervisor, ticket,
        {"--mode", "serve", "--crash-after-ms", "750", "--exit-code", "37"});
    REQUIRE(runtime);
    REQUIRE(sink.wait_for(HLIERuntimeEventKind::Ready, ticket));

    const auto exited = sink.wait_for(HLIERuntimeEventKind::Exited, ticket);
    REQUIRE(exited);
    CHECK(exited->exit_code == 37);
    CHECK_FALSE(exited->expected_exit);
    CHECK(supervisor.metrics().unexpected_exits == 1);
}

TEST_CASE("WorkerSupervisor force-terminates a Worker that misses shutdown deadline",
          "[component][fake-worker][worker-supervisor][forced-shutdown]") {
    HLIEProcessManager manager;
    RecordingRuntimeEventSink sink;
    auto options = test_options();
    options.graceful_shutdown_timeout = 100ms;
    HLIEWorkerSupervisor supervisor(manager, options, sink);
    supervisor.start();

    const HLIEModelLaunchTicket ticket{"model-hung", 14, "hung-token"};
    const auto runtime = spawn_fake_worker(
        supervisor, ticket,
        {"--mode", "serve", "--shutdown-grace-ms", "5000"});
    REQUIRE(runtime);
    REQUIRE(sink.wait_for(HLIERuntimeEventKind::Ready, ticket));

    const auto stopped = supervisor.request_stop(*runtime, "forced_component_test");
    REQUIRE(stopped.confirmed);
    CHECK_FALSE(stopped.graceful);
    CHECK_FALSE(manager.get_process(static_cast<ProcessID>(runtime->runtime_id)));
    CHECK(supervisor.metrics().forced_shutdowns == 1);
}

TEST_CASE("WorkerSupervisor confirms cleanup before emitting READY timeout",
          "[component][fake-worker][worker-supervisor][ready-timeout]") {
    HLIEProcessManager manager;
    RecordingRuntimeEventSink sink;
    auto options = test_options();
    options.ready_timeout = 1s;
    HLIEWorkerSupervisor supervisor(manager, options, sink);
    supervisor.start();

    const HLIEModelLaunchTicket ticket{"model-ready-timeout", 15, "timeout-token"};
    const auto runtime = spawn_fake_worker(
        supervisor, ticket,
        {"--mode", "crash-before-ready", "--crash-after-ms", "10000", "--exit-code", "41"});
    REQUIRE(runtime);

    const auto timeout = sink.wait_for(HLIERuntimeEventKind::ReadyTimeout, ticket, 5s);
    REQUIRE(timeout);
    CHECK(timeout->exit_confirmed);
    CHECK(timeout->exit_code == -2);
    CHECK_FALSE(manager.get_process(static_cast<ProcessID>(runtime->runtime_id)));
    CHECK(supervisor.metrics().ready_timeouts == 1);
}

TEST_CASE("ProcessModelRuntime projects ready drain and confirmed stop",
          "[component][fake-worker][process-runtime]") {
    HLIEProcessManager manager;
    RecordingRuntimeEventSink sink;
    HLIEWorkerSupervisor supervisor(manager, test_options(), sink);
    supervisor.start();

    const HLIEModelLaunchTicket ticket{"runtime-adapter", 16, "adapter-token"};
    const auto worker = spawn_fake_worker(supervisor, ticket, {"--mode", "serve"});
    REQUIRE(worker);
    REQUIRE(sink.wait_for(HLIERuntimeEventKind::Ready, ticket));

    HLIEModelCapabilities capabilities;
    capabilities.text = true;
    capabilities.audio = true;
    HLIEProcessModelRuntime runtime(supervisor, *worker, capabilities);
    REQUIRE(runtime.identity().valid());
    CHECK(runtime.identity().kind == HLIERuntimeKind::Process);
    CHECK(runtime.capabilities().audio);
    CHECK(runtime.health().state == HLIERuntimeHealthState::Ready);

    runtime.begin_draining();
    CHECK(runtime.health().state == HLIERuntimeHealthState::Draining);
    HLIERuntimeStopRequest stop_request;
    stop_request.deadline = std::chrono::steady_clock::now() + 2s;
    stop_request.reason = HLIERuntimeStopReason::ManualUnload;
    const auto stopped = runtime.stop(stop_request);
    REQUIRE(stopped.confirmed);
    CHECK(stopped.graceful);
    CHECK(runtime.health().state == HLIERuntimeHealthState::Stopped);

    const auto repeated = runtime.stop(stop_request);
    CHECK(repeated.confirmed);
    CHECK(repeated.already_stopped);
}

#include <catch_amalgamated.hpp>

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "hlie_app/hlie_process_manager.h"

using namespace hm_HLIE;
using namespace std::chrono_literals;

namespace {

struct ChildCommand {
    std::string executable;
    std::vector<std::string> arguments;
};

ChildCommand short_lived_child() {
#ifdef _WIN32
    const char* system_root = std::getenv("SystemRoot");
    const std::string executable = std::string(system_root ? system_root : "C:\\Windows") +
                                   "\\System32\\PING.EXE";
    return {executable, {"127.0.0.1", "-n", "2", "-w", "100"}};
#else
    return {"/bin/sh", {"-c", "sleep 0.3"}};
#endif
}

ChildCommand long_lived_child() {
#ifdef _WIN32
    const char* system_root = std::getenv("SystemRoot");
    const std::string executable = std::string(system_root ? system_root : "C:\\Windows") +
                                   "\\System32\\PING.EXE";
    return {executable, {"127.0.0.1", "-n", "30", "-w", "1000"}};
#else
    // Execute sleep directly so terminating the tracked PID cannot leave a
    // shell-spawned grandchild holding the test runner's output descriptors.
    return {"/bin/sleep", {"30"}};
#endif
}

} // namespace

TEST_CASE("Process manager preserves expected-exit classification",
          "[process-manager][worker-supervisor][graceful-shutdown]") {
    HLIEProcessManager manager;
    const auto child = short_lived_child();
    const HLIEModelLaunchTicket ticket{"model-a", 7, "test-token"};
    const auto pid = manager.create_process(child.executable, child.arguments,
                                            {"model-a"}, 1, ticket);
    REQUIRE(pid != 0);
    REQUIRE(pid != static_cast<ProcessID>(-1));

    const auto running = manager.get_process(pid);
    REQUIRE(running);
    CHECK_FALSE(running->expected_exit);
    REQUIRE(manager.mark_expected_exit(pid));
    REQUIRE(manager.wait_for_exit(pid, 5s));
    CHECK_FALSE(manager.get_process(pid));

    const auto events = manager.drain_exit_events();
    REQUIRE(events.size() == 1);
    CHECK(events.front().process.expected_exit);
    CHECK(events.front().process.launch_ticket.matches(ticket));
}

TEST_CASE("Process manager forced termination removes the process deterministically",
          "[process-manager][worker-supervisor][forced-shutdown]") {
    HLIEProcessManager manager;
    const auto child = long_lived_child();
    const auto pid = manager.create_process(child.executable, child.arguments,
                                            {"model-a"}, 1);
    REQUIRE(pid != 0);
    REQUIRE(pid != static_cast<ProcessID>(-1));

    // Give the spawned process a brief opportunity to enter its steady state
    // before exercising forced termination.
    std::this_thread::sleep_for(50ms);

    REQUIRE(manager.terminate_process(pid));
    CHECK(manager.wait_for_exit(pid, 0ms));
    CHECK_FALSE(manager.get_process(pid));
}

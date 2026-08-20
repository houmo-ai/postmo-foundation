#include "hlie_core/runtime/hlie_model_runtime_registry.h"

#include <catch_amalgamated.hpp>

#include <string>
#include <utility>

namespace {

using namespace hm_HLIE;

HLIEModelRuntimeRegistry::Registration registration_for(
    const std::string& key, const std::string& model_name = {}) {
    HLIEModelRuntimeRegistry::Registration registration;
    registration.snapshot.identity.model_key = key;
    registration.snapshot.identity.model_name = model_name;
    registration.snapshot.identity.display_name = "alias-" + key;
    registration.snapshot.identity.backend_id = "test";
    registration.snapshot.estimated_mem_mb = 2048;
    return registration;
}

} // namespace

TEST_CASE("Core runtime registry owns lifecycle facts and generation fencing",
          "[model-runtime-registry][p1][core]") {
    std::uint64_t token_sequence = 0;
    HLIEModelRuntimeRegistry registry(
        [&token_sequence] { return "token-" + std::to_string(++token_sequence); });
    REQUIRE(registry.register_if_absent(registration_for("model-a", "shared")));

    const auto first = registry.try_begin_loading("alias-model-a");
    REQUIRE(first);
    CHECK(first->generation == 1);
    REQUIRE(registry.mark_ready_if_current(*first, 7902));

    auto snapshot = registry.get_snapshot("shared");
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Ready);
    CHECK(snapshot->port == 7902);
    CHECK(snapshot->estimated_mem_mb == 2048);

    const auto stale_request = registry.begin_request("model-a");
    REQUIRE(stale_request);
    REQUIRE(registry.mark_failed_if_current(*first, 37));

    const auto second = registry.try_begin_loading("model-a");
    REQUIRE(second);
    REQUIRE(second->generation == first->generation + 1);
    REQUIRE(registry.mark_ready_if_current(*second, 7903));
    const auto current_request = registry.begin_request("model-a");
    REQUIRE(current_request);

    registry.end_request(*stale_request);
    snapshot = registry.get_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->generation == second->generation);
    CHECK(snapshot->active_requests == 1);
    CHECK_FALSE(registry.try_begin_eviction("model-a"));

    registry.end_request(*current_request);
    REQUIRE(registry.try_begin_eviction("model-a"));
    registry.abort_eviction("model-a");
}

TEST_CASE("Core runtime registry resolves aliases without catalog metadata",
          "[model-runtime-registry][p1][core]") {
    std::uint64_t token_sequence = 0;
    HLIEModelRuntimeRegistry registry(
        [&token_sequence] { return "token-" + std::to_string(++token_sequence); });
    registry.upsert(registration_for("model-a", "shared"));
    registry.upsert(registration_for("model-b", "shared"));

    CHECK_FALSE(registry.get_snapshot("shared"));
    const auto ticket = registry.try_begin_loading("model-b");
    REQUIRE(ticket);
    REQUIRE(registry.mark_ready_if_current(*ticket, 7904));

    const auto resolved = registry.get_snapshot("shared");
    REQUIRE(resolved);
    CHECK(resolved->identity.model_key == "model-b");
}

TEST_CASE("Core runtime registry restores ready records with a fenced ticket",
          "[model-runtime-registry][p1][core]") {
    std::uint64_t token_sequence = 0;
    HLIEModelRuntimeRegistry registry(
        [&token_sequence] { return "token-" + std::to_string(++token_sequence); });
    auto ready = registration_for("ready-model");
    ready.snapshot.state = HLIEModelLifecycleState::Ready;
    ready.snapshot.port = 7905;
    registry.upsert(std::move(ready));

    const auto snapshot = registry.get_snapshot("ready-model");
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Ready);
    CHECK(snapshot->generation == 1);
    CHECK(registry.begin_request("ready-model"));

    auto failed = registration_for("failed-model");
    failed.snapshot.state = HLIEModelLifecycleState::Failed;
    failed.snapshot.failure_code = 9;
    failed.snapshot.generation = 7;
    failed.launch_token = "stale-token";
    registry.upsert(std::move(failed));
    CHECK_FALSE(registry.get_launch_ticket("failed-model"));
}

TEST_CASE("Core runtime registry fails closed when ticket generation is unavailable",
          "[model-runtime-registry][p1][core]") {
    HLIEModelRuntimeRegistry registry([] { return std::string{}; });
    auto ready = registration_for("model-a");
    ready.snapshot.state = HLIEModelLifecycleState::Ready;
    ready.snapshot.port = 7906;
    registry.upsert(std::move(ready));

    const auto snapshot = registry.get_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Failed);
    CHECK(snapshot->failure_code != 0);
    CHECK(snapshot->port == 0);
    CHECK_FALSE(registry.begin_request("model-a"));
}

TEST_CASE("Core runtime registry preserves ownership evidence for orphaned runtime",
          "[model-runtime-registry][p1][controller]") {
    HLIEModelRuntimeRegistry registry([] { return std::string{"token"}; });
    REQUIRE(registry.register_if_absent(registration_for("orphaned-model")));
    const auto ticket = registry.try_begin_loading("orphaned-model");
    REQUIRE(ticket);
    REQUIRE(registry.mark_ready_if_current(*ticket, 7907));

    REQUIRE(registry.mark_orphaned_if_current(*ticket, -91));
    const auto snapshot = registry.get_snapshot("orphaned-model");
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Orphaned);
    CHECK(snapshot->failure_code == -91);
    CHECK(snapshot->port == 7907);
    CHECK_FALSE(snapshot->evictable);
    CHECK(registry.is_current_launch(*ticket));
    CHECK_FALSE(registry.try_begin_loading("orphaned-model"));
    CHECK_FALSE(registry.try_begin_eviction("orphaned-model"));
}

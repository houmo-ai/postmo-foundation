#include "hlie_app/hlie_model_residency_manager.h"

#include <catch_amalgamated.hpp>

namespace {

using namespace hm_HLIE;

HLIEModelRegistry::ModelInstance resident(const std::string& key, const std::time_t access,
                                          const std::time_t loaded, const std::uint64_t hits = 0) {
    HLIEModelInfo info;
    info.build_sd("sd", "", "", "clip", "", "", "");
    HLIEModelRegistry::ModelInstance instance;
    instance.key = key;
    instance.meta.name = "alias-" + key;
    instance.meta.status = SERVER_MODEL_STATUS_LOADED;
    instance.meta.port = 7900;
    instance.model_info = std::move(info);
    instance.last_access_time = access;
    instance.last_load_time = loaded;
    instance.hit_count = hits;
    instance.estimated_mem_mb = 4096;
    return instance;
}

} // namespace

TEST_CASE("Residency selects least recently used model with deterministic tie breaking", "[residency]") {
    HLIEModelResidencyManager manager({});
    auto newer = resident("newer", 200, 100, 3);
    auto older = resident("older", 100, 90, 8);
    auto tied_fewer_hits = resident("tied", 100, 95, 1);

    const auto candidate = manager.select_lru_candidate({newer, older, tied_fewer_hits});
    REQUIRE(candidate);
    CHECK(candidate->key == "tied");
    CHECK(candidate->reason == "lru");
    CHECK(candidate->estimated_mem_mb == 4096);
}

TEST_CASE("Residency protects default pinned active and gated models", "[residency]") {
    HLIEModelResidencyPolicy policy;
    policy.default_model = " DEFAULT ";
    policy.pinned_models = {"alias-pinned"};
    HLIEModelResidencyManager manager(policy);

    auto default_model = resident("default", 10, 10);
    auto pinned = resident("pinned", 20, 20);
    auto active = resident("active", 30, 30);
    active.active_requests = 1;
    auto gated = resident("gated", 40, 40);
    gated.evictable = false;
    auto available = resident("available", 50, 50);

    const auto candidate = manager.select_lru_candidate({default_model, pinned, active, gated, available});
    REQUIRE(candidate);
    CHECK(candidate->key == "available");
    CHECK_FALSE(manager.select_lru_candidate({default_model, pinned, active, gated}, "available"));
}

TEST_CASE("Residency enforces loaded model limit only at capacity", "[residency]") {
    HLIEModelResidencyPolicy policy;
    policy.max_loaded_models = 2;
    HLIEModelResidencyManager manager(policy);

    CHECK_FALSE(manager.select_for_loaded_limit({resident("one", 10, 10)}));
    const auto candidate = manager.select_for_loaded_limit({resident("one", 10, 10), resident("two", 20, 20)});
    REQUIRE(candidate);
    CHECK(candidate->key == "one");
    CHECK(candidate->reason == "max_loaded_models");
}

TEST_CASE("Residency TTL uses last access and honors the exact boundary", "[residency]") {
    HLIEModelResidencyPolicy policy;
    policy.idle_ttl_sec = 60;
    HLIEModelResidencyManager manager(policy);

    auto expired = resident("expired", 40, 10);
    auto boundary = resident("boundary", 40, 10);
    auto fresh = resident("fresh", 41, 10);
    const auto candidate = manager.select_for_idle_ttl({fresh, expired}, 100);
    REQUIRE(candidate);
    CHECK(candidate->key == "expired");
    CHECK(candidate->reason == "idle_ttl");
    CHECK(manager.select_for_idle_ttl({boundary}, 100)->key == "boundary");
    CHECK_FALSE(manager.select_for_idle_ttl({fresh}, 100));
}

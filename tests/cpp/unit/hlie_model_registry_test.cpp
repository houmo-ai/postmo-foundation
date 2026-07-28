#include "hlie_app/hlie_model_registry.h"

#include <catch_amalgamated.hpp>

#include <thread>
#include <vector>

namespace {

using namespace hm_HLIE;

HLIEModelRegistry::ModelInstance make_instance(const std::string& key,
                                                const server_model_status status = SERVER_MODEL_STATUS_UNLOADED) {
    HLIEModelInfo info;
    info.build_sd("sd", "vae", "t5", "clip", "clip-l", "mmdit", "");
    HLIEModelRegistry::ModelInstance instance;
    instance.key = key;
    instance.meta.name = "alias-" + key;
    instance.meta.status = status;
    instance.model_info = std::move(info);
    instance.estimated_mem_mb = 2048;
    return instance;
}

} // namespace

TEST_CASE("Registry upsert creates stable metadata snapshots", "[model-registry]") {
    HLIEModelRegistry registry;
    auto instance = make_instance("model-a");
    registry.upsert(std::move(instance));

    CHECK(registry.contains("model-a"));
    const auto snapshot = registry.get_instance_snapshot("  MODEL-A  ");
    REQUIRE(snapshot);
    CHECK(snapshot->key == "model-a");
    CHECK(snapshot->meta.name == "alias-model-a");
    CHECK(snapshot->model_info.model_key_ == "model-a");
    CHECK(snapshot->meta.loaded_info["service_kind"] == "sd");
    CHECK(snapshot->meta.loaded_info["model_family"] == "sd3");
    CHECK(snapshot->estimated_mem_mb == 2048);
}

TEST_CASE("Registry registration captures memory through the production entry point", "[model-registry][registration]") {
    HLIEModelInfo info;
    info.build_sd("sd", "vae", "t5", "clip", "clip-l", "mmdit", "0");
    HLIEModelRegistry registry;
    registry.register_model(info, 3072);

    const auto snapshot = registry.get_instance_snapshot("sd3");
    REQUIRE(snapshot);
    CHECK(snapshot->key == "sd3");
    CHECK(snapshot->estimated_mem_mb == 3072);
    CHECK(snapshot->meta.loaded_info["service_kind"] == "sd");
}

TEST_CASE("Registry updates resolved llama metadata without changing runtime state", "[model-registry][llm-preset]") {
    HLIEModelInfo info;
    REQUIRE(info.parse_identity_from_gguf_filename("HiModel_xh2_qwen3.5_9b_32k_b1_1chip_v1.2.0.gguf") == 0);
    HLIEModelRegistry registry;
    registry.register_model(info, 8192);
    REQUIRE(registry.mark_loaded(info.model_key_, 7902));

    HLIELlmParamSet params;
    params.set("--ctx-size", int64_t{32768});
    info.set_effective_llama_params(params, {"qwen35-9b"}, {"official"}, params.stable_hash());
    REQUIRE(registry.update_model_info(info.model_key_, info));

    const auto snapshot = registry.get_instance_snapshot(info.model_key_);
    REQUIRE(snapshot);
    CHECK(snapshot->meta.status == SERVER_MODEL_STATUS_LOADED);
    CHECK(snapshot->meta.port == 7902);
    CHECK(snapshot->meta.loaded_info["role"] == "vision_language");
    CHECK(snapshot->meta.loaded_info["llama_presets"] == nlohmann::json::array({"qwen35-9b"}));
    CHECK(snapshot->meta.loaded_info["llama_params_summary"]["ctx_size"] == 32768);
}

TEST_CASE("Discovery registration preserves an existing runtime instance", "[model-registry][discovery]") {
    HLIEModelInfo original;
    original.build_sd("sd", "vae", "t5", "clip", "clip-l", "mmdit", "0");
    HLIEModelRegistry registry;
    REQUIRE(registry.register_model_if_absent(original, 3072));
    REQUIRE(registry.mark_loaded("sd3", 7902));
    REQUIRE(registry.begin_request("sd3"));

    HLIEModelInfo rediscovered;
    rediscovered.build_sd("sd", "new-vae", "new-t5", "new-clip", "new-clip-l", "new-mmdit", "1");
    CHECK_FALSE(registry.register_model_if_absent(rediscovered, 4096));

    const auto snapshot = registry.get_instance_snapshot("sd3");
    REQUIRE(snapshot);
    CHECK(snapshot->meta.status == SERVER_MODEL_STATUS_LOADED);
    CHECK(snapshot->meta.port == 7902);
    CHECK(snapshot->active_requests == 1);
    CHECK(snapshot->estimated_mem_mb == 3072);
    CHECK(snapshot->model_info.get_devices() == "0");
    registry.end_request("sd3");
}

TEST_CASE("Registry rejects ambiguous aliases while preserving exact keys", "[model-registry][routing]") {
    HLIEModelRegistry registry;
    auto first = make_instance("model-a", SERVER_MODEL_STATUS_LOADED);
    auto second = make_instance("model-b", SERVER_MODEL_STATUS_LOADED);
    first.meta.name = "shared-alias";
    second.meta.name = "shared-alias";
    registry.upsert(std::move(first));
    registry.upsert(std::move(second));

    CHECK(registry.get_instance_snapshot("model-a")->key == "model-a");
    CHECK(registry.get_instance_snapshot("model-b")->key == "model-b");
    CHECK_FALSE(registry.get_instance_snapshot("shared-alias"));
    CHECK_FALSE(registry.get_meta("shared-alias"));
}

TEST_CASE("Registry state transitions preserve request invariants", "[model-registry]") {
    HLIEModelRegistry registry;
    registry.upsert(make_instance("model-a"));

    CHECK_FALSE(registry.begin_request("model-a"));
    REQUIRE(registry.mark_loading("model-a"));
    CHECK_FALSE(registry.begin_request("model-a"));
    REQUIRE(registry.mark_loaded("model-a", 7902));
    REQUIRE(registry.begin_request("model-a"));

    auto snapshot = registry.get_instance_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->meta.status == SERVER_MODEL_STATUS_LOADED);
    CHECK(snapshot->meta.port == 7902);
    CHECK(snapshot->active_requests == 1);
    CHECK(snapshot->hit_count == 1);
    CHECK(snapshot->last_load_time > 0);

    registry.end_request("model-a");
    registry.end_request("model-a");
    REQUIRE(registry.mark_unloaded("model-a"));
    snapshot = registry.get_instance_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->active_requests == 0);
    CHECK(snapshot->last_load_time == 0);
    CHECK(snapshot->meta.status == SERVER_MODEL_STATUS_UNLOADED);
}

TEST_CASE("Registry eviction gate excludes active and new requests", "[model-registry]") {
    HLIEModelRegistry registry;
    registry.upsert(make_instance("model-a", SERVER_MODEL_STATUS_LOADED));

    REQUIRE(registry.begin_request("model-a"));
    CHECK_FALSE(registry.try_begin_eviction("model-a"));
    registry.end_request("model-a");

    REQUIRE(registry.try_begin_eviction("model-a"));
    CHECK_FALSE(registry.begin_request("model-a"));
    registry.abort_eviction("model-a");
    CHECK(registry.begin_request("model-a"));
    registry.end_request("model-a");
}

TEST_CASE("Registry request accounting is thread-safe", "[model-registry][concurrency]") {
    HLIEModelRegistry registry;
    registry.upsert(make_instance("model-a", SERVER_MODEL_STATUS_LOADED));

    constexpr int thread_count = 8;
    constexpr int requests_per_thread = 250;
    std::vector<std::thread> workers;
    for (int thread = 0; thread < thread_count; ++thread) {
        workers.emplace_back([&] {
            for (int request = 0; request < requests_per_thread; ++request) {
                if (registry.begin_request("model-a")) {
                    registry.end_request("model-a");
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    const auto snapshot = registry.get_instance_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->active_requests == 0);
    CHECK(snapshot->hit_count == thread_count * requests_per_thread);
}

TEST_CASE("Registry prepares an isolated device snapshot", "[model-registry]") {
    HLIEModelRegistry registry;
    registry.upsert(make_instance("model-a"));
    const auto prepared = registry.prepare_model_info_with_devices("alias-model-a", "2,3");
    REQUIRE(prepared);
    CHECK(prepared->get_devices() == "2,3");
    CHECK(registry.get_instance_snapshot("model-a")->model_info.get_devices() == "2,3");
    CHECK_FALSE(registry.prepare_model_info_with_devices("missing", "0"));
}

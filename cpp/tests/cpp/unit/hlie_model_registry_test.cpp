#include "hlie_app/hlie_model_registry.h"
#include "hlie_base/hlie_util.h"

#include <catch_amalgamated.hpp>

#include <atomic>
#include <filesystem>
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

TEST_CASE("Registry publishes an SDK-free lifecycle snapshot",
          "[model-registry][p1][core-snapshot]") {
    HLIEModelRegistry registry;
    registry.upsert(make_instance("model-a"));

    auto snapshot = registry.get_runtime_snapshot("alias-model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->identity.model_key == "model-a");
    CHECK(snapshot->identity.model_name == "sd3");
    CHECK(snapshot->identity.display_name == "alias-model-a");
    CHECK(snapshot->identity.backend_id == "sd");
    CHECK(snapshot->state == HLIEModelLifecycleState::Unloaded);
    CHECK(snapshot->generation == 0);
    CHECK(snapshot->estimated_mem_mb == 2048);

    const auto ticket = registry.try_begin_loading_ticket("model-a");
    REQUIRE(ticket);
    snapshot = registry.get_runtime_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Starting);
    CHECK(snapshot->generation == ticket->generation);
    CHECK(hlie_model_state_is_loading(snapshot->state));
    CHECK(hlie_model_state_protects_reservation(snapshot->state));

    REQUIRE(registry.mark_loaded_if_current(*ticket, 7902));
    snapshot = registry.get_runtime_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Ready);
    CHECK(snapshot->port == 7902);
    CHECK(hlie_model_state_is_ready(snapshot->state));

    REQUIRE(registry.update_status_if_current(
        *ticket, SERVER_MODEL_STATUS_SLEEPING, 7902));
    snapshot = registry.get_runtime_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Sleeping);
    CHECK(hlie_model_state_protects_reservation(snapshot->state));

    REQUIRE(registry.mark_failed_if_current(*ticket, 17));
    snapshot = registry.get_runtime_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->state == HLIEModelLifecycleState::Failed);
    CHECK(snapshot->failure_code == 17);
    CHECK_FALSE(hlie_model_state_protects_reservation(snapshot->state));

    const auto all = registry.get_all_runtime_snapshots();
    REQUIRE(all.size() == 1);
    CHECK(all.front().identity.model_key == "model-a");
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

TEST_CASE("Registry exposes configured model device ids", "[model-registry][devices]") {
    HLIEModelInfo info;
    REQUIRE(info.parse_identity_from_gguf_filename(
        "HiModel_xh2_qwen3_1b_2k_b1_2chips_v1.0.0.gguf") == 0);
    info.set_devices("0,1");

    HLIEModelRegistry registry;
    registry.register_model(info, 8192);

    const auto snapshot = registry.get_instance_snapshot(info.model_key_);
    REQUIRE(snapshot);
    REQUIRE(snapshot->meta.loaded_info.contains("device_ids"));
    CHECK(snapshot->meta.loaded_info["device_ids"] == nlohmann::json::array({0, 1}));

    REQUIRE(registry.update_runtime_devices(info.model_key_, {2}));
    const auto runtime_snapshot = registry.get_instance_snapshot(info.model_key_);
    REQUIRE(runtime_snapshot);
    CHECK(runtime_snapshot->meta.loaded_info["device_ids"] == nlohmann::json::array({2}));
    CHECK(runtime_snapshot->model_info.get_devices() == "2");
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
    const auto request_ticket = registry.begin_request_ticket("sd3");
    REQUIRE(request_ticket);

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
    registry.end_request(*request_ticket);
}

TEST_CASE("Exact multimodal variants are reused without overwriting runtime state",
          "[model-registry][routing][multimodal]") {
    const std::string model_file =
        "HiModel_xh2_qwen3.5_2b_256_256k_b1_1chip_2cores_v1.3.0_20260429.gguf";

    HLIEModelInfo vit448;
    REQUIRE(vit448.parse_identity_from_gguf_filename(model_file) == 0);
    vit448.mm_proj_path_ =
        "mmproj_xh2_qwen3.5_2b_2cores_vit_448x448_v1.3.0_20260429.gguf";
    vit448.vl_spec_ = HLIEModelInfo::parse_mmproj_vl_spec(vit448.mm_proj_path_);
    vit448.is_multi_model_ = true;
    vit448.model_key_ = HLIEModelInfo::build_model_key(vit448);

    HLIEModelInfo vit896;
    REQUIRE(vit896.parse_identity_from_gguf_filename(model_file) == 0);
    vit896.mm_proj_path_ =
        "mmproj_xh2_qwen3.5_2b_2cores_vit_896x896_v1.3.0_20260429.gguf";
    vit896.vl_spec_ = HLIEModelInfo::parse_mmproj_vl_spec(vit896.mm_proj_path_);
    vit896.is_multi_model_ = true;
    vit896.model_key_ = HLIEModelInfo::build_model_key(vit896);

    REQUIRE(vit448.model_key_ != vit896.model_key_);
    HLIEModelRegistry registry;
    REQUIRE(registry.register_model_if_absent(vit448, 1900));
    REQUIRE(registry.register_model_if_absent(vit896, 2300));

    CHECK_FALSE(registry.get_instance_snapshot(vit448.get_model_name()));
    REQUIRE(registry.mark_loaded(vit448.model_key_, 7901));

    HLIEModelInfo repeated_vit448 = vit448;
    const std::string requested_start_key = HLIEModelInfo::build_model_key(repeated_vit448);
    CHECK(requested_start_key == vit448.model_key_);
    const auto exact_start_target = registry.get_exact_instance_snapshot(repeated_vit448);
    REQUIRE(exact_start_target);
    CHECK(exact_start_target->key == vit448.model_key_);
    CHECK(exact_start_target->path_mmproj.find("vit_448x448") != std::string::npos);
    CHECK_FALSE(registry.register_model_if_absent(repeated_vit448, 9999));

    const auto reused = registry.get_instance_snapshot(vit448.model_key_);
    REQUIRE(reused);
    CHECK(reused->meta.status == SERVER_MODEL_STATUS_LOADED);
    CHECK(reused->meta.port == 7901);
    CHECK(reused->estimated_mem_mb == 1900);
    CHECK(reused->path_mmproj.find("vit_448x448") != std::string::npos);

    const auto other_variant = registry.get_instance_snapshot(vit896.model_key_);
    REQUIRE(other_variant);
    CHECK(other_variant->meta.status == SERVER_MODEL_STATUS_UNLOADED);
    CHECK(other_variant->estimated_mem_mb == 2300);
}

TEST_CASE("Model path comparison normalizes equivalent request paths", "[model-routing][paths]") {
    const auto base_dir = std::filesystem::current_path().string();
    CHECK(HlieUtil::paths_refer_to_same_file(
        "models/main.gguf", "models/./main.gguf", base_dir));
    CHECK(HlieUtil::paths_refer_to_same_file(
        "models/mmproj.gguf", "./models/mmproj.gguf", base_dir));
    CHECK_FALSE(HlieUtil::paths_refer_to_same_file(
        "models/main.gguf", "other/main.gguf", base_dir));
    CHECK(HlieUtil::paths_refer_to_same_file({}, {}, base_dir));
    CHECK_FALSE(HlieUtil::paths_refer_to_same_file({}, "models/main.gguf", base_dir));
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

    CHECK_FALSE(registry.begin_request_ticket("model-a"));
    REQUIRE(registry.mark_loading("model-a"));
    CHECK_FALSE(registry.begin_request_ticket("model-a"));
    REQUIRE(registry.mark_loaded("model-a", 7902));
    const auto request_ticket = registry.begin_request_ticket("model-a");
    REQUIRE(request_ticket);

    auto snapshot = registry.get_instance_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->meta.status == SERVER_MODEL_STATUS_LOADED);
    CHECK(snapshot->meta.port == 7902);
    CHECK(snapshot->active_requests == 1);
    CHECK(snapshot->hit_count == 1);
    CHECK(snapshot->last_load_time > 0);

    registry.end_request(*request_ticket);
    registry.end_request(*request_ticket);
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

    const auto first_request = registry.begin_request_ticket("model-a");
    REQUIRE(first_request);
    CHECK_FALSE(registry.try_begin_eviction("model-a"));
    registry.end_request(*first_request);

    REQUIRE(registry.try_begin_eviction("model-a"));
    CHECK_FALSE(registry.begin_request_ticket("model-a"));
    registry.abort_eviction("model-a");
    const auto second_request = registry.begin_request_ticket("model-a");
    REQUIRE(second_request);
    registry.end_request(*second_request);
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
                if (const auto request_ticket = registry.begin_request_ticket("model-a")) {
                    registry.end_request(*request_ticket);
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

TEST_CASE("Registry grants only one concurrent model loading claim", "[model-registry][concurrency][loading]") {
    HLIEModelRegistry registry;
    registry.upsert(make_instance("model-a"));

    constexpr int thread_count = 16;
    std::atomic<int> successful_claims{0};
    std::vector<std::thread> workers;
    for (int thread = 0; thread < thread_count; ++thread) {
        workers.emplace_back([&] {
            if (registry.try_begin_loading("model-a")) {
                successful_claims.fetch_add(1);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    CHECK(successful_claims.load() == 1);
    const auto snapshot = registry.get_instance_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->meta.status == SERVER_MODEL_STATUS_LOADING);
    CHECK_FALSE(registry.try_begin_loading("model-a"));
    REQUIRE(registry.mark_loaded("model-a", 7901));
    CHECK_FALSE(registry.try_begin_loading("model-a"));
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

TEST_CASE("Registry records subprocess failure without advertising a loaded model", "[model-registry]") {
    HLIEModelRegistry registry;
    registry.upsert(make_instance("model-a"));
    REQUIRE(registry.mark_loading("model-a"));
    REQUIRE(registry.mark_failed("model-a", 17));

    const auto snapshot = registry.get_instance_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->meta.status == SERVER_MODEL_STATUS_UNLOADED);
    CHECK(snapshot->meta.exit_code == 17);
    CHECK(snapshot->meta.is_failed());

    REQUIRE(registry.try_begin_loading("model-a"));
    CHECK(registry.get_instance_snapshot("model-a")->meta.exit_code == 0);
}

TEST_CASE("Registry rejects stale worker callbacks from an older generation",
          "[model-registry][generation][worker-supervisor]") {
    HLIEModelRegistry registry;
    registry.upsert(make_instance("model-a"));

    const auto first = registry.try_begin_loading_ticket("model-a");
    REQUIRE(first);
    REQUIRE(registry.mark_loaded_if_current(*first, 7902));
    REQUIRE(registry.mark_unloaded_if_current(*first));

    const auto second = registry.try_begin_loading_ticket("model-a");
    REQUIRE(second);
    REQUIRE(second->generation > first->generation);
    CHECK_FALSE(second->token == first->token);

    CHECK_FALSE(registry.mark_loaded_if_current(*first, 7902));
    CHECK_FALSE(registry.mark_unloaded_if_current(*first));
    CHECK_FALSE(registry.mark_failed_if_current(*first, 17));

    const auto loading = registry.get_instance_snapshot("model-a");
    REQUIRE(loading);
    CHECK(loading->meta.status == SERVER_MODEL_STATUS_LOADING);
    CHECK(loading->generation == second->generation);

    REQUIRE(registry.mark_loaded_if_current(*second, 7903));
    CHECK_FALSE(registry.mark_failed_if_current(*first, 23));
    const auto loaded = registry.get_instance_snapshot("model-a");
    REQUIRE(loaded);
    CHECK(loaded->meta.status == SERVER_MODEL_STATUS_LOADED);
    CHECK(loaded->meta.port == 7903);
    CHECK(loaded->meta.exit_code == 0);
}

TEST_CASE("Registry launch token is required in addition to generation",
          "[model-registry][generation][launch-ticket]") {
    HLIEModelRegistry registry;
    registry.upsert(make_instance("model-a"));

    const auto ticket = registry.try_begin_loading_ticket("model-a");
    REQUIRE(ticket);
    auto forged = *ticket;
    forged.token = "wrong-token";

    CHECK_FALSE(registry.is_current_launch(forged));
    CHECK_FALSE(registry.mark_loaded_if_current(forged, 7902));
    REQUIRE(registry.is_current_launch(*ticket));
    REQUIRE(registry.mark_loaded_if_current(*ticket, 7902));
}

TEST_CASE("Stale request lease cannot decrement a reloaded generation",
          "[model-registry][generation][request-lease]") {
    HLIEModelRegistry registry;
    registry.upsert(make_instance("model-a"));

    const auto first = registry.try_begin_loading_ticket("model-a");
    REQUIRE(first);
    REQUIRE(registry.mark_loaded_if_current(*first, 7902));
    const auto stale_request = registry.begin_request_ticket("model-a");
    REQUIRE(stale_request);

    REQUIRE(registry.mark_failed_if_current(*first, 37));
    const auto second = registry.try_begin_loading_ticket("model-a");
    REQUIRE(second);
    REQUIRE(registry.mark_loaded_if_current(*second, 7903));
    const auto current_request = registry.begin_request_ticket("model-a");
    REQUIRE(current_request);

    registry.end_request(*stale_request);
    auto snapshot = registry.get_instance_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->generation == second->generation);
    CHECK(snapshot->active_requests == 1);
    CHECK_FALSE(registry.try_begin_eviction("model-a"));

    registry.end_request(*current_request);
    snapshot = registry.get_instance_snapshot("model-a");
    REQUIRE(snapshot);
    CHECK(snapshot->active_requests == 0);
    CHECK(registry.try_begin_eviction("model-a"));
}

#include <catch_amalgamated.hpp>

#include <string>

#include "hlie_app/catalog/hlie_model_catalog.h"
#include "json.hpp"

using namespace hm_HLIE;

namespace {
HLIEModelCatalogEntry catalog_entry() {
    HLIEModelCatalogEntry entry;
    entry.model_info.build_sd(
        "sd", "vae", "t5", "clip", "clip-l", "mmdit", "0");
    entry.model_info.model_key_ =
        HLIEModelInfo::build_model_key(entry.model_info);
    entry.meta.name = "Stable Diffusion";
    entry.meta.status = SERVER_MODEL_STATUS_LOADED;
    entry.meta.port = 7901;
    entry.meta.exit_code = 7;
    entry.spec = entry.model_info.to_model_spec();
    return entry;
}
} // namespace

TEST_CASE("ModelCatalog exposes immutable core specs without runtime state",
          "[p1][model-catalog]") {
    HLIEModelCatalog catalog;
    auto entry = catalog_entry();
    const auto key = entry.model_info.model_key_;

    REQUIRE(catalog.insert_if_absent(entry));
    CHECK_FALSE(catalog.insert_if_absent(entry));
    CHECK(catalog.contains(key));

    const IHLIEModelCatalog& core_port = catalog;
    const auto spec = core_port.resolve("  SD3  ");
    REQUIRE(spec);
    CHECK(spec->identity.model_key == key);
    CHECK(spec->identity.display_name == "Stable Diffusion");
    CHECK(spec->backend_alias == "sd");
    REQUIRE(core_port.snapshots().size() == 1);

    const auto stored = catalog.get(key);
    REQUIRE(stored);
    CHECK(stored->meta.status == SERVER_MODEL_STATUS_UNLOADED);
    CHECK(stored->meta.port == 0);
    CHECK(stored->meta.exit_code == 0);
}

TEST_CASE("ModelCatalog updates static payload without retaining lifecycle fields",
          "[p1][model-catalog]") {
    HLIEModelCatalog catalog;
    auto entry = catalog_entry();
    const auto key = entry.model_info.model_key_;
    catalog.upsert(std::move(entry));

    REQUIRE(catalog.update(key, [](HLIEModelCatalogEntry& stored) {
        stored.meta.loaded_info["device_ids"] = {0, 1};
        stored.meta.status = SERVER_MODEL_STATUS_LOADED;
        stored.meta.port = 7999;
    }));

    const auto stored = catalog.get(key);
    REQUIRE(stored);
    CHECK(stored->meta.loaded_info["device_ids"] == json({0, 1}));
    CHECK(stored->meta.status == SERVER_MODEL_STATUS_UNLOADED);
    CHECK(stored->meta.port == 0);
    CHECK_FALSE(catalog.update("missing", [](HLIEModelCatalogEntry&) {}));
}

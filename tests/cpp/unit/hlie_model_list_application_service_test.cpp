#include <catch_amalgamated.hpp>

#include <string>

#include "hlie_app/models/hlie_model_list_application_service.h"
#include "json.hpp"

namespace {

using namespace hm_HLIE;

server_model_meta meta(const std::string& name,
                       const server_model_status status) {
    server_model_meta result;
    result.name = name;
    result.status = status;
    result.aliases.insert(name + "-alias");
    result.tags.insert("test");
    result.loaded_info = {
        {"service_kind", "llm"},
        {"capabilities", {{"vision", true}}},
    };
    return result;
}

} // namespace

TEST_CASE("Model list application service preserves compatibility DTOs",
          "[p1][model-list][application-service]") {
    auto loaded = meta("chat", SERVER_MODEL_STATUS_LOADED);
    loaded.port = 7901;
    HLIEModelListApplicationService service({
        [loaded] { return std::vector<server_model_meta>{
            meta("chat", SERVER_MODEL_STATUS_UNLOADED), loaded}; },
        [] {
            return std::vector<HLIEModelListLocalInstance>{{"chat", {0, 1}}};
        },
        [] {
            return std::vector<HLIEModelListProcessSnapshot>{
                {2, 77, 7903, {"chat-alias"}, {2}}};
        },
        [](const std::string& name) -> std::optional<std::string> {
            return name == "chat-alias"
                ? std::optional<std::string>{"chat"} : std::nullopt;
        },
        [] { return 7901; },
    });

    const auto response = service.handle();
    REQUIRE(response);
    CHECK(response->status == 200);
    const auto body = nlohmann::json::parse(response->data);
    REQUIRE(body["object"] == "list");
    REQUIRE(body["data"].size() == 1);
    const auto& model = body["data"].at(0);
    CHECK(model["id"] == "chat");
    CHECK(model["status"]["value"] == "loaded");
    CHECK(model["capabilities"]["vision"] == true);
    REQUIRE(model["instances"].size() == 2);
    CHECK(model["instances"].at(0)["instance_id"] == 0);
    CHECK(model["instances"].at(0)["device_ids"] ==
          nlohmann::json::array({0, 1}));
    CHECK(model["instances"].at(1)["instance_id"] == 2);
    CHECK(model["instances"].at(1)["process_id"] == 77);
    CHECK(model["instances"].at(1)["port"] == 7903);
}

TEST_CASE("Model list application service falls back to loaded device metadata",
          "[p1][model-list][application-service]") {
    auto loaded = meta("chat", SERVER_MODEL_STATUS_SLEEPING);
    loaded.loaded_info["device_ids"] = {3};
    HLIEModelListApplicationService service({
        [loaded] { return std::vector<server_model_meta>{loaded}; },
        [] {
            return std::vector<HLIEModelListLocalInstance>{{"chat", {}}};
        },
        {}, {}, [] { return 7901; },
    });

    const auto response = service.handle();
    REQUIRE(response);
    const auto body = nlohmann::json::parse(response->data);
    CHECK(body["data"].at(0)["status"]["value"] == "sleeping");
    CHECK(body["data"].at(0)["instances"].at(0)["device_ids"] ==
          nlohmann::json::array({3}));
}

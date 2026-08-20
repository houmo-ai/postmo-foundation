#include <catch_amalgamated.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "hlie_app/startup/hlie_bootstrap_application_service.h"

namespace {

using namespace hm_HLIE;

HLIEModelInfo llm_model(const std::string& filename) {
    HLIEModelInfo model;
    REQUIRE(model.parse_identity_from_gguf_filename(filename) == 0);
    return model;
}

struct BootstrapFixture {
    bool prepare_result = true;
    bool resolve_result = true;
    bool worker_initialize_result = true;
    HLIEDefaultModelInitializationStatus default_status =
        HLIEDefaultModelInitializationStatus::Initialized;
    std::optional<HLIEModelInfo> parsed_manifest;
    std::vector<HLIEModelInfo> discovered_models;
    std::vector<std::string> events;
    std::vector<std::string> registered_models;
    std::string parsed_path;
    std::string discarded_path;
    HLIEDefaultModelInitializationOptions observed_options;
    int default_calls = 0;

    HLIEBootstrapApplicationDependencies dependencies() {
        return {
            [this] {
                events.push_back("prepare");
                return prepare_result;
            },
            [this](const std::string& path)
                -> std::optional<HLIEModelInfo> {
                events.push_back("parse");
                parsed_path = path;
                return parsed_manifest;
            },
            [this](const std::string& path) {
                events.push_back("discard");
                discarded_path = path;
            },
            [this] {
                events.push_back("discover");
                return discovered_models;
            },
            [this](HLIEModelInfo&) {
                events.push_back("resolve");
                return resolve_result;
            },
            [this](HLIEModelInfo& model) {
                events.push_back("register");
                registered_models.push_back(model.model_key_);
            },
            [this](const HLIEModelInfo&) {
                events.push_back("worker_initialize");
                return worker_initialize_result;
            },
            [this] { events.push_back("baseline"); },
            [this](const std::vector<HLIEModelInfo>&,
                   const HLIEDefaultModelInitializationOptions& options) {
                events.push_back("defaults");
                ++default_calls;
                observed_options = options;
                return HLIEDefaultModelInitializationResult{default_status};
            },
        };
    }
};

} // namespace

TEST_CASE("Bootstrap rejects an invalid launch ticket before manifest access",
          "[p1][bootstrap][application-service]") {
    BootstrapFixture fixture;
    fixture.parsed_manifest = llm_model("worker.gguf");
    HLIEBootstrapApplicationService service(fixture.dependencies());
    HLIEBootstrapRequest request;
    request.arguments = {
        {"--launch_generation", "invalid"},
        {"--launch_token", "secret"},
        {"--launch_model_key", "worker-key"},
        {"--launch_manifest", "worker.json"},
    };

    const auto result = service.initialize(request);

    CHECK(result.status == HLIEBootstrapStatus::Failed);
    CHECK_FALSE(result.worker_process());
    CHECK(result.legacy_return_code() == -1);
    CHECK(fixture.events == std::vector<std::string>{"prepare"});
    CHECK(fixture.parsed_path.empty());
    CHECK(fixture.discarded_path.empty());
}

TEST_CASE("Bootstrap initializes a ticketed Worker from one-time manifest",
          "[p1][bootstrap][application-service][worker]") {
    BootstrapFixture fixture;
    fixture.parsed_manifest = llm_model("worker.gguf");
    HLIEBootstrapApplicationService service(fixture.dependencies());
    HLIEBootstrapRequest request;
    request.model_directory = "models";
    request.arguments = {
        {"--launch_generation", "7"},
        {"--launch_token", "secret"},
        {"--launch_model_key", "worker-key"},
        {"--launch_manifest", "worker.json"},
    };

    const auto result = service.initialize(request);

    CHECK(result.status == HLIEBootstrapStatus::Ready);
    CHECK(result.worker_process());
    CHECK(result.legacy_return_code() == 0);
    CHECK(result.launch_ticket.generation == 7);
    CHECK(result.launch_ticket.token == "secret");
    CHECK(result.launch_ticket.model_key == "worker-key");
    CHECK(fixture.parsed_path == "worker.json");
    CHECK(fixture.discarded_path == "worker.json");
    REQUIRE(fixture.registered_models.size() == 1);
    CHECK(fixture.events == std::vector<std::string>{
        "prepare", "parse", "discard", "resolve", "register",
        "worker_initialize"});
    CHECK(fixture.default_calls == 0);
}

TEST_CASE("Bootstrap consumes only one-time manifests on parse failure",
          "[p1][bootstrap][application-service][worker]") {
    SECTION("one-time launch manifest") {
        BootstrapFixture fixture;
        HLIEBootstrapApplicationService service(fixture.dependencies());
        HLIEBootstrapRequest request;
        request.arguments = {{"--launch_manifest", "broken.json"}};

        const auto result = service.initialize(request);

        CHECK(result.status == HLIEBootstrapStatus::Failed);
        CHECK(result.worker_process());
        CHECK(fixture.discarded_path == "broken.json");
        CHECK(fixture.events == std::vector<std::string>{
            "prepare", "parse", "discard"});
    }

    SECTION("legacy json info") {
        BootstrapFixture fixture;
        fixture.parsed_manifest = llm_model("legacy.gguf");
        HLIEBootstrapApplicationService service(fixture.dependencies());
        HLIEBootstrapRequest request;
        request.arguments = {{"--json_info", "legacy.json"}};

        const auto result = service.initialize(request);

        CHECK(result.status == HLIEBootstrapStatus::Ready);
        CHECK(result.worker_process());
        CHECK(fixture.discarded_path.empty());
        CHECK(fixture.events == std::vector<std::string>{
            "prepare", "parse", "resolve", "register",
            "worker_initialize"});
    }
}

TEST_CASE("Bootstrap registers the main catalog before default selection",
          "[p1][bootstrap][application-service][main]") {
    BootstrapFixture fixture;
    fixture.discovered_models = {
        llm_model("first.gguf"), llm_model("second.gguf")};
    HLIEBootstrapApplicationService service(fixture.dependencies());
    HLIEBootstrapRequest request;
    request.model_directory = "models";
    request.default_model = "first";
    request.default_mmproj = "vision";
    request.enable_sd3 = true;

    const auto result = service.initialize(request);

    CHECK(result.status == HLIEBootstrapStatus::Ready);
    CHECK_FALSE(result.worker_process());
    CHECK(result.legacy_return_code() == 0);
    CHECK(fixture.registered_models.size() == 2);
    CHECK(fixture.events == std::vector<std::string>{
        "prepare", "discover", "register", "register", "baseline",
        "defaults"});
    CHECK(fixture.observed_options.model_directory == "models");
    CHECK(fixture.observed_options.default_model == "first");
    CHECK(fixture.observed_options.default_mmproj == "vision");
    CHECK(fixture.observed_options.enable_sd3);
    CHECK(fixture.observed_options.zero_load_allowed);
}

TEST_CASE("Bootstrap preserves zero-load and instance failure mapping",
          "[p1][bootstrap][application-service][main]") {
    SECTION("main process zero-load") {
        BootstrapFixture fixture;
        fixture.default_status =
            HLIEDefaultModelInitializationStatus::ZeroLoad;
        HLIEBootstrapApplicationService service(fixture.dependencies());

        const auto result = service.initialize({});

        CHECK(result.status == HLIEBootstrapStatus::ZeroLoad);
        CHECK(result.started());
        CHECK(result.legacy_return_code() == 8);
        CHECK(fixture.observed_options.zero_load_allowed);
    }

    SECTION("instance process cannot zero-load") {
        BootstrapFixture fixture;
        fixture.default_status =
            HLIEDefaultModelInitializationStatus::Failed;
        HLIEBootstrapApplicationService service(fixture.dependencies());
        HLIEBootstrapRequest request;
        request.arguments = {{"--instance_id", "3"}};

        const auto result = service.initialize(request);

        CHECK(result.status == HLIEBootstrapStatus::Failed);
        CHECK_FALSE(result.started());
        CHECK(result.legacy_return_code() == -1);
        CHECK_FALSE(fixture.observed_options.zero_load_allowed);
    }
}

TEST_CASE("Bootstrap stops when launch configuration preparation fails",
          "[p1][bootstrap][application-service]") {
    BootstrapFixture fixture;
    fixture.prepare_result = false;
    HLIEBootstrapApplicationService service(fixture.dependencies());

    const auto result = service.initialize({});

    CHECK(result.status == HLIEBootstrapStatus::Failed);
    CHECK(fixture.events == std::vector<std::string>{"prepare"});
    CHECK(fixture.default_calls == 0);
}

TEST_CASE("Bootstrap rejects incomplete composition adapters",
          "[p1][bootstrap][application-service]") {
    SECTION("main process requires catalog and baseline adapters") {
        BootstrapFixture fixture;
        auto dependencies = fixture.dependencies();
        dependencies.initialize_discovery_baseline = {};
        HLIEBootstrapApplicationService service(std::move(dependencies));

        const auto result = service.initialize({});

        CHECK(result.status == HLIEBootstrapStatus::Failed);
        CHECK(fixture.events == std::vector<std::string>{"prepare"});
        CHECK(fixture.default_calls == 0);
    }

    SECTION("Worker consumes one-time manifest before adapter failure") {
        BootstrapFixture fixture;
        fixture.parsed_manifest = llm_model("worker.gguf");
        auto dependencies = fixture.dependencies();
        dependencies.register_model = {};
        HLIEBootstrapApplicationService service(std::move(dependencies));
        HLIEBootstrapRequest request;
        request.arguments = {{"--launch_manifest", "worker.json"}};

        const auto result = service.initialize(request);

        CHECK(result.status == HLIEBootstrapStatus::Failed);
        CHECK(result.worker_process());
        CHECK(fixture.discarded_path == "worker.json");
        CHECK(fixture.events == std::vector<std::string>{
            "prepare", "parse", "discard", "resolve"});
    }
}

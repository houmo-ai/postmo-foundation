#include <catch_amalgamated.hpp>

#include <chrono>
#include <memory>

#include "hlie_core/domain/hlie_model_command.h"

using namespace hm_HLIE;

TEST_CASE("control result keeps typed success and failure disjoint", "[p1][core][contract]") {
    HLIEModelOperationReceipt receipt;
    receipt.identity.model_key = "model-key";
    receipt.generation = 7;
    receipt.state = HLIEModelLifecycleState::Starting;

    auto success = HLIEModelOperationResult::success(receipt);
    REQUIRE(success);
    REQUIRE(success.value_if() != nullptr);
    REQUIRE(success.error_if() == nullptr);
    CHECK(success.value_if()->generation == 7);

    auto failure = HLIEModelOperationResult::failure({
        HLIEControlErrorCode::CapacityExceeded,
        HLIEControlErrorStage::Reservation,
        "insufficient device memory",
        80004,
        true,
    });
    REQUIRE_FALSE(failure);
    REQUIRE(failure.value_if() == nullptr);
    REQUIRE(failure.error_if() != nullptr);
    CHECK(failure.error_if()->code == HLIEControlErrorCode::CapacityExceeded);
    CHECK(failure.error_if()->stage == HLIEControlErrorStage::Reservation);
    CHECK(failure.error_if()->retryable);
}

TEST_CASE("model commands express identity placement and deadline without SDK types",
          "[p1][core][contract]") {
    HLIEModelLoadCommand load;
    load.spec.identity.model_name = "qwen3-asr";
    load.spec.required_device_count = 2;
    load.placement.requested_device_ids = {0, 1};
    load.placement.required_device_count = 2;
    load.placement.allow_auto_selection = false;
    load.preferred_runtime = HLIERuntimeKind::Process;
    REQUIRE(load.valid());

    load.placement.requested_device_ids = {0};
    CHECK_FALSE(load.valid());

    load.placement.requested_device_ids = {0, 0};
    CHECK_FALSE(load.valid());

    load.placement.requested_device_ids = {0, -1};
    CHECK_FALSE(load.valid());

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    HLIEModelUnloadCommand unload{HLIEModelSelector::by_key("model-key"), deadline, false};
    CHECK(unload.valid());

    unload.selector.value.clear();
    CHECK_FALSE(unload.valid());
}

TEST_CASE("control result supports move-only payloads", "[p1][core][contract]") {
    auto result = HLIEControlResult<std::unique_ptr<int>>::success(std::make_unique<int>(42));
    REQUIRE(result);
    REQUIRE(result.value_if() != nullptr);
    CHECK(**result.value_if() == 42);
}

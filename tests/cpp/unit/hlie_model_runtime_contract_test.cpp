#include <catch_amalgamated.hpp>

#include "hlie_core/runtime/hlie_model_runtime.h"

namespace {

using namespace hm_HLIE;

HLIEModelLaunchTicket runtime_ticket(const std::string& key,
                                     const HLIEModelGeneration generation) {
    return {key, generation, "runtime-token-" + std::to_string(generation)};
}

} // namespace

TEST_CASE("runtime identity validates embedded and process transports",
          "[p1][runtime]") {
    const HLIERuntimeIdentity embedded{
        1, HLIERuntimeKind::Embedded, runtime_ticket("embedded", 1), 0, 0};
    REQUIRE(embedded.valid());
    CHECK(embedded.matches(embedded));

    const HLIERuntimeIdentity process{
        42, HLIERuntimeKind::Process, runtime_ticket("worker", 2), 3, 7904};
    REQUIRE(process.valid());
    CHECK(process.matches(process));

    auto invalid_process = process;
    invalid_process.endpoint_port = 0;
    CHECK_FALSE(invalid_process.valid());
    auto invalid_embedded = embedded;
    invalid_embedded.instance_id = 1;
    CHECK_FALSE(invalid_embedded.valid());
    invalid_embedded = embedded;
    invalid_embedded.endpoint_port = 7901;
    CHECK_FALSE(invalid_embedded.valid());
}

TEST_CASE("runtime health and stop commands are protocol neutral",
          "[p1][runtime]") {
    const HLIERuntimeIdentity identity{
        7, HLIERuntimeKind::Embedded, runtime_ticket("model", 9), 0, 0};
    const HLIERuntimeHealth ready{identity, HLIERuntimeHealthState::Ready, 0};
    CHECK(ready.ready());

    auto failed = ready;
    failed.state = HLIERuntimeHealthState::Failed;
    failed.native_code = -17;
    CHECK_FALSE(failed.ready());

    HLIERuntimeStopRequest request;
    CHECK_FALSE(request.valid());
    request.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    request.reason = HLIERuntimeStopReason::Eviction;
    CHECK(request.valid());
}

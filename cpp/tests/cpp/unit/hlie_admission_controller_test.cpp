#include <catch_amalgamated.hpp>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "hlie_core/control/hlie_admission_controller.h"

using namespace hm_HLIE;

TEST_CASE("admission controller enforces concurrent in-flight limit", "[admission]") {
    HLIEAdmissionController controller(2);
    auto first = controller.try_acquire();
    auto second = controller.try_acquire();
    auto rejected = controller.try_acquire();

    REQUIRE(first);
    REQUIRE(second);
    CHECK_FALSE(rejected);
    auto snapshot = controller.snapshot();
    CHECK(snapshot.in_flight == 2);
    CHECK(snapshot.accepted == 2);
    CHECK(snapshot.rejected == 1);

    first.reset();
    auto replacement = controller.try_acquire();
    REQUIRE(replacement);
    CHECK(controller.snapshot().in_flight == 2);
}

TEST_CASE("admission permit move releases capacity exactly once", "[admission]") {
    HLIEAdmissionController controller(1);
    auto permit = controller.try_acquire();
    REQUIRE(permit);
    HLIEAdmissionController::Permit moved = std::move(*permit);
    permit.reset();
    CHECK(controller.snapshot().in_flight == 1);
    moved.reset();
    CHECK(controller.snapshot().in_flight == 0);
}

#include <catch_amalgamated.hpp>

#include "hlie_base/hlie_blocking_queue.h"

using namespace hm_HLIE;

TEST_CASE("Blocking queue non-blocking admission respects capacity", "[blocking-queue][admission]") {
    HLIEBlockingQueue<int> queue(2);
    CHECK(queue.max_size() == 2);
    CHECK(queue.try_push(1));
    CHECK(queue.try_push(2));
    CHECK_FALSE(queue.try_push(3));
    CHECK(queue.size() == 2);
    CHECK(queue.pop() == 1);
    CHECK(queue.try_push(3));
}

TEST_CASE("Blocking queue rejects admission after close", "[blocking-queue][admission]") {
    HLIEBlockingQueue<int> queue(1);
    queue.close();
    CHECK_THROWS_AS(queue.try_push(1), std::runtime_error);
}

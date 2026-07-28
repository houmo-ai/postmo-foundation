#ifndef HLIE_MODEL_DISCOVERY_SYNC_H
#define HLIE_MODEL_DISCOVERY_SYNC_H

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class HLIEModelDiscoverySync {
public:
    uint64_t request() {
        std::lock_guard lock(mutex_);
        return ++requested_generation_;
    }

    [[nodiscard]] uint64_t latest_requested() const {
        std::lock_guard lock(mutex_);
        return requested_generation_;
    }

    void complete_through(const uint64_t generation) {
        {
            std::lock_guard lock(mutex_);
            completed_generation_ = std::max(completed_generation_, generation);
        }
        cv_.notify_all();
    }

    [[nodiscard]] bool wait_for(const uint64_t generation, const std::chrono::milliseconds timeout) {
        if (generation == 0) {
            return false;
        }

        std::unique_lock lock(mutex_);
        cv_.wait_for(lock, timeout, [this, generation] {
            return stopping_ || completed_generation_ >= generation;
        });
        return completed_generation_ >= generation;
    }

    void reset() {
        std::lock_guard lock(mutex_);
        stopping_ = false;
    }

    void stop() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
    }

private:
    uint64_t requested_generation_ = 0;
    uint64_t completed_generation_ = 0;
    bool stopping_ = false;
    std::condition_variable cv_;
    mutable std::mutex mutex_;
};

HLIE_NAMESPACE_END

#endif // HLIE_MODEL_DISCOVERY_SYNC_H

#ifndef HLIE_BACKGROUND_TASK_GROUP_H
#define HLIE_BACKGROUND_TASK_GROUP_H

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

// Owns background tasks whose callbacks may reference their enclosing service.
// Destruction waits for every task, preventing callbacks from outliving that service.
class BackgroundTaskGroup {
public:
    BackgroundTaskGroup() = default;
    ~BackgroundTaskGroup();

    BackgroundTaskGroup(const BackgroundTaskGroup&) = delete;
    BackgroundTaskGroup& operator=(const BackgroundTaskGroup&) = delete;

    bool launch(std::function<void()> task);
    void stop_and_join();

    [[nodiscard]] std::size_t tracked_task_count() const;

private:
    struct TaskEntry {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> completed;
    };

    mutable std::mutex mutex_;
    std::vector<TaskEntry> tasks_;
    bool stopping_{false};
};

HLIE_NAMESPACE_END

#endif // HLIE_BACKGROUND_TASK_GROUP_H

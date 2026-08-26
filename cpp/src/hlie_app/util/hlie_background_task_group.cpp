#include "hlie_app/util/hlie_background_task_group.h"

#include <utility>

#include "hlie_base/hlie_logger.h"

HLIE_NAMESPACE_BEGIN

BackgroundTaskGroup::~BackgroundTaskGroup() {
    stop_and_join();
}

bool BackgroundTaskGroup::launch(std::function<void()> task) {
    std::vector<std::thread> completed_threads;
    {
        std::lock_guard lock(mutex_);
        if (stopping_) {
            return false;
        }

        auto entry = tasks_.begin();
        while (entry != tasks_.end()) {
            if (entry->completed->load(std::memory_order_acquire)) {
                completed_threads.push_back(std::move(entry->thread));
                entry = tasks_.erase(entry);
            } else {
                ++entry;
            }
        }
    }

    for (auto& thread : completed_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    auto completed = std::make_shared<std::atomic<bool>>(false);
    std::lock_guard lock(mutex_);
    if (stopping_) {
        return false;
    }
    try {
        tasks_.reserve(tasks_.size() + 1);
        tasks_.push_back(TaskEntry{
            std::thread([task = std::move(task), completed] () mutable {
                try {
                    task();
                } catch (const std::exception& error) {
                    LOG_ERROR("Background task threw: {}", error.what());
                } catch (...) {
                    LOG_ERROR("Background task threw an unknown exception");
                }
                completed->store(true, std::memory_order_release);
            }),
            std::move(completed),
        });
    } catch (...) {
        return false;
    }
    return true;
}

void BackgroundTaskGroup::stop_and_join() {
    std::vector<TaskEntry> tasks;
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        tasks.swap(tasks_);
    }

    for (auto& task : tasks) {
        if (task.thread.joinable()) {
            task.thread.join();
        }
    }
}

std::size_t BackgroundTaskGroup::tracked_task_count() const {
    std::lock_guard lock(mutex_);
    return tasks_.size();
}

HLIE_NAMESPACE_END

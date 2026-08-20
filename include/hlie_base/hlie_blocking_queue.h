#ifndef HLIE_BLOCKING_QUEUE_H
#define HLIE_BLOCKING_QUEUE_H

#include "hlie_base.h"

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <optional>
#include <stdexcept>

HLIE_NAMESPACE_BEGIN

template<typename T>
class HLIEBlockingQueue {
public:
    explicit HLIEBlockingQueue(const size_t max_size = std::numeric_limits<size_t>::max())
        : max_size_(max_size) {
        if (max_size == 0) {
            throw std::invalid_argument("BlockingQueue max_size cannot be zero");
        }
    }

    // 阻塞式插入，队列满时等待
    void push(const T& item) {
        try {
            std::unique_lock lock(mutex_);
            not_full_.wait(lock, [this]() { return queue_.size() < max_size_ || closed_; });

            if (closed_) {
                throw std::runtime_error("BlockingQueue is closed");
            }

            queue_.push(item);
            not_empty_.notify_one();
        } catch (...) {
            not_empty_.notify_all(); // 通知所有等待线程
            throw; // 重新抛出异常
        }
    }

    // Non-blocking insertion used by request admission. Returns false when the
    // queue is at capacity; a closed queue is still an operational error.
    bool try_push(const T& item) {
        std::lock_guard lock(mutex_);
        if (closed_) {
            throw std::runtime_error("BlockingQueue is closed");
        }
        if (queue_.size() >= max_size_) {
            return false;
        }

        queue_.push(item);
        not_empty_.notify_one();
        return true;
    }

    // 阻塞式取出，队列空时等待
    T pop() {
        try {
            std::unique_lock lock(mutex_);
            not_empty_.wait(lock, [this]() { return !queue_.empty() || closed_; });

            if (closed_ && queue_.empty()) {
                throw std::runtime_error("BlockingQueue is closed and empty");
            }

            T item = std::move(queue_.front());
            queue_.pop();
            not_full_.notify_one();
            return item;
        } catch (...) {
            not_full_.notify_all(); // 通知所有等待线程
            throw; // 重新抛出异常
        }
    }

    // 非阻塞式尝试取出
    std::optional<T> try_pop() {
        try {
            std::lock_guard lock(mutex_);
            if (queue_.empty()) {
                return std::nullopt;
            }

            T item = std::move(queue_.front());
            queue_.pop();
            not_full_.notify_one();
            return item;
        } catch (...) {
            not_full_.notify_all();
            throw;
        }
    }

    // 带超时的取出操作
    template<typename Rep, typename Period>
    std::optional<T> pop_for(const std::chrono::duration<Rep, Period>& timeout) {
        try {
            std::unique_lock lock(mutex_);
            if (!not_empty_.wait_for(lock, timeout, [this]() { return !queue_.empty() || closed_; })) {
                return std::nullopt; // 超时返回空
            }

            if (closed_ && queue_.empty()) {
                throw std::runtime_error("BlockingQueue is closed and empty");
            }

            T item = std::move(queue_.front());
            queue_.pop();
            not_full_.notify_one();
            return item;
        } catch (...) {
            not_full_.notify_all();
            throw;
        }
    }

    // 关闭队列，唤醒所有等待线程
    void close() {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
            not_empty_.notify_all();
            not_full_.notify_all();
        } catch (...) {
            not_empty_.notify_all();
            not_full_.notify_all();
            throw;
        }
    }

    size_t size() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::lock_guard lock(mutex_);
        return queue_.empty();
    }

    bool is_closed() const {
        std::lock_guard lock(mutex_);
        return closed_;
    }

    size_t max_size() const {
        return max_size_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    size_t max_size_;
    bool closed_ = false;
};

HLIE_NAMESPACE_END

#endif //HLIE_BLOCKING_QUEUE_H

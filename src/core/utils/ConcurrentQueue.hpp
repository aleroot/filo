#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <optional>

namespace core::utils {

/**
 * @brief A simple thread-safe queue for producer-consumer patterns.
 * 
 * This is a minimal, modern C++ implementation that avoids the complexity
 * of multiple mutexes and condition variables. It's designed for the
 * specific use case of background threads sending updates to a UI thread.
 */
template<typename T>
class ConcurrentQueue {
public:
    void push(T item) {
        {
            std::lock_guard lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }

    std::optional<T> try_pop() {
        std::lock_guard lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    T pop_blocking() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    bool empty() const {
        std::lock_guard lock(mutex_);
        return queue_.empty();
    }

    void clear() {
        std::lock_guard lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
};

} // namespace core::utils

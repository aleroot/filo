#include "DeadlineScheduler.hpp"

#include "core/logging/Logger.hpp"

#include <algorithm>
#include <exception>
#include <utility>

namespace tui {

DeadlineScheduler::~DeadlineScheduler() {
    shutdown();
}

bool DeadlineScheduler::schedule(std::string key,
                                 Clock::time_point when,
                                 std::function<void()> task) {
    {
        std::lock_guard lock(mutex_);
        if (stopping_) {
            return false;
        }
        entries_.insert_or_assign(std::move(key), Entry{when, std::move(task)});
        // Started on first use: a session that never needs a deadline never
        // pays for the thread.
        if (!thread_.joinable()) {
            thread_ = std::thread([this] { run(); });
        }
    }
    cv_.notify_one();
    return true;
}

void DeadlineScheduler::cancel(std::string_view key) {
    {
        std::lock_guard lock(mutex_);
        const auto it = entries_.find(std::string(key));
        if (it == entries_.end()) {
            return;
        }
        entries_.erase(it);
    }
    cv_.notify_one();
}

std::size_t DeadlineScheduler::pending() const {
    std::lock_guard lock(mutex_);
    return entries_.size();
}

void DeadlineScheduler::shutdown() {
    std::thread thread;
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
        entries_.clear();
        thread = std::move(thread_);
    }
    cv_.notify_one();
    if (thread.joinable()) {
        thread.join();
    }
}

void DeadlineScheduler::run() {
    std::unique_lock lock(mutex_);
    while (!stopping_) {
        if (entries_.empty()) {
            cv_.wait(lock, [this] { return stopping_ || !entries_.empty(); });
            continue;
        }
        const auto next = std::ranges::min_element(
            entries_, {}, [](const auto& entry) { return entry.second.when; });
        const auto when = next->second.when;
        if (Clock::now() < when) {
            // Woken early by a schedule/cancel, or spuriously: re-evaluate.
            cv_.wait_until(lock, when);
            continue;
        }
        auto task = std::move(next->second.task);
        entries_.erase(next);
        lock.unlock();
        try {
            task();
        } catch (const std::exception& error) {
            core::logging::warn("Deadline task failed: {}", error.what());
        } catch (...) {
            core::logging::warn("Deadline task failed: unknown exception");
        }
        lock.lock();
    }
}

} // namespace tui

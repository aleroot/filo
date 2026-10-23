#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

namespace tui {

/// Runs keyed one-shot tasks at wall-clock instants on one background thread.
///
/// The TUI renders nothing while idle (see select_animation_cadence), so a
/// reminder that must fire while the user is away needs its own wake-up.
/// This scheduler sleeps until the earliest pending instant and not a moment
/// before: with nothing scheduled it blocks indefinitely, so it adds no idle
/// wake-ups. Instants are system_clock because the deadlines it serves
/// (provider cache lifetimes) run in wall time, including while the machine
/// sleeps.
///
/// Scheduling an existing key replaces it, so callers re-arm without first
/// cancelling. Tasks run outside the lock, one at a time, and may schedule or
/// cancel; they must not block for long, since they delay later tasks.
class DeadlineScheduler final {
public:
    using Clock = std::chrono::system_clock;

    DeadlineScheduler() = default;
    ~DeadlineScheduler();

    DeadlineScheduler(const DeadlineScheduler&) = delete;
    DeadlineScheduler& operator=(const DeadlineScheduler&) = delete;

    /// Returns false once shut down; the task is then dropped.
    bool schedule(std::string key, Clock::time_point when, std::function<void()> task);
    void cancel(std::string_view key);
    [[nodiscard]] std::size_t pending() const;

    /// Drops pending tasks and joins the thread. A task already running
    /// finishes first. Idempotent; must not be called from a task.
    void shutdown();

private:
    struct Entry {
        Clock::time_point when;
        std::function<void()> task;
    };

    void run();

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<std::string, Entry> entries_;
    bool stopping_ = false;
    std::thread thread_;
};

} // namespace tui

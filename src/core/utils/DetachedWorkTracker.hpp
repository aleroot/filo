#pragma once

/**
 * @file DetachedWorkTracker.hpp
 * @brief A bounded observation point for threads that are detached on purpose.
 *
 * Some work must never block its caller. A task worker cannot be joined without
 * letting a non-cooperative tool hang the cancellation of that task, and an
 * agent's tool loop is detached for the same reason, so both outlive the call
 * that started them. Whatever such a thread still touches — session state, or
 * the singletons a process destroys on exit — can therefore be freed underneath
 * it, which is a crash that arrives *after* the caller reported success.
 *
 * This does not make those threads joinable and does not change why they are
 * detached. It counts them, so a caller that is about to destroy such state can
 * wait for them within a bound and be told whether they finished. Waiting is
 * always optional and always time-limited: a thread that never ends costs its
 * waiter the timeout and nothing more.
 */

#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <mutex>

namespace core::utils {

class DetachedWorkTracker {
public:
    /// One unit of detached work in flight, for as long as this lives. Declare it
    /// first in the thread body so it is destroyed last, which makes "idle" mean
    /// the thread has really finished rather than merely stopped doing work.
    class Guard {
    public:
        explicit Guard(DetachedWorkTracker& tracker) noexcept;
        ~Guard();

        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
        Guard(Guard&&) = delete;
        Guard& operator=(Guard&&) = delete;

    private:
        DetachedWorkTracker& tracker_;
    };

    /// Wait until every guarded thread has exited, or @p timeout elapses.
    /// @return false when work was still in flight at that point.
    bool wait_for_idle(std::chrono::milliseconds timeout);

    /// Units of detached work currently in flight.
    [[nodiscard]] std::size_t active() const noexcept;

    /**
     * @brief The process-wide tracker for detached agent and task threads.
     *
     * Deliberately never destroyed: a detached thread can report its exit while
     * the process is tearing down its statics, and destroying the tracker then
     * would trade one use-after-free for another.
     */
    [[nodiscard]] static DetachedWorkTracker& instance() noexcept;

private:
    void enter() noexcept;
    void leave() noexcept;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::size_t active_ = 0;
};

} // namespace core::utils

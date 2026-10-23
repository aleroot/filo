#include <catch2/catch_test_macros.hpp>

#include "tui/DeadlineScheduler.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

using namespace std::chrono_literals;

namespace {

/// Collects fired keys and lets a test wait for a count.
class FiredLog {
public:
    std::function<void()> record(std::string key) {
        return [this, key = std::move(key)]() {
            {
                std::lock_guard lock(mutex_);
                keys_.push_back(key);
            }
            cv_.notify_all();
        };
    }

    [[nodiscard]] bool wait_for(std::size_t count, std::chrono::milliseconds timeout = 2s) {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, timeout, [&] { return keys_.size() >= count; });
    }

    [[nodiscard]] std::vector<std::string> keys() {
        std::lock_guard lock(mutex_);
        return keys_;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::string> keys_;
};

} // namespace

TEST_CASE("DeadlineScheduler fires tasks in deadline order", "[deadline-scheduler]") {
    tui::DeadlineScheduler scheduler;
    FiredLog log;
    const auto now = tui::DeadlineScheduler::Clock::now();

    scheduler.schedule("late", now + 60ms, log.record("late"));
    scheduler.schedule("early", now + 20ms, log.record("early"));
    scheduler.schedule("past", now - 1s, log.record("past"));

    REQUIRE(log.wait_for(3));
    CHECK(log.keys() == std::vector<std::string>{"past", "early", "late"});
    CHECK(scheduler.pending() == 0);
}

TEST_CASE("DeadlineScheduler replaces and cancels by key", "[deadline-scheduler]") {
    tui::DeadlineScheduler scheduler;
    FiredLog log;
    const auto now = tui::DeadlineScheduler::Clock::now();

    scheduler.schedule("notice", now + 10min, log.record("stale"));
    scheduler.schedule("notice", now + 20ms, log.record("fresh"));
    scheduler.schedule("dropped", now + 30ms, log.record("dropped"));
    scheduler.cancel("dropped");
    scheduler.cancel("never-scheduled");

    REQUIRE(log.wait_for(1));
    CHECK_FALSE(log.wait_for(2, 100ms));
    CHECK(log.keys() == std::vector<std::string>{"fresh"});
}

TEST_CASE("DeadlineScheduler shutdown drops pending work and refuses more",
          "[deadline-scheduler]") {
    tui::DeadlineScheduler scheduler;
    FiredLog log;
    const auto now = tui::DeadlineScheduler::Clock::now();
    REQUIRE(scheduler.schedule("later", now + 1h, log.record("later")));

    scheduler.shutdown();
    CHECK(scheduler.pending() == 0);
    CHECK_FALSE(scheduler.schedule("after", now, log.record("after")));
    CHECK_FALSE(log.wait_for(1, 50ms));
    scheduler.shutdown();  // idempotent
}

TEST_CASE("A task may re-arm itself", "[deadline-scheduler]") {
    tui::DeadlineScheduler scheduler;
    FiredLog log;
    auto second = log.record("second");
    scheduler.schedule("tick", tui::DeadlineScheduler::Clock::now(), [&]() {
        log.record("first")();
        scheduler.schedule("tick", tui::DeadlineScheduler::Clock::now() + 10ms, second);
    });
    REQUIRE(log.wait_for(2));
    CHECK(log.keys() == std::vector<std::string>{"first", "second"});
}

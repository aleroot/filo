#include <catch2/catch_test_macros.hpp>

#include "tui/ActivityTimer.hpp"

#include <chrono>

using namespace tui;

TEST_CASE("ActivityTimerRegistry tracks elapsed time per operation id",
          "[tui][timer]") {
    ActivityTimerRegistry timer;

    const auto start = ActivityTimerRegistry::Clock::now();
    timer.start_at("assistant-turn-1", start);

    const auto elapsed = timer.elapsed(
        "assistant-turn-1",
        start + std::chrono::seconds(76));

    REQUIRE(elapsed.has_value());
    REQUIRE(elapsed->count() == 76);

    timer.stop("assistant-turn-1");
    REQUIRE_FALSE(timer.elapsed("assistant-turn-1").has_value());
}

TEST_CASE("format_elapsed_compact renders concise human-friendly durations",
          "[tui][timer]") {
    REQUIRE(format_elapsed_compact(std::chrono::seconds(0)) == "0s");
    REQUIRE(format_elapsed_compact(std::chrono::seconds(59)) == "59s");
    REQUIRE(format_elapsed_compact(std::chrono::seconds(76)) == "1m 16s");
    REQUIRE(format_elapsed_compact(std::chrono::seconds(3661)) == "1h 01m 01s");
}

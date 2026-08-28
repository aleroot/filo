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
    REQUIRE(format_elapsed_compact(std::chrono::hours(49))
            == "2d 01h 00m 00s");
}

TEST_CASE("format_elapsed_compact can reduce precision for relative times",
          "[tui][timer]") {
    REQUIRE(format_elapsed_compact(
                std::chrono::minutes(1), ElapsedFormat::humanized)
            == "1m");
    REQUIRE(format_elapsed_compact(
                std::chrono::hours(1), ElapsedFormat::humanized)
            == "1h");
    REQUIRE(format_elapsed_compact(
                std::chrono::minutes(253) + std::chrono::seconds(49),
                ElapsedFormat::humanized)
            == "4h 13m");
    REQUIRE(format_elapsed_compact(
                std::chrono::hours(49), ElapsedFormat::humanized)
            == "2d 1h");
}

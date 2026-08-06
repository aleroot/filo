#include <catch2/catch_test_macros.hpp>

#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/transport/StreamResilience.hpp"

using namespace std::chrono_literals;
using core::llm::transport::RetryController;
using core::llm::transport::RetryPolicy;
using core::llm::transport::StreamTimeoutKind;
using core::llm::transport::StreamTimeoutPolicy;
using core::llm::transport::StreamWatchdog;

TEST_CASE("StreamWatchdog distinguishes response-start and inactivity deadlines",
          "[http][resilience][timeout]") {
    const StreamWatchdog::TimePoint started{};

    SECTION("late first activity is rejected as a response-start timeout") {
        StreamWatchdog watchdog(
            StreamTimeoutPolicy{.response_start = 100ms, .inactivity = 250ms},
            started);

        CHECK(watchdog.poll(started + 99ms));
        CHECK_FALSE(watchdog.observe_activity(started + 100ms));
        CHECK(watchdog.timeout_kind() == StreamTimeoutKind::ResponseStart);
    }

    SECTION("activity starts the independent inactivity deadline") {
        StreamWatchdog watchdog(
            StreamTimeoutPolicy{.response_start = 100ms, .inactivity = 250ms},
            started);

        REQUIRE(watchdog.observe_activity(started + 50ms));
        CHECK(watchdog.poll(started + 299ms));
        CHECK_FALSE(watchdog.poll(started + 300ms));
        CHECK(watchdog.timeout_kind() == StreamTimeoutKind::Inactivity);
    }

    SECTION("new activity advances the inactivity deadline") {
        StreamWatchdog watchdog(
            StreamTimeoutPolicy{.response_start = 100ms, .inactivity = 250ms},
            started);

        REQUIRE(watchdog.observe_activity(started + 50ms));
        REQUIRE(watchdog.observe_activity(started + 250ms));
        CHECK(watchdog.poll(started + 499ms));
    }
}

TEST_CASE("RetryController centralizes retry safety and exponential backoff",
          "[http][resilience][retry]") {
    RetryController retries(RetryPolicy{
        .max_retries = 3,
        .initial_backoff = 500ms,
        .maximum_backoff = 30s,
        .minimum_delay = 0ms,
        .server_delay_padding = 100ms,
        .jitter_ratio = 0.0,
    });

    CHECK_FALSE(retries.schedule(false, false).has_value());
    CHECK_FALSE(retries.schedule(true, true).has_value());

    const auto first = retries.schedule(true, false);
    REQUIRE(first.has_value());
    CHECK(first->attempt == 1);
    CHECK(first->delay == 500ms);

    const auto second = retries.schedule(true, false);
    REQUIRE(second.has_value());
    CHECK(second->attempt == 2);
    CHECK(second->delay == 1000ms);

    const auto third = retries.schedule(true, false, 3s);
    REQUIRE(third.has_value());
    CHECK(third->attempt == 3);
    CHECK(third->delay == 3100ms);
    CHECK_FALSE(retries.schedule(true, false).has_value());
}

TEST_CASE("Terminal-event requirements are protocol capabilities",
          "[http][resilience][protocol]") {
    CHECK_FALSE(core::llm::protocols::OpenAIProtocol{}
                    .requires_terminal_event());
    CHECK(core::llm::protocols::AnthropicProtocol{}
              .requires_terminal_event());
}

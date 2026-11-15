#include <catch2/catch_test_macros.hpp>

#include "core/llm/protocols/ClaudeProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/transport/StreamResilience.hpp"

using namespace std::chrono_literals;
using core::llm::transport::RetryController;
using core::llm::transport::RetryPolicy;
using core::llm::transport::StreamOutputState;
using core::llm::transport::StreamTimeoutKind;
using core::llm::transport::StreamTimeoutPolicy;
using core::llm::transport::StreamWatchdog;
using core::llm::transport::timeout_user_message;

TEST_CASE("Retry jitter never shortens the server retry deadline", "[http][resilience][retry]") {
    for (int i = 0; i < 100; ++i) {
        RetryController retries(RetryPolicy{.server_delay_padding = 0ms, .jitter_ratio = 0.99});
        const auto retry = retries.schedule(true, StreamOutputState::None, 10s);
        REQUIRE(retry);
        CHECK(retry->delay >= 10s);
    }
}

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

    CHECK_FALSE(retries.schedule(false, StreamOutputState::None).has_value());
    CHECK_FALSE(retries.schedule(true, StreamOutputState::Visible).has_value());

    const auto first = retries.schedule(true, StreamOutputState::None);
    REQUIRE(first.has_value());
    CHECK(first->attempt == 1);
    CHECK(first->delay == 500ms);

    const auto second = retries.schedule(true, StreamOutputState::None);
    REQUIRE(second.has_value());
    CHECK(second->attempt == 2);
    CHECK(second->delay == 1000ms);

    const auto third = retries.schedule(true, StreamOutputState::None, 3s);
    REQUIRE(third.has_value());
    CHECK(third->attempt == 3);
    CHECK(third->delay == 3100ms);
    CHECK_FALSE(retries.schedule(true, StreamOutputState::None).has_value());
}

TEST_CASE("RetryController replays a stream cut during the thinking phase",
          "[http][resilience][retry]") {
    RetryController retries(RetryPolicy{
        .max_retries = 2,
        .initial_backoff = 500ms,
        .maximum_backoff = 30s,
        .minimum_delay = 0ms,
        .server_delay_padding = 100ms,
        .jitter_ratio = 0.0,
    });

    // The model's thinking is retractable: the attempt reset drops it, so a
    // replay cannot duplicate anything the caller kept. A gateway that closes
    // a long-lived SSE connection mid-thinking costs the turn otherwise.
    const auto first = retries.schedule(true, StreamOutputState::ThinkingOnly);
    REQUIRE(first.has_value());
    CHECK(first->attempt == 1);
    const auto second = retries.schedule(true, StreamOutputState::ThinkingOnly);
    REQUIRE(second.has_value());
    CHECK(second->attempt == 2);
    CHECK_FALSE(retries.schedule(true, StreamOutputState::ThinkingOnly).has_value());
}

TEST_CASE("RetryController can restart after streamed output",
          "[http][resilience][retry]") {
    RetryController after_output(RetryPolicy{
        .max_retries = 2,
        .initial_backoff = 500ms,
        .maximum_backoff = 30s,
        .minimum_delay = 0ms,
        .server_delay_padding = 100ms,
        .jitter_ratio = 0.0,
        .retry_only_before_output = false,
    });
    const auto first = after_output.schedule(true, StreamOutputState::Visible);
    REQUIRE(first.has_value());
    CHECK(first->attempt == 1);
    CHECK(after_output.schedule(true, StreamOutputState::Visible).has_value());
    CHECK_FALSE(after_output.schedule(true, StreamOutputState::Visible).has_value());
}

TEST_CASE("StreamWatchdog bounds one attempt by its lifetime",
          "[http][resilience][timeout]") {
    const StreamWatchdog::TimePoint started{};

    // Activity keeps the idle deadline alive, but it cannot keep the attempt
    // alive forever: a stream that trickles would otherwise hold the turn
    // open without end.
    StreamWatchdog watchdog(
        StreamTimeoutPolicy{.response_start = 1s, .inactivity = 10s, .max_lifetime = 500ms},
        started);

    REQUIRE(watchdog.observe_activity(started + 100ms));
    CHECK(watchdog.poll(started + 400ms));
    CHECK_FALSE(watchdog.observe_activity(started + 500ms));
    CHECK(watchdog.timeout_kind() == StreamTimeoutKind::MaxLifetime);
    CHECK(std::string_view(timeout_user_message(StreamTimeoutKind::MaxLifetime))
          == "stream ran past the maximum lifetime of one response");
}

TEST_CASE("Terminal-event requirements are protocol capabilities",
          "[http][resilience][protocol]") {
    CHECK_FALSE(core::llm::protocols::OpenAIProtocol{}
                    .requires_terminal_event());
    CHECK(core::llm::protocols::ClaudeProtocol{}
              .requires_terminal_event());
}

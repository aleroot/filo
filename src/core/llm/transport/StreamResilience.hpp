#pragma once

#include <atomic>
#include <chrono>
#include <optional>
#include <string_view>

namespace core::llm::transport {

enum class StreamTimeoutKind {
    None,
    ResponseStart,
    Inactivity,
};

struct StreamTimeoutPolicy {
    std::chrono::milliseconds response_start = std::chrono::seconds(120);
    std::chrono::milliseconds inactivity = std::chrono::seconds(240);
};

/** Tracks response-start and inter-activity deadlines for one stream attempt. */
class StreamWatchdog {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    explicit StreamWatchdog(
        StreamTimeoutPolicy policy,
        TimePoint started_at = Clock::now()) noexcept;

    /** Check the current deadline without recording transport activity. */
    [[nodiscard]] bool poll(TimePoint now = Clock::now()) noexcept;

    /** Check the deadline, then record a header/body activity timestamp. */
    [[nodiscard]] bool observe_activity(TimePoint now = Clock::now()) noexcept;

    [[nodiscard]] StreamTimeoutKind timeout_kind() const noexcept {
        return timeout_kind_;
    }

private:
    [[nodiscard]] bool expire_if_needed(TimePoint now) noexcept;

    StreamTimeoutPolicy policy_;
    TimePoint started_at_;
    TimePoint last_activity_at_;
    bool response_started_ = false;
    StreamTimeoutKind timeout_kind_ = StreamTimeoutKind::None;
};

[[nodiscard]] std::string_view timeout_log_label(
    StreamTimeoutKind kind) noexcept;
[[nodiscard]] std::string_view timeout_user_message(
    StreamTimeoutKind kind) noexcept;

struct RetryPolicy {
    int max_retries = 3;
    std::chrono::milliseconds initial_backoff = std::chrono::milliseconds(500);
    std::chrono::milliseconds maximum_backoff = std::chrono::seconds(30);
    std::chrono::milliseconds minimum_delay = std::chrono::milliseconds(100);
    std::chrono::milliseconds server_delay_padding =
        std::chrono::milliseconds(100);
    double jitter_ratio = 0.25;
};

struct RetrySchedule {
    int attempt = 0;
    int max_retries = 0;
    std::chrono::milliseconds delay{};
};

/** Owns retry accounting and backoff calculation for one logical request. */
class RetryController {
public:
    explicit RetryController(RetryPolicy policy = {}) noexcept;

    [[nodiscard]] std::optional<RetrySchedule> schedule(
        bool classified_retryable,
        bool output_emitted,
        std::chrono::seconds server_delay = {});

    [[nodiscard]] int retries_completed() const noexcept {
        return retries_completed_;
    }

private:
    [[nodiscard]] std::chrono::milliseconds calculate_delay(
        std::chrono::seconds server_delay) const;

    RetryPolicy policy_;
    int retries_completed_ = 0;
};

[[nodiscard]] bool wait_for_retry(
    std::chrono::milliseconds delay,
    const std::atomic_bool& cancel_requested);

struct HttpStreamResiliencePolicy {
    StreamTimeoutPolicy timeouts;
    RetryPolicy retries;
};

} // namespace core::llm::transport

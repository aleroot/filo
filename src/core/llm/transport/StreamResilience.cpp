#include "StreamResilience.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <thread>

namespace core::llm::transport {

StreamWatchdog::StreamWatchdog(
    StreamTimeoutPolicy policy,
    TimePoint started_at) noexcept
    : policy_(policy)
    , started_at_(started_at)
    , last_activity_at_(started_at) {}

bool StreamWatchdog::poll(TimePoint now) noexcept {
    return !expire_if_needed(now);
}

bool StreamWatchdog::observe_activity(TimePoint now) noexcept {
    if (expire_if_needed(now)) return false;
    response_started_ = true;
    last_activity_at_ = now;
    return true;
}

bool StreamWatchdog::expire_if_needed(TimePoint now) noexcept {
    if (timeout_kind_ != StreamTimeoutKind::None) return true;

    if (!response_started_) {
        if (policy_.response_start.count() > 0
            && now - started_at_ >= policy_.response_start) {
            timeout_kind_ = StreamTimeoutKind::ResponseStart;
            return true;
        }
        return false;
    }

    if (policy_.inactivity.count() > 0
        && now - last_activity_at_ >= policy_.inactivity) {
        timeout_kind_ = StreamTimeoutKind::Inactivity;
        return true;
    }
    return false;
}

std::string_view timeout_log_label(StreamTimeoutKind kind) noexcept {
    switch (kind) {
        case StreamTimeoutKind::ResponseStart:
            return "response start timeout";
        case StreamTimeoutKind::Inactivity:
            return "stream inactivity timeout";
        case StreamTimeoutKind::None:
            return {};
    }
    return {};
}

std::string_view timeout_user_message(StreamTimeoutKind kind) noexcept {
    switch (kind) {
        case StreamTimeoutKind::ResponseStart:
            return "response did not start before the request timeout";
        case StreamTimeoutKind::Inactivity:
            return "stream was inactive past the idle timeout";
        case StreamTimeoutKind::None:
            return {};
    }
    return {};
}

RetryController::RetryController(RetryPolicy policy) noexcept
    : policy_(policy) {
    policy_.max_retries = std::max(0, policy_.max_retries);
    policy_.jitter_ratio = std::clamp(policy_.jitter_ratio, 0.0, 1.0);
    policy_.initial_backoff = std::max(
        std::chrono::milliseconds{}, policy_.initial_backoff);
    policy_.maximum_backoff = std::max(
        policy_.initial_backoff, policy_.maximum_backoff);
    policy_.minimum_delay = std::max(
        std::chrono::milliseconds{}, policy_.minimum_delay);
    policy_.server_delay_padding = std::max(
        std::chrono::milliseconds{}, policy_.server_delay_padding);
}

std::optional<RetrySchedule> RetryController::schedule(
    bool classified_retryable,
    bool output_emitted,
    std::chrono::seconds server_delay) {
    if (!classified_retryable || output_emitted
        || retries_completed_ >= policy_.max_retries) {
        return std::nullopt;
    }

    ++retries_completed_;
    return RetrySchedule{
        .attempt = retries_completed_,
        .max_retries = policy_.max_retries,
        .delay = calculate_delay(server_delay),
    };
}

std::chrono::milliseconds RetryController::calculate_delay(
    std::chrono::seconds server_delay) const {
    std::chrono::milliseconds delay;
    if (server_delay.count() > 0) {
        delay = std::chrono::duration_cast<std::chrono::milliseconds>(
                    server_delay)
            + policy_.server_delay_padding;
    } else {
        delay = policy_.initial_backoff;
        for (int retry = 1; retry < retries_completed_; ++retry) {
            if (delay >= policy_.maximum_backoff / 2) {
                delay = policy_.maximum_backoff;
                break;
            }
            delay *= 2;
        }
        delay = std::min(delay, policy_.maximum_backoff);
    }

    if (policy_.jitter_ratio > 0.0 && delay.count() > 0) {
        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> distribution(
            -policy_.jitter_ratio, policy_.jitter_ratio);
        delay = std::chrono::milliseconds(static_cast<std::int64_t>(
            std::llround(static_cast<double>(delay.count())
                         * (1.0 + distribution(rng)))));
    }
    return std::max(delay, policy_.minimum_delay);
}

bool wait_for_retry(
    std::chrono::milliseconds delay,
    const std::atomic_bool& cancel_requested) {
    constexpr auto kPollInterval = std::chrono::milliseconds(100);
    while (delay.count() > 0) {
        if (cancel_requested.load(std::memory_order_acquire)) return false;
        const auto slice = std::min(delay, kPollInterval);
        std::this_thread::sleep_for(slice);
        delay -= slice;
    }
    return !cancel_requested.load(std::memory_order_acquire);
}

} // namespace core::llm::transport

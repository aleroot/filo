#include <catch2/catch_test_macros.hpp>

#include "core/llm/routing/ProviderHealth.hpp"

#include <chrono>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
using core::llm::routing::CooldownKind;
using core::llm::routing::ProviderHealthRegistry;
using core::llm::routing::rate_limit_cooldown_deadline;
using Clock = ProviderHealthRegistry::Clock;

namespace {

core::llm::protocols::RateLimitInfo limited_info(int retry_after_s,
                                                 bool limited = true) {
    core::llm::protocols::RateLimitInfo info;
    info.is_rate_limited = limited;
    info.retry_after = retry_after_s;
    return info;
}

core::llm::protocols::UsageWindow window(std::string label, float utilization,
                                         int64_t resets_at) {
    return core::llm::protocols::UsageWindow{
        .label = std::move(label),
        .utilization = utilization,
        .resets_at = resets_at,
    };
}

Clock::time_point at_seconds(int64_t unix_seconds) {
    return Clock::time_point{std::chrono::seconds{unix_seconds}};
}

int64_t unix_of(Clock::time_point tp) {
    return std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch())
        .count();
}

} // namespace

// ─── rate_limit_cooldown_deadline ─────────────────────────────────────────────

TEST_CASE("rate_limit_cooldown_deadline", "[provider_health]") {
    const auto now = at_seconds(1'000'000);

    SECTION("non-limited info yields no deadline") {
        CHECK_FALSE(rate_limit_cooldown_deadline(limited_info(30, false), now)
                        .has_value());

        core::llm::protocols::RateLimitInfo unified;
        unified.unified_status = "allowed";
        CHECK_FALSE(rate_limit_cooldown_deadline(unified, now).has_value());
    }

    SECTION("retry_after drives the deadline") {
        const auto deadline = rate_limit_cooldown_deadline(limited_info(120), now);
        REQUIRE(deadline.has_value());
        CHECK(*deadline == now + 120s);
    }

    SECTION("saturated usage windows drive the deadline") {
        const auto reset_5h = now + 90min;
        core::llm::protocols::RateLimitInfo info;
        info.unified_status = "rate_limited";
        info.usage_windows.push_back(window("5h", 1.0f, unix_of(reset_5h)));
        // A partially-used 7d window must not extend the cooldown.
        info.usage_windows.push_back(window("7d", 0.4f, unix_of(now + 48h)));

        const auto deadline = rate_limit_cooldown_deadline(info, now);
        REQUIRE(deadline.has_value());
        CHECK(*deadline == reset_5h);
    }

    SECTION("unified rate_limited without timestamps falls back") {
        core::llm::protocols::RateLimitInfo info;
        info.unified_status = "rate_limited";

        const auto deadline = rate_limit_cooldown_deadline(info, now);
        REQUIRE(deadline.has_value());
        CHECK(*deadline == now + ProviderHealthRegistry::kFallbackRateLimitCooldown);
    }

    SECTION("deadline is clamped to the maximum") {
        core::llm::protocols::RateLimitInfo info;
        info.is_rate_limited = true;
        info.usage_windows.push_back(window("7d", 1.0f, unix_of(now + std::chrono::days{30})));

        const auto deadline = rate_limit_cooldown_deadline(info, now);
        REQUIRE(deadline.has_value());
        CHECK(*deadline == now + ProviderHealthRegistry::kMaxRateLimitCooldown);
    }
}

// ─── Registry cooldown lifecycle ──────────────────────────────────────────────

TEST_CASE("ProviderHealthRegistry cooldown lifecycle", "[provider_health]") {
    ProviderHealthRegistry registry;
    const auto now = at_seconds(2'000'000);

    SECTION("unknown provider is available") {
        const auto snap = registry.snapshot("unknown");
        CHECK(snap.kind == CooldownKind::None);
        CHECK(snap.available(now));
    }

    SECTION("rate limit puts the provider in cooldown until the deadline") {
        registry.observe_rate_limit("anthropic", limited_info(3600), now);

        const auto snap = registry.snapshot("anthropic");
        CHECK(snap.kind == CooldownKind::RateLimit);
        CHECK_FALSE(snap.available(now));
        CHECK(snap.available(now + 3601s));
        CHECK(snap.cooldown_remaining(now) == 3600s);
        CHECK(snap.reason.find("rate limited") != std::string::npos);
    }

    SECTION("non-limited report clears a rate-limit cooldown") {
        registry.observe_rate_limit("p", limited_info(600), now);
        registry.observe_rate_limit("p", limited_info(0, false), now + 10s);
        CHECK(registry.snapshot("p").kind == CooldownKind::None);
    }

    SECTION("stale limited reports never shorten an active cooldown") {
        registry.observe_rate_limit("p", limited_info(600), now);
        registry.observe_rate_limit("p", limited_info(30), now);
        CHECK(registry.snapshot("p").cooldown_remaining(now) == 600s);
    }

    SECTION("success clears cooldown and failure counts") {
        registry.observe_rate_limit("p", limited_info(600), now);
        registry.observe_failure("p", now);
        registry.observe_failure("p", now);

        registry.observe_success("p");

        const auto snap = registry.snapshot("p");
        CHECK(snap.kind == CooldownKind::None);
        CHECK(snap.consecutive_failures == 0);
    }
}

// ─── Circuit breaker ──────────────────────────────────────────────────────────

TEST_CASE("ProviderHealthRegistry circuit breaker", "[provider_health]") {
    ProviderHealthRegistry registry;
    const auto now = at_seconds(3'000'000);

    SECTION("below threshold stays available while counting failures") {
        for (int i = 0; i < ProviderHealthRegistry::kBreakerFailureThreshold - 1; ++i) {
            registry.observe_failure("p", now);
        }
        const auto snap = registry.snapshot("p");
        CHECK(snap.kind == CooldownKind::None);
        CHECK(snap.consecutive_failures
              == ProviderHealthRegistry::kBreakerFailureThreshold - 1);
    }

    SECTION("threshold trips escalating cooldowns") {
        for (int i = 0; i < ProviderHealthRegistry::kBreakerFailureThreshold; ++i) {
            registry.observe_failure("p", now);
        }
        auto snap = registry.snapshot("p");
        REQUIRE(snap.kind == CooldownKind::Breaker);
        CHECK(snap.cooldown_remaining(now)
              == ProviderHealthRegistry::kBreakerBaseCooldown);

        registry.observe_failure("p", now); // one past the threshold doubles it
        snap = registry.snapshot("p");
        CHECK(snap.cooldown_remaining(now)
              == ProviderHealthRegistry::kBreakerBaseCooldown * 2);
    }

    SECTION("cooldown escalates no further than the maximum") {
        for (int i = 0; i < 20; ++i) {
            registry.observe_failure("p", now);
        }
        CHECK(registry.snapshot("p").cooldown_remaining(now)
              == ProviderHealthRegistry::kBreakerMaxCooldown);
    }

    SECTION("rate-limit cooldown wins over the breaker") {
        registry.observe_rate_limit("p", limited_info(600), now);
        for (int i = 0; i < ProviderHealthRegistry::kBreakerFailureThreshold + 2; ++i) {
            registry.observe_failure("p", now);
        }
        const auto snap = registry.snapshot("p");
        CHECK(snap.kind == CooldownKind::RateLimit);
        CHECK(snap.cooldown_remaining(now) == 600s);
    }
}

// ─── Soonest reset ────────────────────────────────────────────────────────────

TEST_CASE("ProviderHealthRegistry soonest_rate_limit_reset", "[provider_health]") {
    ProviderHealthRegistry registry;
    const auto now = at_seconds(4'000'000);

    SECTION("picks the earliest future rate-limit reset") {
        registry.observe_rate_limit("slow", limited_info(600), now);
        registry.observe_rate_limit("fast", limited_info(60), now);

        std::string provider;
        const auto soonest = registry.soonest_rate_limit_reset(now, &provider);
        REQUIRE(soonest.has_value());
        CHECK(provider == "fast");
        CHECK(*soonest == now + 60s);
    }

    SECTION("ignores expired cooldowns and breaker cooldowns") {
        registry.observe_rate_limit("expired", limited_info(1), now);
        for (int i = 0; i < ProviderHealthRegistry::kBreakerFailureThreshold; ++i) {
            registry.observe_failure("breaker-only", now);
        }

        CHECK_FALSE(registry.soonest_rate_limit_reset(now + 5s).has_value());
    }
}

// ─── Concurrency smoke ────────────────────────────────────────────────────────

TEST_CASE("ProviderHealthRegistry concurrent access", "[provider_health]") {
    ProviderHealthRegistry registry;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < 200; ++i) {
                registry.observe_rate_limit("p" + std::to_string(t % 2),
                                            limited_info(i % 7 + 1), Clock::now());
                registry.observe_failure("p" + std::to_string((t + 1) % 2), Clock::now());
                registry.observe_success("s" + std::to_string(t));
                (void)registry.snapshot("p0");
                (void)registry.soonest_rate_limit_reset(Clock::now());
            }
        });
    }
    for (auto& thread : threads) thread.join();

    // No crash/deadlock; state remains queryable.
    CHECK(registry.snapshot("s0").available(Clock::now()));
}

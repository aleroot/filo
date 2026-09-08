#pragma once

#include "../protocols/ApiProtocol.hpp"

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace core::llm::routing {

// ─────────────────────────────────────────────────────────────────────────────
// Provider health — cross-request memory of provider availability.
//
// Rate-limit knowledge used to evaporate after each request: a provider that
// answered 429 with "retry in one hour" was re-selected seconds later by the
// next request.  The registry keeps a small per-provider state machine fed by
// structured RateLimitInfo (parsed from real response headers — no terminal
// scraping) plus a lightweight circuit breaker for generic failures.
//
// All mutating methods take an explicit `now` so behaviour is deterministic
// under test.  Thread-safe: providers stream on worker threads while the
// render thread reads snapshots.
// ─────────────────────────────────────────────────────────────────────────────

enum class CooldownKind {
    None,       ///< Provider considered available.
    RateLimit,  ///< Provider reported a rate limit; usable again at `cooldown_until`.
    Breaker,    ///< Consecutive generic failures tripped the circuit breaker.
};

struct ProviderHealthSnapshot {
    CooldownKind kind = CooldownKind::None;
    std::chrono::system_clock::time_point cooldown_until{};
    std::string reason;
    int consecutive_failures = 0;

    [[nodiscard]] bool available(
        std::chrono::system_clock::time_point now) const noexcept {
        return kind == CooldownKind::None || now >= cooldown_until;
    }

    [[nodiscard]] std::chrono::seconds cooldown_remaining(
        std::chrono::system_clock::time_point now) const noexcept {
        if (kind == CooldownKind::None) return std::chrono::seconds{0};
        const auto remaining = cooldown_until - now;
        if (remaining <= std::chrono::system_clock::duration{0}) {
            return std::chrono::seconds{0};
        }
        return std::chrono::duration_cast<std::chrono::seconds>(remaining);
    }
};

/**
 * @brief Computes when a provider becomes usable again after a rate-limit
 *        response, from structured header data.
 *
 * A response counts as limited when `is_rate_limited` is set or the unified
 * status is "rate_limited".  Candidate deadlines are combined conservatively
 * (the latest wins): `retry_after` seconds, RPM/TPM reset timestamps, and any
 * subscription window whose utilization reached 1.0.  When the provider is
 * limited but reports no usable timestamp, a short fallback cooldown is used
 * so routing still backs off instead of hammering.
 *
 * @return The cooldown deadline, or nullopt when the info does not indicate a
 *         rate limit.  Always clamped to at most 24 h from `now`.
 */
[[nodiscard]] std::optional<std::chrono::system_clock::time_point>
rate_limit_cooldown_deadline(const protocols::RateLimitInfo& info,
                             std::chrono::system_clock::time_point now);

class ProviderHealthRegistry {
public:
    using Clock = std::chrono::system_clock;

    // Circuit breaker: after this many consecutive retryable failures a
    // provider enters an escalating cooldown so chains stop burning attempts.
    static constexpr int kBreakerFailureThreshold = 3;
    static constexpr std::chrono::seconds kBreakerBaseCooldown{30};
    static constexpr std::chrono::seconds kBreakerMaxCooldown{600};

    // Used when a limited response carries no usable reset timestamp, and as
    // the upper bound for any rate-limit cooldown (a week-long 7d window is
    // real, but waiting on it is the wait-policy's decision, not the
    // registry's; routing simply treats the provider as unavailable).
    static constexpr std::chrono::seconds kFallbackRateLimitCooldown{60};
    static constexpr std::chrono::hours kMaxRateLimitCooldown{24};

    /**
     * @brief Records the latest rate-limit state for a provider.
     *
     * Limited info puts the provider into a RateLimit cooldown until the
     * computed deadline.  Non-limited info clears any active RateLimit
     * cooldown (the provider answered normally) but deliberately leaves
     * breaker state untouched — only a successful request does that.
     */
    void observe_rate_limit(std::string_view provider,
                            const protocols::RateLimitInfo& info,
                            Clock::time_point now);

    /// Records one retryable generic failure (circuit breaker accounting).
    void observe_failure(std::string_view provider, Clock::time_point now);

    /// Records a successful request: clears cooldowns and failure counts.
    void observe_success(std::string_view provider);

    [[nodiscard]] ProviderHealthSnapshot snapshot(std::string_view provider) const;

    /**
     * @brief Soonest future RateLimit cooldown expiry across providers.
     *
     * This powers the wait-for-reset policy: "every candidate is limited, but
     * one of them comes back at 15:02".  Breaker cooldowns are excluded — a
     * failing provider is not something to wait for.
     *
     * @param now          Current time; expired cooldowns are ignored.
     * @param provider_out When non-null, receives the provider name.
     */
    [[nodiscard]] std::optional<Clock::time_point> soonest_rate_limit_reset(
        Clock::time_point now,
        std::string* provider_out = nullptr) const;

    /// Clears all state (test helper).
    void reset();

private:
    [[nodiscard]] ProviderHealthSnapshot& entry_locked(const std::string& provider);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, ProviderHealthSnapshot> entries_;
};

} // namespace core::llm::routing

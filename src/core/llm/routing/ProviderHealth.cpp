#include "ProviderHealth.hpp"

#include <algorithm>
#include <format>

namespace core::llm::routing {

namespace {

[[nodiscard]] bool indicates_rate_limit(const protocols::RateLimitInfo& info) noexcept {
    return info.is_rate_limited || info.unified_status == "rate_limited";
}

[[nodiscard]] std::chrono::system_clock::time_point unix_to_time_point(
    int64_t unix_seconds) noexcept {
    return std::chrono::system_clock::time_point{
        std::chrono::seconds{unix_seconds}};
}

} // namespace

std::optional<std::chrono::system_clock::time_point> rate_limit_cooldown_deadline(
    const protocols::RateLimitInfo& info,
    std::chrono::system_clock::time_point now) {
    if (!indicates_rate_limit(info)) return std::nullopt;

    std::optional<std::chrono::system_clock::time_point> deadline;

    const auto consider = [&](std::chrono::system_clock::time_point candidate) {
        if (candidate <= now) return; // stale timestamp — ignore
        deadline = deadline.has_value() ? std::max(*deadline, candidate) : candidate;
    };

    if (info.retry_after > 0) {
        consider(now + std::chrono::seconds{info.retry_after});
    }
    if (info.requests_reset > 0) {
        consider(unix_to_time_point(info.requests_reset));
    }
    if (info.tokens_reset > 0) {
        consider(unix_to_time_point(info.tokens_reset));
    }
    for (const auto& window : info.usage_windows) {
        // Only saturated windows block usage; a 0.4-utilized 7d window must
        // not extend a cooldown triggered by the 5h window.
        if (window.utilization >= 1.0f && window.resets_at > 0) {
            consider(unix_to_time_point(window.resets_at));
        }
    }

    if (!deadline.has_value()) {
        deadline = now + ProviderHealthRegistry::kFallbackRateLimitCooldown;
    }
    return std::min(*deadline,
                    now + ProviderHealthRegistry::kMaxRateLimitCooldown);
}

ProviderHealthSnapshot& ProviderHealthRegistry::entry_locked(
    const std::string& provider) {
    auto [it, inserted] = entries_.try_emplace(provider);
    if (inserted) {
        it->second = ProviderHealthSnapshot{};
    }
    return it->second;
}

void ProviderHealthRegistry::observe_rate_limit(
    std::string_view provider,
    const protocols::RateLimitInfo& info,
    Clock::time_point now) {
    std::lock_guard lock(mutex_);
    auto& snap = entry_locked(std::string{provider});

    const auto deadline = rate_limit_cooldown_deadline(info, now);
    if (deadline.has_value()) {
        // Rate-limit cooldowns are authoritative; never shorten an existing
        // one from an older response, and keep breaker failure counts intact.
        if (snap.kind != CooldownKind::RateLimit || *deadline > snap.cooldown_until) {
            snap.kind = CooldownKind::RateLimit;
            snap.cooldown_until = *deadline;
            if (info.retry_after > 0) {
                snap.reason = std::format("rate limited (retry after {}s)",
                                          info.retry_after);
            } else if (!info.usage_windows.empty()) {
                snap.reason = "rate limited (quota window reset)";
            } else {
                snap.reason = "rate limited";
            }
        }
        return;
    }

    // A non-limited report clears a rate-limit cooldown but not breaker state.
    if (snap.kind == CooldownKind::RateLimit) {
        snap.kind = CooldownKind::None;
        snap.cooldown_until = {};
        snap.reason.clear();
    }
}

void ProviderHealthRegistry::observe_failure(std::string_view provider,
                                             Clock::time_point now) {
    std::lock_guard lock(mutex_);
    auto& snap = entry_locked(std::string{provider});

    ++snap.consecutive_failures;
    if (snap.consecutive_failures < kBreakerFailureThreshold) return;
    if (snap.kind == CooldownKind::RateLimit) return; // rate limit owns the cooldown

    const auto escalation = static_cast<unsigned long>(
        snap.consecutive_failures - kBreakerFailureThreshold);
    auto cooldown = kBreakerBaseCooldown * (1u << std::min<unsigned long>(escalation, 5u));
    cooldown = std::min(cooldown, kBreakerMaxCooldown);

    snap.kind = CooldownKind::Breaker;
    snap.cooldown_until = now + cooldown;
    snap.reason = std::format("{} consecutive failures",
                              snap.consecutive_failures);
}

void ProviderHealthRegistry::observe_success(std::string_view provider) {
    std::lock_guard lock(mutex_);
    auto& snap = entry_locked(std::string{provider});
    snap = ProviderHealthSnapshot{};
}

ProviderHealthSnapshot ProviderHealthRegistry::snapshot(
    std::string_view provider) const {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(std::string{provider});
    return it != entries_.end() ? it->second : ProviderHealthSnapshot{};
}

std::optional<ProviderHealthRegistry::Clock::time_point>
ProviderHealthRegistry::soonest_rate_limit_reset(
    Clock::time_point now,
    std::string* provider_out) const {
    std::lock_guard lock(mutex_);

    std::optional<Clock::time_point> soonest;
    std::string soonest_provider;
    for (const auto& [name, snap] : entries_) {
        if (snap.kind != CooldownKind::RateLimit) continue;
        if (snap.cooldown_until <= now) continue; // already expired
        if (!soonest.has_value() || snap.cooldown_until < *soonest) {
            soonest = snap.cooldown_until;
            soonest_provider = name;
        }
    }

    if (soonest.has_value() && provider_out != nullptr) {
        *provider_out = soonest_provider;
    }
    return soonest;
}

void ProviderHealthRegistry::reset() {
    std::lock_guard lock(mutex_);
    entries_.clear();
}

} // namespace core::llm::routing

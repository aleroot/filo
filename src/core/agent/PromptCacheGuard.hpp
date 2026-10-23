#pragma once

#include "../llm/PromptCacheProfile.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string_view>

namespace core::config {
struct AppConfig;
}

namespace core::agent {

/// What happens when an idle heavy cache is about to expire.
enum class PromptCacheKeepWarm : std::uint8_t {
    /// Ask: keep warm, compact now, or let it expire.
    Ask,
    /// Renew it without asking while that is cheaper than letting it go cold
    /// (PromptCacheGuard::keep_warm_budget), then ask as usual.
    Auto,
};

/// The `prompt_cache_keep_warm` spellings.
inline constexpr std::string_view kPromptCacheKeepWarmAsk = "ask";
inline constexpr std::string_view kPromptCacheKeepWarmAuto = "auto";

/// User-tunable thresholds for cold-cache protection. Pure data: the policy
/// below never reads configuration or clocks on its own.
struct PromptCacheGuardSettings {
    /// Context size from which a cold cache is worth a question. Zero turns
    /// every cold-cache prompt off.
    std::int64_t min_context_tokens = 100'000;
    /// How long before expiry an idle heavy session is offered to keep its
    /// cache warm or compact. Zero turns the expiry notice off.
    std::chrono::seconds notice_lead{std::chrono::minutes{1}};
    /// Applies at the notice instant, so a zero notice_lead turns it off too.
    PromptCacheKeepWarm keep_warm = PromptCacheKeepWarm::Ask;

    [[nodiscard]] constexpr bool enabled() const noexcept {
        return min_context_tokens > 0;
    }
};

/// The configured thresholds: `prompt_cache_guard_tokens`,
/// `prompt_cache_notice_seconds` and `prompt_cache_keep_warm`, each falling
/// back to the default above.
[[nodiscard]] PromptCacheGuardSettings prompt_cache_guard_settings(
    const core::config::AppConfig& config);

/// What a session's prompt cache looks like, as the agent knows it.
struct PromptCacheState {
    /// When the conversation last touched the provider's cache: the start of
    /// the latest successful model request. Unset before the first exchange
    /// and after a history rewrite (compaction, clear) made it irrelevant.
    std::optional<std::chrono::system_clock::time_point> last_exchange_at;
    /// Tokens the next request re-reads.
    std::int64_t context_tokens = 0;
    core::llm::PromptCacheProfile profile;
};

enum class PromptCacheWarmth {
    /// No billed remote cache, or no exchange yet: nothing can go cold.
    Untracked,
    Warm,
    Expired,
};

struct PromptCacheAssessment {
    PromptCacheWarmth warmth = PromptCacheWarmth::Untracked;
    std::chrono::seconds idle{0};
    /// Time left before expiry; zero once expired.
    std::chrono::seconds remaining{0};
    std::int64_t context_tokens = 0;
    core::llm::PromptCacheProfile profile;

    [[nodiscard]] constexpr bool expired() const noexcept {
        return warmth == PromptCacheWarmth::Expired;
    }
};

/// When an idle session needs attention. Both instants are absolute so a
/// scheduler can sleep until them without polling.
struct PromptCacheDeadlines {
    std::chrono::system_clock::time_point notice;
    std::chrono::system_clock::time_point expiry;
};

/// Decides, from a state snapshot and a clock reading, whether a session's
/// prompt cache is warm, and when it needs the user. Stateless and free of
/// I/O so every front end shares one rule and tests drive it with any clock.
class PromptCacheGuard {
public:
    explicit constexpr PromptCacheGuard(PromptCacheGuardSettings settings) noexcept
        : settings_(settings) {}

    [[nodiscard]] constexpr const PromptCacheGuardSettings& settings() const noexcept {
        return settings_;
    }

    [[nodiscard]] static PromptCacheAssessment assess(
        const PromptCacheState& state,
        std::chrono::system_clock::time_point now) noexcept;

    /// True when a prompt about to be sent would re-write a heavy cache that
    /// has expired.
    [[nodiscard]] bool should_confirm_send(
        const PromptCacheAssessment& assessment) const noexcept;

    /// The notice and expiry instants for an idle heavy session, or nullopt
    /// when the notice is off, the session is light or untracked, or the
    /// lead does not fit inside the cache lifetime.
    [[nodiscard]] std::optional<PromptCacheDeadlines> deadlines(
        const PromptCacheState& state) const noexcept;

    /// Keeping a cache warm means re-sending the last request; it is offered
    /// only where that renews the lifetime without side effects.
    [[nodiscard]] static constexpr bool can_keep_warm(
        const core::llm::PromptCacheProfile& profile) noexcept {
        return profile.remote() && profile.replay_renews;
    }

    /// How many replays in a row keep a cache warm for less than the cold
    /// write they avoid. A cold prompt pays a write instead of a read, so the
    /// saving is (write - read) and each replay costs one read: 11 replays on
    /// a 5-minute Anthropic cache (1.25x vs 0.1x), 19 on a 1-hour one. Zero
    /// where keeping warm is impossible or the prices are unknown.
    [[nodiscard]] static constexpr int keep_warm_budget(
        const core::llm::PromptCacheProfile& profile) noexcept {
        const int read = profile.read_cost_percent;
        const int write = profile.write_cost_percent;
        if (!can_keep_warm(profile) || read == 0 || write <= read) return 0;
        return (write - read) / read;
    }

    /// True when the cache should be renewed without asking, @p replays
    /// automatic renewals having already run since the user last sent.
    [[nodiscard]] constexpr bool should_keep_warm(
        const core::llm::PromptCacheProfile& profile, int replays) const noexcept {
        return settings_.keep_warm == PromptCacheKeepWarm::Auto
            && replays < keep_warm_budget(profile);
    }

private:
    [[nodiscard]] bool is_heavy(std::int64_t context_tokens) const noexcept {
        return settings_.enabled() && context_tokens >= settings_.min_context_tokens;
    }

    PromptCacheGuardSettings settings_;
};

} // namespace core::agent

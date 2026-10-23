#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string_view>

namespace core::llm {

/// How long the provider is asked to keep the prompt cache. Only protocols
/// that expose a choice on the wire honour Extended (Anthropic: `"ttl":"1h"`);
/// everywhere else the provider's own retention applies.
enum class PromptCacheTtl : std::uint8_t {
    Standard,
    Extended,
};

/// Accepts the provider-config spellings: "" / "5m" / "standard" and
/// "1h" / "extended". Anything else is nullopt so a typo is reported rather
/// than silently paying for the wrong retention.
[[nodiscard]] std::optional<PromptCacheTtl> parse_prompt_cache_ttl(
    std::string_view value) noexcept;

/// Where a profile's lifetime comes from. The wording shown to the user
/// follows it: a documented lifetime "has expired", an estimated one "has
/// likely expired".
enum class PromptCacheBasis : std::uint8_t {
    Documented,
    Estimated,
    Observed,
};

/// What a provider's prompt cache does with an idle conversation. Owned by
/// the wire protocol, which is the only component that knows what it asked
/// the server for.
struct PromptCacheProfile {
    /// Lifetime after the last request that read or wrote the cache; every
    /// hit restarts it. Zero means there is no billed remote cache to lose
    /// (local inference, unknown providers).
    std::chrono::seconds ttl{0};
    /// Re-sending the last request renews the lifetime without side effects
    /// on conversation state, so it can keep a cache warm. False by default:
    /// protocols that chain server-side responses (previous_response_id)
    /// would chain onto the replay, and models that reason before their
    /// first visible token would bill that reasoning.
    bool replay_renews = false;
    PromptCacheBasis basis = PromptCacheBasis::Estimated;
    /// What reading and writing the cached prefix costs, in percent of the
    /// uncached input price; zero when the provider publishes no price.
    std::uint16_t read_cost_percent = 0;
    std::uint16_t write_cost_percent = 0;

    [[nodiscard]] constexpr bool remote() const noexcept {
        return ttl.count() > 0;
    }

    [[nodiscard]] static constexpr PromptCacheProfile none() noexcept {
        return {};
    }

    [[nodiscard]] static constexpr PromptCacheProfile estimated(
        std::chrono::seconds lifetime) noexcept {
        return {.ttl = lifetime, .basis = PromptCacheBasis::Estimated};
    }

    [[nodiscard]] static constexpr PromptCacheProfile documented(
        std::chrono::seconds lifetime, bool replay_renews) noexcept {
        return {
            .ttl = lifetime,
            .replay_renews = replay_renews,
            .basis = PromptCacheBasis::Documented,
        };
    }

    /// This profile with the provider's published cache prices.
    [[nodiscard]] constexpr PromptCacheProfile with_costs(
        std::uint16_t read_percent, std::uint16_t write_percent) const noexcept {
        auto priced = *this;
        priced.read_cost_percent = read_percent;
        priced.write_cost_percent = write_percent;
        return priced;
    }

    bool operator==(const PromptCacheProfile&) const = default;
};

/// Lifetime assumed for a remote provider that documents none. Five minutes
/// is the shortest common lifetime (Anthropic default, OpenAI in-memory,
/// DashScope explicit), so the estimate errs towards warning.
inline constexpr std::chrono::seconds kEstimatedPromptCacheTtl{std::chrono::minutes{5}};

} // namespace core::llm

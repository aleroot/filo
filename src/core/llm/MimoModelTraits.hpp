#pragma once

/**
 * @file MimoModelTraits.hpp
 * @brief Xiaomi MiMo endpoint, region, and gateway-error vocabulary.
 *
 * MiMo ships one model family behind two commercial surfaces:
 *   - pay-as-you-go  `https://api.xiaomimimo.com/v1`
 *   - Token Plan     `https://token-plan-{cn,ams,sgp}.xiaomimimo.com/v1`
 *
 * Both speak OpenAI Chat Completions, so the wire differences live in
 * MimoProtocol and the commercial differences live in the provider presets.
 * This header holds the vendor facts both of them need, keeping
 * `api.xiaomimimo.com` knowledge out of the generic OpenAI protocol.
 */

#include <chrono>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::llm {

struct TokenUsage;

/// Regional Token Plan gateway. One subscription is served by all three.
enum class MimoRegion {
    Europe,     ///< token-plan-ams.xiaomimimo.com
    Singapore,  ///< token-plan-sgp.xiaomimimo.com
    China,      ///< token-plan-cn.xiaomimimo.com
};

/// Token Plan base URL for @p region.
[[nodiscard]] std::string_view mimo_token_plan_endpoint(MimoRegion region) noexcept;

/// Built-in provider preset name that serves @p region.
[[nodiscard]] std::string_view mimo_region_provider_name(MimoRegion region) noexcept;

/// Human-facing label for @p region, used in login output.
[[nodiscard]] std::string_view mimo_region_label(MimoRegion region) noexcept;

/// Parse a region name ("ams"/"europe", "sgp"/"singapore", "cn"/"china").
[[nodiscard]] std::optional<MimoRegion> mimo_region_from_string(
    std::string_view value) noexcept;

/**
 * Region used when the caller has no better information.
 *
 * Resolution order: `MIMO_REGION`, then the mainland gateway for a zh_CN
 * locale, then Europe. Mirrors how Kimi Code resolves its regional hosts.
 */
[[nodiscard]] MimoRegion resolve_mimo_region();

/// Region owning @p base_url, when it is a Token Plan gateway.
[[nodiscard]] std::optional<MimoRegion> mimo_region_for_endpoint(
    std::string_view base_url);

/// True when @p base_url is any regional Token Plan gateway.
[[nodiscard]] bool is_mimo_token_plan_endpoint(std::string_view base_url);

/// True when @p base_url is any MiMo-operated host (plan or pay-as-you-go).
[[nodiscard]] bool is_mimo_endpoint(std::string_view base_url);

/// True when @p model is a MiMo model id (`mimo-v2.6-pro`, `MiMo-V2-Flash`, ...).
[[nodiscard]] bool is_mimo_model(std::string_view model) noexcept;

/// Token Plan credits charged per token of one model.
struct MimoTokenPlanRates {
    double cache_hit = 0.0;
    double cache_miss = 0.0;
    double output = 0.0;
};

/// Credit rates of a model the Token Plan serves; nullopt for anything else.
[[nodiscard]] std::optional<MimoTokenPlanRates> mimo_token_plan_rates(
    std::string_view model) noexcept;

/// Every model id the Token Plan serves, in catalog order.
[[nodiscard]] std::vector<std::string_view> mimo_token_plan_model_ids();

/// True inside the Token Plan off-peak window, 00:00-08:00 Beijing time.
[[nodiscard]] bool is_mimo_off_peak(
    std::chrono::system_clock::time_point at) noexcept;

/**
 * Token Plan credits one call consumed: cache hits, cache misses and output
 * (reasoning included) at the model's rates, times 0.8 off-peak. Zero for a
 * model the plan does not serve.
 */
[[nodiscard]] double mimo_token_plan_credits(
    const TokenUsage& usage,
    std::string_view model,
    std::chrono::system_clock::time_point at) noexcept;

/**
 * Console/authorization origin, overridable with `MIMO_PLATFORM_URL` exactly
 * like the reference MiMo Code client.
 */
[[nodiscard]] std::string mimo_platform_url();

/**
 * Friendly label for a MiMo gateway `error.code`.
 *
 * The gateway reports moderation and risk-control blocks under a generic
 * HTTP 400, so the numeric code is the only thing that distinguishes them.
 * Returns an empty view for codes with no special meaning.
 */
[[nodiscard]] std::string_view mimo_gateway_code_label(
    std::string_view code) noexcept;

} // namespace core::llm

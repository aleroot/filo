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

#include <optional>
#include <string>
#include <string_view>

namespace core::llm {

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

#pragma once

/**
 * @file KimiClientIdentity.hpp
 * @brief Managed Kimi Code client identity (platform, version, User-Agent).
 *
 * The Kimi Code subscription API requires `X-Msh-Platform: kimi_code_cli` and
 * a matching `kimi-code-cli/<version>` User-Agent on both OAuth and chat
 * calls. Keep that contract here — the same pattern as
 * `XaiGrokClientIdentity.hpp` — so KimiOAuthFlow, KimiProtocol, and the web
 * search backend cannot drift.
 *
 * Compatibility version audited against MoonshotAI/kimi-code 0.42.0
 * (`apps/kimi-code` package version; OAuth client id and regional hosts from
 * `packages/oauth/src/constants.ts` + `region.ts`).
 */

#include <string>
#include <string_view>

namespace core::auth::kimi_code {

inline constexpr std::string_view kClientId =
    "17e5f671-d194-4dfb-9706-5516cb48c098";
inline constexpr std::string_view kPlatform = "kimi_code_cli";
inline constexpr std::string_view kUserAgentProduct = "kimi-code-cli";
// Bare CLI version sent as X-Msh-Version. Track kimi-code, not Filo.
inline constexpr std::string_view kClientVersion = "0.42.0";

inline constexpr std::string_view kMainlandOAuthHost = "https://auth.kimi.com";
inline constexpr std::string_view kGlobalOAuthHost = "https://auth.kimi.ai";
inline constexpr std::string_view kMainlandCodingBaseUrl =
    "https://api.kimi.com/coding/v1";
inline constexpr std::string_view kGlobalCodingBaseUrl =
    "https://api.kimi.ai/coding/v1";

[[nodiscard]] inline const std::string& user_agent() {
    static const std::string value = std::string(kUserAgentProduct)
        + "/" + std::string(kClientVersion);
    return value;
}

} // namespace core::auth::kimi_code

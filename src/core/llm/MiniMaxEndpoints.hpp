#pragma once

#include <string_view>

namespace core::llm {

// Official MiniMax endpoints, single-sourced for providers, quota, search and
// user-facing links. Credentials are only ever sent to these URLs or to the
// configured generation endpoint; never derive them from a custom base URL.
inline constexpr std::string_view kMiniMaxApiOrigin = "https://api.minimax.io";
inline constexpr std::string_view kMiniMaxBaseUrl = "https://api.minimax.io/anthropic";
// Both plan endpoints authenticate with `Authorization: Bearer`, unlike the
// Messages API (`x-api-key`). The api host answers in about a second; the www
// host can take several seconds on a cold connection.
inline constexpr std::string_view kMiniMaxQuotaUrl = "https://api.minimax.io/v1/token_plan/remains";
inline constexpr std::string_view kMiniMaxSearchUrl = "https://api.minimax.io/v1/coding_plan/search";
// Where a subscriber manages the Subscription Key and sees plan usage.
inline constexpr std::string_view kMiniMaxPlanConsoleUrl = "https://platform.minimax.io/console/plan";
static_assert(kMiniMaxBaseUrl.starts_with(kMiniMaxApiOrigin));
static_assert(kMiniMaxQuotaUrl.starts_with(kMiniMaxApiOrigin));
static_assert(kMiniMaxSearchUrl.starts_with(kMiniMaxApiOrigin));

/// True only for the official Messages base URL, ignoring trailing slashes
/// (which the generation path also ignores). Plan endpoints accept the same
/// key, so they are contacted only then; any other host, path or scheme is a
/// proxy whose credential must never reach MiniMax.
[[nodiscard]] constexpr bool is_official_minimax_base_url(std::string_view base_url) noexcept {
    while (base_url.ends_with('/')) base_url.remove_suffix(1);
    return base_url == kMiniMaxBaseUrl;
}

} // namespace core::llm

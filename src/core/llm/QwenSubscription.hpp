#pragma once

/**
 * @file QwenSubscription.hpp
 * @brief Fixed contract of the Qwen Subscription ("Qwen Account") route.
 *
 * Qwen Code v0.27 (QwenLM/qwen-code, packages/core/src/qwen/oauth-config.ts
 * and subscription-*.ts) serves its Plus/Pro subscription through a device
 * authorization grant at auth.qwen.ai and a fixed OpenAI Chat Completions
 * gateway at api.qwen.ai. Nothing here is configurable on purpose: the bearer
 * token minted for that gateway must never be sent to another host.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace core::llm::qwen_subscription {

/// Provider key, login identifier and token-store key.
inline constexpr std::string_view kProviderName = "qwen-subscription";
inline constexpr std::string_view kAuthType = "oauth_qwen_subscription";
inline constexpr std::string_view kDisplayName = "Qwen Subscription";

inline constexpr std::string_view kBaseUrl = "https://api.qwen.ai/compatible-mode/v1";
inline constexpr std::string_view kWebSearchUrl = "https://api.qwen.ai/api/v1/mcp/web_search";
inline constexpr std::string_view kAuthOrigin = "https://auth.qwen.ai";
inline constexpr std::string_view kWebOrigin = "https://chat.qwen.ai";
inline constexpr std::string_view kClientId = "qwen-code-cli";
inline constexpr std::string_view kScope = "model:invoke profile:read";

inline constexpr std::string_view kSignInRequiredMessage =
    "Your Qwen Subscription credentials are no longer valid. "
    "Run /auth qwen-subscription to sign in again.";

/// Browser page where the user approves @p user_code (`paywallScene=cli`).
[[nodiscard]] std::string authorization_url(std::string_view user_code);

/// Shared plans/upgrade page for the Qwen Code membership.
[[nodiscard]] std::string plans_url();

/// Only Plus and Pro memberships (case-insensitive) include the gateway.
[[nodiscard]] bool membership_qualifies(std::string_view member_type);

/// "plus" → "Plus"; other names keep their spelling apart from the first letter.
[[nodiscard]] std::string display_plan(std::string_view member_type);

/// Thinking budget the gateway reads for a Filo effort; nullopt disables thinking.
/// Unset efforts use the route default (medium).
[[nodiscard]] std::optional<int> thinking_budget(std::string_view effort);

/**
 * Message for a gateway error body, or nullopt when it carries none.
 *
 * HTTP 402 with `error.code = insufficient_quota` is the only quota signal;
 * it becomes weekly-credit guidance using `error.quota.member_type` and
 * `error.quota.reset_at` (Unix seconds) when present. Other errors show the
 * public `error.message`, falling back to `error_description`.
 */
[[nodiscard]] std::optional<std::string> gateway_error_message(
    int status, std::string_view body, std::int64_t now_unix_seconds);

} // namespace core::llm::qwen_subscription

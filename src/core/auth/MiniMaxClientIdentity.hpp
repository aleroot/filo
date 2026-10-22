#pragma once

/**
 * @file MiniMaxClientIdentity.hpp
 * @brief Official MiniMax client identities, one per endpoint family.
 *
 * Filo authenticates with a user-supplied key, so it presents what MiniMax's
 * own clients send on their own-key path — the same pattern as
 * `XaiGrokClientIdentity.hpp` and `KimiClientIdentity.hpp`.
 *
 * - Messages API (generation, token counting): MiniMax Code
 *   (MiniMax-AI/minimax-code, audited at da22f11). Its own-key provider is
 *   pi-mono's Anthropic SDK client (`createClient` in
 *   third_party/pi-mono/packages/ai/src/providers/anthropic.ts, SDK 0.91.1),
 *   plus the `X-Mavis-*` headers of `buildLocalProviderHeaders`
 *   (packages/local-runtime-v2/src/service/model-system/resolution/
 *   model-resolver-helpers.ts). `User-Agent: MiniMaxAgent` is reserved there
 *   for the managed MiniMax-login backend, so it is deliberately not sent.
 * - Plan endpoints (quota, web search): MiniMax Code never calls them; the
 *   official client that does is MiniMax-AI/cli (`mmx-cli`, `request` in
 *   src/client/http.ts).
 */

#include "core/utils/Uuid.hpp"

#include <charconv>
#include <ctime>
#include <string>
#include <string_view>

namespace core::auth::minimax_code {

inline constexpr std::string_view kSessionIdHeader = "X-Mavis-Session-Id";
inline constexpr std::string_view kAgentIdHeader = "X-Mavis-Agent-Id";
inline constexpr std::string_view kTimezoneOffsetHeader = "X-Mavis-Timezone-Offset";
/// minimax-code's id for the primary agent (`readAgentHeaderId` default).
inline constexpr std::string_view kMainAgentId = "main";

/// The Anthropic SDK minimax-code pins (pnpm-lock `@anthropic-ai/sdk@0.91.1`).
inline constexpr std::string_view kAnthropicSdkVersion = "0.91.1";
inline constexpr std::string_view kUserAgent = "Anthropic/JS 0.91.1";
static_assert(kUserAgent.ends_with(kAnthropicSdkVersion));
/// minimax-code runs on the user's Node (engines ">=22.19 <23 || >=24.2 <27");
/// the oldest supported release stands in for it.
inline constexpr std::string_view kNodeRuntimeVersion = "v22.19.0";
/// The SDK's streaming default (600000 ms) in the seconds it reports.
inline constexpr std::string_view kRequestTimeoutSeconds = "600";

/// The SDK's `normalizePlatform(process.platform)`.
[[nodiscard]] inline constexpr std::string_view sdk_platform_os() noexcept {
#if defined(__APPLE__)
    return "MacOS";
#elif defined(_WIN32)
    return "Windows";
#elif defined(__linux__)
    return "Linux";
#elif defined(__FreeBSD__)
    return "FreeBSD";
#else
    return "Unknown";
#endif
}

/// The SDK's `normalizeArch(process.arch)`.
[[nodiscard]] inline constexpr std::string_view sdk_platform_arch() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x64";
#else
    return "unknown";
#endif
}

/// Local UTC offset in seconds east of Greenwich, as minimax-code sends it
/// (`String(new Date().getTimezoneOffset() * -60)`), e.g. "7200" or "-18000".
[[nodiscard]] inline std::string timezone_offset_seconds(std::time_t now = std::time(nullptr)) {
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char offset[16] = {};
    // "%z" is "+hhmm"/"-hhmm"; anything else means the zone is unknown.
    if (std::strftime(offset, sizeof(offset), "%z", &local) != 5
        || (offset[0] != '+' && offset[0] != '-')) {
        return "0";
    }
    int hours = 0, minutes = 0;
    std::from_chars(offset + 1, offset + 3, hours);
    std::from_chars(offset + 3, offset + 5, minutes);
    const int seconds = (hours * 3600 + minutes * 60) * (offset[0] == '-' ? -1 : 1);
    return std::to_string(seconds);
}

/// Stable per-process session when a request carries no session id.
[[nodiscard]] inline const std::string& process_session_id() {
    static const std::string value = core::utils::random_uuid_v4();
    return value;
}

/// The runtime headers minimax-code adds to every model request.
template <typename HeaderMap>
void apply_session_headers(HeaderMap& headers, std::string_view session_id) {
    headers[std::string(kSessionIdHeader)] =
        session_id.empty() ? process_session_id() : std::string(session_id);
    headers[std::string(kAgentIdHeader)] = std::string(kMainAgentId);
    headers[std::string(kTimezoneOffsetHeader)] = timezone_offset_seconds();
}

/// MiniMax Code's own-key Messages request: the SDK's fixed headers, the
/// defaults pi-mono's `createClient` adds, then the session headers.
/// minimax-code never lets the SDK retry (`maxRetries: 0`), so every attempt
/// reports retry count 0. No `anthropic-beta` is sent: pi-mono adds betas
/// only for models without eager tool streaming or adaptive thinking.
template <typename HeaderMap>
void apply_identity_headers(HeaderMap& headers, std::string_view session_id) {
    headers["Accept"] = "application/json";
    headers["User-Agent"] = std::string(kUserAgent);
    headers["X-Stainless-Retry-Count"] = "0";
    headers["X-Stainless-Timeout"] = std::string(kRequestTimeoutSeconds);
    headers["X-Stainless-Lang"] = "js";
    headers["X-Stainless-Package-Version"] = std::string(kAnthropicSdkVersion);
    headers["X-Stainless-OS"] = std::string(sdk_platform_os());
    headers["X-Stainless-Arch"] = std::string(sdk_platform_arch());
    headers["X-Stainless-Runtime"] = "node";
    headers["X-Stainless-Runtime-Version"] = std::string(kNodeRuntimeVersion);
    headers["anthropic-dangerous-direct-browser-access"] = "true";
    apply_session_headers(headers, session_id);
}

} // namespace core::auth::minimax_code

namespace core::auth::minimax_cli {

/// MiniMax-AI/cli's `User-Agent: mmx-cli/${CLI_VERSION}`, at the npm release
/// audited (mmx-cli 1.0.28).
inline constexpr std::string_view kUserAgent = "mmx-cli/1.0.28";

/// mmx-cli's headers for a JSON plan request (it always sets Content-Type).
template <typename HeaderMap>
void apply_identity_headers(HeaderMap& headers) {
    headers["User-Agent"] = std::string(kUserAgent);
    headers["Content-Type"] = "application/json";
}

} // namespace core::auth::minimax_cli

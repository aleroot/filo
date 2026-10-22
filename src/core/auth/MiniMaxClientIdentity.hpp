#pragma once

/**
 * @file MiniMaxClientIdentity.hpp
 * @brief MiniMax Code client identity sent to every MiniMax endpoint.
 *
 * MiniMax's own coding agent (MiniMax-AI/minimax-code) identifies itself with
 * `User-Agent: MiniMaxAgent` plus the `X-Mavis-Session-Id`, `X-Mavis-Agent-Id`
 * and `X-Mavis-Timezone-Offset` headers (`buildLocalProviderHeaders` in
 * packages/local-runtime-v2/src/service/model-system/resolution/
 * model-resolver-helpers.ts). Keep that contract here — the same pattern as
 * `XaiGrokClientIdentity.hpp` and `KimiClientIdentity.hpp` — so generation,
 * token counting, quota and web search cannot drift apart.
 *
 * Audited against minimax-code da22f11.
 */

#include "core/utils/Uuid.hpp"

#include <charconv>
#include <ctime>
#include <string>
#include <string_view>

namespace core::auth::minimax_code {

inline constexpr std::string_view kUserAgent = "MiniMaxAgent";
inline constexpr std::string_view kSessionIdHeader = "X-Mavis-Session-Id";
inline constexpr std::string_view kAgentIdHeader = "X-Mavis-Agent-Id";
inline constexpr std::string_view kTimezoneOffsetHeader = "X-Mavis-Timezone-Offset";
/// minimax-code's id for the primary agent (`readAgentHeaderId` default).
inline constexpr std::string_view kMainAgentId = "main";

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

template <typename HeaderMap>
void apply_identity_headers(HeaderMap& headers, std::string_view session_id) {
    headers["User-Agent"] = std::string(kUserAgent);
    headers[std::string(kSessionIdHeader)] =
        session_id.empty() ? process_session_id() : std::string(session_id);
    headers[std::string(kAgentIdHeader)] = std::string(kMainAgentId);
    headers[std::string(kTimezoneOffsetHeader)] = timezone_offset_seconds();
}

} // namespace core::auth::minimax_code

#pragma once

#include "core/utils/AsciiUtils.hpp"
#include "core/utils/Uuid.hpp"

#include <string>
#include <string_view>

namespace core::auth::xai_grok {

inline constexpr std::string_view kClientIdentifier = "grok-shell";
inline constexpr std::string_view kClientVersion = "1.0.1";
inline constexpr std::string_view kClientMode = "interactive";

[[nodiscard]] inline constexpr std::string_view platform_os() noexcept {
#if defined(__APPLE__)
    return "macos";
#elif defined(_WIN32)
    return "windows";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

[[nodiscard]] inline constexpr std::string_view platform_arch() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unknown";
#endif
}

[[nodiscard]] inline const std::string& user_agent() {
    static const std::string value = std::string(kClientIdentifier)
        + "/" + std::string(kClientVersion)
        + " (" + std::string(platform_os())
        + "; " + std::string(platform_arch()) + ")";
    return value;
}

[[nodiscard]] inline const std::string& process_agent_id() {
    static const std::string value = core::utils::random_uuid_v4();
    return value;
}

template <typename HeaderMap>
void apply_proxy_identity_headers(HeaderMap& headers) {
    headers["X-XAI-Token-Auth"] = "xai-grok-cli";
    headers["x-authenticateresponse"] = "authenticate-response";
    headers["x-grok-client-identifier"] = kClientIdentifier;
    headers["x-grok-client-version"] = kClientVersion;
    headers["x-grok-client-mode"] = kClientMode;
    headers["x-grok-agent-id"] = process_agent_id();
    headers["User-Agent"] = user_agent();
}

template <typename HeaderMap>
void remove_proxy_identity_headers(HeaderMap& headers) {
    for (auto it = headers.begin(); it != headers.end();) {
        const std::string_view name = it->first;
        const bool belongs_to_grok_build =
            core::utils::ascii::istarts_with(name, "x-grok-")
            || core::utils::ascii::iequals(name, "X-XAI-Token-Auth")
            || core::utils::ascii::iequals(name, "x-authenticateresponse")
            || core::utils::ascii::iequals(name, "User-Agent");
        if (belongs_to_grok_build) it = headers.erase(it);
        else ++it;
    }
}

} // namespace core::auth::xai_grok

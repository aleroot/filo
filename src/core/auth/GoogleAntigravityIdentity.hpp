#pragma once

/**
 * @file GoogleAntigravityIdentity.hpp
 * @brief Self-reported identity of the unofficial "Antigravity" client.
 *
 * Both the OAuth/control-plane layer (`core/auth`) and the streaming protocol
 * (`core/llm/protocols`) must present the exact same client identity: the
 * backend gates newer models on the `User-Agent` version and any divergence
 * between control-plane and streaming calls is a detection signal. The helpers
 * live here so neither layer depends on the other for them.
 *
 * Format captured from the real darwin/arm64 `antigravity/hub` client:
 * `antigravity/hub/<version> (aidev_client; os_type=darwin; arch=arm64; cl=963137146)`.
 * os/arch stay pinned to that reference build regardless of the host; the
 * backend does not validate `cl` (only the version gates model access).
 *
 * Environment overrides (all optional):
 *   - `ANTIGRAVITY_CLI_VERSION` — reported client version.
 *   - `ANTIGRAVITY_USER_AGENT`  — full `User-Agent` replacement.
 */

#include <cstdlib>
#include <string>
#include <string_view>

namespace core::auth::antigravity {

/// Cloud Code Assist hosts the real Antigravity client uses, in rotation
/// order (daily first, sandbox as streaming failover).
inline constexpr std::string_view kAntigravityEndpoint =
    "https://daily-cloudcode-pa.googleapis.com";
inline constexpr std::string_view kAntigravitySandboxEndpoint =
    "https://daily-cloudcode-pa.sandbox.googleapis.com";

/// Latest Antigravity client release (update manifest, 2026-10-01). The
/// backend gates newer models on this, so bump it when Google ships a new
/// build; the electron-builder manifest always carries the current value.
inline constexpr std::string_view kAntigravityClientVersion = "2.18.1";

/// Client version reported on the wire (`ANTIGRAVITY_CLI_VERSION` override,
/// else the pinned release).
[[nodiscard]] inline std::string client_version() {
    if (const char* raw = std::getenv("ANTIGRAVITY_CLI_VERSION");
        raw && raw[0] != '\0') {
        return raw;
    }
    return std::string(kAntigravityClientVersion);
}

/// `User-Agent` identifying the Antigravity hub client
/// (`ANTIGRAVITY_USER_AGENT` replaces the whole string).
[[nodiscard]] inline std::string user_agent() {
    if (const char* raw = std::getenv("ANTIGRAVITY_USER_AGENT");
        raw && raw[0] != '\0') {
        return raw;
    }
    return "antigravity/hub/" + client_version()
        + " (aidev_client; os_type=darwin; arch=arm64; cl=963137146)";
}

} // namespace core::auth::antigravity

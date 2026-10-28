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
 * `antigravity/hub/<version> (aidev_client; os_type=darwin; arch=arm64; cl=1007512578)`.
 * os/arch stay pinned to that reference build regardless of the host, and `cl`
 * is the changelist the 2.27.0 `language_server` was built from — it must be
 * bumped together with `kAntigravityClientVersion`, because a
 * version/changelist pair the shipped client never produced is a fingerprint.
 *
 * Environment overrides (all optional):
 *   - `ANTIGRAVITY_CLI_VERSION` — reported client version.
 *   - `ANTIGRAVITY_USER_AGENT`  — full `User-Agent` replacement.
 */

#include <cstdlib>
#include <string>
#include <string_view>

namespace core::auth::antigravity {

inline constexpr std::string_view kAntigravityEndpoint = "https://daily-cloudcode-pa.googleapis.com";
inline constexpr std::string_view kAntigravitySandboxEndpoint = "https://daily-cloudcode-pa.sandbox.googleapis.com";
inline constexpr std::string_view kAntigravityClientVersion = "2.27.0";

// The OAuth client identity the genuine `antigravity/hub` client presents at
// accounts.google.com — the `client_id` that appears as the `aud` of the
// id_token Google issues for an Antigravity sign-in, plus the secret of that
// *confidential* client (the token endpoint answers `client_secret is missing.`
// without it) — is deliberately NOT embedded here. GitHub secret scanning
// matches both halves, as "Google OAuth Client ID" and "Google OAuth Client
// Secret", and push protection rejects any commit carrying either, so a public
// repository cannot ship them. Both are configured per machine through
// `GOOGLE_ANTIGRAVITY_CLIENT_ID` / `_CLIENT_SECRET` or through
// `<config_dir>/antigravity.json`; see AntigravityClientCredentials.hpp.

/// Client version reported on the wire (`ANTIGRAVITY_CLI_VERSION` override, else the pinned release).
[[nodiscard]] inline std::string client_version() {
    if (const char* raw = std::getenv("ANTIGRAVITY_CLI_VERSION");
        raw && raw[0] != '\0') {
        return raw;
    }
    return std::string(kAntigravityClientVersion);
}

/// `User-Agent` identifying the Antigravity hub client (`ANTIGRAVITY_USER_AGENT` replaces the whole string).
[[nodiscard]] inline std::string user_agent() {
    if (const char* raw = std::getenv("ANTIGRAVITY_USER_AGENT");
        raw && raw[0] != '\0') {
        return raw;
    }
    return "antigravity/hub/" + client_version()
        + " (aidev_client; os_type=darwin; arch=arm64; cl=1007512578)";
}

} // namespace core::auth::antigravity

#pragma once

/**
 * @file AntigravityClientCredentials.hpp
 * @brief OAuth client credentials for the Antigravity (Cloud Code Assist) path.
 *
 * Filo embeds no Antigravity client identity — see GoogleAntigravityIdentity.hpp
 * for why — so `"auth_type": "oauth_google"` costs a one-time configuration
 * instead of none. After that step login and refresh are native: no Antigravity
 * installation, and no access to any app or keychain, is involved.
 *
 * Two layers supply the four fields, applied per field, highest priority first
 * — a field left empty in a layer never masks a lower one:
 *
 *   1. `GOOGLE_ANTIGRAVITY_CLIENT_ID` / `_CLIENT_SECRET` / `_ACCESS_TOKEN` /
 *      `_REFRESH_TOKEN`;
 *   2. `<config_dir>/antigravity.json` (the directory holding
 *      `oauth_google.json`) with the same four keys.
 *
 */

#include "OAuthToken.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace core::auth::antigravity {

struct ClientCredentials {
    std::string client_id;
    std::string client_secret;
    /// Session token supplied by the user, when there is one.
    std::string access_token;
    std::string refresh_token;
    /// `"env"`, `"unset"`, or the config path the credentials came from.
    std::string source;

    /// Both halves of the confidential client are present.
    [[nodiscard]] bool complete() const noexcept {
        return !client_id.empty() && !client_secret.empty();
    }
    /// A token was supplied explicitly, so no browser login is needed.
    [[nodiscard]] bool has_imported_token() const noexcept {
        return !access_token.empty();
    }
};

/// `<config_dir>/antigravity.json`; empty when no config dir is known.
[[nodiscard]] std::string credentials_path(std::string_view config_dir);

/// Credentials with no client identity: the base both layers are applied to.
[[nodiscard]] ClientCredentials default_client_credentials();

/// Empty credentials with the config-file and environment layers applied.
[[nodiscard]] ClientCredentials load_client_credentials(std::string_view config_dir = {});

/**
 * @brief Throws unless both halves of the confidential client are configured.
 *
 * Google answers an incomplete pair with an opaque `client_secret is missing.`
 * or `invalid_client`, so the missing configuration is reported here instead,
 * naming both places it can be supplied.
 */
void require_client_credentials(const ClientCredentials& credentials,
                                std::string_view config_dir);

/**
 * @brief OAuthToken for an explicitly supplied access token.
 *
 * Google publishes no lifetime for a pasted token, so the usual one-hour access
 * token lifetime is assumed: an expired paste then falls through to a refresh
 * (when a refresh token came with it) or to a browser login, instead of being
 * replayed until the endpoint rejects it.
 */
[[nodiscard]] std::optional<OAuthToken>
imported_token(const ClientCredentials& credentials, std::int64_t now_unix_seconds);

} // namespace core::auth::antigravity

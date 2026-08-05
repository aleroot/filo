#pragma once

#include "OAuthToken.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace core::auth {

/**
 * Shared OAuth token-endpoint helpers used by provider flows and MCP OAuth.
 *
 * Kept intentionally small and side-effect free beyond HTTP: no browser,
 * no token store, no provider-specific claims parsing.
 */
struct OAuthTokenEndpointRequest {
    std::string token_url;
    std::string client_id;
    std::string client_secret;   // optional (confidential clients)
    std::string redirect_uri;    // required for authorization_code
    std::string resource;        // optional RFC 8707 resource indicator
    std::string issuer;          // optional metadata stamped onto the token
    std::chrono::milliseconds timeout{std::chrono::seconds(20)};
};

/**
 * Parse a standard OAuth token JSON body.
 *
 * Sets expires_at = request_time_unix + expires_in (default expires_in=3600).
 * Stamps client_id/issuer when provided.
 */
[[nodiscard]] OAuthToken parse_oauth_token_response(
    std::string_view json,
    int64_t request_time_unix,
    std::string_view client_id = {},
    std::string_view issuer = {});

/** authorization_code grant (+ optional PKCE verifier / resource). */
[[nodiscard]] OAuthToken exchange_authorization_code(
    const OAuthTokenEndpointRequest& request,
    std::string_view code,
    std::string_view code_verifier = {});

/** refresh_token grant (+ optional resource). */
[[nodiscard]] OAuthToken refresh_access_token(
    const OAuthTokenEndpointRequest& request,
    std::string_view refresh_token);

/** Split a space/comma-separated scope string. */
[[nodiscard]] std::vector<std::string> split_oauth_scopes(std::string_view raw);

/** Join scopes with a single space. */
[[nodiscard]] std::string join_oauth_scopes(const std::vector<std::string>& scopes);

} // namespace core::auth

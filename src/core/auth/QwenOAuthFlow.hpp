#pragma once

#include "IOAuthFlow.hpp"
#include "OAuthToken.hpp"
#include <string>
#include <string_view>
#include <cstdint>

namespace core::auth {

/**
 * @brief Qwen OAuth2 device code flow (RFC 8628) with PKCE S256 (RFC 7636).
 *
 * Mirrors the authentication used by qwen-code CLI against chat.qwen.ai.
 * Provides access to the free tier (1000 req/day via "coder-model").
 *
 * Flow:
 *   1. Generate PKCE code_verifier (43-char base64url) + S256 code_challenge
 *   2. POST /api/v1/oauth2/device/code → device_code + user_code
 *   3. Display user_code and open verification_uri in browser
 *   4. Poll /api/v1/oauth2/token with device_code + code_verifier until approved
 *   5. Store access_token and refresh_token
 *
 * login() is blocking and may throw std::runtime_error on failure or timeout.
 * refresh() is blocking and may throw std::runtime_error on HTTP error.
 *
 * The static pure functions are exposed for unit testing.
 */
class QwenOAuthFlow : public IOAuthFlow {
public:
    QwenOAuthFlow() = default;

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;

    // ── Pure functions exposed for unit testing ───────────────────────────

    /// Generates a UUID v4 string for the x-request-id header.
    static std::string generate_request_id();

    /// Parses a Qwen token JSON response body into an OAuthToken.
    /// @param json             Raw JSON response body.
    /// @param request_time_unix Unix timestamp of the request (for expiry calc).
    static OAuthToken parse_token_response(std::string_view json,
                                           int64_t request_time_unix);

private:
    struct DeviceAuthorization {
        std::string device_code;
        std::string user_code;
        std::string verification_uri;
        std::string verification_uri_complete;
        int expires_in  = 900;
        int interval    = 5;
        std::string code_verifier; ///< Retained for the token exchange step
    };

    static DeviceAuthorization request_device_authorization(
        std::string_view code_challenge,
        std::string_view code_verifier);

    static OAuthToken poll_for_token(const DeviceAuthorization& auth);
    static OAuthToken do_refresh(std::string_view refresh_token);
};

} // namespace core::auth

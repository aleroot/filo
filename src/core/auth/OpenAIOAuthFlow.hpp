#pragma once

#include "IOAuthFlow.hpp"
#include <string>
#include <vector>

namespace core::auth {

/**
 * @brief OpenAI OAuth2 PKCE flow (RFC 7636) for ChatGPT Plus/Pro/Team users.
 *
 * Mirrors the Codex CLI authentication approach:
 * - PKCE with SHA-256 code challenge — no client secret is required for
 *   installed-application flows (RFC 6749 §4.1 + RFC 7636)
 * - Loopback redirect on a dynamically-chosen port in [54321, 54400]
 * - Random state token for CSRF protection
 *
 * login() opens the user's browser, starts a local HTTP server to capture
 * the redirect callback, and exchanges the authorisation code for tokens.
 * It blocks until the flow completes or times out (5 minutes).
 *
 * OpenAI token responses include a refresh_token; refresh() uses it to
 * obtain a new access_token without a browser round-trip.
 *
 * NOTE: This flow requires a registered OAuth2 client_id. Pass one
 * explicitly or set OPENAI_OAUTH_CLIENT_ID in the environment. The
 * token_url and auth_url can also be overridden for private deployments.
 *
 * The static pure functions are exposed for unit testing.
 */
class OpenAIOAuthFlow : public IOAuthFlow {
public:
    // Reads client_id from OPENAI_OAUTH_CLIENT_ID. Throws if not set.
    OpenAIOAuthFlow();

    OpenAIOAuthFlow(std::string client_id,
                    std::string auth_url  = "https://auth.openai.com/authorize",
                    std::string token_url = "https://auth.openai.com/oauth/token",
                    std::vector<std::string> scopes = {"openid", "email", "profile"},
                    int port_start = 54321,
                    int port_end   = 54400);

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;

    // ── Pure functions exposed for unit testing ───────────────────────────

    /// Generates a 128-char PKCE code_verifier (RFC 7636 §4.1).
    static std::string generate_code_verifier();

    /// Returns BASE64URL(SHA256(verifier)) — the S256 code_challenge.
    static std::string compute_code_challenge(std::string_view verifier);

    static std::string build_auth_url(std::string_view client_id,
                                      std::string_view redirect_uri,
                                      const std::vector<std::string>& scopes,
                                      std::string_view state,
                                      std::string_view code_challenge,
                                      std::string_view auth_base_url);

    static OAuthToken parse_token_response(std::string_view json,
                                           int64_t request_time_unix);

private:
    OAuthToken exchange_code(const std::string& code,
                             const std::string& redirect_uri,
                             const std::string& code_verifier);

    std::string              client_id_;
    std::string              auth_url_;
    std::string              token_url_;
    std::vector<std::string> scopes_;
    int                      port_start_;
    int                      port_end_;
};

} // namespace core::auth

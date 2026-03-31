#pragma once

#include "IOAuthFlow.hpp"
#include "ui/AuthUI.hpp"
#include <memory>
#include <string>
#include <vector>

namespace core::auth {

/**
 * @brief Claude OAuth2 authorization-code flow with PKCE.
 *
 * Supports:
 * - Browser-based login against Claude/Anthropic OAuth endpoints
 * - Loopback redirect callback on localhost
 * - Manual callback/code paste fallback for remote browser setups
 * - PKCE S256 challenge
 * - Refresh token grant
 *
 * For automation/backward-compatibility, login() also accepts
 * CLAUDE_CODE_OAUTH_TOKEN (optionally with CLAUDE_CODE_OAUTH_REFRESH_TOKEN)
 * or ANTHROPIC_AUTH_TOKEN from environment.
 */
class ClaudeOAuthFlow : public IOAuthFlow {
public:
    explicit ClaudeOAuthFlow(std::shared_ptr<ui::AuthUI> ui = nullptr);
    ClaudeOAuthFlow(std::string client_id,
                    std::string auth_url,
                    std::string token_url,
                    std::vector<std::string> scopes,
                    int port_start,
                    int port_end,
                    std::shared_ptr<ui::AuthUI> ui = nullptr);

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;

private:
    static std::string build_auth_url(std::string_view client_id,
                                      std::string_view redirect_uri,
                                      const std::vector<std::string>& scopes,
                                      std::string_view state,
                                      std::string_view code_challenge,
                                      std::string_view auth_base_url);
    static OAuthToken parse_token_response(std::string_view json,
                                           int64_t request_time_unix);
    OAuthToken exchange_code(const std::string& code,
                             const std::string& redirect_uri,
                             const std::string& code_verifier,
                             const std::string& state);

    std::string client_id_;
    std::string auth_url_;
    std::string token_url_;
    std::vector<std::string> scopes_;
    int port_start_ = 0;
    int port_end_ = 0;
    std::shared_ptr<ui::AuthUI> ui_;
};

} // namespace core::auth

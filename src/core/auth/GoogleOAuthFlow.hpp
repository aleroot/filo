#pragma once

#include "IOAuthFlow.hpp"
#include <string>
#include <vector>

namespace core::auth {

/**
 * @brief Google OAuth2 flow using the loopback redirect method.
 *
 * The default constructor uses the same application credentials as gemini-cli
 * (public desktop-app credentials — intentionally not secret per Google's
 * own guidelines for installed applications).
 *
 * login() opens the user's browser, starts a local HTTP server to capture
 * the redirect, and exchanges the authorization code for tokens. It blocks
 * until the flow completes or times out (5 minutes).
 *
 * The static methods build_auth_url() and parse_token_response() are pure
 * functions exposed for unit testing without network or browser side-effects.
 */
class GoogleOAuthFlow : public IOAuthFlow {
public:
    GoogleOAuthFlow();

    GoogleOAuthFlow(std::string client_id,
                    std::string client_secret,
                    std::vector<std::string> scopes,
                    int port_start = 54321,
                    int port_end   = 54400);

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;

    static std::string build_auth_url(std::string_view client_id,
                                      std::string_view redirect_uri,
                                      const std::vector<std::string>& scopes,
                                      std::string_view state);

    static OAuthToken parse_token_response(std::string_view json,
                                           int64_t request_time_unix);

private:
    OAuthToken exchange_code(const std::string& code,
                             const std::string& redirect_uri);

    std::string              client_id_;
    std::string              client_secret_;
    std::vector<std::string> scopes_;
    int                      port_start_;
    int                      port_end_;
};

} // namespace core::auth

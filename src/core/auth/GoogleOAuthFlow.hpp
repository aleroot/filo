#pragma once

#include "IOAuthFlow.hpp"
#include "ui/AuthUI.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace core::auth {

struct GoogleManualAuthInput {
    std::string code;
    std::optional<std::string> state;
};

/**
 * @brief Google OAuth2 flow for Gemini Code Assist authentication.
 *
 * The default constructor uses the same application credentials as gemini-cli
 * (public desktop-app credentials — intentionally not secret per Google's
 * own guidelines for installed applications).
 *
 * login() normally opens the user's browser, starts a local HTTP server to
 * capture the redirect, and exchanges the authorization code for tokens.
 * When browser launch is suppressed (for example `NO_BROWSER=1`), it switches
 * to an immediate manual Code Assist code flow instead of waiting for a
 * loopback callback that will never arrive.
 *
 * The static methods build_auth_url() and parse_token_response() are pure
 * functions exposed for unit testing without network or browser side-effects.
 */
class GoogleOAuthFlow : public IOAuthFlow {
public:
    explicit GoogleOAuthFlow(std::shared_ptr<ui::AuthUI> ui = nullptr);

    GoogleOAuthFlow(std::string client_id,
                    std::string client_secret,
                    std::vector<std::string> scopes,
                    int port_start = 54321,
                    int port_end   = 54400,
                    std::shared_ptr<ui::AuthUI> ui = nullptr);

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;

    static std::string build_auth_url(std::string_view client_id,
                                      std::string_view redirect_uri,
                                      const std::vector<std::string>& scopes,
                                      std::string_view state,
                                      std::string_view code_challenge = {});

    static OAuthToken parse_token_response(std::string_view json,
                                           int64_t request_time_unix);

    static GoogleManualAuthInput parse_manual_auth_input(std::string_view raw_input);
    static std::string build_loopback_redirect_uri(std::string_view listener_host,
                                                   int port,
                                                   std::string_view path = "/oauth2callback",
                                                   std::string_view redirect_host_override = {});

private:
    OAuthToken exchange_code(const std::string& code,
                             const std::string& redirect_uri,
                             std::string_view code_verifier = {});

    std::string              client_id_;
    std::string              client_secret_;
    std::vector<std::string> scopes_;
    int                      port_start_;
    int                      port_end_;
    std::shared_ptr<ui::AuthUI> ui_;
};

} // namespace core::auth

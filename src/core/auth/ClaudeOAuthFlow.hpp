#pragma once

#include "IOAuthFlow.hpp"
#include "ui/AuthUI.hpp"
#include <memory>

namespace core::auth {

/**
 * @brief Lightweight OAuth flow for Claude bearer tokens.
 *
 * The flow reads a bearer token from ANTHROPIC_AUTH_TOKEN during login().
 * Tokens are persisted through OAuthTokenManager + FileTokenStore so users can
 * run --login claude once and reuse the token until it expires.
 *
 * refresh() is intentionally unsupported because Anthropic bearer tokens used
 * in this mode do not expose a standard refresh token flow.
 */
class ClaudeOAuthFlow : public IOAuthFlow {
public:
    explicit ClaudeOAuthFlow(std::shared_ptr<ui::AuthUI> ui = nullptr);

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;

private:
    std::shared_ptr<ui::AuthUI> ui_;
};

} // namespace core::auth

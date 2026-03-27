#pragma once

#include "IOAuthFlow.hpp"
#include "ui/AuthUI.hpp"
#include <memory>

namespace core::auth {

/**
 * @brief API-key-based auth flow for OpenAI.
 *
 * login() checks OPENAI_API_KEY first (headless / CI support), then falls
 * back to an interactive prompt. The key is stored as access_token so it
 * survives restarts without the user pasting it each time.
 *
 * OpenAI API keys do not expire on their own, so expires_at is set ~1 year
 * out and refresh() is unsupported (keys are rotated manually, not via a
 * token-refresh round-trip).
 */
class OpenAIAuthFlow : public IOAuthFlow {
public:
    explicit OpenAIAuthFlow(std::shared_ptr<ui::AuthUI> ui = nullptr);

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;

private:
    std::shared_ptr<ui::AuthUI> ui_;
};

} // namespace core::auth

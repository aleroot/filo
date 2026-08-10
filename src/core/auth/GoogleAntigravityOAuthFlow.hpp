#pragma once

#include "IOAuthFlow.hpp"
#include "IOAuthTokenRevoker.hpp"
#include "ui/AuthUI.hpp"
#include <memory>
#include <string>

namespace core::auth {

/**
 * @brief Google OAuth2 flow that impersonates Google's official "Antigravity"
 *        IDE/CLI OAuth client instead of the gemini-cli client used by
 *        GoogleOAuthFlow.
 *
 * ── Why this exists ──────────────────────────────────────────────────────
 * Google deprecated the gemini-cli OAuth flow for individual / Google AI
 * Pro / Google AI Ultra accounts and replaced it with a first-party
 * "Antigravity CLI" (`agy`). Antigravity CLI is closed-source, so there is
 * no officially published client_id/endpoint to integrate against. This
 * flow authenticates against Google's Antigravity backend using OAuth
 * client credentials supplied via environment variables.
 *
 * ── IMPORTANT: this is NOT an officially supported integration ─────────
 * Google's Antigravity Terms of Service explicitly prohibit third-party
 * tools/proxies from authenticating against Antigravity's backend, and
 * Google has run real, automated account-ban waves against exactly this
 * pattern. This flow is opt-in, clearly labelled, and every login()
 * invocation surfaces that risk to the user before proceeding. Do not
 * enable this auth type unless you understand and accept that your Google
 * account could be suspended or banned.
 *
 * The Cloud Code Assist backend it talks to (`cloudcode-pa.googleapis.com`)
 * is the same production host gemini-cli uses; only the OAuth client
 * identity, requested scopes, and per-request "Antigravity" client
 * metadata differ.
 *
 * Configure the OAuth client credentials via the
 * GOOGLE_ANTIGRAVITY_CLIENT_ID and GOOGLE_ANTIGRAVITY_CLIENT_SECRET
 * environment variables.
 */
class GoogleAntigravityOAuthFlow : public IOAuthFlow, public IOAuthTokenRevoker {
public:
    explicit GoogleAntigravityOAuthFlow(std::shared_ptr<ui::AuthUI> ui = nullptr);

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;

    /// Best-effort revocation via https://oauth2.googleapis.com/revoke.
    void revoke(const OAuthToken& token) override;

private:
    void warn_about_risk() const;

    std::shared_ptr<ui::AuthUI> ui_;
};

} // namespace core::auth

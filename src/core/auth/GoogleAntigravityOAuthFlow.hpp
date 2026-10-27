#pragma once

#include "IOAuthFlow.hpp"
#include "IOAuthTokenRevoker.hpp"
#include "ui/AuthUI.hpp"
#include <memory>
#include <string>
#include <string_view>

namespace core::auth {

/**
 * @brief Google OAuth2 flow that impersonates Google's official "Antigravity"
 *        IDE/CLI OAuth client instead of the gemini-cli client used by
 *        GoogleOAuthFlow.
 *
 * ── Why this exists ──────────────────────────────────────────────────────
 * Google deprecated the gemini-cli OAuth flow for individual / Google AI
 * Pro / Google AI Ultra accounts and replaced it with a first-party
 * "Antigravity CLI" (`agy`). Antigravity CLI is closed-source and publishes
 * no integrable client_id, so this flow authenticates against Google's
 * Antigravity backend natively, with that client identity supplied by
 * configuration rather than embedded in Filo (see
 * AntigravityClientCredentials.hpp). It never needs the Antigravity app, or any
 * app session, on the machine it runs on.
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
 * The Cloud Code Assist backend it talks to is the daily host
 * (`daily-cloudcode-pa.googleapis.com`, sandbox as streaming failover) the
 * real Antigravity client uses, not gemini-cli's production host; the OAuth
 * client identity, requested scopes, and per-request "Antigravity" client
 * metadata differ too.
 *
 * The client pair is configured per field with the GOOGLE_ANTIGRAVITY_CLIENT_ID
 * and GOOGLE_ANTIGRAVITY_CLIENT_SECRET environment variables, or persisted in
 * `<config_dir>/antigravity.json` (the directory that also holds
 * `oauth_google.json`) as `client_id` / `client_secret`; the environment wins
 * when both are present. Filo embeds neither half, so login() and refresh()
 * report the missing configuration instead of failing opaquely at Google. That
 * file (or `GOOGLE_ANTIGRAVITY_ACCESS_TOKEN`) may also carry an access token,
 * which login() then adopts instead of opening a browser.
 */
class GoogleAntigravityOAuthFlow : public IOAuthFlow, public IOAuthTokenRevoker {
public:
    explicit GoogleAntigravityOAuthFlow(std::shared_ptr<ui::AuthUI> ui = nullptr,
                                        std::string config_dir = {});

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;

    /// Best-effort revocation via https://oauth2.googleapis.com/revoke.
    void revoke(const OAuthToken& token) override;

private:
    void warn_about_risk() const;

    std::shared_ptr<ui::AuthUI> ui_;
    /// Filo config directory that may hold `antigravity.json`.
    std::string config_dir_;
};

} // namespace core::auth

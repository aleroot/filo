#include "GoogleAntigravityOAuthFlow.hpp"
#include "AuthBrowserLauncher.hpp"
#include "GoogleCodeAssist.hpp"
#include "GoogleOAuthFlow.hpp"
#include "OAuthErrors.hpp"
#include "OAuthLoopback.hpp"
#include "OAuthPkce.hpp"
#include "../logging/Logger.hpp"
#include <cpr/cpr.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace core::auth {

namespace {

// OAuth client credentials for the Google Antigravity flow are supplied via
// environment variables. This avoids hardcoding secrets in source control.
// Set GOOGLE_ANTIGRAVITY_CLIENT_ID and GOOGLE_ANTIGRAVITY_CLIENT_SECRET
// before running Filo / Lampo. See GoogleAntigravityOAuthFlow.hpp for the
// risk disclosure this class surfaces before every login.

[[nodiscard]] std::string resolve_client_id() {
    if (const char* env = std::getenv("GOOGLE_ANTIGRAVITY_CLIENT_ID");
        env && env[0] != '\0') {
        return std::string(env);
    }
    throw std::runtime_error(
        "Google (Antigravity) OAuth client ID is not configured. "
        "Set the GOOGLE_ANTIGRAVITY_CLIENT_ID environment variable.");
}

[[nodiscard]] std::string resolve_client_secret() {
    if (const char* env = std::getenv("GOOGLE_ANTIGRAVITY_CLIENT_SECRET");
        env && env[0] != '\0') {
        return std::string(env);
    }
    throw std::runtime_error(
        "Google (Antigravity) OAuth client secret is not configured. "
        "Set the GOOGLE_ANTIGRAVITY_CLIENT_SECRET environment variable.");
}

const std::vector<std::string>& antigravity_scopes() {
    static const std::vector<std::string> scopes = {
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",
        "https://www.googleapis.com/auth/experimentsandconfigs",
    };
    return scopes;
}

constexpr int kLoopbackPortStart = 51121;
constexpr int kLoopbackPortEnd = 51140;
constexpr const char* kCallbackPath = "/oauth-callback";

constexpr const char* kRiskWarning =
    "\xE2\x9A\xA0  UNOFFICIAL: This signs in using Google's internal "
    "\"Antigravity\" IDE OAuth client — it is not an integration Google "
    "publishes or supports. Google's Antigravity Terms of Service "
    "explicitly prohibit third-party tools from doing this, and Google has "
    "suspended/banned real accounts for it in the past. Only continue if "
    "you understand and accept that risk.";

[[nodiscard]] bool env_truthy(const char* key) {
    const char* raw = std::getenv(key);
    if (!raw || raw[0] == '\0') return false;
    return std::string_view(raw) != "0";
}

[[nodiscard]] std::int64_t now_unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

} // namespace

GoogleAntigravityOAuthFlow::GoogleAntigravityOAuthFlow(std::shared_ptr<ui::AuthUI> ui)
    : ui_(std::move(ui)) {}

void GoogleAntigravityOAuthFlow::warn_about_risk() const {
    if (ui_) {
        ui_->show_header("Google (Antigravity) OAuth Login — Unofficial");
        ui_->show_instructions(kRiskWarning);
    } else {
        std::cout << "\n" << kRiskWarning << "\n\n";
        std::cout.flush();
    }
}

OAuthToken GoogleAntigravityOAuthFlow::login() {
    warn_about_risk();

    const std::string client_id = resolve_client_id();
    const std::string client_secret = resolve_client_secret();

    const std::string state = oauth_pkce::generate_correlation_token();
    const std::string code_verifier = oauth_pkce::generate_code_verifier();
    const std::string challenge = oauth_pkce::compute_code_challenge(code_verifier);

    OAuthLoopbackOptions loop_opts;
    loop_opts.bind_host = "127.0.0.1";
    loop_opts.port_start = kLoopbackPortStart;
    loop_opts.port_end = kLoopbackPortEnd;
    loop_opts.callback_path = kCallbackPath;
    loop_opts.success_html =
        "<html><body style=\"font-family:system-ui;padding:2rem\">"
        "<h2>Login successful</h2>"
        "<p>You can close this tab and return to Filo.</p></body></html>";

    OAuthLoopbackServer loopback(std::move(loop_opts));
    const std::string redirect_uri = loopback.redirect_uri();

    const std::string auth_url = GoogleOAuthFlow::build_auth_url(
        client_id,
        redirect_uri,
        antigravity_scopes(),
        state,
        challenge);

    if (ui_) {
        ui_->show_url(auth_url, "Open this URL in a browser:");
    } else {
        std::cout << "Open this URL in a browser and complete the Google login:\n  "
                   << auth_url << "\n\n";
        std::cout.flush();
    }

    if (!env_truthy("NO_BROWSER")) {
        open_browser(auth_url);
    }

    loopback.start();
    auto result = loopback.wait();
    if (!complete_oauth_loopback_with_manual_fallback(result, state, ui_.get())) {
        if (!result.error.empty()) {
            throw std::runtime_error("Google (Antigravity) authorization failed: " + result.error);
        }
        if (result.timed_out) {
            throw std::runtime_error(
                "Google (Antigravity) OAuth: timed out waiting for browser callback");
        }
        throw std::runtime_error("Google (Antigravity) OAuth: authorization code missing");
    }

    const auto request_time = now_unix_seconds();
    cpr::Response token_response = cpr::Post(
        cpr::Url{"https://oauth2.googleapis.com/token"},
        cpr::Payload{
            {"client_id", client_id},
            {"client_secret", client_secret},
            {"code", result.code},
            {"redirect_uri", redirect_uri},
            {"grant_type", "authorization_code"},
            {"code_verifier", code_verifier},
        });
    if (token_response.status_code != 200) {
        throw std::runtime_error(
            "Google (Antigravity) token exchange failed (" +
            std::to_string(token_response.status_code) + "): " + token_response.text);
    }

    OAuthToken token = GoogleOAuthFlow::parse_token_response(token_response.text, request_time);
    if (!token.has_refresh_token()) {
        throw std::runtime_error(
            "Google (Antigravity) OAuth did not return a refresh token. "
            "Try logging in again and make sure to grant offline access.");
    }
    token.client_id = client_id;
    token.scopes = antigravity_scopes();

    try {
        token.project_id = google_code_assist::setup_user(
            token.access_token, ui_, /*ide_type=*/"ANTIGRAVITY");
    } catch (const std::exception& e) {
        core::logging::warn(
            "Google (Antigravity) Cloud Code Assist project setup failed: {}", e.what());
    }

    if (ui_) {
        ui_->show_success(
            "Google (Antigravity) login successful. Remember: this is an unofficial, "
            "ToS-violating integration — use at your own risk.");
    }
    return token;
}

OAuthToken GoogleAntigravityOAuthFlow::refresh(std::string_view refresh_token) {
    const std::string client_id = resolve_client_id();
    const std::string client_secret = resolve_client_secret();

    const auto request_time = now_unix_seconds();

    cpr::Response r = cpr::Post(
        cpr::Url{"https://oauth2.googleapis.com/token"},
        cpr::Payload{
            {"client_id", client_id},
            {"client_secret", client_secret},
            {"refresh_token", std::string(refresh_token)},
            {"grant_type", "refresh_token"},
        });

    if (r.status_code != 200) {
        if (oauth_error_is_invalid_grant(r.text)) {
            throw OAuthRefreshRejected::by_provider("Google (Antigravity)");
        }
        throw std::runtime_error(
            "Google (Antigravity) token refresh failed (" +
            std::to_string(r.status_code) + "): " + r.text);
    }

    OAuthToken token = GoogleOAuthFlow::parse_token_response(r.text, request_time);
    if (token.refresh_token.empty()) {
        token.refresh_token = std::string(refresh_token);
    }
    token.client_id = client_id;
    return token;
}

void GoogleAntigravityOAuthFlow::revoke(const OAuthToken& token) {
    const std::string& target = token.has_refresh_token()
        ? token.refresh_token
        : token.access_token;
    if (target.empty()) return;

    cpr::Response r = cpr::Post(
        cpr::Url{"https://oauth2.googleapis.com/revoke"},
        cpr::Payload{{"token", target}},
        cpr::Timeout{10000});

    if (r.status_code != 200) {
        throw std::runtime_error(
            "Google (Antigravity) token revocation failed (" +
            std::to_string(r.status_code) + "): " + r.text);
    }
}

} // namespace core::auth

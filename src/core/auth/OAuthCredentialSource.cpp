#include "OAuthCredentialSource.hpp"
#include <cstdlib>

namespace core::auth {

namespace {

std::string join_scopes(const std::vector<std::string>& scopes) {
    std::string out;
    for (std::size_t i = 0; i < scopes.size(); ++i) {
        if (i > 0) out += ' ';
        out += scopes[i];
    }
    return out;
}

} // namespace

OAuthCredentialSource::OAuthCredentialSource(std::shared_ptr<OAuthTokenManager> manager)
    : manager_(std::move(manager)) {}

AuthInfo OAuthCredentialSource::get_auth() {
    OAuthToken token = manager_->get_valid_token();
    AuthInfo auth;

    const bool is_session_key =
        token.access_token.starts_with("sk-ant-sid");

    if (is_session_key) {
        auth.headers["Cookie"] = "sessionKey=" + token.access_token;
        std::string resolved_org_uuid = token.organization_id;
        if (const char* org_uuid_env = std::getenv("CLAUDE_CODE_ORGANIZATION_UUID");
            org_uuid_env && org_uuid_env[0] != '\0') {
            // Environment variable takes precedence so users can force org routing.
            resolved_org_uuid = org_uuid_env;
        }
        if (!resolved_org_uuid.empty()) {
            auth.headers["X-Organization-Uuid"] = resolved_org_uuid;
            auth.properties["organization_id"] = resolved_org_uuid;
            // Keep legacy property key for any downstream integrations expecting it.
            auth.properties["organization_uuid"] = resolved_org_uuid;
        }
        auth.properties["auth_mode"] = "session_cookie";
    } else {
        const std::string token_type =
            token.token_type.empty() ? std::string("Bearer") : token.token_type;
        auth.headers["Authorization"] = token_type + " " + token.access_token;
        auth.properties["auth_mode"] = "bearer";
    }

    auth.properties["oauth"] = "1";
    if (!token.scopes.empty()) {
        auth.properties["oauth_scopes"] = join_scopes(token.scopes);
    }
    
    // Pass device_id from token to protocol via properties.
    // This is critical for Kimi OAuth: the X-Msh-Device-Id header must match
    // the device_id claim embedded in the JWT token, or the server returns 401.
    if (!token.device_id.empty()) {
        auth.properties["device_id"] = token.device_id;
    }

    // Provider protocols may translate this generic account identifier into
    // provider-specific transport headers when required.
    if (!token.account_id.empty()) {
        auth.properties["account_id"] = token.account_id;
    }
    
    return auth;
}

bool OAuthCredentialSource::refresh_on_auth_failure() {
    try {
        manager_->force_refresh();
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace core::auth

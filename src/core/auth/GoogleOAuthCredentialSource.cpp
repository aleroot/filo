#include "GoogleOAuthCredentialSource.hpp"
#include "GoogleCodeAssist.hpp"
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

GoogleOAuthCredentialSource::GoogleOAuthCredentialSource(
    std::shared_ptr<OAuthTokenManager> manager)
    : manager_(std::move(manager)) {}

AuthInfo GoogleOAuthCredentialSource::get_auth() {
    OAuthToken token = manager_->get_valid_token();

    std::string project_id;
    if (const auto configured = google_code_assist::configured_project_override();
        configured.has_value()) {
        project_id = *configured;
    }

    {
        std::scoped_lock lock(mutex_);
        if (project_id.empty()) {
            if (!token.project_id.empty()) {
                project_id = token.project_id;
                project_initialized_ = true;
            } else if (!cached_project_id_.empty()) {
                project_id = cached_project_id_;
            } else if (!project_initialized_) {
                project_id = google_code_assist::setup_user(token.access_token);
                project_initialized_ = true;
            }
        }

        cached_project_id_ = project_id;
    }

    if (!project_id.empty() && project_id != token.project_id) {
        token.project_id = project_id;
        manager_->save_token(token);
    }

    AuthInfo auth;
    const std::string token_type =
        token.token_type.empty() ? std::string("Bearer") : token.token_type;
    auth.headers["Authorization"] = token_type + " " + token.access_token;
    auth.properties["auth_mode"] = "bearer";
    auth.properties["oauth"] = "1";
    if (!project_id.empty()) {
        auth.properties["project_id"] = project_id;
    }
    if (!token.scopes.empty()) {
        auth.properties["oauth_scopes"] = join_scopes(token.scopes);
    }
    return auth;
}

bool GoogleOAuthCredentialSource::refresh_on_auth_failure() {
    try {
        manager_->force_refresh();
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace core::auth

#include "OAuthTokenManager.hpp"
#include <stdexcept>

namespace core::auth {
namespace {

void inherit_refresh_context(OAuthToken& refreshed, const OAuthToken& stored) {
    if (refreshed.refresh_token.empty()) refreshed.refresh_token = stored.refresh_token;
    if (refreshed.device_id.empty()) refreshed.device_id = stored.device_id;
    if (refreshed.account_id.empty()) refreshed.account_id = stored.account_id;
    if (refreshed.organization_id.empty()) refreshed.organization_id = stored.organization_id;
    if (refreshed.project_id.empty()) refreshed.project_id = stored.project_id;
    if (refreshed.scopes.empty()) refreshed.scopes = stored.scopes;
    if (refreshed.issuer.empty()) refreshed.issuer = stored.issuer;
    if (refreshed.client_id.empty()) refreshed.client_id = stored.client_id;
    if (refreshed.user_id.empty()) refreshed.user_id = stored.user_id;
    if (refreshed.email.empty()) refreshed.email = stored.email;
    if (refreshed.principal_type.empty()) refreshed.principal_type = stored.principal_type;
    if (refreshed.principal_id.empty()) refreshed.principal_id = stored.principal_id;
    if (refreshed.team_id.empty()) refreshed.team_id = stored.team_id;
}

} // namespace

OAuthTokenManager::OAuthTokenManager(std::string provider_id,
                                     std::shared_ptr<IOAuthFlow>  flow,
                                     std::shared_ptr<ITokenStore> store,
                                     bool allow_interactive_login)
    : provider_id_(std::move(provider_id))
    , flow_(std::move(flow))
    , store_(std::move(store))
    , allow_interactive_login_(allow_interactive_login) {}

OAuthToken OAuthTokenManager::get_valid_token() {
    std::unique_lock lock(mutex_);

    // Serialize disk reload + refresh-token rotation across Filo processes.
    auto refresh_lock = store_->acquire_refresh_lock(provider_id_);

    // 1. Disk (another process may have refreshed or removed it while we waited).
    auto stored = store_->load(provider_id_);
    if (stored) {
        if (stored->is_valid()) {
            return *stored;
        }
        // 2. Expired but refresh token available
        if (stored->has_refresh_token()) {
            OAuthToken refreshed = flow_->refresh(stored->refresh_token);
            inherit_refresh_context(refreshed, *stored);
            store_->save(provider_id_, refreshed);
            return refreshed;
        }
    }

    // 3. Full login (opens browser)
    if (!allow_interactive_login_) {
        throw std::runtime_error(
            "No usable OAuth session for provider '" + provider_id_
            + "'. Run `filo auth login " + provider_id_ + "` first.");
    }
    OAuthToken token = flow_->login();
    store_->save(provider_id_, token);
    return token;
}

void OAuthTokenManager::save_token(const OAuthToken& token) {
    std::unique_lock lock(mutex_);
    auto store_lock = store_->acquire_refresh_lock(provider_id_);
    store_->save(provider_id_, token);
}

void OAuthTokenManager::force_refresh() {
    std::unique_lock lock(mutex_);
    auto refresh_lock = store_->acquire_refresh_lock(provider_id_);

    auto stored = store_->load(provider_id_);
    if (stored && stored->has_refresh_token()) {
        OAuthToken refreshed = flow_->refresh(stored->refresh_token);
        inherit_refresh_context(refreshed, *stored);
        store_->save(provider_id_, refreshed);
        return;
    }

    throw std::runtime_error(
        "No refresh token available for provider '" + provider_id_ + "'.");
}

void OAuthTokenManager::login() {
    std::unique_lock lock(mutex_);
    OAuthToken token = flow_->login();
    auto store_lock = store_->acquire_refresh_lock(provider_id_);
    store_->save(provider_id_, token);
}

void OAuthTokenManager::logout() {
    std::unique_lock lock(mutex_);
    auto store_lock = store_->acquire_refresh_lock(provider_id_);
    store_->clear(provider_id_);
}

} // namespace core::auth

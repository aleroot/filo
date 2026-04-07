#include "OAuthTokenManager.hpp"
#include <stdexcept>

namespace core::auth {

OAuthTokenManager::OAuthTokenManager(std::string provider_id,
                                     std::shared_ptr<IOAuthFlow>  flow,
                                     std::shared_ptr<ITokenStore> store)
    : provider_id_(std::move(provider_id))
    , flow_(std::move(flow))
    , store_(std::move(store)) {}

OAuthToken OAuthTokenManager::get_valid_token() {
    std::unique_lock lock(mutex_);

    // 1. In-memory cache
    if (cached_token_ && cached_token_->is_valid()) {
        return *cached_token_;
    }

    // 2. Disk
    auto stored = store_->load(provider_id_);
    if (stored) {
        if (stored->is_valid()) {
            cached_token_ = *stored;
            return *cached_token_;
        }
        // 3. Expired but refresh token available
        if (stored->has_refresh_token()) {
            OAuthToken refreshed = flow_->refresh(stored->refresh_token);
            if (refreshed.refresh_token.empty()) {
                refreshed.refresh_token = stored->refresh_token;
            }
            if (refreshed.device_id.empty()) {
                refreshed.device_id = stored->device_id;
            }
            if (refreshed.account_id.empty()) {
                refreshed.account_id = stored->account_id;
            }
            if (refreshed.scopes.empty()) {
                refreshed.scopes = stored->scopes;
            }
            store_->save(provider_id_, refreshed);
            cached_token_ = refreshed;
            return *cached_token_;
        }
    }

    // 4. Full login (opens browser)
    OAuthToken token = flow_->login();
    store_->save(provider_id_, token);
    cached_token_ = token;
    return *cached_token_;
}

void OAuthTokenManager::force_refresh() {
    std::unique_lock lock(mutex_);

    auto stored = store_->load(provider_id_);
    if (stored && stored->has_refresh_token()) {
        OAuthToken refreshed = flow_->refresh(stored->refresh_token);
        if (refreshed.refresh_token.empty()) {
            refreshed.refresh_token = stored->refresh_token;
        }
        if (refreshed.device_id.empty()) {
            refreshed.device_id = stored->device_id;
        }
        if (refreshed.account_id.empty()) {
            refreshed.account_id = stored->account_id;
        }
        if (refreshed.scopes.empty()) {
            refreshed.scopes = stored->scopes;
        }
        store_->save(provider_id_, refreshed);
        cached_token_ = refreshed;
        return;
    }

    throw std::runtime_error(
        "No refresh token available for provider '" + provider_id_ + "'.");
}

void OAuthTokenManager::login() {
    std::unique_lock lock(mutex_);
    OAuthToken token = flow_->login();
    store_->save(provider_id_, token);
    cached_token_ = token;
}

void OAuthTokenManager::logout() {
    std::unique_lock lock(mutex_);
    store_->clear(provider_id_);
    cached_token_ = std::nullopt;
}

} // namespace core::auth

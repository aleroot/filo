#include "OAuthCredentialSource.hpp"

namespace core::auth {

OAuthCredentialSource::OAuthCredentialSource(std::shared_ptr<OAuthTokenManager> manager)
    : manager_(std::move(manager)) {}

AuthInfo OAuthCredentialSource::get_auth() {
    OAuthToken token = manager_->get_valid_token();
    AuthInfo auth;
    auth.headers["Authorization"] = token.token_type + " " + token.access_token;
    
    // Pass device_id from token to protocol via properties.
    // This is critical for Kimi OAuth: the X-Msh-Device-Id header must match
    // the device_id claim embedded in the JWT token, or the server returns 401.
    if (!token.device_id.empty()) {
        auth.properties["device_id"] = token.device_id;
    }
    
    return auth;
}

} // namespace core::auth

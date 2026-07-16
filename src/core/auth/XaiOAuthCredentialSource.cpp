#include "XaiOAuthCredentialSource.hpp"

#include "XaiGrokClientIdentity.hpp"

#include <stdexcept>

namespace core::auth {

XaiOAuthCredentialSource::XaiOAuthCredentialSource(
    std::shared_ptr<ICredentialSource> inner)
    : inner_(std::move(inner)) {
    if (!inner_) {
        throw std::invalid_argument("xAI OAuth credential decorator requires a source");
    }
}

AuthInfo XaiOAuthCredentialSource::get_auth() {
    AuthInfo auth = inner_->get_auth();
    xai_grok::apply_proxy_identity_headers(auth.headers);
    return auth;
}

bool XaiOAuthCredentialSource::uses_subscription_billing() const noexcept {
    return inner_->uses_subscription_billing();
}

bool XaiOAuthCredentialSource::refresh_on_auth_failure() {
    return inner_->refresh_on_auth_failure();
}

} // namespace core::auth

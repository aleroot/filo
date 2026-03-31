#pragma once

#include "ICredentialSource.hpp"
#include "OAuthTokenManager.hpp"
#include <memory>

namespace core::auth {

/**
 * @brief Credential source backed by an OAuthTokenManager.
 *
 * get_auth() calls get_valid_token() which may block on the first call
 * to refresh or login. This is safe to call from a background thread
 * (e.g. inside GeminiProvider's stream thread).
 */
class OAuthCredentialSource : public ICredentialSource {
public:
    explicit OAuthCredentialSource(std::shared_ptr<OAuthTokenManager> manager);
    AuthInfo get_auth() override;
    [[nodiscard]] bool uses_subscription_billing() const noexcept override { return true; }
    bool refresh_on_auth_failure() override;

private:
    std::shared_ptr<OAuthTokenManager> manager_;
};

} // namespace core::auth

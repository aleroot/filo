#pragma once

#include "ICredentialSource.hpp"
#include "OAuthTokenManager.hpp"
#include <memory>
#include <mutex>
#include <string>

namespace core::auth {

class GoogleOAuthCredentialSource : public ICredentialSource {
public:
    explicit GoogleOAuthCredentialSource(std::shared_ptr<OAuthTokenManager> manager);

    AuthInfo get_auth() override;
    [[nodiscard]] bool uses_subscription_billing() const noexcept override { return true; }
    bool refresh_on_auth_failure() override;

private:
    std::shared_ptr<OAuthTokenManager> manager_;
    std::mutex                         mutex_;
    std::string                        cached_project_id_;
    bool                               project_initialized_ = false;
};

} // namespace core::auth

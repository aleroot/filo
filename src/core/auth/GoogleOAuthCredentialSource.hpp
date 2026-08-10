#pragma once

#include "ICredentialSource.hpp"
#include "OAuthTokenManager.hpp"
#include <memory>
#include <mutex>
#include <string>

namespace core::auth {

class GoogleOAuthCredentialSource : public ICredentialSource {
public:
    /**
     * @param manager   Backing OAuth token manager.
     * @param ide_type  Cloud Code Assist "ideType" metadata used only for the
     *                  lazy first-time project setup fallback (login() flows
     *                  normally already resolve and cache project_id on the
     *                  token). Defaults to "IDE_UNSPECIFIED" (gemini-cli);
     *                  pass "ANTIGRAVITY" for the Antigravity OAuth strategy.
     */
    explicit GoogleOAuthCredentialSource(std::shared_ptr<OAuthTokenManager> manager,
                                         std::string ide_type = "IDE_UNSPECIFIED");

    AuthInfo get_auth() override;
    [[nodiscard]] bool uses_subscription_billing() const noexcept override { return true; }
    bool refresh_on_auth_failure() override;

private:
    std::shared_ptr<OAuthTokenManager> manager_;
    std::string                        ide_type_;
    std::mutex                         mutex_;
    std::string                        cached_project_id_;
    bool                               project_initialized_ = false;
};

} // namespace core::auth

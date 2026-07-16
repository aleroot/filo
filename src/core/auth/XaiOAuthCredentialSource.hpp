#pragma once

#include "ICredentialSource.hpp"

#include <memory>

namespace core::auth {

/** Adds the Grok proxy transport contract to a generic OAuth credential source. */
class XaiOAuthCredentialSource final : public ICredentialSource {
public:
    explicit XaiOAuthCredentialSource(std::shared_ptr<ICredentialSource> inner);

    AuthInfo get_auth() override;
    [[nodiscard]] bool uses_subscription_billing() const noexcept override;
    bool refresh_on_auth_failure() override;

private:
    std::shared_ptr<ICredentialSource> inner_;
};

} // namespace core::auth

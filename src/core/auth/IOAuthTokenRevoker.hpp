#pragma once

#include "OAuthToken.hpp"

namespace core::auth {

/**
 * Optional capability for OAuth providers that support server-side token
 * revocation. Kept separate from IOAuthFlow so login/refresh implementations
 * are not forced to expose provider-specific logout behaviour.
 */
class IOAuthTokenRevoker {
public:
    virtual ~IOAuthTokenRevoker() = default;
    virtual void revoke(const OAuthToken& token) = 0;
};

} // namespace core::auth

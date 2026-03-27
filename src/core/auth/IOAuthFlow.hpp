#pragma once

#include "OAuthToken.hpp"
#include <string>

namespace core::auth {

/**
 * @brief Abstract OAuth2 flow. Implementations open a browser, handle device
 *        code flows, or use any other mechanism to obtain tokens.
 *
 * login() is blocking and may throw std::runtime_error on failure or timeout.
 * refresh() is blocking and may throw std::runtime_error on HTTP error.
 */
class IOAuthFlow {
public:
    virtual ~IOAuthFlow() = default;
    virtual OAuthToken login() = 0;
    virtual OAuthToken refresh(std::string_view refresh_token) = 0;
};

} // namespace core::auth

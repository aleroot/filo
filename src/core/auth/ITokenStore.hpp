#pragma once

#include "OAuthToken.hpp"
#include <optional>
#include <string>

namespace core::auth {

class ITokenStore {
public:
    virtual ~ITokenStore() = default;
    virtual std::optional<OAuthToken> load(std::string_view provider_id) = 0;
    virtual void save(std::string_view provider_id, const OAuthToken& token) = 0;
    virtual void clear(std::string_view provider_id) = 0;
};

} // namespace core::auth

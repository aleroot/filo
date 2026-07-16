#pragma once

#include "OAuthToken.hpp"
#include <optional>
#include <memory>
#include <string>

namespace core::auth {

class ITokenStoreLock {
public:
    virtual ~ITokenStoreLock() = default;
};

class ITokenStore {
public:
    virtual ~ITokenStore() = default;
    virtual std::optional<OAuthToken> load(std::string_view provider_id) = 0;
    virtual void save(std::string_view provider_id, const OAuthToken& token) = 0;
    virtual void clear(std::string_view provider_id) = 0;

    /**
     * Hold an inter-process refresh lock for this credential family.
     * Stores without cross-process coordination may return nullptr.
     */
    [[nodiscard]] virtual std::unique_ptr<ITokenStoreLock>
    acquire_refresh_lock(std::string_view /*provider_id*/) {
        return nullptr;
    }
};

} // namespace core::auth

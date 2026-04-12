#pragma once

#include "OAuthToken.hpp"
#include "ITokenStore.hpp"
#include "IOAuthFlow.hpp"
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace core::auth {

/**
 * @brief Thread-safe orchestrator that produces valid OAuth tokens.
 *
 * Resolution order on get_valid_token():
 *   1. In-memory cache (no I/O)
 *   2. Stored token (disk read, valid → return)
 *   3. Stored token expired + has refresh_token → flow.refresh()
 *   4. No usable token → flow.login()  (opens browser, blocks)
 *
 * The mutex serialises the entire resolution so concurrent callers never
 * trigger a double-login.
 */
class OAuthTokenManager {
public:
    OAuthTokenManager(std::string provider_id,
                      std::shared_ptr<IOAuthFlow>  flow,
                      std::shared_ptr<ITokenStore> store);

    OAuthToken get_valid_token();
    void save_token(const OAuthToken& token);
    void force_refresh();
    void login();
    void logout();

private:
    std::string                  provider_id_;
    std::shared_ptr<IOAuthFlow>  flow_;
    std::shared_ptr<ITokenStore> store_;
    std::mutex                   mutex_;
    std::optional<OAuthToken>    cached_token_;
};

} // namespace core::auth

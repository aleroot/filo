#pragma once

#include "IOAuthFlow.hpp"
#include <string>
#include <string_view>
#include <unordered_map>

namespace core::auth {

/**
 * @brief Kimi Code OAuth2 device flow implementation.
 *
 * This implements the OAuth2 device authorization grant flow used by the
 * official Kimi CLI. It requires special headers for device identification
 * and uses the Kimi authentication server at auth.kimi.com.
 *
 * The flow:
 *   1. Request device authorization from auth.kimi.com
 *   2. Display user_code and verification_uri to user
 *   3. Poll token endpoint until user authorizes or timeout
 *   4. Store access_token and refresh_token
 *
 * login() is blocking and may throw std::runtime_error on failure or timeout.
 * refresh() is blocking and may throw std::runtime_error on HTTP error.
 */
class KimiOAuthFlow : public IOAuthFlow {
public:
    KimiOAuthFlow();

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;

    // Exposed for unit testing and protocol use
    static std::string generateDeviceId();
    static std::string getDeviceModel();
    
    /**
     * @brief Generate common headers required for Kimi API requests.
     *
     * These headers are required for both OAuth token operations AND
     * API calls to api.moonshot.cn. The official Kimi CLI sends these
     * headers with every request for device identification.
     *
     * @return Map of header names to values:
     *   - X-Msh-Platform: "kimi_cli"
     *   - X-Msh-Version: CLI version
     *   - X-Msh-Device-Name: hostname
     *   - X-Msh-Device-Model: OS and architecture info
     *   - X-Msh-Os-Version: OS version
     *   - X-Msh-Device-Id: unique device identifier
     */
    static std::unordered_map<std::string, std::string> getCommonHeaders();

private:
    struct DeviceAuthorization {
        std::string user_code;
        std::string device_code;
        std::string verification_uri;
        std::string verification_uri_complete;
        int expires_in = 0;
        int interval = 5;
    };

    std::string device_id_;

    DeviceAuthorization requestDeviceAuthorization();
    OAuthToken pollForToken(const DeviceAuthorization& auth);
    OAuthToken exchangeRefreshToken(std::string_view refresh_token);
    
    // Common headers required by Kimi OAuth
    std::string getUserAgent() const;
    std::string getDeviceName() const;
};

} // namespace core::auth

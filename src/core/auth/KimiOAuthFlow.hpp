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
 * official Kimi Code CLI. It requires special headers for device identification
 * and talks to the regional auth host (`auth.kimi.ai` internationally,
 * `auth.kimi.com` in mainland China).
 *
 * The flow:
 *   1. Resolve the Kimi Code region (env, persisted host, locale)
 *   2. Request device authorization from that auth host
 *   3. Display user_code and verification_uri to user
 *   4. Poll token endpoint until user authorizes or timeout
 *   5. Store access_token, refresh_token, and the oauth host (token.issuer)
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
    static std::string getUserAgent();

    /**
     * @brief Generate common Kimi metadata headers.
     *
     * The no-argument form matches the official Kimi Code client: it reads or
     * creates a stable local device id and includes it as X-Msh-Device-Id.
     * Callers with an OAuth token device id may pass it explicitly.
     *
     * @return Map of header names to values:
     *   - X-Msh-Platform: "kimi_code_cli"
     *   - X-Msh-Version: bare CLI version
     *   - X-Msh-Device-Name: hostname
     *   - X-Msh-Device-Model: OS and architecture info
     *   - X-Msh-Os-Version: OS version
     *   - X-Msh-Device-Id: stable local or caller-provided device id
     */
    static std::unordered_map<std::string, std::string> getCommonHeaders();
    static std::unordered_map<std::string, std::string> getCommonHeaders(
        std::string_view device_id);

    /// Last oauth host written by login(), if any.
    static std::string persisted_oauth_host();
    static void persist_oauth_host(std::string_view oauth_host);

    /// Region-resolved auth host for a new device-code login.
    static std::string resolved_oauth_host_for_login();
    /// Region-resolved auth host for token refresh / API routing.
    static std::string resolved_oauth_host_for_session();

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
    std::string oauth_host_;

    static std::string loadOrCreatePersistentDeviceId();

    DeviceAuthorization requestDeviceAuthorization();
    OAuthToken pollForToken(const DeviceAuthorization& auth);
    OAuthToken exchangeRefreshToken(std::string_view refresh_token);

    std::string getDeviceName() const;
};

} // namespace core::auth

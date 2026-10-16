#pragma once

#include "IOAuthFlow.hpp"
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace core::auth {

class IKimiSessionState;

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
 *   5. Return tokens and session metadata for the token manager to persist
 *
 * login() is blocking and may throw std::runtime_error on failure or timeout.
 * refresh() is blocking and may throw std::runtime_error on HTTP error.
 */
class KimiOAuthFlow : public IOAuthFlow {
public:
    explicit KimiOAuthFlow(std::shared_ptr<IKimiSessionState> session_state);

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;

    // Exposed for unit testing and protocol use
    static std::string generateDeviceId();
    static std::string getDeviceModel();
    static std::string getUserAgent();

    /**
     * @brief Generate common Kimi metadata headers.
     *
     * Device identity is supplied by the caller's session or OAuth token.
     *
     * @return Map of header names to values:
     *   - X-Msh-Platform: "kimi_code_cli"
     *   - X-Msh-Version: bare CLI version
     *   - X-Msh-Device-Name: hostname
     *   - X-Msh-Device-Model: OS and architecture info
     *   - X-Msh-Os-Version: OS version
     *   - X-Msh-Device-Id: stable local or caller-provided device id
     */
    static std::unordered_map<std::string, std::string> getCommonHeaders(
        std::string_view device_id);

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
    std::shared_ptr<IKimiSessionState> session_state_;

    void load_session_identity();

    DeviceAuthorization requestDeviceAuthorization();
    OAuthToken pollForToken(const DeviceAuthorization& auth);
    OAuthToken exchangeRefreshToken(std::string_view refresh_token);
};

} // namespace core::auth

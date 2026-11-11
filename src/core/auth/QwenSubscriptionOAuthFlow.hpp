#pragma once

/**
 * @file QwenSubscriptionOAuthFlow.hpp
 * @brief Qwen Subscription sign-in: OAuth 2.0 device authorization (RFC 8628).
 *
 * Mirrors Qwen Code's subscription client
 * (packages/core/src/qwen/subscription-oauth-client.ts):
 *
 *   POST {auth}/api/oauth/device/code   client_id + scope       → user code
 *   POST {auth}/api/oauth/token         device_code grant       → token pair
 *   POST {auth}/api/oauth/token         refresh_token grant     → token pair
 *   GET  {auth}/api/oauth/userinfo      Bearer                  → member_type
 *   POST {auth}/api/oauth/revoke        token + type hint
 *
 * The grant uses no PKCE. Polling honours `authorization_pending`,
 * `slow_down` (+5 s), and backs off on 429/5xx/network failures, respecting
 * Retry-After, until the device code expires.
 */

#include "IOAuthFlow.hpp"
#include "IOAuthTokenRevoker.hpp"

#include <chrono>
#include <functional>
#include <iosfwd>
#include <string>
#include <string_view>

namespace core::auth {

class QwenSubscriptionOAuthFlow final : public IOAuthFlow, public IOAuthTokenRevoker {
public:
    struct Options {
        /// Authorization server origin. Production is fixed; tests inject a local server.
        std::string auth_origin;
        /// Where the link and user code are printed.
        std::ostream* output = nullptr;
        /// Opens the approval page; best effort.
        std::function<void(std::string_view)> open_browser;
        /// Waits between polls; injectable so tests do not sleep.
        std::function<void(std::chrono::milliseconds)> sleep;
        std::chrono::milliseconds request_timeout{std::chrono::seconds(30)};
    };

    struct DeviceAuthorization {
        std::string device_code;
        std::string user_code;
        std::string verification_uri_complete;
        std::int64_t expires_in = 0;
        std::int64_t interval = 5;
    };

    struct UserInfo {
        std::string subject;
        std::string name;
        std::string email;
        std::string member_type;
    };

    QwenSubscriptionOAuthFlow();
    explicit QwenSubscriptionOAuthFlow(Options options);

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;
    void revoke(const OAuthToken& token) override;

    [[nodiscard]] DeviceAuthorization request_device_authorization() const;
    [[nodiscard]] OAuthToken poll_for_token(const DeviceAuthorization& authorization) const;
    /// Identity and membership of the signed-in account.
    [[nodiscard]] UserInfo user_info(std::string_view access_token) const;

private:
    Options options_;
};

} // namespace core::auth

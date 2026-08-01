#pragma once

#include "IOAuthFlow.hpp"
#include "IOAuthTokenRevoker.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace core::auth {

/** xAI OAuth2/OIDC flow compatible with the public Grok Build client. */
class XaiOAuthFlow final : public IOAuthFlow, public IOAuthTokenRevoker {
public:
    struct DiscoveryDocument {
        std::string authorization_endpoint;
        std::string token_endpoint;
        std::string revocation_endpoint; // optional — empty when not advertised
        // RFC 8628 device-authorization endpoint (optional — empty when the
        // issuer does not advertise device flow).
        std::string device_authorization_endpoint;
    };

    /// RFC 8628 device-authorization response.
    struct DeviceAuthorization {
        std::string device_code;
        std::string user_code;
        std::string verification_uri;
        std::string verification_uri_complete;
        int expires_in = 0;   // seconds
        int interval    = 5;  // minimum polling interval (seconds)
    };

    static constexpr std::string_view kIssuer = "https://auth.x.ai";
    static constexpr std::string_view kClientId =
        "b1a00492-073a-47ea-816f-4c329264a828";
    static constexpr std::string_view kReferrer = "grok-build";

    /// @param prefer_device_flow  Force the RFC 8628 device flow instead of the
    ///                            browser/localhost flow. Useful for SSH, Docker,
    ///                            and other headless environments.
    XaiOAuthFlow();
    explicit XaiOAuthFlow(bool prefer_device_flow);
    XaiOAuthFlow(std::string issuer,
                 std::string client_id,
                 std::vector<std::string> scopes,
                 std::string referrer = std::string(kReferrer),
                 bool prefer_device_flow = false);

    OAuthToken login() override;
    OAuthToken refresh(std::string_view refresh_token) override;

    /// Best-effort RFC 7009 revocation via the OIDC discovery
    /// `revocation_endpoint`, when the issuer advertises one.
    void revoke(const OAuthToken& token) override;

    [[nodiscard]] static std::vector<std::string> default_scopes();
    [[nodiscard]] static DiscoveryDocument parse_discovery_response(std::string_view json);
    [[nodiscard]] static DeviceAuthorization parse_device_authorization_response(
        std::string_view json);
    [[nodiscard]] static OAuthToken parse_token_response(
        std::string_view json,
        std::int64_t request_time_unix,
        std::string_view issuer,
        std::string_view client_id,
        std::string_view expected_nonce = {});
    [[nodiscard]] static std::string build_authorization_url(
        std::string_view authorization_endpoint,
        std::string_view client_id,
        std::string_view redirect_uri,
        const std::vector<std::string>& scopes,
        std::string_view state,
        std::string_view nonce,
        std::string_view code_challenge,
        std::string_view referrer);

private:
    [[nodiscard]] DiscoveryDocument discover() const;
    [[nodiscard]] OAuthToken browser_login(const DiscoveryDocument& discovery);
    [[nodiscard]] OAuthToken device_login(const DiscoveryDocument& discovery);
    [[nodiscard]] DeviceAuthorization request_device_authorization(
        const DiscoveryDocument& discovery) const;
    [[nodiscard]] OAuthToken exchange_code(const DiscoveryDocument& discovery,
                                           std::string_view code,
                                           std::string_view redirect_uri,
                                           std::string_view verifier,
                                           std::string_view nonce) const;

    std::string issuer_;
    std::string client_id_;
    std::vector<std::string> scopes_;
    std::string referrer_;
    bool prefer_device_flow_ = false;
};

} // namespace core::auth

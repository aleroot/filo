#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace core::auth {

/**
 * A provider has definitively rejected the refresh credential.
 *
 * This is deliberately narrower than a generic OAuth/network failure: callers
 * may use it to request interactive sign-in without prompting on timeouts,
 * rate limits, or provider outages.
 */
class OAuthRefreshRejected final : public std::runtime_error {
public:
    explicit OAuthRefreshRejected(std::string message)
        : std::runtime_error(std::move(message)) {}

    [[nodiscard]] static OAuthRefreshRejected by_provider(
        std::string_view provider_name) {
        return OAuthRefreshRejected(
            std::string(provider_name)
            + " rejected the saved refresh token. Sign in again to continue.");
    }
};

/**
 * The credential manager cannot recover without user interaction.
 *
 * OAuth flows throw OAuthRefreshRejected at the wire boundary; the token
 * manager enriches it with the provider/store identity before it escapes into
 * the request layer.
 */
class ReauthenticationRequired final : public std::runtime_error {
public:
    ReauthenticationRequired(std::string provider_id, std::string message)
        : std::runtime_error(std::move(message))
        , provider_id_(std::move(provider_id)) {}

    [[nodiscard]] std::string_view provider_id() const noexcept {
        return provider_id_;
    }

private:
    std::string provider_id_;
};

[[nodiscard]] inline bool oauth_error_is_invalid_grant(
    std::string_view response_body) noexcept {
    static constexpr std::array terminal_markers{
        std::string_view{"\"invalid_grant\""},
        std::string_view{"\"expired_token\""},
        std::string_view{"Refresh token expired"},
        std::string_view{"refresh token expired"},
        std::string_view{"token has been revoked"},
    };
    return std::ranges::any_of(
        terminal_markers,
        [response_body](std::string_view marker) {
            return response_body.contains(marker);
        });
}

} // namespace core::auth

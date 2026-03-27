#pragma once

#include <string>
#include <unordered_map>

namespace core::auth {

/**
 * @brief Auth information to inject into an HTTP request.
 *
 * Providers apply headers directly and append query_params to the URL.
 * Either map may be empty — e.g. API-key-in-URL providers leave headers
 * empty, OAuth providers leave query_params empty.
 *
 * Additional properties can be used by protocols for provider-specific
 * authentication requirements (e.g., Kimi OAuth device_id for X-Msh-Device-Id).
 */
struct AuthInfo {
    std::unordered_map<std::string, std::string> headers;
    std::unordered_map<std::string, std::string> query_params;
    
    /**
     * @brief Additional authentication properties.
     *
     * These are provider-specific values extracted from OAuth tokens
     * that protocols may need for constructing headers.
     * Example: "device_id" for Kimi OAuth (X-Msh-Device-Id header).
     */
    std::unordered_map<std::string, std::string> properties;
};

/**
 * @brief Produces AuthInfo for an HTTP request. Implementations may block
 *        on the first call (e.g. to refresh an expired token or open a
 *        browser login flow).
 */
class ICredentialSource {
public:
    virtual ~ICredentialSource() = default;
    virtual AuthInfo get_auth() = 0;

    /**
     * @brief Whether this credential source is backed by a subscription plan.
     *
     * Subscription/OAuth credentials do not map cleanly to per-token USD
     * billing, so callers should disable synthetic token-cost estimation.
     */
    [[nodiscard]] virtual bool uses_subscription_billing() const noexcept {
        return false;
    }
};

} // namespace core::auth

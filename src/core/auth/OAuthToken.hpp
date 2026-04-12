#pragma once

#include <string>
#include <chrono>
#include <vector>

namespace core::auth {

/**
 * @brief OAuth token with associated metadata.
 *
 * For Kimi OAuth, the device_id is extracted from the JWT token payload
 * and must be used in X-Msh-Device-Id header to match the token's device_id claim.
 * The server validates that the header matches the device_id embedded in the token.
 */
struct OAuthToken {
    std::string access_token;
    std::string refresh_token;
    std::string token_type = "Bearer";
    int64_t     expires_at = 0; // Unix timestamp (seconds)
    std::string device_id;      // Device ID extracted from token (for Kimi OAuth)
    std::string account_id;     // Optional provider account identifier
    std::string organization_id; // Optional organization UUID (Claude Team/Enterprise)
    std::string project_id;     // Optional Google Code Assist project identifier
    std::vector<std::string> scopes; // Optional OAuth scopes granted by provider

    bool is_valid() const noexcept {
        if (access_token.empty()) return false;
        auto now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        return now < expires_at - 300; // 5-minute safety buffer
    }

    bool has_refresh_token() const noexcept {
        return !refresh_token.empty();
    }

    /**
     * @brief Get the device ID to use for API requests.
     *
     * For Kimi OAuth, this returns the device_id embedded in the token,
     * which must match the X-Msh-Device-Id header sent with requests.
     * Falls back to the locally stored device ID if not available.
     */
    std::string get_effective_device_id() const {
        if (!device_id.empty()) {
            return device_id;
        }
        return {};
    }
};

} // namespace core::auth

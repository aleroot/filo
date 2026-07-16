#pragma once

#include "ICredentialSource.hpp"
#include "core/config/ConfigManager.hpp"
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core::auth {

class IOAuthTokenRevoker;

struct LoginResult {
    std::string provider;
    std::string login_provider;
    std::vector<std::string> hints;
};

class IAuthStrategy {
public:
    virtual ~IAuthStrategy() = default;

    virtual std::string_view login_provider() const noexcept = 0;
    virtual std::string_view display_name() const noexcept = 0;

    virtual bool supports(std::string_view provider_type,
                          std::string_view auth_type) const noexcept = 0;

    virtual std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& provider_config,
        std::string_view config_dir) const = 0;

    virtual void login(std::string_view config_dir) const = 0;

    virtual std::vector<std::string> post_login_hints() const {
        return {};
    }

    /** Persistent token-store key, or empty when this strategy is not token-backed. */
    [[nodiscard]] virtual std::string_view token_store_key() const noexcept {
        return {};
    }

    /**
     * Flow used for best-effort server-side token revocation on logout.
     * Return nullptr when the provider has no public revocation endpoint
     * (logout then only clears local credentials).
     */
    [[nodiscard]] virtual std::shared_ptr<IOAuthTokenRevoker>
    logout_revocation_flow() const {
        return nullptr;
    }
};

/**
 * @brief Registry + orchestrator for authentication strategies.
 *
 * Strategy pattern:
 *  - `--login <provider>` is dispatched via login_provider()
 *  - runtime credential resolution uses (provider.type, provider.auth_type)
 *
 * This removes provider-specific auth branches from app entrypoints and
 * centralises auth behaviour in one extensible component.
 */
class AuthenticationManager {
public:
    explicit AuthenticationManager(std::string config_dir);

    static AuthenticationManager create_with_defaults(std::string config_dir);

    void register_strategy(std::shared_ptr<IAuthStrategy> strategy);

    LoginResult login(std::string_view provider) const;

    /**
     * Sign out of a login provider: best-effort server-side token revocation
     * (when the provider supports it), then clear cached local credentials.
     *
     * @param revoke_remote  Set to false to skip the network revocation step
     *                       (used by tests and offline flows).
     * @return The strategy display name for user feedback.
     */
    std::string logout(std::string_view provider, bool revoke_remote = true) const;

    /**
     * @param canonical_type  The canonical provider kind string (e.g. "claude",
     *                        "gemini") used to match OAuth strategies.  For
     *                        built-in providers pass the registry prefix; for
     *                        user-defined providers pass the full map key.
     */
    std::shared_ptr<ICredentialSource> create_credential_source(
        std::string_view canonical_type,
        const core::config::ProviderConfig& provider_config) const;

    std::vector<std::string> available_login_providers() const;

private:
    std::string config_dir_;
    std::vector<std::shared_ptr<IAuthStrategy>> strategies_;
};

} // namespace core::auth

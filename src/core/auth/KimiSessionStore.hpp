#pragma once

#include "FileTokenStore.hpp"
#include "IKimiSessionState.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace core::auth {

/// Owns Kimi's session persistence. Tokens and refresh locking are delegated to
/// FileTokenStore; the pre-login device identity and last host use the existing
/// companion files in that same directory.
class KimiSessionStore final : public ITokenStore, public IKimiSessionState {
public:
    explicit KimiSessionStore(std::string directory = {});

    std::optional<OAuthToken> load(std::string_view provider_id) override;
    void save(std::string_view provider_id, const OAuthToken& token) override;
    void clear(std::string_view provider_id) override;
    [[nodiscard]] std::unique_ptr<ITokenStoreLock> acquire_refresh_lock(std::string_view provider_id) override;

    [[nodiscard]] KimiSessionIdentity load_identity() override;
    [[nodiscard]] std::string device_id();
    [[nodiscard]] std::string persisted_oauth_host() const;

private:
    std::filesystem::path directory_;
    FileTokenStore tokens_;
};

} // namespace core::auth

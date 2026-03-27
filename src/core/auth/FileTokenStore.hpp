#pragma once

#include "ITokenStore.hpp"
#include <string>

namespace core::auth {

/**
 * @brief Persists OAuth tokens to ~/.config/filo/oauth_<provider_id>.json
 *        with file permissions 0600.
 */
class FileTokenStore : public ITokenStore {
public:
    explicit FileTokenStore(std::string config_dir);

    std::optional<OAuthToken> load(std::string_view provider_id) override;
    void save(std::string_view provider_id, const OAuthToken& token) override;
    void clear(std::string_view provider_id) override;

private:
    std::string token_path(std::string_view provider_id) const;
    std::string config_dir_;
};

} // namespace core::auth

#pragma once

#include "ITokenStore.hpp"
#include <optional>
#include <string>
#include <string_view>

namespace core::auth {

/**
 * @brief Persists OAuth tokens to ~/.config/filo/oauth_<provider_id>.json
 *        with file permissions 0600.
 */
class FileTokenStore : public ITokenStore {
public:
    explicit FileTokenStore(std::string config_dir);

    /// The file a provider's token lives in: oauth_<provider_id>.json.
    [[nodiscard]] static std::string file_name(std::string_view provider_id);
    /// Inverse of file_name; nullopt for any other file name.
    [[nodiscard]] static std::optional<std::string> provider_id_from_file_name(
        std::string_view file_name);
    /// Whether untrusted text can safely become a provider id (and so part of
    /// a file name): ASCII letters, digits, '-', '_' and '.', not leading '.'.
    [[nodiscard]] static bool is_valid_provider_id(std::string_view provider_id) noexcept;

    std::optional<OAuthToken> load(std::string_view provider_id) override;
    void save(std::string_view provider_id, const OAuthToken& token) override;
    void clear(std::string_view provider_id) override;
    [[nodiscard]] std::unique_ptr<ITokenStoreLock>
    acquire_refresh_lock(std::string_view provider_id) override;

private:
    std::string token_path(std::string_view provider_id) const;
    std::string config_dir_;
};

} // namespace core::auth

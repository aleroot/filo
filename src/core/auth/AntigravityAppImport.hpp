#pragma once

/**
 * @file AntigravityAppImport.hpp
 * @brief Reads the Google credential an installed Antigravity IDE holds.
 *
 * The Antigravity app signs in with the *same* OAuth client Filo authenticates
 * as — the `aud` of its id_token is the Antigravity client id — so its token can
 * bootstrap Filo's `google` session once (`--auth google --from …`; see
 * AppCredentialImport.hpp for the shared pipeline). Nothing here is consulted at
 * inference time.
 *
 * Two things the app cannot supply:
 *   - the client *secret*. The literals inside the app bundle do not authenticate
 *     against the token endpoint, so renewing an imported refresh token still
 *     needs `GOOGLE_ANTIGRAVITY_CLIENT_SECRET` or `<config_dir>/antigravity.json`
 *     (see AntigravityClientCredentials.hpp);
 *   - a fresh token in its file on a machine where the app has not run
 *     recently, because the app writes that file only at launch.
 */

#include "StoredAppCredentialReader.hpp"

#include <filesystem>
#include <memory>
#include <string_view>
#include <utility>

namespace core::auth::antigravity {

/**
 * @brief IAppCredentialReader for the Antigravity IDE.
 *
 * The app stores its sign-in through go-keyring under `service=gemini`,
 * `account=antigravity` — the macOS login keychain, or the Secret Service on
 * Linux — and keeps that item refreshed while it runs. It also writes
 * `~/.gemini/jetski-standalone-oauth-token`, but only at launch. Sources are
 * resolved by StoredAppCredentialReader, which prefers the keyring.
 */
class AppCredentialReader final : public StoredAppCredentialReader {
public:
    explicit AppCredentialReader(std::shared_ptr<const IKeyring> keyring = system_keyring())
        : StoredAppCredentialReader(std::move(keyring)) {}

    [[nodiscard]] std::string_view app_name() const noexcept override { return "Antigravity"; }

protected:
    [[nodiscard]] std::filesystem::path default_directory() const override;
    [[nodiscard]] std::string_view file_name() const noexcept override;
    [[nodiscard]] std::optional<KeyringEntry> keyring_entry(
        const std::filesystem::path& directory) const override;
    [[nodiscard]] AppCredential parse(std::string_view payload,
                                      std::string_view origin) const override;
};

/**
 * @brief Parse the app's credential JSON.
 *
 * Accepts both the raw document and go-keyring's `go-keyring-base64:` wrapper
 * (used in the macOS keychain),
 * and both the app's `{"id_token":…, "token":{…}}` nesting and a flat document.
 * The client id and account email come from the id_token's claims. An absent or
 * unparseable `expiry` leaves `expires_at` at 0 (unknown).
 *
 * @throws std::runtime_error when the payload is not a usable credential.
 */
[[nodiscard]] AppCredential parse_app_token(std::string_view payload, std::string_view origin);

} // namespace core::auth::antigravity

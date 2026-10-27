#pragma once

/**
 * @file ChatGptAppImport.hpp
 * @brief Reads the OpenAI credential an installed ChatGPT/Codex app holds.
 *
 * OpenAI's desktop app (bundle id `com.openai.codex`, shipped as ChatGPT.app)
 * and the Codex CLI sign in with the same public OAuth client as Filo's
 * `openai` login. They keep the result, depending on Codex's
 * `cli_auth_credentials_store` setting, in `$CODEX_HOME/auth.json`
 * (`~/.codex/auth.json` by default; the default `file` mode) or in the OS
 * keyring (`keyring`/`auto` modes) as service `Codex Auth`, account
 * `cli|<first 16 hex digits of SHA-256(canonical CODEX_HOME)>`. Writing the
 * keyring item deletes the file, and Codex reads the keyring first. Both hold
 * the same document:
 *
 *     {"auth_mode":"chatgpt","OPENAI_API_KEY":null,
 *      "tokens":{"id_token":…,"access_token":…,"refresh_token":…,"account_id":…},
 *      "last_refresh":…}
 *
 * so that session can bootstrap Filo's `openai` login once
 * (`--auth openai --from …`; see AppCredentialImport.hpp). The `tokens` object
 * has the shape of an OpenAI token response, so it is decoded by
 * OpenAIOAuthFlow::parse_token_response — the account id, email and expiry
 * come from the same claim parsing a browser login uses.
 */

#include "StoredAppCredentialReader.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace core::auth::chatgpt {

/**
 * @brief IAppCredentialReader for the ChatGPT/Codex app.
 *
 * Sources are resolved by StoredAppCredentialReader; a directory is a
 * `CODEX_HOME`, so `--from ~/.codex` also finds a keyring-stored sign-in.
 */
class AppCredentialReader final : public StoredAppCredentialReader {
public:
    explicit AppCredentialReader(std::shared_ptr<const IKeyring> keyring = system_keyring())
        : StoredAppCredentialReader(std::move(keyring)) {}

    [[nodiscard]] std::string_view app_name() const noexcept override { return "ChatGPT"; }

protected:
    [[nodiscard]] std::filesystem::path default_directory() const override;
    [[nodiscard]] std::string_view file_name() const noexcept override { return "auth.json"; }
    [[nodiscard]] std::optional<KeyringEntry> keyring_entry(
        const std::filesystem::path& directory) const override;
    [[nodiscard]] AppCredential parse(std::string_view payload,
                                      std::string_view origin) const override;
};

/// `$CODEX_HOME` when set, else `~/.codex`.
[[nodiscard]] std::filesystem::path codex_home();

/// The keyring item Codex keeps for @p codex_home (see the file comment).
[[nodiscard]] KeyringEntry keyring_entry_for(const std::filesystem::path& codex_home);

/**
 * @brief Parse an `auth.json` document.
 *
 * The expiry is the access token's `exp` claim (else the id_token's); a token
 * carrying neither is reported as already expired, never given a guessed life.
 *
 * @throws std::runtime_error when the file holds no ChatGPT sign-in — including
 *         an API-key-only login, which is reported without echoing the key.
 */
[[nodiscard]] AppCredential parse_auth_file(std::string_view payload, std::string_view origin);

} // namespace core::auth::chatgpt

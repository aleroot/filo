#pragma once

/**
 * @file StoredAppCredentialReader.hpp
 * @brief The IAppCredentialReader for apps that keep their sign-in in a file,
 *        in the OS keyring, or in either.
 *
 * Every such app is read the same way; only where it stores the credential and
 * how the payload is encoded differ. This class owns the shared algorithm
 * (Template Method) and asks a subclass for the app-specific parts:
 *
 *   - default_directory(): the app's data directory (`~/.gemini`, `$CODEX_HOME`);
 *   - file_name():         its credential file inside that directory;
 *   - keyring_entry():     the keyring item it keeps for a directory, if any;
 *   - parse():             decoding the payload (same format in file and keyring).
 *
 * `read(source)` accepts:
 *   - `auto` or empty: the keyring item, else the file in the default directory.
 *     The keyring comes first because apps that use one keep it current, and
 *     either leave no file or one written less often;
 *   - `keyring` (alias `keychain`): the keyring item only;
 *   - a path to the credential file;
 *   - a directory: the file inside it, else the keyring item kept for that
 *     directory, else auto-detection. A directory without the file is either a
 *     data directory whose app stores into the keyring, or an app bundle, which
 *     holds no user credential.
 *
 * When nothing is found the error lists every location tried, and why a
 * keyring could not be consulted (no backend, helper missing, ...).
 */

#include "AppCredentialImport.hpp"
#include "SystemKeyring.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace core::auth {

class StoredAppCredentialReader : public IAppCredentialReader {
public:
    [[nodiscard]] AppCredential read(std::string_view source) const final;

protected:
    explicit StoredAppCredentialReader(std::shared_ptr<const IKeyring> keyring);

    /// Where the app keeps its data when the user names no directory.
    [[nodiscard]] virtual std::filesystem::path default_directory() const = 0;

    /// The credential file's name inside an app data directory.
    [[nodiscard]] virtual std::string_view file_name() const noexcept = 0;

    /// The keyring item the app keeps for @p directory; nullopt when it never
    /// stores its credential in a keyring.
    [[nodiscard]] virtual std::optional<KeyringEntry> keyring_entry(
        const std::filesystem::path& directory) const = 0;

    /**
     * @brief Decode a payload read from @p origin (a `~`-abbreviated path or a
     *        keyring description).
     * @throws std::runtime_error when it is not a usable credential.
     */
    [[nodiscard]] virtual AppCredential parse(std::string_view payload,
                                              std::string_view origin) const = 0;

private:
    std::shared_ptr<const IKeyring> keyring_;
};

} // namespace core::auth

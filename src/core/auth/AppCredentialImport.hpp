#pragma once

/**
 * @file AppCredentialImport.hpp
 * @brief One-time adoption of the credential a provider's own installed app holds.
 *
 * A machine where the provider's first-party app (the Antigravity IDE, the
 * ChatGPT/Codex app, ...) is already signed in holds a token for the same OAuth
 * client Filo authenticates as, so `filo --auth <provider> --from <source>` can
 * copy it instead of running a browser login. The copy happens once, on request:
 * afterwards only Filo's own token store is used and nothing reads the app again.
 *
 * Responsibilities are split so that supporting another app means writing one
 * reader and nothing else:
 *   - IAppCredentialReader knows where one app keeps its credential and how to
 *     decode it. It performs no network I/O and stores nothing. Apps that keep
 *     it in a file and/or the OS keyring derive from StoredAppCredentialReader,
 *     which owns source resolution and supplies only locations and a parser.
 *   - IAuthStrategy::app_credential_import() pairs that reader with the
 *     provider's own OAuth flow (which renews the token, now and later) and any
 *     caveat about adopting it.
 *   - import_app_credential() is the single pipeline every provider shares:
 *     read, renew when already expired, store.
 */

#include "IOAuthFlow.hpp"
#include "ITokenStore.hpp"
#include "OAuthToken.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core::auth {

/// A credential read from an installed app, before Filo stores it.
struct AppCredential {
    OAuthToken token;
    /// Where it was read from, for messages (`~`-abbreviated path or keyring item).
    std::string origin;
};

/**
 * @brief Reads the credential one installed app keeps for its own use.
 *
 * `read()` resolves @p source — empty or `auto` to discover the app's default
 * location, otherwise an app-specific keyword or a path — and decodes what it
 * finds into an OAuthToken. A token whose expiry cannot be determined must keep
 * `expires_at == 0` rather than guess, so the pipeline renews it instead of
 * storing a credential it knows nothing about.
 *
 * @throws std::runtime_error naming what was tried when nothing usable is found.
 */
class IAppCredentialReader {
public:
    virtual ~IAppCredentialReader() = default;

    /// Product name for messages, e.g. "Antigravity" or "ChatGPT".
    [[nodiscard]] virtual std::string_view app_name() const noexcept = 0;

    [[nodiscard]] virtual AppCredential read(std::string_view source) const = 0;
};

/// A provider's support for `--auth <provider> --from <source>`.
struct AppCredentialImport {
    std::shared_ptr<const IAppCredentialReader> reader;
    /// The provider's own OAuth flow. It renews an imported token that has
    /// already expired, and every later refresh goes through it as usual.
    std::shared_ptr<IOAuthFlow> flow;
    /// Caveats about the adopted credential that the user must hear now rather
    /// than when it stops working (it cannot be renewed, it is shared, ...).
    std::vector<std::string> warnings;
};

/// What an import stored, for the caller to report.
struct AppImportReport {
    std::string app_name;
    std::string origin;
    std::string email;
    /// The access token had expired and was renewed through the provider's flow.
    bool renewed = false;
    /// Expiry of the stored access token, Unix seconds.
    std::int64_t expires_at = 0;
    std::vector<std::string> warnings;
};

/**
 * @brief Read an app credential and store it as Filo's session for @p store_key.
 *
 * A token that has expired, or whose expiry is unknown, is renewed through
 * `import.flow` first, so a credential that would die on its first request is
 * never stored. Nothing is written when reading or renewing fails: an existing
 * session survives a failed import.
 *
 * @throws std::runtime_error explaining why the credential cannot be adopted.
 */
[[nodiscard]] AppImportReport import_app_credential(const AppCredentialImport& import,
                                                    std::string_view source,
                                                    std::string store_key,
                                                    std::shared_ptr<ITokenStore> store);

} // namespace core::auth

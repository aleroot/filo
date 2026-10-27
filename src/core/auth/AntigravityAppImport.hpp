#pragma once

/**
 * @file AntigravityAppImport.hpp
 * @brief One-time bootstrap of Filo's Google credential from an Antigravity install.
 *
 * A machine that already has Google's Antigravity IDE signed in holds a valid
 * OAuth token for the *same* client Filo authenticates as — the `aud` of that
 * token's id_token is the Antigravity client id — so importing it saves one
 * browser login. That is all this does: `--auth google --from …` reads the app's
 * credential once, on request, and from then on only Filo's own token store is
 * used. Nothing here is consulted at inference time.
 *
 * Two things the app cannot supply:
 *   - the client *secret*. The literals inside the app bundle do not authenticate
 *     against the token endpoint, so renewing an imported refresh token still
 *     needs `GOOGLE_ANTIGRAVITY_CLIENT_SECRET` or `<config_dir>/antigravity.json`
 *     (see AntigravityClientCredentials.hpp);
 *   - a fresh token on a machine where the app has not run recently, because the
 *     app writes its file only at launch.
 */

#include <cstdint>
#include <string>
#include <string_view>

namespace core::auth::antigravity {

/// What an Antigravity installation yielded.
struct AppImport {
    std::string access_token;
    std::string refresh_token;
    /// `aud` of the id_token: the OAuth client the app signed in with.
    std::string client_id;
    /// Account the app is signed in with, when the id_token carries one.
    std::string email;
    /// Unix seconds from the stored `expiry`; 0 when absent or unparseable.
    std::int64_t expires_at = 0;
    /// Where the credential came from, for messages (`~`-abbreviated).
    std::string source;
};

/**
 * @brief Read an Antigravity credential from @p source.
 *
 * @p source may be:
 *   - `keychain` — the macOS login keychain item the app keeps refreshed while
 *     it runs (`service=gemini`, `account=antigravity`);
 *   - `auto` or empty — the keychain when it holds one, else
 *     `~/.gemini/jetski-standalone-oauth-token`;
 *   - a path to that JSON file, to a directory containing it (such as
 *     `~/.gemini`), or to an `Antigravity.app` bundle. A bundle carries no user
 *     credential, so it falls back to auto-detection.
 *
 * @throws std::runtime_error naming what was tried when nothing usable is found.
 */
[[nodiscard]] AppImport import_from_app(std::string_view source);

/// The file the Antigravity app writes at launch, below @p home_dir.
[[nodiscard]] std::string app_token_file(std::string_view home_dir);

/**
 * @brief Parse the app's credential JSON.
 *
 * Accepts both the raw document and the keychain's `go-keyring-base64:` wrapper,
 * and both the app's `{"id_token":…, "token":{…}}` nesting and a flat document.
 * The client id is recovered from the id_token's `aud` claim.
 *
 * @throws std::runtime_error when the payload is not a usable credential.
 */
[[nodiscard]] AppImport parse_app_token(std::string_view payload, std::string_view source);

} // namespace core::auth::antigravity

#pragma once

/**
 * @file MimoOAuthFlow.hpp
 * @brief Xiaomi MiMo browser sign-in (sealed-box key delivery).
 *
 * MiMo's console does not implement OAuth 2.0. It provisions a long-lived API
 * key and hands it back sealed to a one-shot X25519 public key that the CLI
 * generates per login:
 *
 *   1. the client generates an ephemeral X25519 keypair and opens
 *      `{platform}/authorize?pk=…&redirect_uri=…&kn=mimocode&key_name=…`
 *   2. after the user approves, the console redirects to the loopback server
 *      with a `u` query parameter
 *   3. `u` is base64url( ephemeral_public_key(32) || nonce(12) || ciphertext ||
 *      tag(16) ), encrypted with AES-256-GCM under
 *      SHA-256( X25519(our_private_key, ephemeral_public_key) )
 *   4. the plaintext is `{"sk": "<api key>", "uid": "…", "url": "<base url>"}`
 *
 * Because the result is a plain API key rather than a refreshable token, the
 * login strategy persists it through the normal API-key overlay; there is no
 * token store or refresh path.
 */

#include "ui/AuthUI.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace core::auth {

/// Decrypted payload of the console's sealed response.
struct MimoAuthorizationGrant {
    std::string api_key;   ///< `sk` — the provisioned API key.
    std::string user_id;   ///< `uid` — the MiMo account identifier.
    std::string base_url;  ///< `url` — the gateway this key is issued for.
};

/**
 * One MiMo browser login attempt.
 *
 * Holds the ephemeral private key for the lifetime of the flow so a pasted
 * code can still be decrypted after the loopback callback times out.
 */
class MimoOAuthFlow {
public:
    MimoOAuthFlow();

    /// `pk` query value: base64url of the SPKI DER encoding of the public key.
    [[nodiscard]] std::string public_key_parameter() const;

    /// Authorization URL for @p redirect_uri, carrying `pk` and `key_name`.
    [[nodiscard]] std::string authorize_url(std::string_view redirect_uri) const;

    /**
     * Authorization URL whose redirect target is the console's own
     * copy-the-code page, for when the loopback server cannot be reached.
     */
    [[nodiscard]] std::string manual_authorize_url() const;

    /// Decrypt a `u` parameter (or pasted code). Empty on any failure.
    [[nodiscard]] std::optional<MimoAuthorizationGrant> decrypt_grant(
        std::string_view sealed_base64url) const;

    /**
     * Run the full interactive login: bind a loopback server, open the
     * browser, wait for the callback, and fall back to a pasted code.
     * Throws std::runtime_error when no grant could be obtained.
     */
    [[nodiscard]] MimoAuthorizationGrant login(ui::AuthUI& ui) const;

    /**
     * Stable per-installation key name, so repeat logins reuse one key on the
     * console instead of provisioning a new one each time. Persisted under
     * @p config_dir; generated on first use.
     */
    [[nodiscard]] static std::string key_name(std::string_view config_dir);

    /// Set the key name used by subsequent authorize URLs.
    void set_key_name(std::string key_name);

private:
    std::array<std::uint8_t, 32> private_key_{};
    std::array<std::uint8_t, 32> public_key_{};
    std::string key_name_;
};

/**
 * Decrypt a MiMo sealed box with an explicit private key.
 * Exposed for tests; `MimoOAuthFlow::decrypt_grant` is the normal entry point.
 */
[[nodiscard]] std::optional<MimoAuthorizationGrant> mimo_decrypt_sealed_grant(
    std::span<const std::uint8_t, 32> private_key,
    std::string_view sealed_base64url);

/// SPKI DER prefix for an X25519 public key (RFC 8410 id-X25519).
inline constexpr std::array<std::uint8_t, 12> kX25519SpkiPrefix{
    0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x6e, 0x03, 0x21, 0x00,
};

} // namespace core::auth

#include "MimoOAuthFlow.hpp"

#include "AuthBrowserLauncher.hpp"
#include "OAuthLoopback.hpp"
#include "OAuthPkce.hpp"
#include "core/llm/MimoModelTraits.hpp"
#include "core/logging/Logger.hpp"
#include "core/utils/Base64.hpp"
#include "core/utils/Crypto.hpp"
#include "core/utils/StringUtils.hpp"
#include "core/utils/JsonUtils.hpp"

#include <simdjson.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace core::auth {
namespace {

namespace crypto = core::utils::crypto;

/// Sealed layout: ephemeral public key (32) || nonce (12) || ciphertext || tag (16).
constexpr std::size_t kEphemeralKeyBytes = 32;
constexpr std::size_t kNonceBytes = 12;
constexpr std::size_t kTagBytes = 16;
constexpr std::size_t kMinimumSealedBytes =
    kEphemeralKeyBytes + kNonceBytes + kTagBytes;

[[nodiscard]] std::optional<MimoAuthorizationGrant> parse_grant_json(
    std::string_view json) {
    thread_local simdjson::dom::parser parser;
    const core::utils::json::ParserRetentionGuard parser_guard{parser};
    simdjson::padded_string padded(json);
    simdjson::dom::element document;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS) {
        return std::nullopt;
    }

    MimoAuthorizationGrant grant;
    std::string_view value;
    if (document["sk"].get(value) == simdjson::SUCCESS) {
        grant.api_key = std::string(value);
    }
    if (document["uid"].get(value) == simdjson::SUCCESS) {
        grant.user_id = std::string(value);
    }
    if (document["url"].get(value) == simdjson::SUCCESS) {
        grant.base_url = std::string(value);
    }

    // A grant without a key is unusable, however well-formed the envelope was.
    if (grant.api_key.empty()) return std::nullopt;
    return grant;
}

[[nodiscard]] std::filesystem::path key_name_path(std::string_view config_dir) {
    return std::filesystem::path(config_dir) / "mimo_key_name";
}

[[nodiscard]] std::string random_key_suffix() {
    static constexpr char kHex[] = "0123456789abcdef";
    const auto bytes = crypto::random_bytes(4);
    std::string suffix;
    suffix.reserve(bytes.size() * 2);
    for (const auto byte : bytes) {
        suffix.push_back(kHex[byte >> 4]);
        suffix.push_back(kHex[byte & 0x0f]);
    }
    return suffix;
}

} // namespace

std::optional<MimoAuthorizationGrant> mimo_decrypt_sealed_grant(
    std::span<const std::uint8_t, 32> private_key,
    std::string_view sealed_base64url) {
    const std::string trimmed =
        core::utils::str::trim_ascii_copy(sealed_base64url);
    if (trimmed.empty()) return std::nullopt;

    const auto decoded = core::utils::Base64::decode_url(trimmed);
    if (!decoded || decoded->size() < kMinimumSealedBytes) return std::nullopt;

    const auto* bytes = reinterpret_cast<const std::uint8_t*>(decoded->data());
    const std::size_t size = decoded->size();

    std::array<std::uint8_t, kEphemeralKeyBytes> ephemeral_public{};
    std::copy_n(bytes, kEphemeralKeyBytes, ephemeral_public.begin());

    const std::span<const std::uint8_t> nonce(bytes + kEphemeralKeyBytes,
                                              kNonceBytes);
    const std::size_t body_offset = kEphemeralKeyBytes + kNonceBytes;
    const std::size_t body_size = size - body_offset;
    const std::size_t ciphertext_size = body_size - kTagBytes;

    const std::span<const std::uint8_t> ciphertext(bytes + body_offset,
                                                   ciphertext_size);
    std::array<std::uint8_t, kTagBytes> tag{};
    std::copy_n(bytes + body_offset + ciphertext_size, kTagBytes, tag.begin());

    // The console derives the content key as SHA-256 of the raw ECDH output.
    const auto shared_secret = crypto::x25519(private_key, ephemeral_public);
    const auto content_key = crypto::sha256(std::span<const std::uint8_t>(
        shared_secret.data(), shared_secret.size()));

    const auto plaintext = crypto::aes_256_gcm_decrypt(
        content_key, nonce, ciphertext, tag);
    if (!plaintext) return std::nullopt;

    return parse_grant_json(*plaintext);
}

MimoOAuthFlow::MimoOAuthFlow() {
    private_key_ = crypto::x25519_generate_private_key();
    public_key_ = crypto::x25519_public_key(private_key_);
}

void MimoOAuthFlow::set_key_name(std::string key_name) {
    key_name_ = std::move(key_name);
}

std::string MimoOAuthFlow::public_key_parameter() const {
    // The console expects the SPKI DER encoding, not the bare 32-byte key.
    std::array<std::uint8_t, kX25519SpkiPrefix.size() + 32> spki{};
    std::copy(kX25519SpkiPrefix.begin(), kX25519SpkiPrefix.end(), spki.begin());
    std::copy(public_key_.begin(), public_key_.end(),
              spki.begin() + static_cast<std::ptrdiff_t>(kX25519SpkiPrefix.size()));
    return core::utils::Base64::encode_url(spki);
}

std::string MimoOAuthFlow::authorize_url(std::string_view redirect_uri) const {
    using core::auth::oauth_pkce::url_encode;

    std::string url = core::llm::mimo_platform_url();
    url += "/authorize?pk=";
    url += url_encode(public_key_parameter());
    url += "&redirect_uri=";
    url += url_encode(redirect_uri);
    // `kn` identifies the client kind the console provisions a key for.
    url += "&kn=mimocode";
    if (!key_name_.empty()) {
        url += "&key_name=";
        url += url_encode(key_name_);
    }
    return url;
}

std::string MimoOAuthFlow::manual_authorize_url() const {
    return authorize_url(core::llm::mimo_platform_url()
                         + "/authorize/code/callback");
}

std::optional<MimoAuthorizationGrant> MimoOAuthFlow::decrypt_grant(
    std::string_view sealed_base64url) const {
    return mimo_decrypt_sealed_grant(private_key_, sealed_base64url);
}

MimoAuthorizationGrant MimoOAuthFlow::login(ui::AuthUI& ui) const {
    OAuthLoopbackOptions options;
    // The console redirects to the bare origin and carries the sealed payload
    // in `u`; it registers no fixed callback port.
    options.fixed_port = 0;
    options.callback_path = "/";
    options.code_param = "u";
    options.timeout = std::chrono::minutes(5);

    std::unique_ptr<OAuthLoopbackServer> server;
    try {
        server = std::make_unique<OAuthLoopbackServer>(std::move(options));
        server->start();
    } catch (const std::exception& error) {
        core::logging::warn("MiMo login: loopback server unavailable ({})",
                            error.what());
        server.reset();
    }

    if (server) {
        const std::string url = authorize_url(server->redirect_uri());
        ui.show_url(url, "Approve the login in your browser:");
        open_browser(url);

        OAuthLoopbackResult result = server->wait();
        if (!result.code.empty()) {
            if (auto grant = decrypt_grant(result.code)) {
                return *grant;
            }
            // A callback that cannot be opened is a real failure, not a
            // timeout: fall through to the manual path rather than silently
            // accepting a corrupted envelope.
            ui.show_error("Could not decrypt the authorization response.");
        } else if (!result.error.empty()) {
            ui.show_error("Browser login failed: " + result.error);
        }
    }

    ui.show_instructions(
        "Open the URL below, approve the login, then paste the code the page "
        "displays.");
    ui.show_url(manual_authorize_url(), "Authorization URL:");
    const std::string pasted = ui.prompt_secret("Authorization code:");
    if (auto grant = decrypt_grant(pasted)) {
        return *grant;
    }

    throw std::runtime_error(
        "MiMo login did not produce a usable API key. The code may have "
        "expired or belong to a different login attempt.");
}

std::string MimoOAuthFlow::key_name(std::string_view config_dir) {
    const std::filesystem::path path = key_name_path(config_dir);

    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
        std::ifstream input(path);
        std::string existing;
        if (input && std::getline(input, existing)) {
            existing = core::utils::str::trim_ascii_copy(existing);
            if (!existing.empty()) return existing;
        }
    }

    const std::string generated = "filo-cli-key-" + random_key_suffix();
    std::filesystem::create_directories(path.parent_path(), ec);
    if (std::ofstream output(path); output) {
        output << generated << '\n';
    }
    return generated;
}

} // namespace core::auth

#include "ChatGptAppImport.hpp"

#include "OpenAIOAuthFlow.hpp"
#include "../utils/Crypto.hpp"
#include "../utils/PathUtils.hpp"

#include <simdjson.h>

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

namespace core::auth::chatgpt {

namespace {

[[nodiscard]] bool has_string(simdjson::dom::element element, std::string_view key) {
    std::string_view value;
    return element[key].get(value) == simdjson::SUCCESS && !value.empty();
}

} // namespace

std::filesystem::path codex_home() {
    if (const char* configured = std::getenv("CODEX_HOME");
        configured != nullptr && configured[0] != '\0') {
        return core::utils::path::expand_user_path(configured);
    }
    return core::utils::path::home_directory() / ".codex";
}

KeyringEntry keyring_entry_for(const std::filesystem::path& codex_home) {
    // Codex hashes the canonical path, falling back to the path as given.
    std::error_code ec;
    std::filesystem::path canonical = std::filesystem::canonical(codex_home, ec);
    if (ec) canonical = codex_home;
    return KeyringEntry{
        .service = "Codex Auth",
        .account = "cli|" + core::utils::crypto::sha256_hex(canonical.string()).substr(0, 16),
    };
}

AppCredential parse_auth_file(std::string_view payload, std::string_view origin) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(payload.data(), payload.size());
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS || !doc.is_object()) {
        throw std::runtime_error(
            "The ChatGPT credential in " + std::string(origin) +
            " is not a JSON object, so it cannot be imported.");
    }

    simdjson::dom::element tokens;
    if (doc["tokens"].get(tokens) != simdjson::SUCCESS || !tokens.is_object() ||
        !has_string(tokens, "access_token")) {
        // An API-key login stores the key instead of tokens. Say so without
        // echoing it: the key is usable as OPENAI_API_KEY, not as a sign-in.
        if (has_string(doc, "OPENAI_API_KEY")) {
            throw std::runtime_error(
                std::string(origin) + " holds an OpenAI API key rather than a ChatGPT "
                "sign-in, so there is no session to import. Sign in to the ChatGPT app with "
                "your ChatGPT account, or configure that key as OPENAI_API_KEY instead.");
        }
        throw std::runtime_error(
            "The ChatGPT credential in " + std::string(origin) +
            " carries no access_token. Sign in to the ChatGPT app once, or drop --from "
            "and complete a browser login instead.");
    }

    // `tokens` has the shape of an OpenAI token response. The file records no
    // issue time, so anchoring a relative lifetime at the epoch makes a token
    // whose claims carry no `exp` count as expired: the import then renews it
    // instead of trusting a guessed lifetime.
    AppCredential imported;
    imported.origin = std::string(origin);
    imported.token = OpenAIOAuthFlow::parse_token_response(
        simdjson::to_string(tokens), /*request_time_unix=*/0);
    return imported;
}

std::filesystem::path AppCredentialReader::default_directory() const {
    return codex_home();
}

std::optional<KeyringEntry> AppCredentialReader::keyring_entry(
    const std::filesystem::path& directory) const {
    return keyring_entry_for(directory);
}

AppCredential AppCredentialReader::parse(std::string_view payload,
                                         std::string_view origin) const {
    return parse_auth_file(payload, origin);
}

} // namespace core::auth::chatgpt

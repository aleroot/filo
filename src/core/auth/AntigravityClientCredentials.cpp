#include "AntigravityClientCredentials.hpp"

#include "../logging/Logger.hpp"

#include <simdjson.h>

#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace core::auth::antigravity {

namespace {

/// Assumed lifetime of a user-supplied access token (Google's standard hour).
constexpr std::int64_t kImportedTokenLifetimeSeconds = 3600;

[[nodiscard]] std::string env_or_empty(const char* name) {
    const char* raw = std::getenv(name);
    return (raw && raw[0] != '\0') ? std::string(raw) : std::string();
}

/// Read a whole file; empty when missing or unreadable.
[[nodiscard]] std::string read_file(std::string_view path) {
    if (path.empty()) return {};
    std::ifstream in(std::string(path), std::ios::binary);
    if (!in) return {};
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

[[nodiscard]] std::string json_string(simdjson::dom::element element,
                                      std::string_view key) {
    std::string_view value;
    if (element[key].get(value) == simdjson::SUCCESS) return std::string(value);
    return {};
}

} // namespace

std::string credentials_path(std::string_view config_dir) {
    if (config_dir.empty()) return {};
    std::string path(config_dir);
    if (path.back() != '/') path += '/';
    path += "antigravity.json";
    return path;
}

ClientCredentials default_client_credentials() {
    // Filo embeds no client identity (see GoogleAntigravityIdentity.hpp), so the
    // base is empty: the environment and config-file layers must supply both
    // halves of the confidential client.
    ClientCredentials credentials;
    credentials.source = "unset";
    return credentials;
}

ClientCredentials load_client_credentials(std::string_view config_dir) {
    ClientCredentials credentials = default_client_credentials();

    if (const std::string path = credentials_path(config_dir); !path.empty()) {
        if (const std::string content = read_file(path); !content.empty()) {
            simdjson::dom::parser parser;
            simdjson::padded_string padded(content.data(), content.size());
            simdjson::dom::element doc;
            if (parser.parse(padded).get(doc) == simdjson::SUCCESS) {
                bool overridden = false;
                const auto apply = [&](std::string_view key, std::string& field) {
                    if (std::string value = json_string(doc, key); !value.empty()) {
                        field = std::move(value);
                        overridden = true;
                    }
                };
                apply("client_id", credentials.client_id);
                apply("client_secret", credentials.client_secret);
                apply("access_token", credentials.access_token);
                apply("refresh_token", credentials.refresh_token);
                if (overridden) credentials.source = path;
            } else {
                core::logging::warn(
                    "Antigravity credentials file {} is not valid JSON; ignoring it.",
                    path);
            }
        }
    }

    // The environment wins so a one-off override never needs a file edit.
    bool from_env = false;
    const auto apply_env = [&from_env](const char* name, std::string& field) {
        if (std::string value = env_or_empty(name); !value.empty()) {
            field = std::move(value);
            from_env = true;
        }
    };
    apply_env("GOOGLE_ANTIGRAVITY_CLIENT_ID", credentials.client_id);
    apply_env("GOOGLE_ANTIGRAVITY_CLIENT_SECRET", credentials.client_secret);
    apply_env("GOOGLE_ANTIGRAVITY_ACCESS_TOKEN", credentials.access_token);
    apply_env("GOOGLE_ANTIGRAVITY_REFRESH_TOKEN", credentials.refresh_token);
    if (from_env) credentials.source = "env";

    return credentials;
}

void require_client_credentials(const ClientCredentials& credentials,
                                std::string_view config_dir) {
    if (credentials.complete()) return;

    std::string missing = credentials.client_id.empty() ? "client_id" : "client_secret";
    if (credentials.client_id.empty() && credentials.client_secret.empty()) {
        missing = "client_id and client_secret";
    }
    throw std::runtime_error(
        "Google (Antigravity) OAuth is missing " + missing + ". Filo does not embed "
        "Google's Antigravity client identity, so supply both halves either as "
        "GOOGLE_ANTIGRAVITY_CLIENT_ID / GOOGLE_ANTIGRAVITY_CLIENT_SECRET or in "
        + (config_dir.empty() ? std::string("<config_dir>") : std::string(config_dir))
        + "/antigravity.json, then run `filo --login google` again.");
}

std::optional<OAuthToken> imported_token(const ClientCredentials& credentials,
                                         std::int64_t now_unix_seconds) {
    if (credentials.access_token.empty()) return std::nullopt;

    OAuthToken token;
    token.access_token  = credentials.access_token;
    token.refresh_token = credentials.refresh_token;
    token.token_type    = "Bearer";
    token.client_id     = credentials.client_id;
    token.expires_at    = now_unix_seconds + kImportedTokenLifetimeSeconds;
    return token;
}

} // namespace core::auth::antigravity

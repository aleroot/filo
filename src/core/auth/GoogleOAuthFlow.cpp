#include "GoogleOAuthFlow.hpp"

#include <simdjson.h>

#include <cctype>
#include <stdexcept>

#include "../utils/StringUtils.hpp"

namespace core::auth {

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string url_encode(std::string_view s) {
    std::string out;
    out.reserve(s.size() * 3);
    const char hex[] = "0123456789ABCDEF";
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 0xF];
        }
    }
    return out;
}


// ── GoogleOAuthFlow ───────────────────────────────────────────────────────────

std::string GoogleOAuthFlow::build_auth_url(std::string_view client_id,
                                             std::string_view redirect_uri,
                                             const std::vector<std::string>& scopes,
                                             std::string_view state,
                                             std::string_view code_challenge) {
    std::string scope_str;
    for (size_t i = 0; i < scopes.size(); ++i) {
        if (i > 0) scope_str += ' ';
        scope_str += scopes[i];
    }

    std::string auth_url = std::string("https://accounts.google.com/o/oauth2/v2/auth")
        + "?client_id="     + url_encode(client_id)
        + "&redirect_uri="  + url_encode(redirect_uri)
        + "&response_type=code"
        + "&scope="         + url_encode(scope_str)
        + "&state="         + url_encode(state)
        + "&access_type=offline"
        + "&prompt=consent";

    if (!code_challenge.empty()) {
        auth_url += "&code_challenge=" + url_encode(code_challenge);
        auth_url += "&code_challenge_method=S256";
    }

    return auth_url;
}

// static
OAuthToken GoogleOAuthFlow::parse_token_response(std::string_view json,
                                                  int64_t request_time_unix) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::dom::element doc = parser.parse(padded);

    OAuthToken token;
    std::string_view sv;
    int64_t v;

    if (doc["access_token"].get_string().get(sv) == simdjson::SUCCESS)
        token.access_token = std::string(sv);
    if (doc["refresh_token"].get_string().get(sv) == simdjson::SUCCESS)
        token.refresh_token = std::string(sv);
    if (doc["token_type"].get_string().get(sv) == simdjson::SUCCESS)
        token.token_type = std::string(sv);
    if (doc["expires_in"].get_int64().get(v) == simdjson::SUCCESS)
        token.expires_at = request_time_unix + v;

    if (token.access_token.empty())
        throw std::runtime_error("No access_token in response: " + std::string(json));

    return token;
}

GoogleManualAuthInput GoogleOAuthFlow::parse_manual_auth_input(std::string_view raw_input) {
    return parse_oauth_manual_auth_input(raw_input);
}

std::string GoogleOAuthFlow::build_loopback_redirect_uri(std::string_view listener_host,
                                                         int port,
                                                         std::string_view path,
                                                         std::string_view redirect_host_override) {
    std::string host = core::utils::str::trim_ascii_copy(redirect_host_override.empty()
        ? listener_host
        : redirect_host_override);

    if (host.empty()
        || host == "0.0.0.0"
        || host == "::"
        || host == "[::]"
        || host == "::0") {
        host = "127.0.0.1";
    }

    if (host.find(':') != std::string::npos
        && !(host.starts_with("[") && host.ends_with("]"))) {
        host = "[" + host + "]";
    }

    std::string normalized_path = path.empty() ? "/oauth2callback" : std::string(path);
    if (normalized_path.front() != '/') {
        normalized_path.insert(normalized_path.begin(), '/');
    }

    return "http://" + host + ":" + std::to_string(port) + normalized_path;
}

} // namespace core::auth

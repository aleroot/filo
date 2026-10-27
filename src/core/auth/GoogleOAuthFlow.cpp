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


static int hex_value(unsigned char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return 10 + (ch - 'a');
    if (ch >= 'A' && ch <= 'F') return 10 + (ch - 'A');
    return -1;
}

static std::string url_decode(std::string_view in) {
    std::string out;
    out.reserve(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(in[i]);
        if (ch == '+') {
            out.push_back(' ');
            continue;
        }
        if (ch == '%' && i + 2 < in.size()) {
            const int hi = hex_value(static_cast<unsigned char>(in[i + 1]));
            const int lo = hex_value(static_cast<unsigned char>(in[i + 2]));
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(static_cast<char>(ch));
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
    const std::string input = core::utils::str::trim_ascii_copy(raw_input);
    if (input.empty()) {
        throw std::runtime_error("No authorization code provided");
    }

    const bool looks_like_url =
        input.rfind("http://", 0) == 0 || input.rfind("https://", 0) == 0;
    if (!looks_like_url) {
        return {.code = input, .state = std::nullopt};
    }

    const std::size_t qmark = input.find('?');
    if (qmark == std::string::npos || qmark + 1 >= input.size()) {
        throw std::runtime_error("Callback URL does not contain query parameters");
    }

    GoogleManualAuthInput parsed{};
    std::size_t pos = qmark + 1;
    while (pos < input.size()) {
        const std::size_t amp = input.find('&', pos);
        const std::size_t end = (amp == std::string::npos) ? input.size() : amp;
        const std::string_view pair = std::string_view(input).substr(pos, end - pos);
        const std::size_t eq = pair.find('=');

        const std::string key = url_decode(eq == std::string::npos ? pair : pair.substr(0, eq));
        const std::string value =
            (eq == std::string::npos) ? std::string() : url_decode(pair.substr(eq + 1));

        if (key == "code") parsed.code = value;
        else if (key == "state") parsed.state = value;

        if (amp == std::string::npos) break;
        pos = amp + 1;
    }

    if (parsed.code.empty()) {
        throw std::runtime_error("Callback URL did not include an authorization code");
    }
    return parsed;
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

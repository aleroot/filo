#include "OpenAIOAuthFlow.hpp"
#include "OAuthErrors.hpp"
#include "AuthBrowserLauncher.hpp"
#include "OAuthLoopback.hpp"
#include "OAuthPkce.hpp"
#include "core/utils/Base64.hpp"
#include "core/utils/JsonUtils.hpp"
#include "core/utils/StringUtils.hpp"
#include <cpr/cpr.h>
#include <simdjson.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <thread>

namespace core::auth {

namespace {

static int64_t now_unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

struct OpenAIClaims {
    std::string account_id;
    std::string user_id;
    std::string email;
    std::string issuer;
    int64_t expires_at = 0;
};

[[nodiscard]] OpenAIClaims parse_openai_claims(std::string_view jwt_token) {
    OpenAIClaims claims;
    const std::size_t first_dot = jwt_token.find('.');
    if (first_dot == std::string_view::npos) return claims;
    const std::size_t second_dot = jwt_token.find('.', first_dot + 1);
    if (second_dot == std::string_view::npos || second_dot <= first_dot + 1) {
        return claims;
    }

    const std::string_view payload_b64url =
        jwt_token.substr(first_dot + 1, second_dot - first_dot - 1);

    const auto payload_json = core::utils::Base64::decode_url(payload_b64url);
    if (!payload_json.has_value()) return claims;

    simdjson::dom::parser parser;
    simdjson::padded_string padded(payload_json->data(), payload_json->size());
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return claims;

    const auto copy_string = [&](std::string_view key, std::string& out) {
        std::string_view value;
        if (doc[key].get_string().get(value) == simdjson::SUCCESS && !value.empty()) {
            out = std::string(value);
        }
    };

    copy_string("email", claims.email);
    copy_string("iss", claims.issuer);
    copy_string("sub", claims.user_id);
    int64_t expiration = 0;
    if (doc["exp"].get_int64().get(expiration) == simdjson::SUCCESS) {
        claims.expires_at = expiration;
    }

    std::string_view account_id;
    if (doc["https://api.openai.com/auth.chatgpt_account_id"]
            .get_string()
            .get(account_id) == simdjson::SUCCESS
        && !account_id.empty()) {
        claims.account_id = std::string(account_id);
    } else {
        copy_string("chatgpt_account_id", claims.account_id);
        if (claims.account_id.empty()) copy_string("account_id", claims.account_id);
    }

    simdjson::dom::object auth;
    if (doc["https://api.openai.com/auth"].get_object().get(auth) == simdjson::SUCCESS) {
        const auto copy_auth_string = [&](std::string_view key, std::string& out) {
            std::string_view value;
            if (auth[key].get_string().get(value) == simdjson::SUCCESS && !value.empty()) {
                out = std::string(value);
            }
        };
        copy_auth_string("chatgpt_account_id", claims.account_id);
        copy_auth_string("chatgpt_user_id", claims.user_id);
        if (claims.user_id.empty()) copy_auth_string("user_id", claims.user_id);
    }

    if (claims.email.empty()) {
        simdjson::dom::object profile;
        if (doc["https://api.openai.com/profile"].get_object().get(profile)
            == simdjson::SUCCESS) {
            std::string_view email;
            if (profile["email"].get_string().get(email) == simdjson::SUCCESS) {
                claims.email = std::string(email);
            }
        }
    }
    return claims;
}

// Public OAuth client id used by the official Codex login flow.
constexpr std::string_view kCodexDefaultClientId = "app_EMoamEEZ73f0CkXaXp7hrann";
constexpr std::string_view kFiloOriginator = "filo";

[[nodiscard]] bool env_truthy(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) return false;
    const std::string lowered = core::utils::str::to_lower_ascii_copy(value);
    return lowered != "0" && lowered != "false" && lowered != "no";
}

[[nodiscard]] std::string oauth_error_detail(std::string_view body) {
    try {
        simdjson::dom::parser parser;
        simdjson::padded_string padded(body.data(), body.size());
        simdjson::dom::element doc = parser.parse(padded);
        std::string_view error;
        std::string_view description;
        (void)doc["error"].get_string().get(error);
        (void)doc["error_description"].get_string().get(description);
        if (!error.empty() && !description.empty()) {
            return std::string(error) + ": " + std::string(description);
        }
        if (!error.empty()) return std::string(error);
    } catch (...) {
    }
    return "request rejected";
}

[[nodiscard]] std::string issuer_from_token_url(std::string_view token_url) {
    std::string issuer = core::utils::str::trim_trailing_slashes(token_url);
    for (const std::string_view suffix : {std::string_view{"/oauth/token"},
                                          std::string_view{"/token"}}) {
        if (issuer.ends_with(suffix)) {
            issuer.resize(issuer.size() - suffix.size());
            break;
        }
    }
    return issuer;
}

[[nodiscard]] std::string json_string(std::string_view value) {
    return "\"" + core::utils::escape_json_string(value) + "\"";
}

std::string resolve_client_id() {
    if (const char* env = std::getenv("OPENAI_OAUTH_CLIENT_ID"); env && env[0] != '\0') {
        return std::string(env);
    }
    return std::string(kCodexDefaultClientId);
}

} // namespace

// ── OpenAIOAuthFlow ───────────────────────────────────────────────────────────

OpenAIOAuthFlow::OpenAIOAuthFlow()
    : OpenAIOAuthFlow(
        resolve_client_id(),
        "https://auth.openai.com/oauth/authorize",
        "https://auth.openai.com/oauth/token",
        {"openid", "profile", "email", "offline_access",
         "api.connectors.read", "api.connectors.invoke"},
        1455,
        1457)
{}

OpenAIOAuthFlow::OpenAIOAuthFlow(std::string client_id,
                                 std::string auth_url,
                                 std::string token_url,
                                 std::vector<std::string> scopes,
                                 int port_start,
                                 int port_end)
    : client_id_(std::move(client_id))
    , auth_url_(std::move(auth_url))
    , token_url_(std::move(token_url))
    , scopes_(std::move(scopes))
    , port_start_(port_start)
    , port_end_(port_end)
{}

// static
std::string OpenAIOAuthFlow::generate_code_verifier() {
    return oauth_pkce::generate_code_verifier();
}

// static
std::string OpenAIOAuthFlow::compute_code_challenge(std::string_view verifier) {
    return oauth_pkce::compute_code_challenge(verifier);
}

// static
std::string OpenAIOAuthFlow::build_auth_url(std::string_view client_id,
                                             std::string_view redirect_uri,
                                             const std::vector<std::string>& scopes,
                                             std::string_view state,
                                             std::string_view code_challenge,
                                             std::string_view auth_base_url) {
    std::string scope_str;
    for (size_t i = 0; i < scopes.size(); ++i) {
        if (i > 0) scope_str += ' ';
        scope_str += scopes[i];
    }

    return std::string(auth_base_url)
        + "?client_id="             + oauth_pkce::url_encode(client_id)
        + "&redirect_uri="          + oauth_pkce::url_encode(redirect_uri)
        + "&response_type=code"
        + "&scope="                 + oauth_pkce::url_encode(scope_str)
        + "&state="                 + oauth_pkce::url_encode(state)
        + "&code_challenge="        + std::string(code_challenge)
        + "&code_challenge_method=S256"
        + "&id_token_add_organizations=true"
        + "&codex_cli_simplified_flow=true"
        + "&originator="            + oauth_pkce::url_encode(kFiloOriginator);
}

// static
OAuthToken OpenAIOAuthFlow::parse_token_response(std::string_view json,
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
    if (doc["account_id"].get_string().get(sv) == simdjson::SUCCESS) {
        token.account_id = std::string(sv);
    } else if (doc["chatgpt_account_id"].get_string().get(sv) == simdjson::SUCCESS) {
        token.account_id = std::string(sv);
    }

    std::string id_token;
    if (doc["id_token"].get_string().get(sv) == simdjson::SUCCESS) {
        id_token = std::string(sv);
    }

    if (token.access_token.empty())
        throw std::runtime_error("OpenAI token response did not include an access token");

    OpenAIClaims claims = parse_openai_claims(id_token);
    const OpenAIClaims access_claims = parse_openai_claims(token.access_token);
    if (claims.account_id.empty()) claims.account_id = access_claims.account_id;
    if (claims.user_id.empty()) claims.user_id = access_claims.user_id;
    if (claims.email.empty()) claims.email = access_claims.email;
    if (claims.issuer.empty()) claims.issuer = access_claims.issuer;

    if (token.account_id.empty()) token.account_id = std::move(claims.account_id);
    token.user_id = std::move(claims.user_id);
    token.email = std::move(claims.email);
    token.issuer = std::move(claims.issuer);
    if (token.expires_at <= 0) {
        token.expires_at = access_claims.expires_at > 0
            ? access_claims.expires_at
            : claims.expires_at > 0
                ? claims.expires_at
            : request_time_unix + std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::hours(1)).count();
    }

    return token;
}

OAuthToken OpenAIOAuthFlow::exchange_code(const std::string& code,
                                           const std::string& redirect_uri,
                                           const std::string& code_verifier) {
    int64_t req_time = now_unix_seconds();

    cpr::Response r = cpr::Post(
        cpr::Url{token_url_},
        cpr::Payload{
            {"client_id",     client_id_},
            {"code",          code},
            {"redirect_uri",  redirect_uri},
            {"grant_type",    "authorization_code"},
            {"code_verifier", code_verifier},
        },
        cpr::Timeout{15000}
    );

    if (r.status_code != 200)
        throw std::runtime_error("Token exchange failed ("
                                 + std::to_string(r.status_code) + "): "
                                 + oauth_error_detail(r.text));

    return parse_token_response(r.text, req_time);
}

OAuthToken OpenAIOAuthFlow::device_code_login() {
    const std::string issuer = issuer_from_token_url(token_url_);
    const std::string device_api = issuer + "/api/accounts/deviceauth";
    const cpr::Header json_headers{{"Content-Type", "application/json"}};

    const cpr::Response code_response = cpr::Post(
        cpr::Url{device_api + "/usercode"},
        json_headers,
        cpr::Body{"{\"client_id\":" + json_string(client_id_) + "}"},
        cpr::Timeout{15000});
    if (code_response.status_code < 200 || code_response.status_code >= 300) {
        const std::string unavailable = code_response.status_code == 404
            ? "device-code login is unavailable; free callback port 1455 or 1457 and retry browser login"
            : oauth_error_detail(code_response.text);
        throw std::runtime_error(
            "OpenAI device-code request failed ("
            + std::to_string(code_response.status_code) + "): " + unavailable);
    }

    simdjson::dom::parser parser;
    simdjson::padded_string padded(code_response.text);
    simdjson::dom::element doc = parser.parse(padded);
    std::string_view device_auth_id;
    std::string_view user_code;
    if (doc["device_auth_id"].get_string().get(device_auth_id) != simdjson::SUCCESS
        || device_auth_id.empty()) {
        throw std::runtime_error("OpenAI device-code response did not include device_auth_id");
    }
    if (doc["user_code"].get_string().get(user_code) != simdjson::SUCCESS) {
        (void)doc["usercode"].get_string().get(user_code);
    }
    if (user_code.empty()) {
        throw std::runtime_error("OpenAI device-code response did not include user_code");
    }

    int64_t interval_seconds = 5;
    if (doc["interval"].get_int64().get(interval_seconds) != simdjson::SUCCESS) {
        std::string_view interval_text;
        if (doc["interval"].get_string().get(interval_text) == simdjson::SUCCESS) {
            try {
                interval_seconds = std::stoll(std::string(interval_text));
            } catch (...) {
                interval_seconds = 5;
            }
        }
    }
    interval_seconds = std::clamp<int64_t>(interval_seconds, 1, 60);

    const std::string verification_url = issuer + "/codex/device";
    std::fprintf(
        stdout,
        "\nOpenAI device login\n"
        "1. Open: %s\n"
        "2. Enter this one-time code (expires in 15 minutes): %.*s\n\n"
        "Continue only if you started this login in Filo.\n\n",
        verification_url.c_str(),
        static_cast<int>(user_code.size()), user_code.data());
    std::fflush(stdout);
    if (!env_truthy("NO_BROWSER")) open_browser(verification_url);

    const std::string poll_body = "{\"device_auth_id\":" + json_string(device_auth_id)
        + ",\"user_code\":" + json_string(user_code) + "}";
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(15);
    while (std::chrono::steady_clock::now() < deadline) {
        const cpr::Response poll_response = cpr::Post(
            cpr::Url{device_api + "/token"},
            json_headers,
            cpr::Body{poll_body},
            cpr::Timeout{15000});

        if (poll_response.status_code >= 200 && poll_response.status_code < 300) {
            simdjson::dom::parser poll_parser;
            simdjson::padded_string poll_padded(poll_response.text);
            simdjson::dom::element poll_doc = poll_parser.parse(poll_padded);
            std::string_view authorization_code;
            std::string_view code_verifier;
            if (poll_doc["authorization_code"].get_string().get(authorization_code)
                    != simdjson::SUCCESS
                || poll_doc["code_verifier"].get_string().get(code_verifier)
                    != simdjson::SUCCESS
                || authorization_code.empty() || code_verifier.empty()) {
                throw std::runtime_error(
                    "OpenAI device-code completion response was incomplete");
            }
            return exchange_code(
                std::string(authorization_code),
                issuer + "/deviceauth/callback",
                std::string(code_verifier));
        }

        if (poll_response.status_code != 403 && poll_response.status_code != 404) {
            throw std::runtime_error(
                "OpenAI device-code polling failed ("
                + std::to_string(poll_response.status_code) + "): "
                + oauth_error_detail(poll_response.text));
        }
        std::this_thread::sleep_for(std::chrono::seconds(interval_seconds));
    }
    throw std::runtime_error("OpenAI device login timed out after 15 minutes");
}

OAuthToken OpenAIOAuthFlow::login() {
    if (env_truthy("OPENAI_DEVICE_AUTH")) return device_code_login();

    const std::string state         = oauth_pkce::generate_correlation_token();
    const std::string code_verifier = generate_code_verifier();
    const std::string challenge      = compute_code_challenge(code_verifier);

    OAuthLoopbackOptions loopback_options;
    loopback_options.redirect_host = "localhost";
    loopback_options.port_start = port_start_;
    loopback_options.port_end = port_end_;
    // These are the callback ports registered for the public Codex client.
    // Do not improvise an intermediate port that the authorization server may
    // reject even when it is locally available.
    loopback_options.candidate_ports = {port_start_, port_end_};
    loopback_options.callback_path = "/auth/callback";
    loopback_options.extra_paths = {"/callback"};
    loopback_options.expected_state = state;
    loopback_options.success_html =
        "<html><body><h2>OpenAI login successful!</h2>"
        "<p>You can close this tab and return to Filo.</p></body></html>";

    std::optional<OAuthLoopbackServer> loopback;
    try {
        loopback.emplace(std::move(loopback_options));
        loopback->start();
    } catch (const std::exception&) {
        // Headless systems and machines with occupied callback ports still have
        // a supported authentication path.
        return device_code_login();
    }

    const std::string redirect_uri = loopback->redirect_uri();
    const std::string auth_url       = build_auth_url(
        client_id_, redirect_uri, scopes_, state, challenge, auth_url_);

    // Print the URL regardless — if xdg-open isn't available the user can paste it
    fprintf(stdout,
            "\nOpening browser for OpenAI login.\n"
            "If your browser does not open automatically, visit:\n  %s\n\n",
            auth_url.c_str());
    fflush(stdout);

    if (!env_truthy("NO_BROWSER")) open_browser(auth_url);

    OAuthLoopbackResult result = loopback->wait();
    if (result.timed_out) {
        throw std::runtime_error("OpenAI login timed out after 5 minutes");
    }
    if (!result.error.empty()) {
        throw std::runtime_error("OpenAI login failed: " + result.error);
    }
    return exchange_code(result.code, redirect_uri, code_verifier);
}

std::string OpenAIOAuthFlow::build_refresh_request_body(
    std::string_view client_id, std::string_view refresh_token) {
    return "{\"client_id\":" + json_string(client_id)
        + ",\"grant_type\":\"refresh_token\",\"refresh_token\":"
        + json_string(refresh_token) + "}";
}

std::string OpenAIOAuthFlow::build_revoke_request_body(
    std::string_view client_id,
    std::string_view token,
    std::string_view token_type_hint) {
    return "{\"token\":" + json_string(token)
        + ",\"token_type_hint\":" + json_string(token_type_hint)
        + ",\"client_id\":" + json_string(client_id) + "}";
}

OAuthToken OpenAIOAuthFlow::refresh(std::string_view refresh_token) {
    int64_t req_time = now_unix_seconds();

    cpr::Response r = cpr::Post(
        cpr::Url{token_url_},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{build_refresh_request_body(client_id_, refresh_token)},
        cpr::Timeout{15000}
    );

    if (r.status_code != 200) {
        if (oauth_error_is_invalid_grant(r.text)) {
            throw OAuthRefreshRejected::by_provider("OpenAI");
        }
        throw std::runtime_error("Token refresh failed ("
                                 + std::to_string(r.status_code) + "): "
                                 + oauth_error_detail(r.text));
    }

    OAuthToken token = parse_token_response(r.text, req_time);

    // Refresh responses may omit the refresh_token — preserve the existing one
    if (token.refresh_token.empty())
        token.refresh_token = std::string(refresh_token);

    return token;
}

void OpenAIOAuthFlow::revoke(const OAuthToken& token) {
    // Revoke the refresh token when available (which also invalidates derived
    // access tokens), otherwise fall back to the access token (RFC 7009).
    const bool use_refresh = token.has_refresh_token();
    const std::string& target = use_refresh ? token.refresh_token
                                            : token.access_token;
    if (target.empty()) return;

    // Default token_url is https://auth.openai.com/oauth/token; the sibling
    // revocation endpoint is /oauth/revoke.
    std::string revoke_url = token_url_;
    if (constexpr std::string_view suffix = "/token"; revoke_url.ends_with(suffix)) {
        revoke_url.resize(revoke_url.size() - suffix.size());
    }
    revoke_url += "/revoke";

    cpr::Response r = cpr::Post(
        cpr::Url{revoke_url},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{build_revoke_request_body(
            client_id_, target, use_refresh ? "refresh_token" : "access_token")},
        cpr::Timeout{10000});

    if (r.status_code < 200 || r.status_code >= 300)
        throw std::runtime_error("OpenAI token revocation failed ("
                                 + std::to_string(r.status_code) + "): "
                                 + oauth_error_detail(r.text));
}

} // namespace core::auth

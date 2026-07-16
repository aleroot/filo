#include "XaiOAuthFlow.hpp"

#include "AuthBrowserLauncher.hpp"
#include "OAuthPkce.hpp"
#include "core/logging/Logger.hpp"
#include "core/utils/Base64.hpp"

#include <cpr/cpr.h>
#include <httplib.h>
#include <simdjson.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace core::auth {
namespace {

constexpr std::string_view kClientVersion = "0.1.0";
[[nodiscard]] std::int64_t now_unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

[[nodiscard]] std::string join_scopes(const std::vector<std::string>& scopes) {
    std::string joined;
    for (std::size_t i = 0; i < scopes.size(); ++i) {
        if (i > 0) joined.push_back(' ');
        joined += scopes[i];
    }
    return joined;
}

[[nodiscard]] cpr::Header xai_oauth_headers() {
    return cpr::Header{
        {"Accept", "application/json"},
        {"x-grok-client-version", std::string(kClientVersion)},
    };
}

[[nodiscard]] std::string endpoint(std::string_view issuer, std::string_view path) {
    std::string result(issuer);
    while (!result.empty() && result.back() == '/') result.pop_back();
    result += path;
    return result;
}

struct OAuthErrorResponse {
    std::string code;
    std::string description;

    [[nodiscard]] std::string message() const {
        if (!description.empty()) return description;
        return code;
    }
};

[[nodiscard]] OAuthErrorResponse parse_oauth_error(std::string_view body) {
    OAuthErrorResponse result;
    try {
        simdjson::dom::parser parser;
        simdjson::padded_string padded(body);
        simdjson::dom::element doc = parser.parse(padded);
        std::string_view value;
        if (doc["error"].get_string().get(value) == simdjson::SUCCESS) {
            result.code = std::string(value);
        }
        if (doc["error_description"].get_string().get(value) == simdjson::SUCCESS) {
            result.description = std::string(value);
        }
    } catch (...) {
    }
    if (result.code.empty() && result.description.empty()) {
        result.description = std::string(body);
    }
    return result;
}

struct IdTokenClaims {
    std::string subject;
    std::string email;
    std::string nonce;
    std::string principal_type;
    std::string principal_id;
    std::string team_id;
};

[[nodiscard]] IdTokenClaims parse_id_token_claims(std::string_view jwt) {
    IdTokenClaims claims;
    const auto first_dot = jwt.find('.');
    if (first_dot == std::string_view::npos) return claims;
    const auto second_dot = jwt.find('.', first_dot + 1);
    if (second_dot == std::string_view::npos || second_dot <= first_dot + 1) return claims;
    const auto decoded = core::utils::Base64::decode_url(
        jwt.substr(first_dot + 1, second_dot - first_dot - 1));
    if (!decoded.has_value()) return claims;

    simdjson::dom::parser parser;
    simdjson::padded_string padded(decoded->data(), decoded->size());
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return claims;

    const auto read = [&doc](std::string_view key) -> std::string {
        std::string_view value;
        if (doc[key].get_string().get(value) == simdjson::SUCCESS) {
            return std::string(value);
        }
        return {};
    };
    claims.subject = read("sub");
    claims.email = read("email");
    claims.nonce = read("nonce");
    claims.principal_type = read("principal_type");
    if (claims.principal_type.empty()) claims.principal_type = read("principalType");
    claims.principal_id = read("principal_id");
    if (claims.principal_id.empty()) claims.principal_id = read("principalId");
    claims.team_id = read("team_id");
    return claims;
}

} // namespace

XaiOAuthFlow::XaiOAuthFlow()
    : XaiOAuthFlow(std::string(kIssuer),
                   std::string(kClientId),
                   default_scopes(),
                   std::string(kReferrer)) {}

XaiOAuthFlow::XaiOAuthFlow(std::string issuer,
                           std::string client_id,
                           std::vector<std::string> scopes,
                           std::string referrer)
    : issuer_(std::move(issuer))
    , client_id_(std::move(client_id))
    , scopes_(std::move(scopes))
    , referrer_(std::move(referrer)) {}

std::vector<std::string> XaiOAuthFlow::default_scopes() {
    return {
        "openid",
        "profile",
        "email",
        "offline_access",
        "grok-cli:access",
        "api:access",
        "conversations:read",
        "conversations:write",
    };
}

XaiOAuthFlow::DiscoveryDocument
XaiOAuthFlow::parse_discovery_response(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(json);
    simdjson::dom::element doc = parser.parse(padded);
    DiscoveryDocument discovery;
    std::string_view value;
    if (doc["authorization_endpoint"].get_string().get(value) == simdjson::SUCCESS) {
        discovery.authorization_endpoint = std::string(value);
    }
    if (doc["token_endpoint"].get_string().get(value) == simdjson::SUCCESS) {
        discovery.token_endpoint = std::string(value);
    }
    if (doc["revocation_endpoint"].get_string().get(value) == simdjson::SUCCESS) {
        discovery.revocation_endpoint = std::string(value);
    }
    if (discovery.authorization_endpoint.empty() || discovery.token_endpoint.empty()) {
        throw std::runtime_error("xAI OIDC discovery response is missing endpoints");
    }
    return discovery;
}

OAuthToken XaiOAuthFlow::parse_token_response(std::string_view json,
                                               std::int64_t request_time_unix,
                                               std::string_view issuer,
                                               std::string_view client_id,
                                               std::string_view expected_nonce) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(json);
    simdjson::dom::element doc = parser.parse(padded);
    OAuthToken token;
    std::string_view value;
    std::int64_t expires_in = 0;
    std::string id_token;
    if (doc["access_token"].get_string().get(value) == simdjson::SUCCESS)
        token.access_token = std::string(value);
    if (doc["refresh_token"].get_string().get(value) == simdjson::SUCCESS)
        token.refresh_token = std::string(value);
    if (doc["token_type"].get_string().get(value) == simdjson::SUCCESS)
        token.token_type = std::string(value);
    if (doc["id_token"].get_string().get(value) == simdjson::SUCCESS)
        id_token = std::string(value);
    if (doc["expires_in"].get_int64().get(expires_in) == simdjson::SUCCESS)
        token.expires_at = request_time_unix + expires_in;
    if (token.access_token.empty()) {
        throw std::runtime_error("xAI OAuth response does not contain an access token");
    }
    if (token.expires_at == 0) token.expires_at = request_time_unix + 30 * 24 * 60 * 60;
    token.issuer = std::string(issuer);
    token.client_id = std::string(client_id);
    token.scopes = default_scopes();

    if (!id_token.empty()) {
        const auto claims = parse_id_token_claims(id_token);
        if (!expected_nonce.empty() && claims.nonce != expected_nonce) {
            throw std::runtime_error("xAI OIDC nonce mismatch");
        }
        token.user_id = claims.subject;
        token.email = claims.email;
        token.principal_type = claims.principal_type;
        token.principal_id = claims.principal_id;
        token.team_id = claims.team_id;
    }
    return token;
}

std::string XaiOAuthFlow::build_authorization_url(
    std::string_view authorization_endpoint,
    std::string_view client_id,
    std::string_view redirect_uri,
    const std::vector<std::string>& scopes,
    std::string_view state,
    std::string_view nonce,
    std::string_view code_challenge,
    std::string_view referrer) {
    return std::string(authorization_endpoint)
        + "?response_type=code"
        + "&client_id=" + oauth_pkce::url_encode(client_id)
        + "&redirect_uri=" + oauth_pkce::url_encode(redirect_uri)
        + "&scope=" + oauth_pkce::url_encode(join_scopes(scopes))
        + "&code_challenge=" + oauth_pkce::url_encode(code_challenge)
        + "&code_challenge_method=S256"
        + "&state=" + oauth_pkce::url_encode(state)
        + "&nonce=" + oauth_pkce::url_encode(nonce)
        + "&referrer=" + oauth_pkce::url_encode(referrer);
}

XaiOAuthFlow::DiscoveryDocument XaiOAuthFlow::discover() const {
    const std::string url = endpoint(issuer_, "/.well-known/openid-configuration");
    const cpr::Response response = cpr::Get(
        cpr::Url{url}, xai_oauth_headers(), cpr::Timeout{10000});
    if (response.status_code < 200 || response.status_code >= 300) {
        throw std::runtime_error(
            "xAI OIDC discovery failed (" + std::to_string(response.status_code) + ")");
    }
    return parse_discovery_response(response.text);
}

OAuthToken XaiOAuthFlow::exchange_code(const DiscoveryDocument& discovery,
                                        std::string_view code,
                                        std::string_view redirect_uri,
                                        std::string_view verifier,
                                        std::string_view nonce) const {
    const auto request_time = now_unix_seconds();
    const cpr::Response response = cpr::Post(
        cpr::Url{discovery.token_endpoint},
        xai_oauth_headers(),
        cpr::Payload{
            {"grant_type", "authorization_code"},
            {"code", std::string(code)},
            {"redirect_uri", std::string(redirect_uri)},
            {"client_id", client_id_},
            {"code_verifier", std::string(verifier)},
        },
        cpr::Timeout{15000});
    if (response.status_code < 200 || response.status_code >= 300) {
        throw std::runtime_error(
            "xAI token exchange failed (" + std::to_string(response.status_code)
            + "): " + parse_oauth_error(response.text).message());
    }
    return parse_token_response(
        response.text, request_time, issuer_, client_id_, nonce);
}

OAuthToken XaiOAuthFlow::browser_login(const DiscoveryDocument& discovery) {
    std::mutex mutex;
    std::condition_variable signal;
    bool completed = false;
    std::string received_code;
    std::string received_state;
    std::string received_error;
    httplib::Server server;

    const auto callback = [&](const httplib::Request& request, httplib::Response& response) {
        std::lock_guard lock(mutex);
        received_code = request.get_param_value("code");
        received_state = request.get_param_value("state");
        received_error = request.get_param_value("error");
        if (!received_error.empty()) {
            const std::string description = request.get_param_value("error_description");
            if (!description.empty()) received_error += ": " + description;
            response.set_content(
                "<html><body><h2>Grok login failed</h2><p>Return to Filo and try again.</p></body></html>",
                "text/html");
        } else {
            response.set_content(
                "<html><body><h2>Grok login successful</h2><p>You can close this tab and return to Filo.</p></body></html>",
                "text/html");
        }
        completed = true;
        signal.notify_one();
    };
    server.Get("/callback", callback);

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) throw std::runtime_error("Unable to bind the xAI OAuth callback server");

    const std::string redirect_uri =
        "http://127.0.0.1:" + std::to_string(port) + "/callback";
    const std::string verifier = oauth_pkce::generate_code_verifier(43);
    const std::string challenge = oauth_pkce::compute_code_challenge(verifier);
    const std::string state = oauth_pkce::generate_correlation_token();
    const std::string nonce = oauth_pkce::generate_correlation_token();
    const std::string authorization_url = build_authorization_url(
        discovery.authorization_endpoint,
        client_id_,
        redirect_uri,
        scopes_,
        state,
        nonce,
        challenge,
        referrer_);

    std::thread server_thread([&server] { server.listen_after_bind(); });
    std::fprintf(stdout,
        "\nSigning in with Grok...\n\nOpen this URL to sign in:\n  %s\n\n",
        authorization_url.c_str());
    std::fflush(stdout);
    open_browser(authorization_url);

    {
        std::unique_lock lock(mutex);
        const bool callback_received = signal.wait_for(
            lock, std::chrono::minutes(10), [&completed] { return completed; });
        server.stop();
        lock.unlock();
        if (server_thread.joinable()) server_thread.join();
        if (!callback_received) {
            throw std::runtime_error("Grok login timed out after 10 minutes");
        }
    }
    if (!received_error.empty()) {
        throw std::runtime_error("Grok login failed: " + received_error);
    }
    if (received_state != state) {
        throw std::runtime_error("xAI OAuth state mismatch");
    }
    if (received_code.empty()) {
        throw std::runtime_error("xAI OAuth callback did not contain an authorization code");
    }
    return exchange_code(discovery, received_code, redirect_uri, verifier, nonce);
}

OAuthToken XaiOAuthFlow::login() {
    return browser_login(discover());
}

OAuthToken XaiOAuthFlow::refresh(std::string_view refresh_token) {
    const auto discovery = discover();
    std::string last_error;
    for (int attempt = 0; attempt < 3; ++attempt) {
        const auto request_time = now_unix_seconds();
        const cpr::Response response = cpr::Post(
            cpr::Url{discovery.token_endpoint},
            xai_oauth_headers(),
            cpr::Payload{
                {"grant_type", "refresh_token"},
                {"refresh_token", std::string(refresh_token)},
                {"client_id", client_id_},
            },
            cpr::Timeout{15000});
        if (response.status_code >= 200 && response.status_code < 300) {
            OAuthToken token = parse_token_response(
                response.text, request_time, issuer_, client_id_);
            if (token.refresh_token.empty()) token.refresh_token = std::string(refresh_token);
            return token;
        }
        const OAuthErrorResponse error = parse_oauth_error(response.text);
        last_error = error.message();
        if (error.code == "invalid_grant" || error.code == "invalid_client") break;
        if (attempt < 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200 << attempt));
        }
    }
    throw std::runtime_error("xAI token refresh failed: " + last_error);
}

void XaiOAuthFlow::revoke(const OAuthToken& token) {
    // RFC 7009 revocation, best effort: only attempted when the issuer
    // advertises a revocation endpoint in its OIDC discovery document.
    const bool use_refresh = token.has_refresh_token();
    const std::string& target = use_refresh ? token.refresh_token
                                            : token.access_token;
    if (target.empty()) return;

    const auto discovery = discover();
    if (discovery.revocation_endpoint.empty()) {
        core::logging::debug(
            "xAI issuer does not advertise a revocation endpoint; "
            "clearing local credentials only.");
        return;
    }

    const cpr::Response response = cpr::Post(
        cpr::Url{discovery.revocation_endpoint},
        xai_oauth_headers(),
        cpr::Payload{
            {"token", target},
            {"token_type_hint", use_refresh ? "refresh_token" : "access_token"},
            {"client_id", client_id_},
        },
        cpr::Timeout{15000});
    if (response.status_code < 200 || response.status_code >= 300) {
        throw std::runtime_error(
            "xAI token revocation failed (" + std::to_string(response.status_code)
            + "): " + parse_oauth_error(response.text).message());
    }
}

} // namespace core::auth

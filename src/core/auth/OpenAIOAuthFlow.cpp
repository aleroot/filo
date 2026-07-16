#include "OpenAIOAuthFlow.hpp"
#include "AuthBrowserLauncher.hpp"
#include "OAuthPkce.hpp"
#include "core/utils/Base64.hpp"
#include <cpr/cpr.h>
#include <httplib.h>
#include <simdjson.h>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

namespace core::auth {

namespace {

static int64_t now_unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::optional<std::string> parse_openai_account_id_claim(std::string_view jwt_token) {
    const std::size_t first_dot = jwt_token.find('.');
    if (first_dot == std::string_view::npos) return std::nullopt;
    const std::size_t second_dot = jwt_token.find('.', first_dot + 1);
    if (second_dot == std::string_view::npos || second_dot <= first_dot + 1) {
        return std::nullopt;
    }

    const std::string_view payload_b64url =
        jwt_token.substr(first_dot + 1, second_dot - first_dot - 1);

    const auto payload_json = core::utils::Base64::decode_url(payload_b64url);
    if (!payload_json.has_value()) return std::nullopt;

    simdjson::dom::parser parser;
    simdjson::padded_string padded(payload_json->data(), payload_json->size());
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return std::nullopt;

    std::string_view account_id;
    if (doc["https://api.openai.com/auth.chatgpt_account_id"]
            .get_string()
            .get(account_id) == simdjson::SUCCESS
        && !account_id.empty()) {
        return std::string(account_id);
    }

    if (doc["chatgpt_account_id"].get_string().get(account_id) == simdjson::SUCCESS
        && !account_id.empty()) {
        return std::string(account_id);
    }

    if (doc["account_id"].get_string().get(account_id) == simdjson::SUCCESS
        && !account_id.empty()) {
        return std::string(account_id);
    }

    return std::nullopt;
}

// Public OAuth client id used by the official Codex login flow.
constexpr std::string_view kCodexDefaultClientId = "app_EMoamEEZ73f0CkXaXp7hrann";

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
        "https://auth.openai.com/authorize",
        "https://auth.openai.com/oauth/token",
        {"openid", "email", "profile", "offline_access"},
        1455,
        1455)
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
        + "&codex_cli_simplified_flow=true";
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
        throw std::runtime_error("No access_token in response: " + std::string(json));

    if (token.account_id.empty()) {
        if (const auto claim = parse_openai_account_id_claim(id_token);
            claim.has_value()) {
            token.account_id = *claim;
        } else if (const auto claim = parse_openai_account_id_claim(token.access_token);
                   claim.has_value()) {
            token.account_id = *claim;
        }
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
        }
    );

    if (r.status_code != 200)
        throw std::runtime_error("Token exchange failed ("
                                 + std::to_string(r.status_code) + "): " + r.text);

    return parse_token_response(r.text, req_time);
}

OAuthToken OpenAIOAuthFlow::login() {
    // Find a free port for the loopback callback
    int port = -1;
    for (int p = port_start_; p <= port_end_; ++p) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) continue;
        int opt = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(static_cast<uint16_t>(p));
        bool free_port = (bind(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0);
        close(sock);
        if (free_port) { port = p; break; }
    }
    if (port < 0)
        throw std::runtime_error("No free port available for OpenAI OAuth callback");

    const std::string redirect_uri = "http://localhost:" + std::to_string(port) + "/auth/callback";
    const std::string state         = oauth_pkce::generate_correlation_token();
    const std::string code_verifier = generate_code_verifier();
    const std::string challenge      = compute_code_challenge(code_verifier);
    const std::string auth_url       = build_auth_url(
        client_id_, redirect_uri, scopes_, state, challenge, auth_url_);

    // Print the URL regardless — if xdg-open isn't available the user can paste it
    fprintf(stdout,
            "\nOpening browser for OpenAI login.\n"
            "If your browser does not open automatically, visit:\n  %s\n\n",
            auth_url.c_str());
    fflush(stdout);

    open_browser(auth_url);

    std::mutex              cv_mtx;
    std::condition_variable cv;
    bool                    done = false;
    std::string             received_code;
    std::string             received_state;
    std::string             error_msg;

    httplib::Server svr;

    const auto callback_handler = [&](const httplib::Request& req, httplib::Response& res) {
        std::string code  = req.get_param_value("code");
        std::string st    = req.get_param_value("state");
        std::string err   = req.get_param_value("error");

        if (!err.empty()) {
            std::string desc = req.get_param_value("error_description");
            res.set_content("Login failed: " + err + " — " + desc, "text/plain");
            std::unique_lock lk(cv_mtx);
            error_msg = err + ": " + desc;
            done = true;
            cv.notify_one();
            return;
        }

        res.set_content(
            "<html><body><h2>OpenAI login successful!</h2>"
            "<p>You can close this tab and return to filo.</p></body></html>",
            "text/html");

        std::unique_lock lk(cv_mtx);
        received_code  = std::move(code);
        received_state = std::move(st);
        done = true;
        cv.notify_one();
    };
    // Keep legacy /callback for backward compatibility with older redirects.
    svr.Get("/callback", callback_handler);
    svr.Get("/auth/callback", callback_handler);

    std::thread server_thread([&svr, port]() {
        svr.listen("127.0.0.1", port);
    });

    {
        std::unique_lock lk(cv_mtx);
        bool ok = cv.wait_for(lk, std::chrono::minutes(5), [&] { return done; });
        svr.stop();
        if (server_thread.joinable()) server_thread.join();

        if (!ok)
            throw std::runtime_error("OpenAI login timed out after 5 minutes");
        if (!error_msg.empty())
            throw std::runtime_error("OpenAI login failed: " + error_msg);
    }

    if (received_state != state)
        throw std::runtime_error("OAuth state mismatch — possible CSRF attempt");
    if (received_code.empty())
        throw std::runtime_error("No authorisation code received from OpenAI");

    return exchange_code(received_code, redirect_uri, code_verifier);
}

OAuthToken OpenAIOAuthFlow::refresh(std::string_view refresh_token) {
    int64_t req_time = now_unix_seconds();

    cpr::Response r = cpr::Post(
        cpr::Url{token_url_},
        cpr::Payload{
            {"client_id",     client_id_},
            {"refresh_token", std::string(refresh_token)},
            {"grant_type",    "refresh_token"},
        }
    );

    if (r.status_code != 200)
        throw std::runtime_error("Token refresh failed ("
                                 + std::to_string(r.status_code) + "): " + r.text);

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
        cpr::Payload{
            {"token",           target},
            {"token_type_hint", use_refresh ? "refresh_token" : "access_token"},
            {"client_id",       client_id_},
        },
        cpr::Timeout{10000});

    if (r.status_code < 200 || r.status_code >= 300)
        throw std::runtime_error("OpenAI token revocation failed ("
                                 + std::to_string(r.status_code) + "): " + r.text);
}

} // namespace core::auth

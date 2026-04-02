#include "ClaudeOAuthFlow.hpp"
#include "OpenAIOAuthFlow.hpp"
#include "core/utils/JsonUtils.hpp"
#include <cpr/cpr.h>
#include <httplib.h>
#include <simdjson.h>
#include <array>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <iostream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

namespace core::auth {

namespace {

constexpr std::string_view kDefaultClientId = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
constexpr std::string_view kDefaultAuthUrl  = "https://claude.com/cai/oauth/authorize";
constexpr std::string_view kDefaultTokenUrl = "https://platform.claude.com/v1/oauth/token";
constexpr std::string_view kManualRedirectUrl = "https://platform.claude.com/oauth/code/callback";

int64_t now_unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string resolve_client_id() {
    if (const char* env = std::getenv("CLAUDE_CODE_OAUTH_CLIENT_ID");
        env && env[0] != '\0') {
        return std::string(env);
    }
    return std::string(kDefaultClientId);
}

std::vector<std::string> default_scopes() {
    return {
        "org:create_api_key",
        "user:profile",
        "user:inference",
        "user:sessions:claude_code",
        "user:mcp_servers",
        "user:file_upload",
    };
}

std::string base64url_encode(const unsigned char* data, std::size_t len) {
    static constexpr char tbl[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2u) / 3u) * 4u);

    for (std::size_t i = 0; i < len; i += 3u) {
        const unsigned int octet_a = data[i];
        const unsigned int octet_b = (i + 1u < len) ? data[i + 1u] : 0u;
        const unsigned int octet_c = (i + 2u < len) ? data[i + 2u] : 0u;

        const unsigned int triple = (octet_a << 16u) | (octet_b << 8u) | octet_c;
        out.push_back(tbl[(triple >> 18u) & 0x3Fu]);
        out.push_back(tbl[(triple >> 12u) & 0x3Fu]);
        out.push_back((i + 1u < len) ? tbl[(triple >> 6u) & 0x3Fu] : '=');
        out.push_back((i + 2u < len) ? tbl[triple & 0x3Fu] : '=');
    }

    for (char& ch : out) {
        if (ch == '+') ch = '-';
        else if (ch == '/') ch = '_';
    }
    while (!out.empty() && out.back() == '=') out.pop_back();
    return out;
}

std::string url_encode(std::string_view s) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size() * 3u);
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[(c >> 4u) & 0x0Fu];
            out += hex[c & 0x0Fu];
        }
    }
    return out;
}

std::string generate_random_state() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    std::array<unsigned char, 32> bytes{};
    for (auto& b : bytes) {
        b = static_cast<unsigned char>(dist(gen));
    }
    return base64url_encode(bytes.data(), bytes.size());
}

std::string generate_code_verifier() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    std::array<unsigned char, 32> bytes{};
    for (auto& b : bytes) {
        b = static_cast<unsigned char>(dist(gen));
    }
    return base64url_encode(bytes.data(), bytes.size());
}

void open_browser(const std::string& url) {
    pid_t pid = fork();
    if (pid < 0) return;
    if (pid == 0) {
        if (fork() == 0) {
#if defined(__linux__)
            execlp("xdg-open", "xdg-open", url.c_str(), nullptr);
#elif defined(__APPLE__)
            execlp("open", "open", url.c_str(), nullptr);
#endif
            _exit(1);
        }
        _exit(0);
    }
    waitpid(pid, nullptr, 0);
}

std::vector<std::string> parse_scope_list(std::string_view scopes) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < scopes.size()) {
        while (i < scopes.size() && std::isspace(static_cast<unsigned char>(scopes[i]))) {
            ++i;
        }
        if (i >= scopes.size()) break;
        std::size_t j = i;
        while (j < scopes.size() && !std::isspace(static_cast<unsigned char>(scopes[j]))) {
            ++j;
        }
        out.emplace_back(scopes.substr(i, j - i));
        i = j;
    }
    return out;
}

std::optional<OAuthToken> token_from_env() {
    if (const char* oauth = std::getenv("CLAUDE_CODE_OAUTH_TOKEN");
        oauth && oauth[0] != '\0') {
        OAuthToken token;
        token.access_token = oauth;
        token.token_type = "Bearer";
        token.expires_at = now_unix_seconds() + (60 * 60 * 24 * 30);

        if (const char* scope_env = std::getenv("CLAUDE_CODE_OAUTH_SCOPES");
            scope_env && scope_env[0] != '\0') {
            token.scopes = parse_scope_list(scope_env);
        } else {
            token.scopes = {"user:inference"};
        }
        if (const char* refresh_env = std::getenv("CLAUDE_CODE_OAUTH_REFRESH_TOKEN");
            refresh_env && refresh_env[0] != '\0') {
            token.refresh_token = refresh_env;
        }
        return token;
    }

    if (const char* legacy = std::getenv("ANTHROPIC_AUTH_TOKEN");
        legacy && legacy[0] != '\0') {
        OAuthToken token;
        token.access_token = legacy;
        token.token_type = "Bearer";
        token.expires_at = now_unix_seconds() + (60 * 60 * 24 * 30);
        token.scopes = {"user:inference"};
        return token;
    }

    return std::nullopt;
}

std::string join_scopes(const std::vector<std::string>& scopes) {
    std::string out;
    for (std::size_t i = 0; i < scopes.size(); ++i) {
        if (i > 0) out += ' ';
        out += scopes[i];
    }
    return out;
}

std::string trim_copy(std::string_view in) {
    std::size_t begin = 0;
    while (begin < in.size()
        && std::isspace(static_cast<unsigned char>(in[begin]))) {
        ++begin;
    }

    std::size_t end = in.size();
    while (end > begin
        && std::isspace(static_cast<unsigned char>(in[end - 1]))) {
        --end;
    }
    return std::string(in.substr(begin, end - begin));
}

std::string append_rate_limit_headers(const cpr::Response& r) {
    std::string extra;
    auto append_header = [&](std::string_view key) {
        const auto it = r.header.find(std::string(key));
        if (it == r.header.end() || it->second.empty()) return;
        if (!extra.empty()) extra += ", ";
        extra += std::string(key) + "=" + it->second;
    };

    append_header("Retry-After");
    append_header("retry-after");
    append_header("anthropic-ratelimit-unified-reset");
    append_header("anthropic-ratelimit-unified-remaining");
    append_header("anthropic-ratelimit-unified-requests-limit");

    return extra;
}

int hex_value(unsigned char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return 10 + (ch - 'a');
    if (ch >= 'A' && ch <= 'F') return 10 + (ch - 'A');
    return -1;
}

std::string url_decode(std::string_view in) {
    std::string out;
    out.reserve(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(in[i]);
        if (ch == '+' ) {
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

struct ManualAuthInput {
    std::string code;
    std::optional<std::string> state;
};

ManualAuthInput parse_manual_auth_input(std::string_view raw_input) {
    const std::string input = trim_copy(raw_input);
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

    ManualAuthInput parsed{};
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

int pick_free_port(int port_start, int port_end) {
    for (int p = port_start; p <= port_end; ++p) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) continue;

        int opt = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(static_cast<uint16_t>(p));

        const bool free_port =
            (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
        close(sock);
        if (free_port) return p;
    }
    return -1;
}

} // namespace

ClaudeOAuthFlow::ClaudeOAuthFlow(std::shared_ptr<ui::AuthUI> ui)
    : ClaudeOAuthFlow(resolve_client_id(),
                      std::string(kDefaultAuthUrl),
                      std::string(kDefaultTokenUrl),
                      default_scopes(),
                      57890,
                      57990,
                      std::move(ui)) {}

ClaudeOAuthFlow::ClaudeOAuthFlow(std::string client_id,
                                 std::string auth_url,
                                 std::string token_url,
                                 std::vector<std::string> scopes,
                                 int port_start,
                                 int port_end,
                                 std::shared_ptr<ui::AuthUI> ui)
    : client_id_(std::move(client_id))
    , auth_url_(std::move(auth_url))
    , token_url_(std::move(token_url))
    , scopes_(std::move(scopes))
    , port_start_(port_start)
    , port_end_(port_end)
    , ui_(std::move(ui)) {}

std::string ClaudeOAuthFlow::build_auth_url(std::string_view client_id,
                                            std::string_view redirect_uri,
                                            const std::vector<std::string>& scopes,
                                            std::string_view state,
                                            std::string_view code_challenge,
                                            std::string_view auth_base_url) {
    std::string url = std::string(auth_base_url)
        + "?code=true"
        + "&client_id=" + url_encode(client_id)
        + "&response_type=code"
        + "&redirect_uri=" + url_encode(redirect_uri)
        + "&scope=" + url_encode(join_scopes(scopes))
        + "&code_challenge=" + url_encode(code_challenge)
        + "&code_challenge_method=S256"
        + "&state=" + url_encode(state);

    if (const char* org_uuid = std::getenv("CLAUDE_CODE_ORGANIZATION_UUID");
        org_uuid && org_uuid[0] != '\0') {
        url += "&orgUUID=";
        url += url_encode(org_uuid);
    }

    return url;
}

OAuthToken ClaudeOAuthFlow::parse_token_response(std::string_view json,
                                                 int64_t request_time_unix) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::dom::element doc = parser.parse(padded);

    OAuthToken token;
    std::string_view sv;
    int64_t v = 0;

    if (doc["access_token"].get_string().get(sv) == simdjson::SUCCESS)
        token.access_token = std::string(sv);
    if (doc["refresh_token"].get_string().get(sv) == simdjson::SUCCESS)
        token.refresh_token = std::string(sv);
    if (doc["token_type"].get_string().get(sv) == simdjson::SUCCESS)
        token.token_type = std::string(sv);
    if (doc["scope"].get_string().get(sv) == simdjson::SUCCESS)
        token.scopes = parse_scope_list(sv);
    if (doc["expires_in"].get_int64().get(v) == simdjson::SUCCESS)
        token.expires_at = request_time_unix + v;

    if (token.access_token.empty()) {
        throw std::runtime_error("No access_token in Claude OAuth response");
    }

    if (token.token_type.empty()) token.token_type = "Bearer";
    if (token.expires_at <= 0) {
        // Some providers omit expires_in for long-lived tokens.
        token.expires_at = request_time_unix + (60 * 60 * 24 * 30);
    }
    return token;
}

OAuthToken ClaudeOAuthFlow::exchange_code(const std::string& code,
                                          const std::string& redirect_uri,
                                          const std::string& code_verifier,
                                          const std::string& state) {
    const int64_t request_time = now_unix_seconds();

    const std::string request_body = std::string("{")
        + "\"grant_type\":\"authorization_code\","
        + "\"code\":\"" + core::utils::escape_json_string(code) + "\","
        + "\"redirect_uri\":\"" + core::utils::escape_json_string(redirect_uri) + "\","
        + "\"client_id\":\"" + core::utils::escape_json_string(client_id_) + "\","
        + "\"code_verifier\":\"" + core::utils::escape_json_string(code_verifier) + "\","
        + "\"state\":\"" + core::utils::escape_json_string(state) + "\""
        + "}";

    cpr::Response r = cpr::Post(
        cpr::Url{token_url_},
        cpr::Header{
            {"Content-Type", "application/json"},
            {"Accept", "application/json, text/plain, */*"},
            {"User-Agent", "axios/1.6.8"},
        },
        cpr::Body{request_body},
        cpr::Timeout{15000});

    if (r.status_code != 200) {
        std::string message = "Claude token exchange failed ("
                             + std::to_string(r.status_code) + "): " + r.text;
        if (r.status_code == 429) {
            if (const std::string headers = append_rate_limit_headers(r); !headers.empty()) {
                message += " [headers: " + headers + "]";
            }
        }
        throw std::runtime_error(std::move(message));
    }

    return parse_token_response(r.text, request_time);
}

OAuthToken ClaudeOAuthFlow::login() {
    // Headless compatibility path.
    if (auto env_token = token_from_env(); env_token.has_value()) {
        return *env_token;
    }

    if (!ui_) {
        throw std::runtime_error(
            "No stored Claude token and no interactive UI available.\n"
            "Run `filo --login claude` in an interactive terminal or set CLAUDE_CODE_OAUTH_TOKEN.");
    }

    const int port = pick_free_port(port_start_, port_end_);
    if (port < 0) {
        throw std::runtime_error("No free port available for Claude OAuth callback");
    }

    const std::string redirect_uri = "http://localhost:" + std::to_string(port) + "/callback";
    const std::string state = generate_random_state();
    const std::string code_verifier = generate_code_verifier();
    const std::string code_challenge = OpenAIOAuthFlow::compute_code_challenge(code_verifier);
    const std::string auth_url = build_auth_url(
        client_id_, redirect_uri, scopes_, state, code_challenge, auth_url_);
    const std::string manual_auth_url = build_auth_url(
        client_id_, kManualRedirectUrl, scopes_, state, code_challenge, auth_url_);

    ui_->show_header("Claude OAuth Login");
    ui_->show_instructions(
        "Filo will open your browser to authenticate with Claude."
        " After approval, you'll be redirected back automatically.");
    ui_->show_url(auth_url, "If your browser does not open automatically, visit:");
    ui_->show_url(manual_auth_url, "Manual fallback (for remote/headless browser setups):");

    open_browser(auth_url);

    std::mutex cv_mtx;
    std::condition_variable cv;
    bool done = false;
    std::string received_code;
    std::string received_state;
    std::string error_message;

    httplib::Server server;
    server.Get("/callback", [&](const httplib::Request& req, httplib::Response& res) {
        const std::string err = req.get_param_value("error");
        if (!err.empty()) {
            const std::string description = req.get_param_value("error_description");
            res.set_content("Claude OAuth login failed: " + err + " - " + description, "text/plain");
            std::unique_lock lk(cv_mtx);
            error_message = err + ": " + description;
            done = true;
            cv.notify_one();
            return;
        }

        res.set_content(
            "<html><body><h2>Claude login successful.</h2>"
            "<p>You can close this tab and return to filo.</p></body></html>",
            "text/html");

        std::unique_lock lk(cv_mtx);
        received_code = req.get_param_value("code");
        received_state = req.get_param_value("state");
        done = true;
        cv.notify_one();
    });

    std::thread server_thread([&server, port]() {
        server.listen("127.0.0.1", port);
    });

    bool callback_received = false;
    {
        std::unique_lock lk(cv_mtx);
        callback_received =
            cv.wait_for(lk, std::chrono::minutes(5), [&]() { return done; });
    }
    server.stop();
    if (server_thread.joinable()) server_thread.join();

    if (!callback_received) {
        ui_->show_instructions(
            "Automatic callback was not received."
            " Complete login manually, then paste the full callback URL or just the code.");
        const std::string pasted =
            ui_->prompt_secret("Paste callback URL or authorization code:");
        const ManualAuthInput manual = parse_manual_auth_input(pasted);

        if (manual.state.has_value() && *manual.state != state) {
            throw std::runtime_error("OAuth state mismatch in manual callback URL");
        }
        received_code = manual.code;
        received_state = manual.state.value_or(state);
    } else {
        if (!error_message.empty()) {
            throw std::runtime_error("Claude OAuth login failed: " + error_message);
        }
    }

    if (received_state != state) {
        throw std::runtime_error("OAuth state mismatch - possible CSRF attempt");
    }
    if (received_code.empty()) {
        throw std::runtime_error("No authorization code received from Claude OAuth");
    }

    const std::string token_exchange_redirect =
        callback_received ? redirect_uri : std::string(kManualRedirectUrl);
    OAuthToken token = exchange_code(
        received_code, token_exchange_redirect, code_verifier, state);
    ui_->show_success("Claude authentication completed.");
    return token;
}

OAuthToken ClaudeOAuthFlow::refresh(std::string_view refresh_token) {
    if (refresh_token.empty()) {
        throw std::runtime_error("Claude OAuth refresh requires a refresh token");
    }

    const int64_t request_time = now_unix_seconds();
    auto post_refresh = [&](bool include_scope) {
        std::string request_body = std::string("{")
            + "\"grant_type\":\"refresh_token\","
            + "\"refresh_token\":\"" + core::utils::escape_json_string(refresh_token) + "\","
            + "\"client_id\":\"" + core::utils::escape_json_string(client_id_) + "\"";
        if (include_scope) {
            request_body += ",\"scope\":\""
                + core::utils::escape_json_string(join_scopes(scopes_)) + "\"";
        }
        request_body += "}";

        return cpr::Post(
            cpr::Url{token_url_},
            cpr::Header{
                {"Content-Type", "application/json"},
                {"Accept", "application/json, text/plain, */*"},
                {"User-Agent", "axios/1.6.8"},
            },
            cpr::Body{request_body},
            cpr::Timeout{15000});
    };

    cpr::Response r = post_refresh(true);
    if (r.status_code == 400
        && r.text.find("invalid_scope") != std::string::npos) {
        // OAuth 2.0 allows omitting scope on refresh; the server then reuses
        // the originally granted scopes. This avoids hard failures when one of
        // our default scopes is no longer accepted by the provider.
        r = post_refresh(false);
    }

    if (r.status_code != 200) {
        std::string message = "Claude token refresh failed ("
                             + std::to_string(r.status_code) + "): " + r.text;
        if (r.status_code == 429) {
            if (const std::string headers = append_rate_limit_headers(r); !headers.empty()) {
                message += " [headers: " + headers + "]";
            }
        }
        throw std::runtime_error(std::move(message));
    }

    OAuthToken token = parse_token_response(r.text, request_time);
    if (token.refresh_token.empty()) {
        token.refresh_token = std::string(refresh_token);
    }
    return token;
}

} // namespace core::auth

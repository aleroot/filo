#include "GoogleOAuthFlow.hpp"
#include <cpr/cpr.h>
#include <httplib.h>
#include <simdjson.h>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "../logging/Logger.hpp"

namespace core::auth {

// ── Constants ────────────────────────────────────────────────────────────────

// OAuth Client ID used to initiate the OAuth2 flow.
static const char* GOOGLE_CLIENT_ID =
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com";

// OAuth Secret value used to initiate the OAuth2 flow.
// as described here: https://developers.google.com/identity/protocols/oauth2#installed
static const char* GOOGLE_CLIENT_SECRET =
    "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";
static const std::vector<std::string> GOOGLE_SCOPES = {
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
};

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

static std::string generate_random_state() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 61);
    const char chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    std::string state;
    state.reserve(32);
    for (int i = 0; i < 32; ++i) state += chars[dist(gen)];
    return state;
}

static void open_browser(const std::string& url) {
    // Double-fork so the grandchild is reparented to init and we avoid zombies.
    pid_t pid = fork();
    if (pid < 0) return; // fork failed — browser just won't open
    if (pid == 0) {
        // Intermediate child
        if (fork() == 0) {
            // Grandchild: exec the browser
#if defined(__linux__)
            execlp("xdg-open", "xdg-open", url.c_str(), nullptr);
#elif defined(__APPLE__)
            execlp("open", "open", url.c_str(), nullptr);
#endif
            _exit(1);
        }
        _exit(0);
    }
    waitpid(pid, nullptr, 0); // reap intermediate child immediately
}

// ── GoogleOAuthFlow ───────────────────────────────────────────────────────────

GoogleOAuthFlow::GoogleOAuthFlow()
    : client_id_(GOOGLE_CLIENT_ID)
    , client_secret_(GOOGLE_CLIENT_SECRET)
    , scopes_(GOOGLE_SCOPES)
    , port_start_(54321)
    , port_end_(54400) {}

GoogleOAuthFlow::GoogleOAuthFlow(std::string client_id,
                                 std::string client_secret,
                                 std::vector<std::string> scopes,
                                 int port_start,
                                 int port_end)
    : client_id_(std::move(client_id))
    , client_secret_(std::move(client_secret))
    , scopes_(std::move(scopes))
    , port_start_(port_start)
    , port_end_(port_end) {}

// static
std::string GoogleOAuthFlow::build_auth_url(std::string_view client_id,
                                             std::string_view redirect_uri,
                                             const std::vector<std::string>& scopes,
                                             std::string_view state) {
    std::string scope_str;
    for (size_t i = 0; i < scopes.size(); ++i) {
        if (i > 0) scope_str += ' ';
        scope_str += scopes[i];
    }

    return std::string("https://accounts.google.com/o/oauth2/v2/auth")
        + "?client_id="     + url_encode(client_id)
        + "&redirect_uri="  + url_encode(redirect_uri)
        + "&response_type=code"
        + "&scope="         + url_encode(scope_str)
        + "&state="         + url_encode(state)
        + "&access_type=offline"
        + "&prompt=consent";
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

OAuthToken GoogleOAuthFlow::exchange_code(const std::string& code,
                                           const std::string& redirect_uri) {
    auto request_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    cpr::Response r = cpr::Post(
        cpr::Url{"https://oauth2.googleapis.com/token"},
        cpr::Payload{
            {"client_id",     client_id_},
            {"client_secret", client_secret_},
            {"code",          code},
            {"redirect_uri",  redirect_uri},
            {"grant_type",    "authorization_code"},
        }
    );

    if (r.status_code != 200)
        throw std::runtime_error("Token exchange failed (" +
                                 std::to_string(r.status_code) + "): " + r.text);

    return parse_token_response(r.text, request_time);
}

OAuthToken GoogleOAuthFlow::login() {
    // Find a free port
    int port = -1;
    for (int p = port_start_; p <= port_end_; ++p) {
        // httplib will try to bind; we probe with a quick test socket
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) continue;
        int opt = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(static_cast<uint16_t>(p));
        bool free = (bind(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0);
        close(sock);
        if (free) { port = p; break; }
    }
    if (port < 0)
        throw std::runtime_error("No free port available for OAuth callback");

    std::string redirect_uri = "http://127.0.0.1:" + std::to_string(port) + "/callback";
    std::string state        = generate_random_state();
    std::string auth_url     = build_auth_url(client_id_, redirect_uri, scopes_, state);

    core::logging::info("Opening browser for Google login.");
    core::logging::info("If your browser does not open, visit: {}", auth_url);

    open_browser(auth_url);

    // Shared state between handler and main thread
    std::mutex              cv_mtx;
    std::condition_variable cv;
    bool                    done = false;
    std::string             received_code;
    std::string             received_state;
    std::string             error_msg;

    httplib::Server svr;

    svr.Get("/callback", [&](const httplib::Request& req, httplib::Response& res) {
        std::string code  = req.get_param_value("code");
        std::string st    = req.get_param_value("state");
        std::string err   = req.get_param_value("error");

        if (!err.empty()) {
            std::string desc = req.get_param_value("error_description");
            res.set_content("Login failed: " + err + " - " + desc, "text/plain");
            std::unique_lock lk(cv_mtx);
            error_msg = err + ": " + desc;
            done = true;
            cv.notify_one();
            return;
        }

        res.set_content(
            "<html><body><h2>Login successful!</h2>"
            "<p>You can close this tab and return to filo.</p></body></html>",
            "text/html");

        std::unique_lock lk(cv_mtx);
        received_code  = std::move(code);
        received_state = std::move(st);
        done = true;
        cv.notify_one();
    });

    std::thread server_thread([&svr, port]() {
        svr.listen("127.0.0.1", port);
    });

    // Wait up to 5 minutes for the callback
    {
        std::unique_lock lk(cv_mtx);
        bool ok = cv.wait_for(lk, std::chrono::minutes(5), [&]{ return done; });
        svr.stop();
        if (server_thread.joinable()) server_thread.join();

        if (!ok)
            throw std::runtime_error("Google login timed out after 5 minutes");
        if (!error_msg.empty())
            throw std::runtime_error("Google login failed: " + error_msg);
    }

    if (received_state != state)
        throw std::runtime_error("OAuth state mismatch — possible CSRF attempt");
    if (received_code.empty())
        throw std::runtime_error("No authorization code received from Google");

    return exchange_code(received_code, redirect_uri);
}

OAuthToken GoogleOAuthFlow::refresh(std::string_view refresh_token) {
    auto request_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    cpr::Response r = cpr::Post(
        cpr::Url{"https://oauth2.googleapis.com/token"},
        cpr::Payload{
            {"client_id",     client_id_},
            {"client_secret", client_secret_},
            {"refresh_token", std::string(refresh_token)},
            {"grant_type",    "refresh_token"},
        }
    );

    if (r.status_code != 200)
        throw std::runtime_error("Token refresh failed (" +
                                 std::to_string(r.status_code) + "): " + r.text);

    OAuthToken token = parse_token_response(r.text, request_time);

    // Refresh responses often omit refresh_token — preserve the existing one
    if (token.refresh_token.empty())
        token.refresh_token = std::string(refresh_token);

    return token;
}

} // namespace core::auth

#include "GoogleOAuthFlow.hpp"
#include "GoogleCodeAssist.hpp"
#include "OpenAIOAuthFlow.hpp"
#include <cpr/cpr.h>
#include <httplib.h>
#include <simdjson.h>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <iostream>
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
static const char* GOOGLE_CODE_ASSIST_REDIRECT_URI =
    "https://codeassist.google.com/authcode";

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

static bool env_truthy(const char* key) {
    const char* raw = std::getenv(key);
    if (!raw || raw[0] == '\0') return false;

    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

static std::string build_form_body(
    const std::vector<std::pair<std::string, std::string>>& fields) {
    std::string body;
    bool first = true;
    for (const auto& [key, value] : fields) {
        if (!first) body.push_back('&');
        first = false;
        body += url_encode(key);
        body.push_back('=');
        body += url_encode(value);
    }
    return body;
}

static std::string trim_copy(std::string_view in) {
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

static std::string oauth_listener_host() {
    if (const char* host = std::getenv("OAUTH_CALLBACK_HOST");
        host && host[0] != '\0') {
        return host;
    }
    return "127.0.0.1";
}

static std::string oauth_redirect_host_override() {
    if (const char* host = std::getenv("OAUTH_CALLBACK_REDIRECT_HOST");
        host && host[0] != '\0') {
        return host;
    }
    return {};
}

static std::optional<int> oauth_requested_port() {
    const char* raw = std::getenv("OAUTH_CALLBACK_PORT");
    if (!raw || raw[0] == '\0') return std::nullopt;

    const std::string text = trim_copy(raw);
    if (text.empty()) return std::nullopt;

    int port = 0;
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), port);
    if (ec != std::errc{} || ptr != text.data() + text.size() || port <= 0 || port > 65535) {
        throw std::runtime_error(
            "Invalid value for OAUTH_CALLBACK_PORT: \"" + text + "\"");
    }
    return port;
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

GoogleOAuthFlow::GoogleOAuthFlow(std::shared_ptr<ui::AuthUI> ui)
    : client_id_(GOOGLE_CLIENT_ID)
    , client_secret_(GOOGLE_CLIENT_SECRET)
    , scopes_(GOOGLE_SCOPES)
    , port_start_(54321)
    , port_end_(54400)
    , ui_(std::move(ui)) {}

GoogleOAuthFlow::GoogleOAuthFlow(std::string client_id,
                                 std::string client_secret,
                                 std::vector<std::string> scopes,
                                 int port_start,
                                 int port_end,
                                 std::shared_ptr<ui::AuthUI> ui)
    : client_id_(std::move(client_id))
    , client_secret_(std::move(client_secret))
    , scopes_(std::move(scopes))
    , port_start_(port_start)
    , port_end_(port_end)
    , ui_(std::move(ui)) {}

// static
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

OAuthToken GoogleOAuthFlow::exchange_code(const std::string& code,
                                           const std::string& redirect_uri,
                                           std::string_view code_verifier) {
    auto request_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    std::vector<std::pair<std::string, std::string>> fields = {
        {"client_id", client_id_},
        {"client_secret", client_secret_},
        {"code", code},
        {"redirect_uri", redirect_uri},
        {"grant_type", "authorization_code"},
    };
    if (!code_verifier.empty()) {
        fields.emplace_back("code_verifier", std::string(code_verifier));
    }

    cpr::Response r = cpr::Post(
        cpr::Url{"https://oauth2.googleapis.com/token"},
        cpr::Header{{"Content-Type", "application/x-www-form-urlencoded"}},
        cpr::Body{build_form_body(fields)}
    );

    if (r.status_code != 200)
        throw std::runtime_error("Token exchange failed (" +
                                 std::to_string(r.status_code) + "): " + r.text);

    return parse_token_response(r.text, request_time);
}

std::string GoogleOAuthFlow::build_loopback_redirect_uri(std::string_view listener_host,
                                                         int port,
                                                         std::string_view path,
                                                         std::string_view redirect_host_override) {
    std::string host = trim_copy(redirect_host_override.empty()
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

OAuthToken GoogleOAuthFlow::login() {
    const bool suppress_browser_launch = env_truthy("NO_BROWSER");
    const auto prompt_manual_auth = [&](std::string_view prompt) -> GoogleManualAuthInput {
        if (ui_) {
            return parse_manual_auth_input(ui_->prompt_secret(std::string(prompt)));
        }

        if (!isatty(STDIN_FILENO)) {
            throw std::runtime_error(
                "Google OAuth manual login requires an interactive terminal. "
                "Re-run `filo --login google` in a terminal to paste the authorization code.");
        }

        std::cout << prompt;
        std::cout.flush();

        std::string raw_input;
        std::getline(std::cin, raw_input);
        if (!std::cin) {
            throw std::runtime_error("Failed to read Google authorization code from stdin");
        }

        return parse_manual_auth_input(raw_input);
    };

    if (suppress_browser_launch) {
        const std::string redirect_uri = GOOGLE_CODE_ASSIST_REDIRECT_URI;
        const std::string state = generate_random_state();
        const std::string code_verifier = OpenAIOAuthFlow::generate_code_verifier();
        const std::string challenge = OpenAIOAuthFlow::compute_code_challenge(code_verifier);
        const std::string auth_url = build_auth_url(
            client_id_,
            redirect_uri,
            scopes_,
            state,
            challenge);

        if (ui_) {
            ui_->show_header("Google OAuth Login");
            ui_->show_instructions(
                "Filo will authenticate with Google using the same public desktop OAuth "
                "client as gemini-cli. Browser launch is disabled via NO_BROWSER, so "
                "complete the Code Assist auth page manually and paste the returned code.");
            ui_->show_url(auth_url, "Open this URL in a browser:");
        } else {
            std::cout
                << "\nGoogle OAuth Login\n"
                << "Browser launch is disabled via NO_BROWSER.\n"
                << "Open this URL in a browser and complete the Code Assist flow:\n  "
                << auth_url << "\n\n";
            std::cout.flush();
        }

        const GoogleManualAuthInput manual = prompt_manual_auth(
            "Paste authorization code or callback URL: ");
        const std::string received_state = manual.state.value_or(state);
        if (received_state != state) {
            throw std::runtime_error("OAuth state mismatch — possible CSRF attempt");
        }

        OAuthToken token = exchange_code(manual.code, redirect_uri, code_verifier);
        token.project_id = google_code_assist::setup_user(token.access_token, ui_);
        if (ui_) {
            ui_->show_success("Google login successful. You can start using Gemini in Filo.");
        }
        return token;
    }

    const std::string listener_host = oauth_listener_host();
    const std::string redirect_host_override = oauth_redirect_host_override();

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
    };
    // Keep /callback for compatibility while matching gemini-cli's callback path.
    svr.Get("/callback", callback_handler);
    svr.Get("/oauth2callback", callback_handler);

    int port = -1;
    if (const auto requested_port = oauth_requested_port(); requested_port.has_value()) {
        if (!svr.bind_to_port(listener_host, *requested_port)) {
            throw std::runtime_error(
                "Could not bind OAuth callback server to " + listener_host + ":"
                + std::to_string(*requested_port));
        }
        port = *requested_port;
    } else {
        for (int candidate = port_start_; candidate <= port_end_; ++candidate) {
            if (svr.bind_to_port(listener_host, candidate)) {
                port = candidate;
                break;
            }
        }
    }
    if (port < 0) {
        throw std::runtime_error("No free port available for Google OAuth callback");
    }

    const std::string redirect_uri = build_loopback_redirect_uri(
        listener_host,
        port,
        "/oauth2callback",
        redirect_host_override);
    const std::string state = generate_random_state();
    const std::string code_verifier = OpenAIOAuthFlow::generate_code_verifier();
    const std::string challenge = OpenAIOAuthFlow::compute_code_challenge(code_verifier);
    const std::string auth_url = build_auth_url(
        client_id_,
        redirect_uri,
        scopes_,
        state,
        challenge);

    std::thread server_thread([&svr]() {
        svr.listen_after_bind();
    });

    // Ensure the callback server is live before asking the browser to redirect into it.
    for (int attempt = 0; attempt < 200 && !svr.is_running(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!svr.is_running()) {
        svr.stop();
        if (server_thread.joinable()) server_thread.join();
        throw std::runtime_error("Google OAuth callback server failed to start");
    }

    if (ui_) {
        ui_->show_header("Google OAuth Login");
        ui_->show_instructions(
            "Filo will authenticate with Google using the same public desktop OAuth "
            "client as gemini-cli. After approval, you should be redirected back automatically.");
        ui_->show_url(auth_url, "If your browser does not open automatically, visit:");
        if (suppress_browser_launch) {
            ui_->show_instructions(
                "Browser launch is disabled via NO_BROWSER. Open the URL manually. "
                "If automatic redirect does not reach this machine, Filo will ask for the "
                "final callback URL or authorization code.");
        }
    } else {
        core::logging::info("Opening browser for Google login.");
        core::logging::info("If your browser does not open, visit: {}", auth_url);
        if (suppress_browser_launch) {
            core::logging::info(
                "Browser launch is disabled via NO_BROWSER. Open the URL manually.");
        }
    }

    if (!suppress_browser_launch) {
        open_browser(auth_url);
    }

    bool callback_received = false;
    {
        std::unique_lock lk(cv_mtx);
        callback_received = cv.wait_for(lk, std::chrono::minutes(5), [&]{ return done; });
    }
    svr.stop();
    if (server_thread.joinable()) server_thread.join();

    if (!callback_received) {
        if (ui_) {
            ui_->show_instructions(
                "Automatic callback was not received."
                " Complete login manually, then paste the full callback URL or just the code.");
        } else {
            core::logging::info(
                "Google OAuth callback was not received after 5 minutes; prompting for a "
                "manual authorization code instead.");
            core::logging::info("If needed, reopen this URL to finish login: {}", auth_url);
        }

        const GoogleManualAuthInput manual =
            prompt_manual_auth("Paste callback URL or authorization code: ");
        if (manual.state.has_value()) {
            received_state = *manual.state;
        } else {
            received_state = state;
        }
        received_code = manual.code;
    } else if (!error_msg.empty()) {
        throw std::runtime_error("Google login failed: " + error_msg);
    }

    if (received_state != state)
        throw std::runtime_error("OAuth state mismatch — possible CSRF attempt");
    if (received_code.empty())
        throw std::runtime_error("No authorization code received from Google");

    OAuthToken token = exchange_code(received_code, redirect_uri, code_verifier);
    token.project_id = google_code_assist::setup_user(token.access_token, ui_);
    if (ui_) {
        ui_->show_success("Google login successful. You can start using Gemini in Filo.");
    }
    return token;
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

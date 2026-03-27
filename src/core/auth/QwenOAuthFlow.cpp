#include "QwenOAuthFlow.hpp"
#include "OpenAIOAuthFlow.hpp"
#include <cpr/cpr.h>
#include <simdjson.h>
#include <array>
#include <chrono>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif
#include "../logging/Logger.hpp"

namespace core::auth {

namespace {

constexpr std::string_view QWEN_AUTH_HOST   = "https://chat.qwen.ai";
constexpr std::string_view QWEN_CLIENT_ID   = "f0304373b74a44d2b584a3fb70ca9e56";
constexpr std::string_view QWEN_SCOPE       = "openid profile email model.completion";
constexpr std::string_view QWEN_DEVICE_EP   = "/api/v1/oauth2/device/code";
constexpr std::string_view QWEN_TOKEN_EP    = "/api/v1/oauth2/token";
constexpr std::string_view QWEN_GRANT_DEVICE =
    "urn:ietf:params:oauth:grant-type:device_code";

cpr::Header make_qwen_headers() {
    return cpr::Header{
        {"Content-Type", "application/x-www-form-urlencoded"},
        {"Accept",       "application/json"},
        {"x-request-id", QwenOAuthFlow::generate_request_id()},
    };
}

void open_browser_best_effort(const std::string& url) {
#if defined(_WIN32)
    (void)url;
    return;
#else
    if (url.empty()) {
        return;
    }

    // Double-fork so the browser launcher is detached and we avoid zombies.
    pid_t pid = fork();
    if (pid < 0) {
        return;
    }
    if (pid == 0) {
        if (fork() == 0) {
#if defined(__linux__)
            execlp("xdg-open", "xdg-open", url.c_str(), nullptr);
            execlp("open", "open", url.c_str(), nullptr);
#elif defined(__APPLE__)
            execlp("open", "open", url.c_str(), nullptr);
            execlp("xdg-open", "xdg-open", url.c_str(), nullptr);
#endif
            _exit(1);
        }
        _exit(0);
    }

    waitpid(pid, nullptr, 0);
#endif
}

} // namespace

// ── Pure helpers ──────────────────────────────────────────────────────────────

std::string QwenOAuthFlow::generate_request_id() {
    // UUID v4: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    // y ∈ {8, 9, a, b}  (variant bits 10xx)
    std::random_device           rd;
    std::mt19937_64              gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, 15);
    std::uniform_int_distribution<uint32_t> dist_y(8, 11);

    static constexpr char hex[] = "0123456789abcdef";
    std::string uuid;
    uuid.reserve(36);

    auto push = [&](int n) {
        for (int i = 0; i < n; ++i)
            uuid += hex[dist(gen)];
    };

    push(8);  uuid += '-';
    push(4);  uuid += '-';
    uuid += '4';   // version nibble
    push(3);  uuid += '-';
    uuid += hex[dist_y(gen)];  // variant nibble
    push(3);  uuid += '-';
    push(12);

    return uuid;
}

OAuthToken QwenOAuthFlow::parse_token_response(std::string_view json,
                                               int64_t request_time_unix) {
    simdjson::dom::parser parser;
    simdjson::padded_string ps(json);
    simdjson::dom::element doc = parser.parse(ps);

    OAuthToken token;
    std::string_view sv;
    int64_t expires_in = 0;

    if (doc["access_token"].get_string().get(sv) == simdjson::SUCCESS)
        token.access_token = std::string(sv);
    if (doc["refresh_token"].get_string().get(sv) == simdjson::SUCCESS)
        token.refresh_token = std::string(sv);
    if (doc["token_type"].get_string().get(sv) == simdjson::SUCCESS)
        token.token_type = std::string(sv);
    else
        token.token_type = "Bearer";
    if (doc["expires_in"].get_int64().get(expires_in) == simdjson::SUCCESS)
        token.expires_at = request_time_unix + expires_in;

    if (token.access_token.empty())
        throw std::runtime_error("Qwen OAuth: no access_token in response");

    return token;
}

// ── Device authorization ──────────────────────────────────────────────────────

QwenOAuthFlow::DeviceAuthorization
QwenOAuthFlow::request_device_authorization(std::string_view code_challenge,
                                            std::string_view code_verifier) {
    cpr::Response r = cpr::Post(
        cpr::Url{std::string(QWEN_AUTH_HOST) + std::string(QWEN_DEVICE_EP)},
        make_qwen_headers(),
        cpr::Payload{
            {"client_id",              std::string(QWEN_CLIENT_ID)},
            {"scope",                  std::string(QWEN_SCOPE)},
            {"code_challenge",         std::string(code_challenge)},
            {"code_challenge_method",  "S256"},
        }
    );

    if (r.status_code != 200) {
        throw std::runtime_error(
            "Qwen device authorization failed ("
            + std::to_string(r.status_code) + "): " + r.text);
    }

    simdjson::dom::parser parser;
    simdjson::padded_string ps(r.text);
    simdjson::dom::element doc = parser.parse(ps);

    DeviceAuthorization auth;
    auth.code_verifier = std::string(code_verifier);

    std::string_view sv;
    int64_t iv = 0;

    if (doc["device_code"].get_string().get(sv) == simdjson::SUCCESS)
        auth.device_code = std::string(sv);
    if (doc["user_code"].get_string().get(sv) == simdjson::SUCCESS)
        auth.user_code = std::string(sv);
    if (doc["verification_uri"].get_string().get(sv) == simdjson::SUCCESS)
        auth.verification_uri = std::string(sv);
    if (doc["verification_uri_complete"].get_string().get(sv) == simdjson::SUCCESS)
        auth.verification_uri_complete = std::string(sv);
    if (doc["expires_in"].get_int64().get(iv) == simdjson::SUCCESS)
        auth.expires_in = static_cast<int>(iv);
    if (doc["interval"].get_int64().get(iv) == simdjson::SUCCESS)
        auth.interval = static_cast<int>(iv);

    if (auth.device_code.empty())
        throw std::runtime_error("Qwen OAuth: no device_code in authorization response");

    return auth;
}

// ── Token polling ─────────────────────────────────────────────────────────────

OAuthToken QwenOAuthFlow::poll_for_token(const DeviceAuthorization& auth) {
    const int max_attempts = (auth.expires_in > 0 && auth.interval > 0)
                             ? auth.expires_in / auth.interval
                             : 60;
    int interval = std::max(auth.interval, 1);

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        std::this_thread::sleep_for(std::chrono::seconds(interval));

        const int64_t request_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        cpr::Response r = cpr::Post(
            cpr::Url{std::string(QWEN_AUTH_HOST) + std::string(QWEN_TOKEN_EP)},
            make_qwen_headers(),
            cpr::Payload{
                {"client_id",    std::string(QWEN_CLIENT_ID)},
                {"device_code",  auth.device_code},
                {"grant_type",   std::string(QWEN_GRANT_DEVICE)},
                {"code_verifier", auth.code_verifier},
            }
        );

        if (r.status_code == 200) {
            auto token = parse_token_response(r.text, request_time);
            core::logging::info("Qwen OAuth authorization successful!");
            return token;
        }

        // Parse error to decide whether to continue polling
        simdjson::dom::parser parser;
        simdjson::padded_string ps(r.text);
        auto doc = parser.parse(ps);

        std::string_view error;
        if (doc["error"].get_string().get(error) == simdjson::SUCCESS) {
            if (error == "authorization_pending") {
                continue;
            } else if (error == "slow_down") {
                interval += 5;
                continue;
            } else if (error == "expired_token") {
                throw std::runtime_error("Qwen OAuth: device code expired. Please try again.");
            } else {
                throw std::runtime_error("Qwen OAuth error: " + std::string(error));
            }
        }

        // Non-200 without a recognized error field
        throw std::runtime_error(
            "Qwen OAuth: unexpected token response ("
            + std::to_string(r.status_code) + "): " + r.text);
    }

    throw std::runtime_error("Qwen OAuth: authorization timed out. Please try again.");
}

// ── Refresh ───────────────────────────────────────────────────────────────────

OAuthToken QwenOAuthFlow::do_refresh(std::string_view refresh_token) {
    const int64_t request_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    cpr::Response r = cpr::Post(
        cpr::Url{std::string(QWEN_AUTH_HOST) + std::string(QWEN_TOKEN_EP)},
        make_qwen_headers(),
        cpr::Payload{
            {"client_id",     std::string(QWEN_CLIENT_ID)},
            {"grant_type",    "refresh_token"},
            {"refresh_token", std::string(refresh_token)},
        }
    );

    if (r.status_code == 401 || r.status_code == 403)
        throw std::runtime_error("Qwen OAuth: token refresh unauthorized. Please login again.");

    if (r.status_code != 200)
        throw std::runtime_error(
            "Qwen OAuth: token refresh failed ("
            + std::to_string(r.status_code) + "): " + r.text);

    OAuthToken token = parse_token_response(r.text, request_time);

    // Refresh responses may omit refresh_token; preserve the current one.
    if (token.refresh_token.empty())
        token.refresh_token = std::string(refresh_token);

    return token;
}

// ── IOAuthFlow interface ──────────────────────────────────────────────────────

OAuthToken QwenOAuthFlow::login() {
    core::logging::info("Starting Qwen OAuth device flow...");

    const std::string verifier  = OpenAIOAuthFlow::generate_code_verifier();
    const std::string challenge = OpenAIOAuthFlow::compute_code_challenge(verifier);

    auto auth = request_device_authorization(challenge, verifier);

    const std::string open_url = auth.verification_uri_complete.empty()
                                 ? auth.verification_uri
                                 : auth.verification_uri_complete;

    core::logging::info("Please visit the following URL to authorize:");
    core::logging::info("  {}", open_url);
    core::logging::info("");
    core::logging::info("User code: {}", auth.user_code);
    core::logging::info("");
    core::logging::info("Waiting for authorization...");

    // Best-effort browser open; failures are non-fatal.
    open_browser_best_effort(open_url);

    return poll_for_token(auth);
}

OAuthToken QwenOAuthFlow::refresh(std::string_view refresh_token) {
    return do_refresh(refresh_token);
}

} // namespace core::auth

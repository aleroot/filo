#include "OpenAIOAuthFlow.hpp"
#include <cpr/cpr.h>
#include <httplib.h>
#include <simdjson.h>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace core::auth {

// ── SHA-256 (FIPS 180-4, self-contained) ─────────────────────────────────────

namespace {

constexpr uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static uint32_t rotr32(uint32_t x, uint32_t n) noexcept {
    return (x >> n) | (x << (32u - n));
}

static std::array<uint8_t, 32> sha256(std::string_view msg) {
    uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    // Pre-processing: build padded message
    std::vector<uint8_t> data(msg.begin(), msg.end());
    const uint64_t bit_len = static_cast<uint64_t>(msg.size()) * 8u;
    data.push_back(0x80u);
    while (data.size() % 64u != 56u) data.push_back(0x00u);
    for (int i = 7; i >= 0; --i)
        data.push_back(static_cast<uint8_t>((bit_len >> (i * 8)) & 0xFFu));

    // Process each 512-bit chunk
    for (size_t off = 0; off < data.size(); off += 64u) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(data[off + i*4u])     << 24u)
                 | (uint32_t(data[off + i*4u + 1u]) << 16u)
                 | (uint32_t(data[off + i*4u + 2u]) <<  8u)
                 |  uint32_t(data[off + i*4u + 3u]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr32(w[i-15], 7u) ^ rotr32(w[i-15], 18u) ^ (w[i-15] >> 3u);
            uint32_t s1 = rotr32(w[i-2],  17u) ^ rotr32(w[i-2],  19u) ^ (w[i-2]  >> 10u);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1    = rotr32(e, 6u)  ^ rotr32(e, 11u) ^ rotr32(e, 25u);
            uint32_t ch    = (e & f) ^ (~e & g);
            uint32_t temp1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0    = rotr32(a, 2u)  ^ rotr32(a, 13u) ^ rotr32(a, 22u);
            uint32_t maj   = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;

            hh = g; g = f; f = e; e = d + temp1;
            d  = c; c = b; b = a; a = temp1 + temp2;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    std::array<uint8_t, 32> digest;
    for (int i = 0; i < 8; ++i) {
        digest[i*4u]     = static_cast<uint8_t>((h[i] >> 24u) & 0xFFu);
        digest[i*4u + 1] = static_cast<uint8_t>((h[i] >> 16u) & 0xFFu);
        digest[i*4u + 2] = static_cast<uint8_t>((h[i] >>  8u) & 0xFFu);
        digest[i*4u + 3] = static_cast<uint8_t>( h[i]         & 0xFFu);
    }
    return digest;
}

// ── Base64URL (no padding, RFC 4648 §5) ──────────────────────────────────────

static std::string base64url_encode(const uint8_t* data, size_t len) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    out.reserve(((len + 2u) / 3u) * 4u);
    for (size_t i = 0; i < len; i += 3u) {
        uint32_t n = uint32_t(data[i]) << 16u;
        if (i + 1u < len) n |= uint32_t(data[i + 1u]) << 8u;
        if (i + 2u < len) n |= uint32_t(data[i + 2u]);
        out += alphabet[(n >> 18u) & 63u];
        out += alphabet[(n >> 12u) & 63u];
        if (i + 1u < len) out += alphabet[(n >>  6u) & 63u];
        if (i + 2u < len) out += alphabet[ n         & 63u];
    }
    // No '=' padding per RFC 7636 §4.2
    return out;
}

// ── URL encoding ─────────────────────────────────────────────────────────────

static std::string url_encode(std::string_view s) {
    std::string out;
    out.reserve(s.size() * 3u);
    static constexpr char hex[] = "0123456789ABCDEF";
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4u];
            out += hex[c & 0xFu];
        }
    }
    return out;
}

// ── Misc helpers ──────────────────────────────────────────────────────────────

static std::string generate_random_state() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 61);
    static constexpr char chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    std::string state;
    state.reserve(32u);
    for (int i = 0; i < 32; ++i) state += chars[dist(gen)];
    return state;
}

static void open_browser(const std::string& url) {
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

static int64_t now_unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
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
    // RFC 7636 §4.1: unreserved chars, 43–128 octets. We always emit 128.
    static constexpr char chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, static_cast<int>(sizeof(chars)) - 2);
    std::string verifier;
    verifier.reserve(128u);
    for (int i = 0; i < 128; ++i) verifier += chars[dist(gen)];
    return verifier;
}

// static
std::string OpenAIOAuthFlow::compute_code_challenge(std::string_view verifier) {
    auto hash = sha256(verifier);
    return base64url_encode(hash.data(), hash.size());
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
        + "?client_id="             + url_encode(client_id)
        + "&redirect_uri="          + url_encode(redirect_uri)
        + "&response_type=code"
        + "&scope="                 + url_encode(scope_str)
        + "&state="                 + url_encode(state)
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

    if (token.access_token.empty())
        throw std::runtime_error("No access_token in response: " + std::string(json));

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
    const std::string state         = generate_random_state();
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

} // namespace core::auth

#include <catch2/catch_test_macros.hpp>
#include "core/auth/QwenOAuthFlow.hpp"
#include "core/auth/OpenAIOAuthFlow.hpp"
#include "core/auth/AuthenticationManager.hpp"
#include "core/config/ConfigManager.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>

using namespace core::auth;

// ── Helpers ───────────────────────────────────────────────────────────────────

static int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// ── QwenOAuthFlow::generate_request_id ───────────────────────────────────────

TEST_CASE("QwenOAuthFlow::generate_request_id — UUID v4 format", "[QwenOAuthFlow]") {
    const std::string id = QwenOAuthFlow::generate_request_id();

    // Length: 8-4-4-4-12 plus 4 dashes = 36
    REQUIRE(id.size() == 36);

    // Shape: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    static const std::regex uuid_re(
        R"([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})");
    REQUIRE(std::regex_match(id, uuid_re));
}

TEST_CASE("QwenOAuthFlow::generate_request_id — each call produces a unique value", "[QwenOAuthFlow]") {
    const std::string a = QwenOAuthFlow::generate_request_id();
    const std::string b = QwenOAuthFlow::generate_request_id();
    const std::string c = QwenOAuthFlow::generate_request_id();
    REQUIRE(a != b);
    REQUIRE(b != c);
}

TEST_CASE("QwenOAuthFlow::generate_request_id — version nibble is always 4", "[QwenOAuthFlow]") {
    for (int i = 0; i < 20; ++i) {
        const std::string id = QwenOAuthFlow::generate_request_id();
        REQUIRE(id[14] == '4');
    }
}

TEST_CASE("QwenOAuthFlow::generate_request_id — variant nibble is in {8,9,a,b}", "[QwenOAuthFlow]") {
    const std::string valid = "89ab";
    for (int i = 0; i < 20; ++i) {
        const std::string id = QwenOAuthFlow::generate_request_id();
        REQUIRE(valid.find(id[19]) != std::string::npos);
    }
}

// ── QwenOAuthFlow::parse_token_response ──────────────────────────────────────

TEST_CASE("QwenOAuthFlow::parse_token_response — full response", "[QwenOAuthFlow]") {
    const int64_t t = now_unix();
    const std::string json = R"({
        "access_token":  "at-xyz",
        "refresh_token": "rt-abc",
        "token_type":    "Bearer",
        "expires_in":    3600
    })";

    auto token = QwenOAuthFlow::parse_token_response(json, t);

    REQUIRE(token.access_token  == "at-xyz");
    REQUIRE(token.refresh_token == "rt-abc");
    REQUIRE(token.token_type    == "Bearer");
    REQUIRE(token.expires_at    == t + 3600);
}

TEST_CASE("QwenOAuthFlow::parse_token_response — missing token_type defaults to Bearer", "[QwenOAuthFlow]") {
    const int64_t t = now_unix();
    const std::string json = R"({"access_token":"tok","expires_in":1800})";

    auto token = QwenOAuthFlow::parse_token_response(json, t);

    REQUIRE(token.access_token == "tok");
    REQUIRE(token.token_type   == "Bearer");
    REQUIRE(token.expires_at   == t + 1800);
}

TEST_CASE("QwenOAuthFlow::parse_token_response — missing refresh_token is empty", "[QwenOAuthFlow]") {
    const std::string json = R"({"access_token":"tok","expires_in":100})";
    auto token = QwenOAuthFlow::parse_token_response(json, 0);
    REQUIRE(token.refresh_token.empty());
}

TEST_CASE("QwenOAuthFlow::parse_token_response — missing access_token throws", "[QwenOAuthFlow]") {
    const std::string json = R"({"refresh_token":"rt","expires_in":3600})";
    REQUIRE_THROWS_AS(QwenOAuthFlow::parse_token_response(json, 0), std::runtime_error);
}

TEST_CASE("QwenOAuthFlow::parse_token_response — empty JSON throws", "[QwenOAuthFlow]") {
    REQUIRE_THROWS(QwenOAuthFlow::parse_token_response("{}", 0));
}

TEST_CASE("QwenOAuthFlow::parse_token_response — expires_at is request_time + expires_in", "[QwenOAuthFlow]") {
    auto token = QwenOAuthFlow::parse_token_response(
        R"({"access_token":"x","expires_in":7200})", 1000);
    REQUIRE(token.expires_at == 8200);
}

// ── PKCE reuse (via OpenAIOAuthFlow statics) ──────────────────────────────────

TEST_CASE("Qwen PKCE — code_verifier is 128 chars of RFC 7636 unreserved alphabet", "[QwenOAuthFlow]") {
    const std::string v = OpenAIOAuthFlow::generate_code_verifier();
    REQUIRE(v.size() == 128);
    // RFC 7636 §4.1 unreserved chars: ALPHA / DIGIT / "-" / "." / "_" / "~"
    static const std::string rfc7636 =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    for (char c : v)
        REQUIRE(rfc7636.find(c) != std::string::npos);
}

TEST_CASE("Qwen PKCE — S256 challenge is deterministic for a known verifier", "[QwenOAuthFlow]") {
    // SHA256("abc") = ba7816bf8f01cfea414140de5dae2ec73b00361bbef0469348423f656b162c8 (hex)
    // base64url      = ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0=
    // Without padding: ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0
    // With URL-safe:   ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0
    const std::string challenge = OpenAIOAuthFlow::compute_code_challenge("abc");
    REQUIRE(challenge == "ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0");
}

TEST_CASE("Qwen PKCE — challenge changes when verifier changes", "[QwenOAuthFlow]") {
    const std::string v1 = OpenAIOAuthFlow::generate_code_verifier();
    const std::string v2 = OpenAIOAuthFlow::generate_code_verifier();
    REQUIRE(OpenAIOAuthFlow::compute_code_challenge(v1)
         != OpenAIOAuthFlow::compute_code_challenge(v2));
}

// ── QwenOAuthStrategy via AuthenticationManager ───────────────────────────────

TEST_CASE("AuthenticationManager — qwen is an available login provider", "[QwenOAuthStrategy]") {
    auto manager  = AuthenticationManager::create_with_defaults("/tmp");
    auto providers = manager.available_login_providers();
    auto it = std::find(providers.begin(), providers.end(), "qwen");
    REQUIRE(it != providers.end());
}

TEST_CASE("AuthenticationManager — oauth_qwen creates a credential source", "[QwenOAuthStrategy]") {
    // Seed a pre-existing token so the OAuthTokenManager doesn't attempt a live login.
    const std::string dir = "/tmp/filo_qwen_oauth_test_" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(dir);

    {
        // Write a valid token file that OAuthTokenManager can load
        std::ofstream f(dir + "/oauth_qwen.json");
        f << R"({"access_token":"test-qwen-token","refresh_token":"rt","token_type":"Bearer","expires_at":)"
          << (now_unix() + 3600) << R"(})";
    }

    core::config::ProviderConfig cfg;
    cfg.auth_type = "oauth_qwen";

    auto manager = AuthenticationManager::create_with_defaults(dir);
    auto cred    = manager.create_credential_source("qwen", cfg);

    REQUIRE(cred != nullptr);

    auto auth = cred->get_auth();
    REQUIRE(auth.headers.count("Authorization") == 1);
    REQUIRE(auth.headers.at("Authorization") == "Bearer test-qwen-token");

    std::filesystem::remove_all(dir);
}

TEST_CASE("AuthenticationManager — oauth_qwen not activated for api_key auth_type", "[QwenOAuthStrategy]") {
    core::config::ProviderConfig cfg;
    cfg.auth_type = "api_key";

    auto manager = AuthenticationManager::create_with_defaults("/tmp");
    auto cred    = manager.create_credential_source("qwen", cfg);
    REQUIRE(cred == nullptr);
}

TEST_CASE("AuthenticationManager — oauth_qwen not activated for wrong provider type", "[QwenOAuthStrategy]") {
    core::config::ProviderConfig cfg;
    cfg.auth_type = "oauth_qwen";

    auto manager = AuthenticationManager::create_with_defaults("/tmp");
    auto cred    = manager.create_credential_source("kimi", cfg);  // wrong provider
    REQUIRE(cred == nullptr);
}

TEST_CASE("AuthenticationManager — oauth_kimi not activated for qwen provider", "[QwenOAuthStrategy]") {
    core::config::ProviderConfig cfg;
    cfg.auth_type = "oauth_kimi";  // wrong auth_type for qwen

    auto manager = AuthenticationManager::create_with_defaults("/tmp");
    auto cred    = manager.create_credential_source("qwen", cfg);
    REQUIRE(cred == nullptr);
}

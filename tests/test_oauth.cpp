#include <catch2/catch_test_macros.hpp>
#include "core/auth/OAuthToken.hpp"
#include "core/auth/ITokenStore.hpp"
#include "core/auth/IOAuthFlow.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/auth/FileTokenStore.hpp"
#include "core/auth/GoogleOAuthFlow.hpp"

#include "core/auth/OpenAIOAuthFlow.hpp"
#include "core/auth/OAuthTokenManager.hpp"
#include "core/auth/OAuthCredentialSource.hpp"
#include "core/auth/ClaudeOAuthFlow.hpp"
#include "core/auth/AuthenticationManager.hpp"
#include "core/config/ConfigManager.hpp"
#include <httplib.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <optional>
#include <thread>
#include <vector>

using namespace core::auth;

// ── Helpers ──────────────────────────────────────────────────────────────────

struct ScopedEnvVar {
    std::string name;
    std::optional<std::string> old_value;

    ScopedEnvVar(std::string env_name, const std::string& value)
        : name(std::move(env_name)) {
        if (const char* existing = std::getenv(name.c_str())) {
            old_value = std::string(existing);
        }
        setenv(name.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvVar() {
        if (old_value.has_value()) {
            setenv(name.c_str(), old_value->c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }
};

static int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static OAuthToken make_token(int64_t expires_offset_secs,
                              const std::string& refresh = "rt",
                              const std::string& access  = "at") {
    OAuthToken t;
    t.access_token  = access;
    t.refresh_token = refresh;
    t.token_type    = "Bearer";
    t.expires_at    = now_unix() + expires_offset_secs;
    return t;
}

// Temporary directory RAII for FileTokenStore tests
struct TempDir {
    std::string path;
    TempDir() {
        path = "/tmp/filo_oauth_test_" + std::to_string(getpid());
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

// ── Stub implementations for OAuthTokenManager tests ─────────────────────────

struct StubStore : ITokenStore {
    std::optional<OAuthToken> stored;
    int load_calls  = 0;
    int save_calls  = 0;
    int clear_calls = 0;

    std::optional<OAuthToken> load(std::string_view) override {
        ++load_calls;
        return stored;
    }
    void save(std::string_view, const OAuthToken& t) override {
        ++save_calls;
        stored = t;
    }
    void clear(std::string_view) override {
        ++clear_calls;
        stored = std::nullopt;
    }
};

struct StubFlow : IOAuthFlow {
    int         login_calls   = 0;
    int         refresh_calls = 0;
    OAuthToken  login_result;
    OAuthToken  refresh_result;

    OAuthToken login() override {
        ++login_calls;
        return login_result;
    }
    OAuthToken refresh(std::string_view) override {
        ++refresh_calls;
        return refresh_result;
    }
};

// ── OAuthToken ────────────────────────────────────────────────────────────────

TEST_CASE("OAuthToken::is_valid() — future expiry", "[OAuthToken]") {
    auto t = make_token(3600);
    REQUIRE(t.is_valid());
}

TEST_CASE("OAuthToken::is_valid() — past expiry", "[OAuthToken]") {
    auto t = make_token(-60); // expired a minute ago
    REQUIRE_FALSE(t.is_valid());
}

TEST_CASE("OAuthToken::is_valid() — within 5-minute buffer", "[OAuthToken]") {
    auto t = make_token(120); // expires in 2 minutes — inside the 5-min buffer
    REQUIRE_FALSE(t.is_valid());
}

TEST_CASE("OAuthToken::is_valid() — empty access token", "[OAuthToken]") {
    OAuthToken t;
    t.expires_at = now_unix() + 3600;
    REQUIRE_FALSE(t.is_valid());
}

TEST_CASE("OAuthToken::has_refresh_token()", "[OAuthToken]") {
    OAuthToken t;
    REQUIRE_FALSE(t.has_refresh_token());
    t.refresh_token = "rt";
    REQUIRE(t.has_refresh_token());
}

// ── ApiKeyCredentialSource ────────────────────────────────────────────────────

TEST_CASE("ApiKeyCredentialSource::as_query_param — default param name", "[ApiKeyCredentialSource]") {
    auto src  = ApiKeyCredentialSource::as_query_param("my-key");
    auto auth = src->get_auth();
    REQUIRE(auth.query_params.count("key") == 1);
    REQUIRE(auth.query_params.at("key") == "my-key");
    REQUIRE(auth.headers.empty());
}

TEST_CASE("ApiKeyCredentialSource::as_query_param — custom param name", "[ApiKeyCredentialSource]") {
    auto src  = ApiKeyCredentialSource::as_query_param("k", "token");
    auto auth = src->get_auth();
    REQUIRE(auth.query_params.count("token") == 1);
    REQUIRE(auth.query_params.at("token") == "k");
}

TEST_CASE("ApiKeyCredentialSource::as_query_param omits empty values", "[ApiKeyCredentialSource]") {
    auto src  = ApiKeyCredentialSource::as_query_param("");
    auto auth = src->get_auth();
    REQUIRE(auth.query_params.empty());
}

TEST_CASE("ApiKeyCredentialSource::as_bearer", "[ApiKeyCredentialSource]") {
    auto src  = ApiKeyCredentialSource::as_bearer("secret");
    auto auth = src->get_auth();
    REQUIRE(auth.headers.count("Authorization") == 1);
    REQUIRE(auth.headers.at("Authorization") == "Bearer secret");
    REQUIRE(auth.query_params.empty());
}

TEST_CASE("ApiKeyCredentialSource::as_bearer omits empty Authorization header", "[ApiKeyCredentialSource]") {
    auto src  = ApiKeyCredentialSource::as_bearer("");
    auto auth = src->get_auth();
    REQUIRE(auth.headers.count("Authorization") == 0);
    REQUIRE(auth.query_params.empty());
}

TEST_CASE("ApiKeyCredentialSource::as_custom_header", "[ApiKeyCredentialSource]") {
    auto src  = ApiKeyCredentialSource::as_custom_header("v", "x-api-key");
    auto auth = src->get_auth();
    REQUIRE(auth.headers.count("x-api-key") == 1);
    REQUIRE(auth.headers.at("x-api-key") == "v");
    REQUIRE(auth.query_params.empty());
}

TEST_CASE("ApiKeyCredentialSource::as_custom_header omits empty header values", "[ApiKeyCredentialSource]") {
    auto src  = ApiKeyCredentialSource::as_custom_header("", "x-api-key");
    auto auth = src->get_auth();
    REQUIRE(auth.headers.count("x-api-key") == 0);
    REQUIRE(auth.query_params.empty());
}

TEST_CASE("ApiKeyCredentialSource::none does not emit auth headers", "[ApiKeyCredentialSource]") {
    auto src  = ApiKeyCredentialSource::none();
    auto auth = src->get_auth();
    REQUIRE(auth.headers.empty());
    REQUIRE(auth.query_params.empty());
}

// ── FileTokenStore ────────────────────────────────────────────────────────────

TEST_CASE("FileTokenStore — save and load round-trips all fields", "[FileTokenStore]") {
    TempDir tmp;
    FileTokenStore store(tmp.path);

    OAuthToken original = make_token(3600, "refresh123", "access456");
    original.token_type = "Bearer";
    original.device_id = "device-123";
    original.account_id = "acct_123";
    store.save("google", original);

    auto loaded = store.load("google");
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->access_token  == "access456");
    REQUIRE(loaded->refresh_token == "refresh123");
    REQUIRE(loaded->token_type    == "Bearer");
    REQUIRE(loaded->expires_at    == original.expires_at);
    REQUIRE(loaded->device_id     == "device-123");
    REQUIRE(loaded->account_id == "acct_123");
}

TEST_CASE("FileTokenStore — save/load round-trips OAuth scopes", "[FileTokenStore]") {
    TempDir tmp;
    FileTokenStore store(tmp.path);

    OAuthToken original = make_token(3600, "refresh123", "access456");
    original.scopes = {"user:profile", "user:inference", "user:file_upload"};
    store.save("claude", original);

    auto loaded = store.load("claude");
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->scopes == original.scopes);
}

TEST_CASE("FileTokenStore — load returns nullopt for unknown provider", "[FileTokenStore]") {
    TempDir tmp;
    FileTokenStore store(tmp.path);
    REQUIRE_FALSE(store.load("nonexistent").has_value());
}

TEST_CASE("FileTokenStore — clear removes the token", "[FileTokenStore]") {
    TempDir tmp;
    FileTokenStore store(tmp.path);

    store.save("google", make_token(3600));
    REQUIRE(store.load("google").has_value());

    store.clear("google");
    REQUIRE_FALSE(store.load("google").has_value());
}

TEST_CASE("FileTokenStore — clear on missing file does not throw", "[FileTokenStore]") {
    TempDir tmp;
    FileTokenStore store(tmp.path);
    REQUIRE_NOTHROW(store.clear("never_saved"));
}

TEST_CASE("FileTokenStore — saved file has permissions 0600", "[FileTokenStore]") {
    TempDir tmp;
    FileTokenStore store(tmp.path);
    store.save("google", make_token(3600));

    std::string path = tmp.path + "/oauth_google.json";
    struct stat st{};
    REQUIRE(stat(path.c_str(), &st) == 0);
    REQUIRE((st.st_mode & 0777) == 0600);
}

TEST_CASE("FileTokenStore — load returns nullopt when file is corrupted", "[FileTokenStore]") {
    TempDir tmp;
    // Write garbage JSON
    std::ofstream f(tmp.path + "/oauth_google.json");
    f << "not valid json {{{{";
    f.close();

    FileTokenStore store(tmp.path);
    REQUIRE_FALSE(store.load("google").has_value());
}

TEST_CASE("FileTokenStore — save/load escapes JSON special characters in tokens", "[FileTokenStore]") {
    TempDir tmp;
    FileTokenStore store(tmp.path);

    OAuthToken original = make_token(3600);
    original.access_token = "a\"b\\c\nd";
    original.refresh_token = "r\"s\\t\nu";
    original.token_type = "Bearer\"x";

    store.save("google", original);
    auto loaded = store.load("google");

    REQUIRE(loaded.has_value());
    REQUIRE(loaded->access_token == original.access_token);
    REQUIRE(loaded->refresh_token == original.refresh_token);
    REQUIRE(loaded->token_type == original.token_type);
}

// ── GoogleOAuthFlow — static methods ─────────────────────────────────────────

TEST_CASE("GoogleOAuthFlow::build_auth_url — contains client_id", "[GoogleOAuthFlow]") {
    auto url = GoogleOAuthFlow::build_auth_url(
        "my-client-id", "http://localhost:54321/callback", {"scope1"}, "state123");
    REQUIRE(url.find("my-client-id") != std::string::npos);
}

TEST_CASE("GoogleOAuthFlow::build_auth_url — contains redirect_uri encoded", "[GoogleOAuthFlow]") {
    auto url = GoogleOAuthFlow::build_auth_url(
        "cid", "http://127.0.0.1:54321/callback", {"s"}, "st");
    // The colon and slashes in the URI must be percent-encoded
    REQUIRE(url.find("redirect_uri=") != std::string::npos);
    REQUIRE(url.find("127.0.0.1") != std::string::npos);
}

TEST_CASE("GoogleOAuthFlow::build_auth_url — contains state", "[GoogleOAuthFlow]") {
    auto url = GoogleOAuthFlow::build_auth_url("c", "http://r", {}, "my-state");
    REQUIRE(url.find("my-state") != std::string::npos);
}

TEST_CASE("GoogleOAuthFlow::build_auth_url — contains access_type=offline", "[GoogleOAuthFlow]") {
    auto url = GoogleOAuthFlow::build_auth_url("c", "http://r", {}, "s");
    REQUIRE(url.find("access_type=offline") != std::string::npos);
}

TEST_CASE("GoogleOAuthFlow::build_auth_url — scopes are joined and encoded", "[GoogleOAuthFlow]") {
    auto url = GoogleOAuthFlow::build_auth_url(
        "c", "http://r",
        {"https://www.googleapis.com/auth/cloud-platform",
         "https://www.googleapis.com/auth/userinfo.email"},
        "s");
    REQUIRE(url.find("scope=") != std::string::npos);
    REQUIRE(url.find("cloud-platform") != std::string::npos);
    REQUIRE(url.find("userinfo.email") != std::string::npos);
}

TEST_CASE("GoogleOAuthFlow::parse_token_response — extracts access_token", "[GoogleOAuthFlow]") {
    std::string json = R"({
        "access_token": "ya29.abc",
        "refresh_token": "1//rt",
        "token_type": "Bearer",
        "expires_in": 3600
    })";
    int64_t req_time = now_unix();
    auto token = GoogleOAuthFlow::parse_token_response(json, req_time);
    REQUIRE(token.access_token  == "ya29.abc");
    REQUIRE(token.refresh_token == "1//rt");
    REQUIRE(token.token_type    == "Bearer");
}

TEST_CASE("GoogleOAuthFlow::parse_token_response — sets expires_at from expires_in", "[GoogleOAuthFlow]") {
    std::string json = R"({"access_token":"at","expires_in":3600})";
    int64_t req_time = now_unix();
    auto token = GoogleOAuthFlow::parse_token_response(json, req_time);
    REQUIRE(token.expires_at == req_time + 3600);
}

TEST_CASE("GoogleOAuthFlow::parse_token_response — throws when access_token absent", "[GoogleOAuthFlow]") {
    std::string json = R"({"error": "invalid_grant"})";
    REQUIRE_THROWS_AS(
        GoogleOAuthFlow::parse_token_response(json, now_unix()),
        std::runtime_error);
}

// ── OAuthTokenManager ─────────────────────────────────────────────────────────

TEST_CASE("OAuthTokenManager — returns valid cached token without hitting flow/store", "[OAuthTokenManager]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    store->stored = make_token(3600);

    OAuthTokenManager mgr("google", flow, store);
    auto tok = mgr.get_valid_token();

    REQUIRE(tok.access_token == "at");
    REQUIRE(flow->login_calls   == 0);
    REQUIRE(flow->refresh_calls == 0);

    // Second call should use in-memory cache — no extra store hits
    int load_before = store->load_calls;
    mgr.get_valid_token();
    REQUIRE(store->load_calls == load_before);
}

TEST_CASE("OAuthTokenManager — calls refresh() when token expired + has refresh_token", "[OAuthTokenManager]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    store->stored = make_token(-60, "old-rt", "old-at"); // expired

    flow->refresh_result = make_token(3600, "old-rt", "new-at");

    OAuthTokenManager mgr("google", flow, store);
    auto tok = mgr.get_valid_token();

    REQUIRE(tok.access_token     == "new-at");
    REQUIRE(flow->refresh_calls  == 1);
    REQUIRE(flow->login_calls    == 0);
    REQUIRE(store->save_calls    >= 1);
}

TEST_CASE("OAuthTokenManager — calls login() when store is empty", "[OAuthTokenManager]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    // store->stored is nullopt by default

    flow->login_result = make_token(3600, "rt-new", "at-new");

    OAuthTokenManager mgr("google", flow, store);
    auto tok = mgr.get_valid_token();

    REQUIRE(tok.access_token  == "at-new");
    REQUIRE(flow->login_calls == 1);
    REQUIRE(store->save_calls >= 1);
}

TEST_CASE("OAuthTokenManager — calls login() when token expired and no refresh_token", "[OAuthTokenManager]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    store->stored = make_token(-60, "" /* no refresh_token */, "old-at");

    flow->login_result = make_token(3600, "new-rt", "new-at");

    OAuthTokenManager mgr("google", flow, store);
    auto tok = mgr.get_valid_token();

    REQUIRE(tok.access_token  == "new-at");
    REQUIRE(flow->login_calls == 1);
    REQUIRE(flow->refresh_calls == 0);
}

TEST_CASE("OAuthTokenManager — login() force re-login", "[OAuthTokenManager]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    store->stored = make_token(3600); // valid token already stored

    flow->login_result = make_token(7200, "new-rt", "new-at");

    OAuthTokenManager mgr("google", flow, store);
    mgr.login();

    REQUIRE(flow->login_calls == 1);
    REQUIRE(store->save_calls >= 1);

    auto tok = mgr.get_valid_token();
    REQUIRE(tok.access_token == "new-at"); // uses the newly-logged-in token
}

TEST_CASE("OAuthTokenManager — logout() clears store and forces next get to login", "[OAuthTokenManager]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    store->stored = make_token(3600);

    flow->login_result = make_token(3600, "fresh-rt", "fresh-at");

    OAuthTokenManager mgr("google", flow, store);
    mgr.get_valid_token(); // loads valid token, caches it

    mgr.logout();
    REQUIRE(store->clear_calls == 1);
    REQUIRE_FALSE(store->stored.has_value());

    // Next call must trigger login
    mgr.get_valid_token();
    REQUIRE(flow->login_calls == 1);
}

TEST_CASE("OAuthTokenManager — force_refresh() refreshes stored token", "[OAuthTokenManager]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    store->stored = make_token(-3600, "refresh-me", "expired-at");
    flow->refresh_result = make_token(3600, "new-rt", "new-at");

    OAuthTokenManager mgr("claude", flow, store);
    mgr.force_refresh();

    REQUIRE(flow->refresh_calls == 1);
    REQUIRE(store->stored.has_value());
    REQUIRE(store->stored->access_token == "new-at");
}

TEST_CASE("OAuthTokenManager — refresh preserves account_id when omitted", "[OAuthTokenManager]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    OAuthToken stored = make_token(-3600, "refresh-me", "expired-at");
    stored.account_id = "acct_789";
    store->stored = stored;

    flow->refresh_result = make_token(3600, "new-rt", "new-at");

    OAuthTokenManager mgr("openai-pkce", flow, store);
    auto token = mgr.get_valid_token();

    REQUIRE(token.account_id == "acct_789");
}

TEST_CASE("OAuthTokenManager — force_refresh() throws without refresh token", "[OAuthTokenManager]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    store->stored = make_token(-3600, "", "expired-at");

    OAuthTokenManager mgr("claude", flow, store);
    REQUIRE_THROWS_AS(mgr.force_refresh(), std::runtime_error);
    REQUIRE(flow->refresh_calls == 0);
}

// ── OAuthCredentialSource ─────────────────────────────────────────────────────

TEST_CASE("OAuthCredentialSource — returns Bearer header from manager's token", "[OAuthCredentialSource]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    store->stored = make_token(3600, "rt", "my-access-token");

    auto manager = std::make_shared<OAuthTokenManager>("google", flow, store);
    OAuthCredentialSource src(manager);

    auto auth = src.get_auth();
    REQUIRE(auth.headers.count("Authorization") == 1);
    REQUIRE(auth.headers.at("Authorization") == "Bearer my-access-token");
    REQUIRE(auth.query_params.empty());
}

TEST_CASE("OAuthCredentialSource — exposes account_id property when present", "[OAuthCredentialSource]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    OAuthToken token = make_token(3600, "rt", "my-openai-access-token");
    token.account_id = "acct_456";
    store->stored = token;

    auto manager = std::make_shared<OAuthTokenManager>("openai-pkce", flow, store);
    OAuthCredentialSource src(manager);

    const auto auth = src.get_auth();
    REQUIRE(auth.headers.count("chatgpt-account-id") == 0);
    REQUIRE(auth.properties.count("account_id") == 1);
    REQUIRE(auth.properties.at("account_id") == "acct_456");
}

TEST_CASE("OAuthCredentialSource — session key uses Cookie auth and org UUID header", "[OAuthCredentialSource]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    store->stored = make_token(3600, "rt", "sk-ant-sid01-session-token");
    ScopedEnvVar org_uuid("CLAUDE_CODE_ORGANIZATION_UUID", "org-test-uuid");

    auto manager = std::make_shared<OAuthTokenManager>("claude", flow, store);
    OAuthCredentialSource src(manager);

    auto auth = src.get_auth();
    REQUIRE(auth.headers.count("Cookie") == 1);
    REQUIRE(auth.headers.at("Cookie") == "sessionKey=sk-ant-sid01-session-token");
    REQUIRE(auth.headers.count("X-Organization-Uuid") == 1);
    REQUIRE(auth.headers.at("X-Organization-Uuid") == "org-test-uuid");
    REQUIRE(auth.headers.count("Authorization") == 0);
    REQUIRE(auth.properties.at("oauth") == "1");
    REQUIRE(auth.properties.at("auth_mode") == "session_cookie");
}

TEST_CASE("OAuthCredentialSource — refresh_on_auth_failure() returns true when refresh succeeds", "[OAuthCredentialSource]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    store->stored = make_token(-3600, "refresh-me", "expired-at");
    flow->refresh_result = make_token(3600, "new-rt", "new-at");

    auto manager = std::make_shared<OAuthTokenManager>("claude", flow, store);
    OAuthCredentialSource src(manager);

    REQUIRE(src.refresh_on_auth_failure());
    REQUIRE(flow->refresh_calls == 1);
}

TEST_CASE("OAuthCredentialSource — refresh_on_auth_failure() returns false without refresh token", "[OAuthCredentialSource]") {
    auto flow  = std::make_shared<StubFlow>();
    auto store = std::make_shared<StubStore>();
    store->stored = make_token(-3600, "", "expired-at");

    auto manager = std::make_shared<OAuthTokenManager>("claude", flow, store);
    OAuthCredentialSource src(manager);

    REQUIRE_FALSE(src.refresh_on_auth_failure());
    REQUIRE(flow->refresh_calls == 0);
}

// ── ClaudeOAuthFlow / AuthenticationManager ──────────────────────────────────

TEST_CASE("ClaudeOAuthFlow::login uses ANTHROPIC_AUTH_TOKEN", "[ClaudeOAuthFlow]") {
    ScopedEnvVar token("ANTHROPIC_AUTH_TOKEN", "test-claude-token");
    ClaudeOAuthFlow flow;
    const auto oauth = flow.login();

    REQUIRE(oauth.access_token == "test-claude-token");
    REQUIRE(oauth.token_type == "Bearer");
    REQUIRE(oauth.is_valid());
}

TEST_CASE("ClaudeOAuthFlow::login uses CLAUDE_CODE_OAUTH_TOKEN with optional scopes/refresh token", "[ClaudeOAuthFlow]") {
    ScopedEnvVar oauth_token("CLAUDE_CODE_OAUTH_TOKEN", "oauth-access-token");
    ScopedEnvVar oauth_scopes("CLAUDE_CODE_OAUTH_SCOPES", "user:profile user:inference");
    ScopedEnvVar oauth_refresh("CLAUDE_CODE_OAUTH_REFRESH_TOKEN", "oauth-refresh-token");
    ClaudeOAuthFlow flow;
    const auto oauth = flow.login();

    REQUIRE(oauth.access_token == "oauth-access-token");
    REQUIRE(oauth.refresh_token == "oauth-refresh-token");
    REQUIRE(oauth.scopes == std::vector<std::string>{"user:profile", "user:inference"});
}

TEST_CASE("ClaudeOAuthFlow::login throws when token is missing", "[ClaudeOAuthFlow]") {
    ScopedEnvVar token("ANTHROPIC_AUTH_TOKEN", "");
    ClaudeOAuthFlow flow;
    REQUIRE_THROWS_AS(flow.login(), std::runtime_error);
}

TEST_CASE("ClaudeOAuthFlow::refresh retries without scope after invalid_scope", "[ClaudeOAuthFlow]") {
    httplib::Server server;
    std::atomic<int> call_count{0};
    std::vector<std::string> request_bodies;
    std::mutex request_mutex;

    server.Post("/oauth/token", [&](const httplib::Request& req, httplib::Response& res) {
        const int call = ++call_count;
        {
            std::lock_guard lock(request_mutex);
            request_bodies.push_back(req.body);
        }

        if (call == 1) {
            res.status = 400;
            res.set_content(
                R"({"error":"invalid_scope","error_description":"The requested scope is invalid"})",
                "application/json");
            return;
        }

        res.status = 200;
        res.set_content(
            R"({"access_token":"new-access-token","token_type":"Bearer","expires_in":3600})",
            "application/json");
    });

    int port = -1;
    for (int candidate = 42000; candidate < 42100; ++candidate) {
        if (server.bind_to_port("127.0.0.1", candidate)) {
            port = candidate;
            break;
        }
    }
    if (port <= 0) {
        SKIP("Loopback port binding unavailable in this test environment");
    }
    std::jthread server_thread([&server]() { server.listen_after_bind(); });

    const std::string base = "http://127.0.0.1:" + std::to_string(port);
    ClaudeOAuthFlow flow(
        "test-client-id",
        base + "/oauth/authorize",
        base + "/oauth/token",
        {"unknown:scope", "user:inference"},
        42000,
        42001,
        nullptr);

    const auto token = flow.refresh("existing-refresh-token");
    server.stop();

    REQUIRE(token.access_token == "new-access-token");
    REQUIRE(token.refresh_token == "existing-refresh-token");
    REQUIRE(call_count.load() == 2);
    REQUIRE(request_bodies.size() == 2);
    REQUIRE(request_bodies[0].find("\"scope\"") != std::string::npos);
    REQUIRE(request_bodies[1].find("\"scope\"") == std::string::npos);
}

TEST_CASE("AuthenticationManager login(claude) stores token and returns hints", "[AuthenticationManager]") {
    TempDir tmp;
    ScopedEnvVar token("ANTHROPIC_AUTH_TOKEN", "test-claude-token");

    auto manager = AuthenticationManager::create_with_defaults(tmp.path);
    auto result = manager.login("claude");

    REQUIRE(result.provider == "Claude");
    REQUIRE_FALSE(result.hints.empty());

    FileTokenStore store(tmp.path);
    auto saved = store.load("claude");
    REQUIRE(saved.has_value());
    REQUIRE(saved->access_token == "test-claude-token");
}

TEST_CASE("AuthenticationManager resolves oauth_claude credential source", "[AuthenticationManager]") {
    TempDir tmp;
    ScopedEnvVar token("ANTHROPIC_AUTH_TOKEN", "test-claude-token");

    auto manager = AuthenticationManager::create_with_defaults(tmp.path);
    manager.login("claude");

    core::config::ProviderConfig provider;
    provider.auth_type = "oauth_claude";
    auto cred = manager.create_credential_source("claude", provider);

    REQUIRE(cred != nullptr);
    auto auth = cred->get_auth();
    REQUIRE(auth.headers.count("Authorization") == 1);
    REQUIRE(auth.headers.at("Authorization") == "Bearer test-claude-token");
}

TEST_CASE("AuthenticationManager login reports unknown providers", "[AuthenticationManager]") {
    TempDir tmp;
    auto manager = AuthenticationManager::create_with_defaults(tmp.path);
    REQUIRE_THROWS_AS(manager.login("unknown-provider"), std::runtime_error);
}

// ── OpenAIOAuthFlow — static pure functions ───────────────────────────────────

TEST_CASE("OpenAIOAuthFlow default constructor works without OPENAI_OAUTH_CLIENT_ID",
          "[OpenAIOAuthFlow]") {
    ScopedEnvVar client_id("OPENAI_OAUTH_CLIENT_ID", "");
    REQUIRE_NOTHROW(OpenAIOAuthFlow{});
}

TEST_CASE("OpenAIOAuthFlow::generate_code_verifier — correct length and charset", "[OpenAIOAuthFlow]") {
    const auto v = OpenAIOAuthFlow::generate_code_verifier();
    REQUIRE(v.size() >= 43u);
    REQUIRE(v.size() <= 128u);
    // RFC 7636 §4.1 allowed chars: ALPHA / DIGIT / "-" / "." / "_" / "~"
    const std::string allowed =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    for (char c : v) {
        REQUIRE(allowed.find(c) != std::string::npos);
    }
}

TEST_CASE("OpenAIOAuthFlow::compute_code_challenge — RFC 7636 appendix B vector", "[OpenAIOAuthFlow]") {
    // Published test vector from RFC 7636 §B:
    //   verifier   = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    //   challenge  = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    const auto challenge = OpenAIOAuthFlow::compute_code_challenge(
        "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk");
    REQUIRE(challenge == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
}

TEST_CASE("OpenAIOAuthFlow::compute_code_challenge — deterministic", "[OpenAIOAuthFlow]") {
    const std::string v = "test-verifier-12345";
    REQUIRE(OpenAIOAuthFlow::compute_code_challenge(v) ==
            OpenAIOAuthFlow::compute_code_challenge(v));
}

TEST_CASE("OpenAIOAuthFlow::build_auth_url — contains PKCE parameters", "[OpenAIOAuthFlow]") {
    const auto url = OpenAIOAuthFlow::build_auth_url(
        "my-client-id",
        "http://localhost:54321/callback",
        {"openid", "email"},
        "state123",
        "challenge456",
        "https://auth.openai.com/authorize"
    );
    REQUIRE(url.find("my-client-id")          != std::string::npos);
    REQUIRE(url.find("challenge456")           != std::string::npos);
    REQUIRE(url.find("S256")                   != std::string::npos);
    REQUIRE(url.find("code_challenge_method")  != std::string::npos);
    REQUIRE(url.find("state123")               != std::string::npos);
    REQUIRE(url.find("response_type=code")     != std::string::npos);
    REQUIRE(url.find("id_token_add_organizations=true") != std::string::npos);
    REQUIRE(url.find("codex_cli_simplified_flow=true")  != std::string::npos);
}

TEST_CASE("OpenAIOAuthFlow::build_auth_url — redirect_uri is percent-encoded", "[OpenAIOAuthFlow]") {
    const auto url = OpenAIOAuthFlow::build_auth_url(
        "cid", "http://127.0.0.1:54321/callback", {}, "s", "c",
        "https://auth.openai.com/authorize");
    REQUIRE(url.find("redirect_uri=") != std::string::npos);
    REQUIRE(url.find("127.0.0.1")     != std::string::npos);
}

TEST_CASE("OpenAIOAuthFlow::parse_token_response — extracts all fields", "[OpenAIOAuthFlow]") {
    const std::string json = R"({
        "access_token":  "sess-abc123",
        "refresh_token": "rt-def456",
        "token_type":    "Bearer",
        "expires_in":    3600
    })";
    const int64_t t = now_unix();
    const auto token = OpenAIOAuthFlow::parse_token_response(json, t);
    REQUIRE(token.access_token  == "sess-abc123");
    REQUIRE(token.refresh_token == "rt-def456");
    REQUIRE(token.token_type    == "Bearer");
    REQUIRE(token.expires_at    == t + 3600);
}

TEST_CASE("OpenAIOAuthFlow::parse_token_response — extracts account_id from id_token claim", "[OpenAIOAuthFlow]") {
    const std::string id_token =
        "eyJhbGciOiJub25lIn0."
        "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGguY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdF9jbGFpbSJ9."
        "sig";
    const std::string json = std::string(R"({
        "access_token": "opaque-token",
        "id_token": ")") + id_token + R"(",
        "expires_in": 3600
    })";

    const auto token = OpenAIOAuthFlow::parse_token_response(json, now_unix());
    REQUIRE(token.account_id == "acct_claim");
}

TEST_CASE("OpenAIOAuthFlow::parse_token_response — extracts account_id from access token JWT", "[OpenAIOAuthFlow]") {
    const std::string access_token =
        "eyJhbGciOiJub25lIn0."
        "eyJjaGF0Z3B0X2FjY291bnRfaWQiOiJhY2N0XzEyMyJ9."
        "sig";
    const std::string json = std::string(R"({
        "access_token": ")") + access_token + R"(",
        "expires_in": 3600
    })";

    const auto token = OpenAIOAuthFlow::parse_token_response(json, now_unix());
    REQUIRE(token.account_id == "acct_123");
}

TEST_CASE("OpenAIOAuthFlow::parse_token_response — throws when access_token absent", "[OpenAIOAuthFlow]") {
    const std::string json = R"({"error": "invalid_grant"})";
    REQUIRE_THROWS_AS(
        OpenAIOAuthFlow::parse_token_response(json, now_unix()),
        std::runtime_error);
}

// ── AuthenticationManager — OpenAI integration ───────────────────────────────
TEST_CASE("AuthenticationManager resolves oauth_openai_pkce credential source", "[AuthenticationManager]") {
    TempDir tmp;

    FileTokenStore store(tmp.path);
    store.save("openai-pkce", make_token(3600, "openai-rt", "sess-openai-token"));

    auto manager = AuthenticationManager::create_with_defaults(tmp.path);
    core::config::ProviderConfig provider;
    provider.auth_type = "oauth_openai_pkce";

    auto cred = manager.create_credential_source("openai", provider);
    REQUIRE(cred != nullptr);

    const auto auth = cred->get_auth();
    REQUIRE(auth.headers.count("Authorization") == 1);
    REQUIRE(auth.headers.at("Authorization") == "Bearer sess-openai-token");
}

TEST_CASE("AuthenticationManager rejects legacy oauth_openai auth type", "[AuthenticationManager]") {
    TempDir tmp;
    auto manager = AuthenticationManager::create_with_defaults(tmp.path);

    core::config::ProviderConfig provider;
    provider.auth_type = "oauth_openai";

    auto cred = manager.create_credential_source("openai", provider);
    REQUIRE(cred == nullptr);
}

TEST_CASE("AuthenticationManager exposes openai login provider without legacy aliases", "[AuthenticationManager]") {
    TempDir tmp;
    auto manager = AuthenticationManager::create_with_defaults(tmp.path);

    const auto providers = manager.available_login_providers();
    REQUIRE(std::find(providers.begin(), providers.end(), "openai") != providers.end());
    REQUIRE(std::find(providers.begin(), providers.end(), "openai-pkce") == providers.end());
    REQUIRE(std::find(providers.begin(), providers.end(), "openai-api-key") == providers.end());
}

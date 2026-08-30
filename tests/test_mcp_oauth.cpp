#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/auth/FileTokenStore.hpp"
#include "core/auth/OAuthErrors.hpp"
#include "core/auth/OAuthLoopback.hpp"
#include "core/auth/OAuthToken.hpp"
#include "core/auth/OAuthTokenEndpoint.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/mcp/McpClientSession.hpp"
#include "core/mcp/McpOAuth.hpp"

#include <cpr/cpr.h>
#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using Catch::Matchers::ContainsSubstring;

namespace {

namespace fs = std::filesystem;

class ScopedServerStop {
public:
    explicit ScopedServerStop(httplib::Server& server)
        : server_(server) {}
    ~ScopedServerStop() { server_.stop(); }

private:
    httplib::Server& server_;
};

void wait_until_running(httplib::Server& server) {
    for (int i = 0; i < 100 && !server.is_running(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(server.is_running());
}

fs::path make_temp_dir(const std::string& label) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    auto path = fs::temp_directory_path()
              / std::format("{}_{}_{}", label, ::getpid(), stamp);
    fs::create_directories(path);
    return path;
}

struct TempDir {
    fs::path path;
    explicit TempDir(std::string label)
        : path(make_temp_dir(std::move(label))) {}
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

struct MockAsState {
    std::mutex mutex;
    std::vector<std::string> registration_bodies;
    std::vector<std::string> token_bodies;
    std::map<std::string, std::string> auth_headers_seen;
    int token_calls = 0;
    std::string last_grant_type;
    std::string last_code_verifier;
    std::string last_resource;
    std::string last_client_id;
    std::string last_refresh_token;
};

/**
 * Linear/Datadog-shaped remote MCP stack:
 *  - /mcp requires Bearer and speaks Streamable HTTP (JSON or SSE)
 *  - PRM + AS metadata + DCR + token endpoint
 */
struct MockRemoteMcpStack {
    httplib::Server server;
    MockAsState state;
    int port = -1;
    std::jthread thread;
    std::unique_ptr<ScopedServerStop> stop;

    std::string base_url() const {
        return std::format("http://127.0.0.1:{}", port);
    }
    std::string mcp_url() const { return base_url() + "/mcp"; }
    std::string resource() const { return mcp_url(); }

    explicit MockRemoteMcpStack(bool sse_mode = false) {
        auto* st = &state;
        const bool use_sse = sse_mode;

        // Unauthorized probe for discovery (initialize without token).
        server.Post("/mcp", [st, use_sse, this](const httplib::Request& req, httplib::Response& res) {
            const auto auth = req.get_header_value("Authorization");
            {
                std::lock_guard lock(st->mutex);
                st->auth_headers_seen["mcp"] = auth;
            }

            if (auth.empty() || auth == "Bearer " || auth.find("Bearer ") != 0) {
                res.status = 401;
                res.set_header(
                    "WWW-Authenticate",
                    std::format(
                        "Bearer realm=\"OAuth\", resource_metadata=\"{}/.well-known/oauth-protected-resource/mcp\", error=\"invalid_token\"",
                        base_url()));
                res.set_content(
                    R"({"error":"invalid_token","error_description":"Missing or invalid access token"})",
                    "application/json");
                return;
            }

            // Authorized MCP JSON-RPC.
            const std::string& body = req.body;
            auto reply_json = [&](const std::string& payload) {
                if (use_sse) {
                    res.set_header("Content-Type", "text/event-stream");
                    res.set_content(
                        "event: message\ndata: " + payload + "\n\n",
                        "text/event-stream");
                } else {
                    res.set_content(payload, "application/json");
                }
            };

            if (body.find("\"method\":\"initialize\"") != std::string::npos
                || body.find("\"method\": \"initialize\"") != std::string::npos) {
                res.set_header("MCP-Session-Id", "sess-test-1");
                reply_json(
                    R"({"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"mock-linear","version":"1"}}})");
                return;
            }
            if (body.find("notifications/initialized") != std::string::npos) {
                res.status = 202;
                res.set_content("", "application/json");
                return;
            }
            if (body.find("tools/list") != std::string::npos) {
                reply_json(
                    R"({"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"list_issues","description":"List tickets","inputSchema":{"type":"object","properties":{"query":{"type":"string"}}}}]}})");
                return;
            }
            if (body.find("tools/call") != std::string::npos) {
                reply_json(
                    R"({"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"[{\"id\":\"ENG-1\",\"title\":\"Fix MCP auth\"}]"}],"isError":false}})");
                return;
            }
            reply_json(R"({"jsonrpc":"2.0","id":9,"result":{}})");
        });

        server.Get("/.well-known/oauth-protected-resource/mcp",
                   [this](const httplib::Request&, httplib::Response& res) {
                       res.set_content(
                           std::format(
                               R"({{"resource":"{}","authorization_servers":["{}"],"scopes_supported":["read","write"],"bearer_methods_supported":["header"]}})",
                               resource(),
                               base_url()),
                           "application/json");
                   });

        server.Get("/.well-known/oauth-protected-resource",
                   [this](const httplib::Request&, httplib::Response& res) {
                       res.set_content(
                           std::format(
                               R"({{"resource":"{}","authorization_servers":["{}"],"scopes_supported":["read","write"]}})",
                               resource(),
                               base_url()),
                           "application/json");
                   });

        server.Get("/.well-known/oauth-authorization-server",
                   [this](const httplib::Request&, httplib::Response& res) {
                       res.set_content(
                           std::format(
                               R"({{"issuer":"{}","authorization_endpoint":"{}/authorize","token_endpoint":"{}/token","registration_endpoint":"{}/register","scopes_supported":["read","write","openid","email"],"response_types_supported":["code"],"grant_types_supported":["authorization_code","refresh_token"],"token_endpoint_auth_methods_supported":["none"],"code_challenge_methods_supported":["S256"]}})",
                               base_url(),
                               base_url(),
                               base_url(),
                               base_url()),
                           "application/json");
                   });

        server.Post("/register", [st](const httplib::Request& req, httplib::Response& res) {
            {
                std::lock_guard lock(st->mutex);
                st->registration_bodies.push_back(req.body);
            }
            res.status = 201;
            res.set_content(
                R"({"client_id":"mock-client-abc","token_endpoint_auth_method":"none","grant_types":["authorization_code","refresh_token"],"response_types":["code"]})",
                "application/json");
        });

        server.Post("/token", [st](const httplib::Request& req, httplib::Response& res) {
            {
                std::lock_guard lock(st->mutex);
                st->token_bodies.push_back(req.body);
                st->token_calls += 1;
                st->last_grant_type.clear();
                st->last_code_verifier.clear();
                st->last_resource.clear();
                st->last_client_id.clear();
                st->last_refresh_token.clear();

                // Parse application/x-www-form-urlencoded body roughly.
                auto take = [&](const std::string& key) -> std::string {
                    const auto needle = key + "=";
                    const auto pos = req.body.find(needle);
                    if (pos == std::string::npos) return {};
                    auto start = pos + needle.size();
                    const auto end = req.body.find('&', start);
                    return req.body.substr(
                        start, end == std::string::npos ? std::string::npos : end - start);
                };
                st->last_grant_type = take("grant_type");
                st->last_code_verifier = take("code_verifier");
                st->last_resource = take("resource");
                st->last_client_id = take("client_id");
                st->last_refresh_token = take("refresh_token");
            }

            if (req.body.find("grant_type=refresh_token") != std::string::npos) {
                if (req.body.find("refresh_token=bad-refresh") != std::string::npos) {
                    res.status = 400;
                    res.set_content(R"({"error":"invalid_grant"})", "application/json");
                    return;
                }
                res.set_content(
                    R"({"access_token":"access-refreshed","refresh_token":"refresh-2","token_type":"Bearer","expires_in":3600,"scope":"read write"})",
                    "application/json");
                return;
            }

            // authorization_code
            if (req.body.find("code_verifier=") == std::string::npos) {
                res.status = 400;
                res.set_content(R"({"error":"invalid_request","error_description":"missing pkce"})",
                                "application/json");
                return;
            }
            if (req.body.find("resource=") == std::string::npos) {
                res.status = 400;
                res.set_content(R"({"error":"invalid_target"})", "application/json");
                return;
            }
            res.set_content(
                R"({"access_token":"access-from-code","refresh_token":"refresh-1","token_type":"Bearer","expires_in":3600,"scope":"read write"})",
                "application/json");
        });

        port = server.bind_to_any_port("127.0.0.1");
        if (port <= 0) {
            return;
        }
        thread = std::jthread([this]() { server.listen_after_bind(); });
        stop = std::make_unique<ScopedServerStop>(server);
        wait_until_running(server);
    }

    [[nodiscard]] bool ok() const { return port > 0; }
};

} // namespace

TEST_CASE("discover_mcp_oauth follows Linear-style 401 resource_metadata",
          "[mcp][oauth][discovery][integration]") {
    MockRemoteMcpStack stack;
    if (!stack.ok()) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    const auto meta = core::mcp::discover_mcp_oauth(stack.mcp_url());
    CHECK(meta.resource == stack.resource());
    CHECK(meta.authorization_server == stack.base_url());
    CHECK(meta.authorization_endpoint == stack.base_url() + "/authorize");
    CHECK(meta.token_endpoint == stack.base_url() + "/token");
    CHECK(meta.registration_endpoint == stack.base_url() + "/register");
    REQUIRE_FALSE(meta.scopes.empty());
    CHECK(std::find(meta.scopes.begin(), meta.scopes.end(), "read") != meta.scopes.end());
    CHECK(std::find(meta.scopes.begin(), meta.scopes.end(), "write") != meta.scopes.end());
}

TEST_CASE("MCP OAuth DCR + token exchange + resource/PKCE requirements",
          "[mcp][oauth][token][integration]") {
    MockRemoteMcpStack stack;
    if (!stack.ok()) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    // Dynamic client registration (as login_mcp_oauth does).
    cpr::Response reg = cpr::Post(
        cpr::Url{stack.base_url() + "/register"},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{R"({"client_name":"filo-mcp-linear","redirect_uris":["http://127.0.0.1:1/callback"],"grant_types":["authorization_code","refresh_token"],"response_types":["code"],"token_endpoint_auth_method":"none"})"},
        cpr::Timeout{5000});
    REQUIRE(reg.status_code == 201);
    REQUIRE_THAT(reg.text, ContainsSubstring("mock-client-abc"));

    core::auth::OAuthTokenEndpointRequest req;
    req.token_url = stack.base_url() + "/token";
    req.client_id = "mock-client-abc";
    req.redirect_uri = "http://127.0.0.1:1/callback";
    req.resource = stack.resource();
    req.issuer = stack.base_url();

    const auto token = core::auth::exchange_authorization_code(
        req, "auth-code-1", "pkce-verifier-value");
    CHECK(token.access_token == "access-from-code");
    CHECK(token.refresh_token == "refresh-1");
    CHECK(token.client_id == "mock-client-abc");
    CHECK(token.issuer == stack.base_url());
    CHECK(token.is_valid());

    {
        std::lock_guard lock(stack.state.mutex);
        CHECK(stack.state.last_grant_type == "authorization_code");
        CHECK(stack.state.last_code_verifier == "pkce-verifier-value");
        // resource is percent-encoded by cpr; ensure it was present.
        CHECK_FALSE(stack.state.last_resource.empty());
        CHECK(stack.state.last_client_id == "mock-client-abc");
    }

    const auto refreshed =
        core::auth::refresh_access_token(req, token.refresh_token);
    CHECK(refreshed.access_token == "access-refreshed");
    CHECK(refreshed.refresh_token == "refresh-2");
    CHECK(refreshed.is_valid());

    REQUIRE_THROWS_AS(
        core::auth::refresh_access_token(req, "bad-refresh"),
        core::auth::OAuthRefreshRejected);
}

TEST_CASE("OAuthLoopbackServer accepts real callback with code+state",
          "[mcp][oauth][loopback][integration]") {
    core::auth::OAuthLoopbackOptions opts;
    opts.fixed_port = 0;
    opts.callback_path = "/callback";
    opts.timeout = std::chrono::seconds(3);

    std::unique_ptr<core::auth::OAuthLoopbackServer> loopback;
    try {
        loopback = std::make_unique<core::auth::OAuthLoopbackServer>(std::move(opts));
    } catch (const core::auth::OAuthLoopbackBindError&) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    const auto redirect = loopback->redirect_uri();
    REQUIRE(redirect.starts_with("http://127.0.0.1:"));

    loopback->start();

    std::thread hit([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // Hit the loopback as the browser would after authorize redirect.
        (void)cpr::Get(
            cpr::Url{redirect + "?code=loop-code-42&state=state-xyz"},
            cpr::Timeout{2000});
    });

    const auto result = loopback->wait();
    hit.join();

    CHECK_FALSE(result.timed_out);
    CHECK(result.error.empty());
    CHECK(result.code == "loop-code-42");
    CHECK(result.state == "state-xyz");
}

TEST_CASE("save/load MCP OAuth meta and resolve/refresh access tokens",
          "[mcp][oauth][store][integration]") {
    MockRemoteMcpStack stack;
    if (!stack.ok()) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    TempDir tmp("filo_mcp_oauth_store");
    const std::string server_name = "linear";
    const std::string config_dir = tmp.path.string();

    core::mcp::McpOAuthMeta meta;
    meta.resource = stack.resource();
    meta.authorization_server = stack.base_url();
    meta.authorization_endpoint = stack.base_url() + "/authorize";
    meta.token_endpoint = stack.base_url() + "/token";
    meta.registration_endpoint = stack.base_url() + "/register";
    meta.client_id = "mock-client-abc";
    meta.scopes = {"read", "write"};
    core::mcp::save_mcp_oauth_meta(server_name, config_dir, meta);

    const auto loaded = core::mcp::load_mcp_oauth_meta(server_name, config_dir);
    REQUIRE(loaded.has_value());
    CHECK(loaded->client_id == "mock-client-abc");
    CHECK(loaded->token_endpoint == meta.token_endpoint);
    CHECK(loaded->resource == meta.resource);

    // Expired token with refresh -> resolve_mcp_access_token refreshes.
    core::auth::OAuthToken expired;
    expired.access_token = "old-access";
    expired.refresh_token = "refresh-1";
    expired.expires_at = 1; // far in the past
    expired.client_id = meta.client_id;
    expired.issuer = meta.authorization_server;
    core::auth::FileTokenStore(config_dir)
        .save(core::mcp::mcp_oauth_provider_id(server_name), expired);

    REQUIRE(core::mcp::has_mcp_oauth_session(server_name, config_dir));

    core::config::McpServerConfig cfg;
    cfg.name = server_name;
    cfg.transport = "http";
    cfg.url = stack.mcp_url();
    cfg.auth = "oauth";

    const auto access = core::mcp::resolve_mcp_access_token(cfg, config_dir);
    REQUIRE(access.has_value());
    CHECK(*access == "access-refreshed");

    const auto resolved = core::mcp::with_resolved_mcp_auth(cfg, config_dir);
    REQUIRE(resolved.headers.size() == 1);
    CHECK(resolved.headers.front().first == "Authorization");
    CHECK(resolved.headers.front().second == "Bearer access-refreshed");

    // Explicit Authorization header wins over OAuth store.
    cfg.headers = {{"Authorization", "Bearer static-key"}};
    const auto static_auth = core::mcp::with_resolved_mcp_auth(cfg, config_dir);
    REQUIRE(static_auth.headers.size() == 1);
    CHECK(static_auth.headers.front().second == "Bearer static-key");

    // auth=none never attaches OAuth tokens.
    core::config::McpServerConfig none_cfg = cfg;
    none_cfg.headers.clear();
    none_cfg.auth = "none";
    const auto none_auth = core::mcp::with_resolved_mcp_auth(none_cfg, config_dir);
    CHECK(none_auth.headers.empty());

    core::mcp::logout_mcp_oauth(server_name, config_dir);
    CHECK_FALSE(core::mcp::has_mcp_oauth_session(server_name, config_dir));
    CHECK_FALSE(core::mcp::load_mcp_oauth_meta(server_name, config_dir).has_value());
}

TEST_CASE("HttpMcpSession + OAuth bearer can list tools and call tools (JSON mode)",
          "[mcp][oauth][http][client][integration]") {
    MockRemoteMcpStack stack(/*sse_mode=*/false);
    if (!stack.ok()) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    TempDir tmp("filo_mcp_http_json");
    core::mcp::McpOAuthMeta meta;
    meta.resource = stack.resource();
    meta.authorization_server = stack.base_url();
    meta.token_endpoint = stack.base_url() + "/token";
    meta.client_id = "mock-client-abc";
    meta.scopes = {"read", "write"};
    core::mcp::save_mcp_oauth_meta("linear", tmp.path.string(), meta);

    core::auth::OAuthToken token;
    token.access_token = "access-from-code";
    token.refresh_token = "refresh-1";
    token.expires_at =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count()
        + 3600;
    core::auth::FileTokenStore(tmp.path.string())
        .save(core::mcp::mcp_oauth_provider_id("linear"), token);

    core::config::McpServerConfig cfg;
    cfg.name = "linear";
    cfg.transport = "http";
    cfg.url = stack.mcp_url();
    cfg.auth = "oauth";
    cfg = core::mcp::with_resolved_mcp_auth(cfg, tmp.path.string());
    REQUIRE_FALSE(cfg.headers.empty());

    core::mcp::HttpMcpSession session(cfg);
    const auto tools = session.initialize();
    REQUIRE(tools.size() == 1);
    CHECK(tools.front().name == "list_issues");

    const auto result = session.call_tool("list_issues", R"({"query":"assignee:me"})");
    REQUIRE_THAT(result, ContainsSubstring("ENG-1"));
    REQUIRE_THAT(result, ContainsSubstring("Fix MCP auth"));

    {
        std::lock_guard lock(stack.state.mutex);
        CHECK(stack.state.auth_headers_seen["mcp"] == "Bearer access-from-code");
    }
}

TEST_CASE("HttpMcpSession handles Streamable HTTP SSE tool results with bearer auth",
          "[mcp][oauth][http][sse][integration]") {
    MockRemoteMcpStack stack(/*sse_mode=*/true);
    if (!stack.ok()) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    core::config::McpServerConfig cfg;
    cfg.name = "datadog";
    cfg.transport = "http";
    cfg.url = stack.mcp_url();
    cfg.headers = {{"Authorization", "Bearer sse-token"}};

    core::mcp::HttpMcpSession session(cfg);
    const auto tools = session.initialize();
    REQUIRE(tools.size() == 1);
    CHECK(tools.front().name == "list_issues");

    const auto result = session.call_tool("list_issues", "{}");
    REQUIRE_THAT(result, ContainsSubstring("ENG-1"));

    {
        std::lock_guard lock(stack.state.mutex);
        CHECK(stack.state.auth_headers_seen["mcp"] == "Bearer sse-token");
    }
}

TEST_CASE("HttpMcpSession without auth fails with authentication guidance",
          "[mcp][oauth][http][auth][integration]") {
    MockRemoteMcpStack stack;
    if (!stack.ok()) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    core::config::McpServerConfig cfg;
    cfg.name = "linear";
    cfg.transport = "http";
    cfg.url = stack.mcp_url();
    cfg.auth = "oauth";

    core::mcp::HttpMcpSession session(cfg);
    try {
        (void)session.initialize();
        FAIL("expected authentication failure");
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        // Either raw 401 body or the specialized authentication message.
        CHECK((msg.find("401") != std::string::npos
               || msg.find("authentication") != std::string::npos
               || msg.find("invalid_token") != std::string::npos));
    }
}

TEST_CASE("env expansion works for Datadog-style API key headers",
          "[mcp][oauth][headers][env]") {
    REQUIRE(::setenv("FILO_TEST_DD_KEY", "dd-secret-key", 1) == 0);

    core::config::McpServerConfig cfg;
    cfg.name = "datadog";
    cfg.transport = "http";
    cfg.url = "https://mcp.datadoghq.com/api/unstable/mcp-server/mcp";
    cfg.auth = "none";
    cfg.headers = {
        {"Authorization", "Bearer ${FILO_TEST_DD_KEY}"},
        {"DD-API-KEY", "$FILO_TEST_DD_KEY"},
    };

    const auto resolved = core::mcp::with_resolved_mcp_auth(cfg, "/tmp");
    REQUIRE(resolved.headers.size() == 2);
    CHECK(resolved.headers[0].second == "Bearer dd-secret-key");
    // Non-Authorization headers are not expanded by with_resolved_mcp_auth today;
    // HttpMcpSession expands all header values at connect time.
    // Verify the session-side expander:
    CHECK(core::mcp::expand_mcp_env_placeholders("$FILO_TEST_DD_KEY")
          == "dd-secret-key");
    CHECK(core::mcp::expand_mcp_env_placeholders("Bearer ${FILO_TEST_DD_KEY}")
          == "Bearer dd-secret-key");

    ::unsetenv("FILO_TEST_DD_KEY");
}

TEST_CASE("refresh_mcp_oauth persists rotated tokens for subsequent connects",
          "[mcp][oauth][refresh][integration]") {
    MockRemoteMcpStack stack;
    if (!stack.ok()) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    TempDir tmp("filo_mcp_oauth_refresh");
    core::mcp::McpOAuthMeta meta;
    meta.resource = stack.resource();
    meta.authorization_server = stack.base_url();
    meta.token_endpoint = stack.base_url() + "/token";
    meta.client_id = "mock-client-abc";
    core::mcp::save_mcp_oauth_meta("linear", tmp.path.string(), meta);

    const auto refreshed =
        core::mcp::refresh_mcp_oauth("linear", tmp.path.string(), "refresh-1");
    CHECK(refreshed.access_token == "access-refreshed");
    CHECK(refreshed.refresh_token == "refresh-2");

    auto stored = core::auth::FileTokenStore(tmp.path.string())
                      .load(core::mcp::mcp_oauth_provider_id("linear"));
    REQUIRE(stored.has_value());
    CHECK(stored->access_token == "access-refreshed");
    CHECK(stored->refresh_token == "refresh-2");
}

TEST_CASE("mcp_server_wants_oauth matrix for Linear/Datadog-style configs",
          "[mcp][oauth][policy]") {
    core::config::McpServerConfig linear;
    linear.name = "linear";
    linear.transport = "http";
    linear.url = "https://mcp.linear.app/mcp";
    CHECK(core::mcp::mcp_server_wants_oauth(linear));

    linear.auth = "oauth";
    CHECK(core::mcp::mcp_server_wants_oauth(linear));

    linear.auth = "none";
    CHECK_FALSE(core::mcp::mcp_server_wants_oauth(linear));

    linear.auth.clear();
    linear.headers = {{"Authorization", "Bearer ${LINEAR_API_KEY}"}};
    CHECK_FALSE(core::mcp::mcp_server_wants_oauth(linear));

    core::config::McpServerConfig dd;
    dd.name = "datadog";
    dd.transport = "http";
    dd.url = "https://mcp.datadoghq.com/api/unstable/mcp-server/mcp";
    CHECK(core::mcp::mcp_server_wants_oauth(dd));

    core::config::McpServerConfig local_http;
    local_http.name = "local";
    local_http.transport = "http";
    local_http.url = "http://127.0.0.1:9999/mcp";
    // Non-HTTPS defaults to no auto-oauth.
    CHECK_FALSE(core::mcp::mcp_server_wants_oauth(local_http));
    local_http.auth = "oauth";
    CHECK(core::mcp::mcp_server_wants_oauth(local_http));

    core::config::McpServerConfig stdio;
    stdio.name = "bridge";
    stdio.transport = "stdio";
    stdio.command = "npx";
    CHECK_FALSE(core::mcp::mcp_server_wants_oauth(stdio));
}


TEST_CASE("mcp_request_timeout resolves defaults and overrides",
          "[mcp][oauth][timeout]") {
    core::config::McpServerConfig http;
    http.transport = "http";
    CHECK(core::mcp::mcp_request_timeout(http, core::mcp::kDefaultMcpHttpTimeout)
          == core::mcp::kDefaultMcpHttpTimeout);
    CHECK(core::mcp::mcp_request_timeout(http, core::mcp::kDefaultMcpHttpTimeout).count()
          == 120'000);

    http.request_timeout_ms = 300'000;
    CHECK(core::mcp::mcp_request_timeout(http, core::mcp::kDefaultMcpHttpTimeout).count()
          == 300'000);

    core::config::McpServerConfig stdio;
    stdio.transport = "stdio";
    CHECK(core::mcp::mcp_request_timeout(stdio, core::mcp::kDefaultMcpStdioTimeout).count()
          == 60'000);
}

TEST_CASE("HttpMcpSession refreshes OAuth token after mid-session 401",
          "[mcp][oauth][http][refresh][integration]") {
    // Resource server that rejects the first authorized call with 401, then
    // accepts the rotated bearer after force-refresh.
    httplib::Server server;
    std::mutex mu;
    int authorized_posts = 0;
    int unauthorized_posts = 0;
    std::string last_auth;
    int token_calls = 0;

    server.Post("/mcp", [&](const httplib::Request& req, httplib::Response& res) {
        const auto auth = req.get_header_value("Authorization");
        {
            std::lock_guard lock(mu);
            last_auth = auth;
        }
        if (auth != "Bearer access-fresh" && auth != "Bearer access-refreshed") {
            ++unauthorized_posts;
            res.status = 401;
            res.set_content(R"({"error":"invalid_token"})", "application/json");
            return;
        }
        // First successful-looking token ("access-fresh") is rejected once to
        // simulate expiry mid-session; after refresh only access-refreshed works.
        if (auth == "Bearer access-fresh") {
            {
                std::lock_guard lock(mu);
                ++authorized_posts;
            }
            res.status = 401;
            res.set_header("WWW-Authenticate", "Bearer error=\"invalid_token\"");
            res.set_content(R"({"error":"invalid_token"})", "application/json");
            return;
        }
        {
            std::lock_guard lock(mu);
            ++authorized_posts;
        }
        // Minimal initialize / tools/list / tools/call responses.
        if (req.body.find("\"method\":\"initialize\"") != std::string::npos) {
            res.set_content(
                R"({"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"mock","version":"0"}}})",
                "application/json");
            return;
        }
        if (req.body.find("\"method\":\"notifications/initialized\"") != std::string::npos) {
            res.status = 202;
            res.set_content("", "application/json");
            return;
        }
        if (req.body.find("\"method\":\"tools/list\"") != std::string::npos) {
            res.set_content(
                R"({"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"ping","description":"p","inputSchema":{"type":"object"}}]}})",
                "application/json");
            return;
        }
        if (req.body.find("\"method\":\"tools/call\"") != std::string::npos) {
            res.set_content(
                R"({"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"pong"}]}})",
                "application/json");
            return;
        }
        res.status = 400;
        res.set_content(R"({"error":"unexpected"})", "application/json");
    });

    server.Post("/token", [&](const httplib::Request& /*req*/, httplib::Response& res) {
        {
            std::lock_guard lock(mu);
            ++token_calls;
        }
        res.set_content(
            R"({"access_token":"access-refreshed","refresh_token":"refresh-2","token_type":"Bearer","expires_in":3600})",
            "application/json");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port < 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread thread([&] { server.listen_after_bind(); });
    for (int i = 0; i < 100 && !server.is_running(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(server.is_running());
    struct Stop {
        httplib::Server& s;
        ~Stop() { s.stop(); }
    } stop{server};

    TempDir tmp("filo_mcp_mid_session_401");
    const std::string base = std::format("http://127.0.0.1:{}", port);

    core::mcp::McpOAuthMeta meta;
    meta.resource = base + "/mcp";
    meta.authorization_server = base;
    meta.token_endpoint = base + "/token";
    meta.client_id = "mock-client";
    meta.scopes = {"read"};
    core::mcp::save_mcp_oauth_meta("linear", tmp.path.string(), meta);

    core::auth::OAuthToken token;
    token.access_token = "access-fresh";
    token.refresh_token = "refresh-1";
    token.expires_at =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count()
        + 3600; // still "valid" by clock — 401 forces refresh
    core::auth::FileTokenStore(tmp.path.string())
        .save(core::mcp::mcp_oauth_provider_id("linear"), token);

    core::config::McpServerConfig cfg;
    cfg.name = "linear";
    cfg.transport = "http";
    cfg.url = base + "/mcp";
    cfg.auth = "oauth";
    cfg.request_timeout_ms = 5'000;

    auto auth = core::mcp::make_mcp_http_auth(cfg, tmp.path.string());
    REQUIRE(auth);
    CHECK(auth->can_refresh());

    core::mcp::HttpMcpSession session(
        cfg, {}, core::mcp::mcp_request_timeout(cfg, core::mcp::kDefaultMcpHttpTimeout), auth);

    // initialize uses access-fresh → 401 → refresh → access-refreshed → success
    const auto tools = session.initialize();
    REQUIRE(tools.size() == 1);
    CHECK(tools.front().name == "ping");

    const auto result = session.call_tool("ping", "{}");
    REQUIRE_THAT(result, ContainsSubstring("pong"));

    {
        std::lock_guard lock(mu);
        CHECK(token_calls >= 1);
        CHECK(last_auth == "Bearer access-refreshed");
    }

    auto stored = core::auth::FileTokenStore(tmp.path.string())
                      .load(core::mcp::mcp_oauth_provider_id("linear"));
    REQUIRE(stored.has_value());
    CHECK(stored->access_token == "access-refreshed");
    CHECK(stored->refresh_token == "refresh-2");
}

TEST_CASE("McpHttpAuth static Authorization does not claim OAuth refresh",
          "[mcp][oauth][auth]") {
    core::config::McpServerConfig cfg;
    cfg.name = "key";
    cfg.transport = "http";
    cfg.url = "https://example.test/mcp";
    cfg.headers = {{"Authorization", "Bearer static"}};
    auto auth = core::mcp::make_mcp_http_auth(cfg, "/tmp");
    REQUIRE(auth);
    CHECK_FALSE(auth->can_refresh());
    const auto headers = auth->request_headers();
    REQUIRE_FALSE(headers.empty());
    bool found = false;
    for (const auto& [n, v] : headers) {
        if (n == "Authorization") {
            found = true;
            CHECK(v == "Bearer static");
        }
    }
    CHECK(found);
}

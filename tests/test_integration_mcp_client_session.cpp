#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <httplib.h>

#include "core/config/ConfigManager.hpp"
#include "core/mcp/McpClientSession.hpp"
#include "core/policy/ApprovalChannel.hpp"
#include "core/policy/PolicyRules.hpp"
#include "core/policy/PolicySnapshot.hpp"

#include <chrono>
#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <signal.h>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

using namespace Catch::Matchers;

namespace {

struct IgnoreSigpipeInIntegrationTests {
    IgnoreSigpipeInIntegrationTests() { ::signal(SIGPIPE, SIG_IGN); }
} g_ignore_sigpipe_in_integration_tests;

class TempScript {
public:
    TempScript(std::string_view prefix, std::string_view content) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path()
              / std::format("{}_{}_{}.py", prefix, static_cast<long long>(::getpid()), stamp);

        std::ofstream out(path_);
        out << content;
        out.close();

        std::filesystem::permissions(
            path_,
            std::filesystem::perms::owner_read
                | std::filesystem::perms::owner_write
                | std::filesystem::perms::owner_exec,
            std::filesystem::perm_options::replace);
    }

    ~TempScript() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

class ScopedServerStop {
public:
    explicit ScopedServerStop(httplib::Server& server) : server_(server) {}
    ~ScopedServerStop() { server_.stop(); }
private:
    httplib::Server& server_;
};

[[nodiscard]] core::config::McpServerConfig make_stdio_server_config(
    const std::filesystem::path& script_path) {
    core::config::McpServerConfig config;
    config.name = "mock-stdio";
    config.transport = "stdio";
    config.command = script_path.string();
    return config;
}

} // namespace

TEST_CASE("StdioMcpSession surfaces write failure after peer closes stdin",
          "[integration][mcp][client][stdio]") {
    const TempScript script(
        "filo_mcp_stdio_write_fail",
        R"PY(#!/usr/bin/env python3
import json
import os
import sys
import time

line = sys.stdin.readline()
request = json.loads(line)
print(json.dumps({
    "jsonrpc": "2.0",
    "id": request.get("id", 1),
    "result": {"protocolVersion": "2025-11-25"},
}), flush=True)
sys.stdin.close()
try:
    os.close(0)
except OSError:
    pass
time.sleep(1)
)PY");

    core::mcp::StdioMcpSession session(make_stdio_server_config(script.path()));
    REQUIRE_THROWS_WITH(session.initialize(),
                        ContainsSubstring("MCP: write() to child stdin failed"));
}

TEST_CASE("an MCP server destination is authorized before the request leaves the process",
          "[integration][mcp][policy]") {
    httplib::Server server;
    std::atomic<int> hits{0};
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    server.Post("/mcp", [&](const auto&, auto& response) {
        ++hits;
        response.set_content("{}", "application/json");
    });
    std::jthread thread([&] { server.listen_after_bind(); });
    ScopedServerStop stop(server);
    core::config::McpServerConfig config;
    config.name = "policy-test";
    config.transport = "http";
    config.url = std::format("http://127.0.0.1:{}/mcp", port);

    // An explicit deny of the server refuses the connection. The server the
    // user configured does not override a matching rule.
    core::policy::PolicyRule deny;
    deny.id = "deny-server";
    deny.effect = core::policy::Effect::deny;
    deny.principal = core::policy::PrincipalKind::mcp_server;
    deny.host = core::policy::HostPattern::parse("127.0.0.1").value();
    deny.port = core::policy::PortRange::exact(static_cast<std::uint16_t>(port));
    {
        core::policy::ScopedActivePolicyForTesting installed(
            std::make_shared<core::policy::PolicySnapshot>(
                core::policy::PolicySet({deny}, core::policy::Effect::ask),
                std::make_shared<core::policy::GrantStore>()));
        core::mcp::HttpMcpSession denied(config, {}, std::chrono::seconds(5));
        REQUIRE_THROWS_WITH(denied.initialize(), ContainsSubstring("Network policy"));
    }
    CHECK(hits == 0);

    // The configured server replaces a default of ask, so the request leaves
    // the process without anyone being asked.
    {
        core::policy::ScopedActivePolicyForTesting installed(
            std::make_shared<core::policy::PolicySnapshot>(
                core::policy::PolicySet({}, core::policy::Effect::ask),
                std::make_shared<core::policy::GrantStore>()));
        core::mcp::HttpMcpSession configured(config, {}, std::chrono::seconds(5));
        REQUIRE_THROWS(configured.initialize()); // The target returns no MCP result.
    }
    CHECK(hits == 1);

    // A policy that names it allows the request to reach the server.
    core::policy::PolicyRule allow;
    allow.id = "allow-server";
    allow.effect = core::policy::Effect::allow;
    allow.principal = core::policy::PrincipalKind::mcp_server;
    allow.principal_id = "policy-test";
    allow.host = core::policy::HostPattern::parse("127.0.0.1").value();
    allow.port = core::policy::PortRange::exact(static_cast<std::uint16_t>(port));
    core::policy::ScopedActivePolicyForTesting installed(
        std::make_shared<core::policy::PolicySnapshot>(
            core::policy::PolicySet({allow}, core::policy::Effect::deny),
            std::make_shared<core::policy::GrantStore>()));
    core::mcp::HttpMcpSession allowed(config, {}, std::chrono::seconds(5));
    REQUIRE_THROWS(allowed.initialize()); // The target returns no MCP result.
    CHECK(hits == 2);

    // The session reports the destination it would reach, so a caller can
    // name it in a policy.
    const auto endpoint = allowed.network_endpoint();
    REQUIRE(endpoint);
    CHECK(endpoint->host == "127.0.0.1");
    CHECK(endpoint->port == static_cast<std::uint16_t>(port));
}

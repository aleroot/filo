#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/mcp/McpClientSession.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <signal.h>
#include <string>
#include <string_view>
#include <sys/types.h>
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

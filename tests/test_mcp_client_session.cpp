#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/mcp/McpClientSession.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <format>
#include <string>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

using namespace Catch::Matchers;

namespace {

struct IgnoreSigpipeInTests {
    IgnoreSigpipeInTests() { ::signal(SIGPIPE, SIG_IGN); }
} g_ignore_sigpipe_in_tests;

class TempScript {
public:
    TempScript(std::string_view prefix, std::string_view content) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path()
              / std::format("{}_{}_{}.sh", prefix, static_cast<long long>(::getpid()), stamp);

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

TEST_CASE("StdioMcpSession initialize and call_tool round-trip", "[mcp][client][stdio]") {
    const TempScript script(
        "filo_mcp_stdio_roundtrip",
        R"SH(#!/bin/sh
set -eu
while IFS= read -r line; do
  [ -n "$line" ] || continue
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')
  [ -n "$id" ] || continue
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{}}}\n' "$id"
      ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"echo tool","inputSchema":{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}}]}}\n' "$id"
      ;;
    *'"method":"tools/call"'*)
      msg=$(printf '%s\n' "$line" | sed -n 's/.*"message"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
      [ -n "$msg" ] || msg=ok
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"%s"}],"isError":false}}\n' "$id" "$msg"
      ;;
    *)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      ;;
  esac
done
)SH");

    core::mcp::StdioMcpSession session(make_stdio_server_config(script.path()));
    const auto tools = session.initialize();
    REQUIRE(tools.size() == 1);
    REQUIRE(tools.front().name == "echo");

    const auto result = session.call_tool("echo", R"({"message":"hello_stdio"})");
    REQUIRE_THAT(result, ContainsSubstring(R"("result":"hello_stdio")"));
}

TEST_CASE("StdioMcpSession handles large single-line responses", "[mcp][client][stdio]") {
    const TempScript script(
        "filo_mcp_stdio_large",
        R"SH(#!/bin/sh
set -eu
while IFS= read -r line; do
  [ -n "$line" ] || continue
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')
  [ -n "$id" ] || continue
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25"}}\n' "$id"
      ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"echo tool","inputSchema":{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}}]}}\n' "$id"
      ;;
    *'"method":"tools/call"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"' "$id"
      i=0
      while [ "$i" -lt 12000 ]; do
        printf 'x'
        i=$((i + 1))
      done
      printf '"}],"isError":false}}\n'
      ;;
    *)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      ;;
  esac
done
)SH");

    core::mcp::StdioMcpSession session(make_stdio_server_config(script.path()));
    [[maybe_unused]] const auto tools = session.initialize();
    const auto result = session.call_tool("echo", R"({"message":"large"})");

    REQUIRE_THAT(result, ContainsSubstring(R"("result":")"));
    const auto x_count = static_cast<std::size_t>(std::count(result.begin(), result.end(), 'x'));
    REQUIRE(x_count >= 12000);
}

TEST_CASE("StdioMcpSession fails fast when child exits before reply", "[mcp][client][stdio]") {
    const TempScript script(
        "filo_mcp_stdio_exit_early",
        R"SH(#!/bin/sh
IFS= read -r _ || exit 0
exit 0
)SH");

    core::mcp::StdioMcpSession session(make_stdio_server_config(script.path()));
    REQUIRE_THROWS_WITH(session.initialize(),
                        ContainsSubstring("MCP: read_fd closed unexpectedly"));
}

TEST_CASE("StdioMcpSession surfaces write failure after peer closes stdin",
          "[mcp][client][stdio]") {
    const TempScript script(
        "filo_mcp_stdio_write_fail",
        R"SH(#!/bin/sh
set -eu
IFS= read -r line || exit 0
id=$(printf '%s\n' "$line" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')
[ -n "$id" ] || id=1
printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25"}}\n' "$id"
exec 0<&-
sleep 1
)SH");

    core::mcp::StdioMcpSession session(make_stdio_server_config(script.path()));
    REQUIRE_THROWS_WITH(session.initialize(),
                        ContainsSubstring("MCP: write() to child stdin failed"));
}

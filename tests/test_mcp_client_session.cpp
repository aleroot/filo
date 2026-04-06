#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/mcp/McpClientSession.hpp"
#include "core/mcp/McpSamplingBridge.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <format>
#include <memory>
#include <string>
#include <vector>
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

class MockSamplingProvider final : public core::llm::LLMProvider {
public:
    using StreamHandler = std::function<void(
        const core::llm::ChatRequest&,
        const std::function<void(const core::llm::StreamChunk&)>&)>;

    explicit MockSamplingProvider(StreamHandler handler, std::string last_model = "mock-model")
        : handler_(std::move(handler))
        , last_model_(std::move(last_model)) {}

    void stream_response(const core::llm::ChatRequest& request,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        handler_(request, callback);
    }

    [[nodiscard]] std::string get_last_model() const override {
        return last_model_;
    }

private:
    StreamHandler handler_;
    std::string last_model_;
};

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

TEST_CASE("StdioMcpSession supports resources and prompts", "[mcp][client][stdio]") {
    const TempScript script(
        "filo_mcp_stdio_features",
        R"SH(#!/bin/sh
set -eu
while IFS= read -r line; do
  [ -n "$line" ] || continue
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')
  [ -n "$id" ] || continue
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{},"resources":{},"prompts":{}}}}\n' "$id"
      ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[]}}\n' "$id"
      ;;
    *'"method":"resources/list"'*)
      case "$line" in
        *'"cursor":"page2"'*)
          printf '{"jsonrpc":"2.0","id":%s,"result":{"resources":[{"uri":"file:///second.txt","name":"second.txt"}]}}\n' "$id"
          ;;
        *)
          printf '{"jsonrpc":"2.0","id":%s,"result":{"resources":[{"uri":"file:///first.txt","name":"first.txt","mimeType":"text/plain"}],"nextCursor":"page2"}}\n' "$id"
          ;;
      esac
      ;;
    *'"method":"resources/templates/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resourceTemplates":[{"uriTemplate":"file:///{path}","name":"Project Files","mimeType":"application/octet-stream"}]}}\n' "$id"
      ;;
    *'"method":"resources/read"'*)
      uri=$(printf '%s\n' "$line" | sed -n 's/.*"uri"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
      [ -n "$uri" ] || uri="unknown://uri"
      printf '{"jsonrpc":"2.0","id":%s,"result":{"contents":[{"uri":"%s","mimeType":"text/plain","text":"content for %s"}]}}\n' "$id" "$uri" "$uri"
      ;;
    *'"method":"prompts/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"prompts":[{"name":"review","description":"Review code","arguments":[{"name":"code","required":true}]}]}}\n' "$id"
      ;;
    *'"method":"prompts/get"'*)
      name=$(printf '%s\n' "$line" | sed -n 's/.*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
      [ -n "$name" ] || name="unknown"
      printf '{"jsonrpc":"2.0","id":%s,"result":{"messages":[{"role":"user","content":{"type":"text","text":"Prompt:%s"}}]}}\n' "$id" "$name"
      ;;
    *)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      ;;
  esac
done
)SH");

    core::mcp::StdioMcpSession session(make_stdio_server_config(script.path()));
    const auto tools = session.initialize();
    REQUIRE(tools.empty());
    REQUIRE(session.supports_resources());
    REQUIRE(session.supports_prompts());

    const auto resources = session.list_resources();
    REQUIRE(resources.size() == 2);
    REQUIRE(resources[0].uri == "file:///first.txt");
    REQUIRE(resources[1].uri == "file:///second.txt");

    const auto templates = session.list_resource_templates();
    REQUIRE(templates.size() == 1);
    REQUIRE(templates[0].uri_template == "file:///{path}");

    const auto resource_read = session.read_resource("file:///first.txt");
    REQUIRE_THAT(resource_read, ContainsSubstring("content for file:///first.txt"));

    const auto prompts = session.list_prompts();
    REQUIRE(prompts.size() == 1);
    REQUIRE(prompts[0].name == "review");
    REQUIRE(prompts[0].arguments.size() == 1);
    REQUIRE(prompts[0].arguments[0].name == "code");
    REQUIRE(prompts[0].arguments[0].required);

    const auto prompt = session.get_prompt("review", R"({"code":"int x = 1;"})");
    REQUIRE_THAT(prompt, ContainsSubstring("Prompt:review"));
}

TEST_CASE("StdioMcpSession handles server sampling/createMessage requests",
          "[mcp][client][stdio][sampling]") {
    const TempScript script(
        "filo_mcp_stdio_sampling",
        R"SH(#!/bin/sh
set -eu
while IFS= read -r line; do
  [ -n "$line" ] || continue
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')
  [ -n "$id" ] || continue
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{}}}}\n' "$id"
      ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"needs_sampling","description":"Tool that triggers sampling","inputSchema":{"type":"object","properties":{}}}]}}\n' "$id"
      ;;
    *'"method":"tools/call"'*)
      call_id="$id"
      printf '{"jsonrpc":"2.0","id":777,"method":"sampling/createMessage","params":{"messages":[{"role":"user","content":{"type":"text","text":"hello"}}],"maxTokens":16}}\n'
      IFS= read -r sampling_response || exit 1
      sampled=$(printf '%s\n' "$sampling_response" | sed -n 's/.*"text"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
      [ -n "$sampled" ] || sampled="missing"
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"sampled=%s"}],"isError":false}}\n' "$call_id" "$sampled"
      ;;
    *)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      ;;
  esac
done
)SH");

    bool sampling_called = false;
    std::string sampling_server;
    std::string sampling_params;
    core::mcp::StdioMcpSession session(
        make_stdio_server_config(script.path()),
        [&](std::string_view server_name, std::string_view params_json) {
            sampling_called = true;
            sampling_server = std::string(server_name);
            sampling_params = std::string(params_json);
            return R"({"role":"assistant","content":{"type":"text","text":"from_sampling"},"model":"mock-model","stopReason":"endTurn"})";
        });

    [[maybe_unused]] const auto tools = session.initialize();
    const auto result = session.call_tool("needs_sampling", "{}");

    REQUIRE(sampling_called);
    REQUIRE(sampling_server == "mock-stdio");
    REQUIRE_THAT(sampling_params, ContainsSubstring(R"("messages")"));
    REQUIRE_THAT(result, ContainsSubstring(R"("result":"sampled=from_sampling")"));
}

TEST_CASE("StdioMcpSession probes tools/list when tools capability is omitted",
          "[mcp][client][stdio][capabilities]") {
    const TempScript script(
        "filo_mcp_stdio_tools_probe",
        R"SH(#!/bin/sh
set -eu
while IFS= read -r line; do
  [ -n "$line" ] || continue
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')
  [ -n "$id" ] || continue
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{"resources":{}}}}\n' "$id"
      ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Echo","inputSchema":{"type":"object","properties":{}}}]}}\n' "$id"
      ;;
    *'"method":"tools/call"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"probe_ok"}],"isError":false}}\n' "$id"
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
    REQUIRE(tools[0].name == "echo");
    REQUIRE(session.supports_resources());

    const auto result = session.call_tool("echo", "{}");
    REQUIRE_THAT(result, ContainsSubstring(R"("result":"probe_ok")"));
}

TEST_CASE("StdioMcpSession keeps resources when tools/list is unavailable",
          "[mcp][client][stdio][capabilities]") {
    const TempScript script(
        "filo_mcp_stdio_no_tools",
        R"SH(#!/bin/sh
set -eu
while IFS= read -r line; do
  [ -n "$line" ] || continue
  id=$(printf '%s\n' "$line" | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')
  [ -n "$id" ] || continue
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-11-25","capabilities":{"resources":{}}}}\n' "$id"
      ;;
    *'"method":"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"Method not found"}}\n' "$id"
      ;;
    *'"method":"resources/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"resources":[{"uri":"file:///doc.txt","name":"doc.txt"}]}}\n' "$id"
      ;;
    *)
      printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id"
      ;;
  esac
done
)SH");

    core::mcp::StdioMcpSession session(make_stdio_server_config(script.path()));
    const auto tools = session.initialize();
    REQUIRE(tools.empty());
    REQUIRE(session.supports_resources());

    const auto resources = session.list_resources();
    REQUIRE(resources.size() == 1);
    REQUIRE(resources[0].uri == "file:///doc.txt");

    REQUIRE_THROWS_WITH(session.call_tool("echo", "{}"),
                        ContainsSubstring("does not advertise tools capability"));
}

TEST_CASE("McpSamplingBridge keeps text alongside tool_use blocks",
          "[mcp][sampling][bridge]") {
    auto provider = std::make_shared<MockSamplingProvider>(
        [](const core::llm::ChatRequest&,
           const std::function<void(const core::llm::StreamChunk&)>& callback) {
            callback(core::llm::StreamChunk::make_content("Need to check one thing."));

            core::llm::ToolCall tool_call;
            tool_call.id = "call_1";
            tool_call.function.name = "lookup";
            tool_call.function.arguments = R"({"query":"status"})";
            callback(core::llm::StreamChunk::make_tools({tool_call}));
            callback(core::llm::StreamChunk::make_final());
        },
        "bridge-model");

    core::mcp::McpSamplingBridge bridge(provider, "bridge-model");
    const auto result = bridge.create_message(
        "mock-server",
        R"({"messages":[{"role":"user","content":{"type":"text","text":"hello"}}]})");

    REQUIRE_THAT(result, ContainsSubstring(R"("type":"text")"));
    REQUIRE_THAT(result, ContainsSubstring(R"("Need to check one thing.")"));
    REQUIRE_THAT(result, ContainsSubstring(R"("type":"tool_use")"));
    REQUIRE_THAT(result, ContainsSubstring(R"("stopReason":"toolUse")"));
}

TEST_CASE("McpSamplingBridge can switch provider/model backend",
          "[mcp][sampling][bridge]") {
    auto provider_a = std::make_shared<MockSamplingProvider>(
        [](const core::llm::ChatRequest& request,
           const std::function<void(const core::llm::StreamChunk&)>& callback) {
            REQUIRE(request.model == "model-a");
            callback(core::llm::StreamChunk::make_content("from_a"));
            callback(core::llm::StreamChunk::make_final());
        },
        "model-a");

    auto provider_b = std::make_shared<MockSamplingProvider>(
        [](const core::llm::ChatRequest& request,
           const std::function<void(const core::llm::StreamChunk&)>& callback) {
            REQUIRE(request.model == "model-b");
            callback(core::llm::StreamChunk::make_content("from_b"));
            callback(core::llm::StreamChunk::make_final());
        },
        "model-b");

    core::mcp::McpSamplingBridge bridge(provider_a, "model-a");

    const auto first = bridge.create_message(
        "mock-server",
        R"({"messages":[{"role":"user","content":{"type":"text","text":"hello"}}]})");
    REQUIRE_THAT(first, ContainsSubstring(R"("text":"from_a")"));
    REQUIRE_THAT(first, ContainsSubstring(R"("model":"model-a")"));

    bridge.set_backend(provider_b, "model-b");
    const auto second = bridge.create_message(
        "mock-server",
        R"({"messages":[{"role":"user","content":{"type":"text","text":"hello"}}]})");
    REQUIRE_THAT(second, ContainsSubstring(R"("text":"from_b")"));
    REQUIRE_THAT(second, ContainsSubstring(R"("model":"model-b")"));
}

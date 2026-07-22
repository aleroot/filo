#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/context/SessionContext.hpp"
#include "core/tools/ShellTool.hpp"
#include "TestSessionContext.hpp"

#include <simdjson.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <future>
#include <string>
#include <thread>
#include <unistd.h>

using namespace core::tools;

namespace {

[[nodiscard]] core::context::SessionContext make_tool_test_context(std::string session_id = {}) {
    core::config::ConfigManager::get_instance().load(std::filesystem::current_path());
    return test_support::make_workspace_session_context(
        core::context::SessionTransport::cli,
        std::move(session_id));
}

} // namespace

#define execute(...) execute(__VA_ARGS__, make_tool_test_context())

TEST_CASE("ShellTool executes a simple command", "[integration][tools][shell]") {
    ShellTool tool;
    auto res = tool.execute(R"({"command":"echo hello"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("hello"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool captures non-zero exit code", "[integration][tools][shell]") {
    ShellTool tool;
    auto res = tool.execute(R"({"command":"exit 42"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":42"));
}

TEST_CASE("ShellTool rejects nonexistent working_dir", "[integration][tools][shell]") {
    ShellTool tool;
    auto res = tool.execute(R"({"command":"echo hi","working_dir":"/nonexistent_filo_test_dir_xyz"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

TEST_CASE("ShellTool returns error for missing command field", "[integration][tools][shell]") {
    ShellTool tool;
    auto res = tool.execute(R"({})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

TEST_CASE("ShellTool session persists working directory across calls",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto res1 = tool.execute(R"({"command":"cd /tmp"})");
    REQUIRE_THAT(res1, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

    auto res2 = tool.execute(R"({"command":"pwd"})");
    REQUIRE_THAT(res2, Catch::Matchers::ContainsSubstring("/tmp"));
    REQUIRE_THAT(res2, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool session persists exported environment variable",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto res1 = tool.execute(R"({"command":"export FILO_TEST_VAR=hello_world_42"})");
    REQUIRE_THAT(res1, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

    auto res2 = tool.execute(R"({"command":"echo $FILO_TEST_VAR"})");
    REQUIRE_THAT(res2, Catch::Matchers::ContainsSubstring("hello_world_42"));
}

TEST_CASE("ShellTool isolates persistent state across MCP session contexts",
          "[integration][tools][shell]") {
    ShellTool tool;
    const auto session_a = make_tool_test_context("session-a");
    const auto session_b = make_tool_test_context("session-b");

#undef execute
    {
        auto res = tool.execute(R"({"command":"cd /tmp"})", session_a);
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    }
    {
        auto res = tool.execute(R"({"command":"cd /var"})", session_b);
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    }
    {
        auto res = tool.execute(R"({"command":"pwd"})", session_a);
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("/tmp"));
    }
    {
        auto res = tool.execute(R"({"command":"pwd"})", session_b);
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("/var"));
    }
#define execute(...) execute(__VA_ARGS__, make_tool_test_context())
}

TEST_CASE("ShellTool clear_mcp_session resets that session shell state",
          "[integration][tools][shell]") {
    ShellTool tool;
    constexpr std::string_view kSession = "session-clear";
    constexpr std::string_view kToken = "filo_session_token_42";
    const auto session = make_tool_test_context(std::string(kSession));

#undef execute
    {
        auto set_res = tool.execute(
            R"({"command":"export FILO_SESSION_CLEAR_VAR=filo_session_token_42"})",
            session);
        REQUIRE_THAT(set_res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

        auto echo_res = tool.execute(R"({"command":"echo $FILO_SESSION_CLEAR_VAR"})", session);
        REQUIRE_THAT(echo_res, Catch::Matchers::ContainsSubstring(std::string(kToken)));
    }

    ShellTool::clear_mcp_session(kSession);

    {
        auto echo_res = tool.execute(R"({"command":"echo $FILO_SESSION_CLEAR_VAR"})", session);
        REQUIRE_THAT(echo_res, !Catch::Matchers::ContainsSubstring(std::string(kToken)));
    }
#define execute(...) execute(__VA_ARGS__, make_tool_test_context())
}

TEST_CASE("ShellTool handles MCP session LRU eviction without crashing",
          "[integration][tools][shell]") {
    ShellTool tool;

#undef execute
    for (int i = 0; i < 70; ++i) {
        const auto session = make_tool_test_context("session-evict-" + std::to_string(i));
        auto res = tool.execute(R"({"command":"echo alive"})", session);
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("alive"));
    }
#define execute(...) execute(__VA_ARGS__, make_tool_test_context())
}

TEST_CASE("ShellTool working_dir runs in subshell and does not affect session cwd",
          "[integration][tools][shell]") {
    ShellTool tool;

    [[maybe_unused]] const auto initial_cd = tool.execute(R"({"command":"cd /tmp"})");

    auto res1 = tool.execute(R"({"command":"pwd","working_dir":"/var"})");
    REQUIRE_THAT(res1, Catch::Matchers::ContainsSubstring("/var"));

    auto res2 = tool.execute(R"({"command":"pwd"})");
    REQUIRE_THAT(res2, Catch::Matchers::ContainsSubstring("/tmp"));
}

TEST_CASE("ShellTool completes a long-running command within timeout",
          "[integration][tools][shell]") {
    ShellTool tool;
    auto res = tool.execute(R"({"command":"sleep 2","timeout_seconds":10})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("TIMEOUT"));
}

TEST_CASE("ShellTool returns timeout message when command exceeds limit",
          "[integration][tools][shell]") {
    ShellTool tool;
    auto res = tool.execute(R"({"command":"sleep 60","timeout_seconds":1})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("TIMEOUT"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":-1"));
}

TEST_CASE("ShellTool session recovers cleanly after a timeout",
          "[integration][tools][shell]") {
    ShellTool tool;
    [[maybe_unused]] const auto timeout_res =
        tool.execute(R"({"command":"sleep 60","timeout_seconds":1})");
    auto res = tool.execute(R"({"command":"echo recovered"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("recovered"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool interrupt_mcp_session cancels a running command",
          "[integration][tools][shell]") {
    ShellTool tool;
    constexpr std::string_view kSession = "session-interrupt";
    const auto session = make_tool_test_context(std::string(kSession));

#undef execute
    auto future = std::async(std::launch::async, [&]() {
        return tool.execute(R"({"command":"sleep 60","timeout_seconds":30})", session);
    });

    bool observed_active = false;
    for (int i = 0; i < 100; ++i) {
        const auto active = ShellTool::active_commands();
        observed_active = std::ranges::any_of(active, [kSession](const ShellTool::ActiveCommand& command) {
            return command.session_id == kSession && command.command == "sleep 60";
        });
        if (observed_active) break;
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
    }
    REQUIRE(observed_active);
    REQUIRE(ShellTool::interrupt_mcp_session(kSession));

    const auto res = future.get();
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("INTERRUPTED"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":-1"));

    auto next = tool.execute(R"({"command":"echo recovered_after_interrupt"})", session);
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("recovered_after_interrupt"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    ShellTool::clear_mcp_session(kSession);
#define execute(...) execute(__VA_ARGS__, make_tool_test_context())
}

TEST_CASE("ShellTool preserves partial-line stdout and stderr",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto res = tool.execute(
        R"({"command":"printf partial_stdout; printf partial_stderr >&2"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("partial_stdout"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("partial_stderr"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool repairs malformed UTF-8 emitted by binary commands",
          "[integration][tools][shell]") {
    ShellTool tool;

    const auto res = tool.execute(
        R"({"command":"printf '\\355\\240\\200\\377'"})");

    simdjson::dom::parser parser;
    simdjson::dom::element document;
    REQUIRE(parser.parse(res).get(document) == simdjson::SUCCESS);
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring(R"(\ufffd)"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool returns when execed command exits but child keeps stdout open",
          "[integration][tools][shell]") {
    ShellTool tool;

    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(
        R"({"command":"exec sh -c '(sleep 5) & exit 23'","timeout_seconds":2})");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":23"));
    REQUIRE(elapsed < std::chrono::milliseconds{1500});

    auto next = tool.execute(R"({"command":"echo recovered_after_exec"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("recovered_after_exec"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool captures output and exit code when command replaces shell",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto res = tool.execute(
        R"({"command":"exec sh -c 'printf exec_stdout; printf exec_stderr >&2; exit 17'","timeout_seconds":2})");

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("exec_stdout"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("exec_stderr"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":17"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("TIMEOUT"));

    auto next = tool.execute(R"({"command":"echo recovered_after_exec_output"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("recovered_after_exec_output"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool reports signal exit for command that replaces shell",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto res = tool.execute(
        R"({"command":"exec sh -c 'kill -TERM $$'","timeout_seconds":2})");

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":143"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("TIMEOUT"));

    auto next = tool.execute(R"({"command":"echo recovered_after_signal"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("recovered_after_signal"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool resets after exit with child holding stdout open",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto res = tool.execute(
        R"({"command":"(sleep 5) & exit 7","timeout_seconds":2})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":7"));

    auto next = tool.execute(R"({"command":"echo recovered_after_exit"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("recovered_after_exit"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool kills child processes spawned by timed-out command",
          "[integration][tools][shell]") {
    ShellTool tool;
    const std::string marker = "/tmp/filo_shell_test_marker_" + std::to_string(::getpid());
    std::filesystem::remove(marker);

    const std::string cmd =
        "(sleep 2 && touch " + marker + ") & sleep 60";
    [[maybe_unused]] const auto timeout_res =
        tool.execute("{\"command\":\"" + cmd + "\",\"timeout_seconds\":1}");

    ::sleep(3);

    REQUIRE_FALSE(std::filesystem::exists(marker));
    std::filesystem::remove(marker);
}

TEST_CASE("ShellTool working_dir subshell preserves exit status and session cwd",
          "[integration][tools][shell]") {
    ShellTool tool;

    [[maybe_unused]] const auto initial_cd = tool.execute(R"({"command":"cd /tmp"})");

    auto res = tool.execute(
        R"({"command":"pwd; exit 12","working_dir":"/var"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("/var"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":12"));

    auto next = tool.execute(R"({"command":"pwd"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("/tmp"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool timed-out command does not leak child output into next command",
          "[integration][tools][shell]") {
    ShellTool tool;

    [[maybe_unused]] const auto timeout_res = tool.execute(
        R"({"command":"(sleep 2; echo should_not_leak_after_timeout) & sleep 60","timeout_seconds":1})");

    ::sleep(3);

    auto res = tool.execute(R"({"command":"echo clean_after_timeout"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("clean_after_timeout"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("should_not_leak_after_timeout"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool recovers after output truncation drains to sentinel",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto res = tool.execute(
        R"({"command":"awk 'BEGIN{for (i=0;i<4300000;i++) printf \"x\"}'","timeout_seconds":10})");
    REQUIRE(res.find("OUTPUT TRUNCATED AT 4MB") != std::string::npos);
    REQUIRE(res.find("\"exit_code\":-1") != std::string::npos);

    auto next = tool.execute(R"({"command":"echo recovered_after_truncation"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("recovered_after_truncation"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool timeout applies while draining truncated output",
          "[integration][tools][shell]") {
    ShellTool tool;

    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(
        R"({"command":"awk 'BEGIN{for (i=0;i<5000000;i++) printf \"x\"; fflush(); while (1) system(\"sleep 1\")}'","timeout_seconds":3})");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    REQUIRE(res.find("OUTPUT TRUNCATED AT 4MB") != std::string::npos);
    REQUIRE(res.find("TIMEOUT") != std::string::npos);
    REQUIRE(res.find("\"exit_code\":-1") != std::string::npos);
    REQUIRE(elapsed < std::chrono::seconds{8});

    auto next = tool.execute(R"({"command":"echo clean_after_truncated_timeout"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("clean_after_truncated_timeout"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool writes large command payloads without truncation",
          "[integration][tools][shell]") {
    ShellTool tool;

    const std::string payload(150000, 'x');
    const std::string command = "printf '" + payload + "' | wc -c";
    const std::string args = "{\"command\":\"" + command + "\"}";

    auto res = tool.execute(args);
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("150000"));
}

TEST_CASE("ShellTool session restarts transparently after exit",
          "[integration][tools][shell]") {
    ShellTool tool;

    [[maybe_unused]] const auto export_res =
        tool.execute(R"({"command":"export FILO_PERSIST=yes"})");
    auto exit_res = tool.execute(R"({"command":"exit 7"})");
    REQUIRE_THAT(exit_res, Catch::Matchers::ContainsSubstring("\"exit_code\":7"));

    auto res = tool.execute(R"({"command":"echo alive"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("alive"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

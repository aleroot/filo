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

TEST_CASE("ShellTool commands that read stdin get EOF instead of hanging",
          "[integration][tools][shell]") {
    ShellTool tool;

    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(
        R"({"command":"read line; echo read_rc=$?","timeout_seconds":10})");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("read_rc=1"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE(elapsed < std::chrono::seconds{8});
}

TEST_CASE("ShellTool stdin-reading command cannot swallow the completion sentinel",
          "[integration][tools][shell]") {
    ShellTool tool;

    // Before stdin isolation, a command reading stdin (ssh, cat, read, ...)
    // consumed the injected sentinel line from the shared control pipe and the
    // call hung until the timeout even though the command itself was instant.
    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(R"({"command":"cat","timeout_seconds":10})");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE(elapsed < std::chrono::seconds{8});

    // The session is still synchronised: the next command runs cleanly.
    auto next = tool.execute(R"({"command":"echo still_in_sync"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("still_in_sync"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool brace wrapping preserves session state and exit codes",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto cd_res = tool.execute(R"({"command":"cd /tmp"})");
    REQUIRE_THAT(cd_res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    auto pwd_res = tool.execute(R"({"command":"pwd"})");
    REQUIRE_THAT(pwd_res, Catch::Matchers::ContainsSubstring("/tmp"));

    // Commands ending with '&' still parse inside the brace group wrapper.
    auto bg_res = tool.execute(R"({"command":"sleep 0.1 & wait; echo bg_done"})");
    REQUIRE_THAT(bg_res, Catch::Matchers::ContainsSubstring("bg_done"));
    REQUIRE_THAT(bg_res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

    // Whitespace-only commands must not produce a syntax error.
    auto empty_res = tool.execute(R"({"command":"   "})");
    REQUIRE_THAT(empty_res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

// ---------------------------------------------------------------------------
// stdin isolation: commands reading stdin must get EOF, never the control pipe
// ---------------------------------------------------------------------------

TEST_CASE("ShellTool pipelines reading stdin see EOF with no leaked control bytes",
          "[integration][tools][shell]") {
    ShellTool tool;

    // If the command could read bash's control pipe, wc would count the bytes
    // of the injected sentinel line instead of reporting zero.
    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(
        R"json({"command":"echo bytes=$(cat | wc -c | tr -d ' ')","timeout_seconds":10})json");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("bytes=0"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE(elapsed < std::chrono::seconds{8});
}

TEST_CASE("ShellTool while-read loop exits immediately on stdin EOF",
          "[integration][tools][shell]") {
    ShellTool tool;

    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(
        R"({"command":"while read line; do echo got:$line; done; echo loop_done","timeout_seconds":10})");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("loop_done"));
    // EOF arrives before any line, so the loop body never runs.
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("got:"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE(elapsed < std::chrono::seconds{8});
}

TEST_CASE("ShellTool partial stdin reader does not desynchronise the session",
          "[integration][tools][shell]") {
    ShellTool tool;

    // head reads a bounded amount of stdin then exits; without stdin
    // isolation it would consume part of the control stream and corrupt the
    // command/sentinel protocol for every subsequent call.
    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(
        R"({"command":"head -c 64; echo after_partial_read","timeout_seconds":10})");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("after_partial_read"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE(elapsed < std::chrono::seconds{8});

    auto next = tool.execute(R"({"command":"echo sync_after_head"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("sync_after_head"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool backgrounded stdin reader terminates on EOF",
          "[integration][tools][shell]") {
    ShellTool tool;

    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(
        R"({"command":"cat & wait; echo bg_reader_done","timeout_seconds":10})");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("bg_reader_done"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE(elapsed < std::chrono::seconds{8});
}

// ---------------------------------------------------------------------------
// No usable terminal: prompts must fail fast, never block.
// The session shell runs in its own session (setsid) with no controlling
// terminal, and command stdin is /dev/null — so any interactive prompt,
// from ANY program, errors out immediately instead of hanging.
// ---------------------------------------------------------------------------

TEST_CASE("ShellTool command streams are never terminals",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto res = tool.execute(
        R"({"command":"test -t 0 || echo stdin_not_tty; test -t 1 || echo stdout_not_tty; test -t 2 || echo stderr_not_tty; tty || true"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("stdin_not_tty"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("stdout_not_tty"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("stderr_not_tty"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("not a tty"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool commands cannot open the controlling terminal",
          "[integration][tools][shell]") {
    ShellTool tool;

    // General proof, not tied to any specific tool: with no controlling
    // terminal, ANY program that opens /dev/tty directly to prompt (ssh
    // passphrases, credential helpers, sudo, pinentry, ...) fails fast with
    // ENXIO instead of blocking forever on input nobody can supply.
    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(
        R"({"command":"read -r x < /dev/tty; echo tty_read_rc=$?; head -c 1 /dev/tty; echo tty_head_rc=$?","timeout_seconds":10})");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("tty_read_rc=1"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("tty_head_rc=1"));
    REQUIRE(elapsed < std::chrono::seconds{8});

    // The session itself is unaffected by the failed prompts.
    auto next = tool.execute(R"({"command":"echo alive_after_tty_probes"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("alive_after_tty_probes"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool git credential lookup fails fast instead of prompting",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto git_check = tool.execute(R"({"command":"git --version"})");
    REQUIRE_THAT(git_check, Catch::Matchers::ContainsSubstring("git version"));

    // The original hang scenario: git needs credentials for an unknown host
    // and tries to prompt on the terminal.  With no controlling terminal the
    // prompt dies immediately (exit 128) instead of blocking forever on an
    // invisible username/password prompt.
    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(
        R"({"command":"echo url=https://example.invalid | git credential fill","timeout_seconds":10})");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    // git dies with "could not read Username ... : Device not configured"
    // (macOS) or "...: No such device or address" (Linux) — assert the
    // platform-independent prefix plus the exit code.
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("could not read Username"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":128"));
    REQUIRE(elapsed < std::chrono::seconds{8});
}

// ---------------------------------------------------------------------------
// Brace-group wrapper: parsing edge cases
// ---------------------------------------------------------------------------

TEST_CASE("ShellTool heredoc content survives the command wrapper",
          "[integration][tools][shell]") {
    ShellTool tool;

    // Multi-line command: the heredoc body is read from bash's script input,
    // not from the /dev/null redirect applied to the brace group.
    auto res = tool.execute(
        R"({"command":"cat <<EOF\nhello_heredoc\nEOF"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("hello_heredoc"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

    auto next = tool.execute(R"({"command":"echo sync_after_heredoc"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("sync_after_heredoc"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool working_dir wrapper preserves a trailing heredoc delimiter",
          "[integration][tools][shell]") {
    ShellTool tool;

    SECTION("unquoted delimiter") {
        // The heredoc deliberately ends the command.  The working-directory
        // subshell must put its closing ')' on a later line or bash sees
        // `EOF)` instead of the required standalone delimiter and waits until
        // timeout.  Expansion also proves the command ran in working_dir.
        auto res = tool.execute(
            R"json({"command":"cat <<EOF\ncwd=$PWD\nEOF","working_dir":"/var","timeout_seconds":2})json");

        // macOS canonicalizes /var to /private/var; Linux keeps /var.
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("cwd="));
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("/var"));
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
        REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("TIMEOUT"));
    }

    SECTION("quoted delimiter") {
        // Quoted delimiters are the common form for embedded Python scripts
        // because they prevent the shell from expanding the script body.
        auto res = tool.execute(
            R"json({"command":"cat <<'PY'\nquoted_heredoc_body\nPY","working_dir":"/var","timeout_seconds":2})json");

        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("quoted_heredoc_body"));
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
        REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("TIMEOUT"));
    }

    // Completion detection must remain aligned after the multiline command.
    auto next = tool.execute(R"({"command":"echo sync_after_working_dir_heredoc"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("sync_after_working_dir_heredoc"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool trailing comment does not break the command wrapper",
          "[integration][tools][shell]") {
    ShellTool tool;

    // The comment must not be able to swallow the closing brace of the
    // wrapper (the brace is emitted on its own line for this reason).
    auto res = tool.execute(
        R"({"command":"echo trailing_comment_ok # done"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("trailing_comment_ok"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

    // The working-directory subshell also has closing syntax.  It must be on
    // the next line so this comment cannot swallow the closing parenthesis.
    auto working_dir_res = tool.execute(
        R"({"command":"echo working_dir_comment_ok # done","working_dir":"/var","timeout_seconds":2})");
    REQUIRE_THAT(
        working_dir_res,
        Catch::Matchers::ContainsSubstring("working_dir_comment_ok"));
    REQUIRE_THAT(
        working_dir_res,
        Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE_THAT(
        working_dir_res,
        !Catch::Matchers::ContainsSubstring("TIMEOUT"));
}

TEST_CASE("ShellTool comment-only command is a harmless no-op",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto setup = tool.execute(R"({"command":"export FILO_STILL_HERE=yes"})");
    REQUIRE_THAT(setup, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

    // A comment-only command wrapped naively would leave an empty brace
    // group — a syntax error that kills the session shell.  It must instead
    // run as a no-op and keep the session (and its state) alive.
    auto res = tool.execute(R"({"command":"# nothing to do"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("syntax error"));

    auto next = tool.execute(R"({"command":"echo kept=$FILO_STILL_HERE"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("kept=yes"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool parse error fails fast and the session recovers",
          "[integration][tools][shell]") {
    ShellTool tool;

    // A syntax error detectable from the buffered line (stray `)`) makes
    // bash exit immediately; the EXIT trap still emits the sentinel, so the
    // call returns promptly with the error instead of hanging.
    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(R"json({"command":"echo )","timeout_seconds":10})json");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("syntax error"));
    // Exit code differs across bash versions (2 vs 258) — just not success.
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("TIMEOUT"));
    REQUIRE(elapsed < std::chrono::seconds{8});

    // The dead shell is transparently replaced on the next call.
    auto next = tool.execute(R"({"command":"echo recovered_after_syntax_error"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("recovered_after_syntax_error"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool incomplete construct times out and the session recovers",
          "[integration][tools][shell]") {
    ShellTool tool;

    // An unterminated quote is different from a parse error: bash keeps
    // reading the control pipe waiting for the closing quote, so no sentinel
    // can arrive.  The timeout must fire, the stuck shell must be killed and
    // the next command must get a fresh, working session.
    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(
        R"({"command":"echo 'unterminated","timeout_seconds":2})");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("TIMEOUT"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":-1"));
    REQUIRE(elapsed < std::chrono::seconds{8});

    auto next = tool.execute(R"({"command":"echo recovered_after_incomplete"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("recovered_after_incomplete"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool command output may contain braces and sentinel-like text",
          "[integration][tools][shell]") {
    ShellTool tool;

    // JSON-looking output (braces) and text resembling the completion
    // sentinel must pass through untouched: the parser matches the full
    // random sentinel, never a prefix.
    auto res = tool.execute(
        R"({"command":"printf '{\"a\":1}\n'; printf '\nFILO_END_deadbeef:99\n'; echo after_fake"})");
    // (result is JSON-encoded, so the quotes in the output appear escaped)
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring(R"({\"a\":1})"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("FILO_END_deadbeef:99"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("after_fake"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

// ---------------------------------------------------------------------------
// Exit-code semantics and session recovery
// ---------------------------------------------------------------------------

TEST_CASE("ShellTool exit code reflects the last command of a sequence",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto ok = tool.execute(R"({"command":"false; true"})");
    REQUIRE_THAT(ok, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

    auto bad = tool.execute(R"({"command":"true; false"})");
    REQUIRE_THAT(bad, Catch::Matchers::ContainsSubstring("\"exit_code\":1"));
}

TEST_CASE("ShellTool kill-signalled session reports signal exit and restarts",
          "[integration][tools][shell]") {
    ShellTool tool;

    // SIGKILL cannot be trapped, so no sentinel is ever printed: the session
    // must detect the dead shell via EOF/reap and report 128+SIGKILL.
    const auto started = std::chrono::steady_clock::now();
    auto res = tool.execute(R"({"command":"kill -KILL $$","timeout_seconds":10})");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":137"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("TIMEOUT"));
    REQUIRE(elapsed < std::chrono::seconds{8});

    auto next = tool.execute(R"({"command":"echo recovered_after_sigkill"})");
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("recovered_after_sigkill"));
    REQUIRE_THAT(next, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool subshell cd does not leak while grouped cd persists",
          "[integration][tools][shell]") {
    ShellTool tool;

    auto setup = tool.execute(R"({"command":"cd /tmp"})");
    REQUIRE_THAT(setup, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

    // A subshell must not leak its working directory into the session.
    auto sub = tool.execute(R"({"command":"(cd /var); pwd"})");
    REQUIRE_THAT(sub, Catch::Matchers::ContainsSubstring("/tmp"));
    REQUIRE_THAT(sub, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

    // An explicit brace group in the user command shares the session shell,
    // so its cd persists — confirming the wrapper uses brace semantics too.
    auto grp = tool.execute(R"({"command":"{ cd /var; }"})");
    REQUIRE_THAT(grp, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    auto pwd = tool.execute(R"({"command":"pwd"})");
    REQUIRE_THAT(pwd, Catch::Matchers::ContainsSubstring("/var"));
    REQUIRE_THAT(pwd, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

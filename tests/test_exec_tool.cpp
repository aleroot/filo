#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/Models.hpp"
#include "core/tools/ExecTool.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace {

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::StartsWith;
using core::tools::ToolScriptCall;
using core::tools::ToolScriptReply;

core::llm::Tool tool(std::string name,
                     std::string description,
                     std::vector<core::tools::ToolParameter> parameters = {}) {
    return core::llm::Tool{.function = core::tools::ToolDefinition{
        .name = std::move(name),
        .description = std::move(description),
        .parameters = std::move(parameters),
    }};
}

/// Records every batch the runtime hands over and answers through @c reply.
struct RecordingHost final : core::tools::ToolScriptHost {
    std::function<ToolScriptReply(const ToolScriptCall&)> reply = [](const ToolScriptCall& call) {
        return ToolScriptReply{.ran = true, .value = R"({"echo":)" + call.arguments + "}"};
    };
    std::vector<std::vector<ToolScriptCall>> batches;
    std::atomic<bool> stop{false};

    std::vector<ToolScriptReply> run(std::span<const ToolScriptCall> calls) override {
        batches.emplace_back(calls.begin(), calls.end());
        std::vector<ToolScriptReply> replies;
        for (const auto& call : calls) replies.push_back(reply(call));
        return replies;
    }
    bool cancelled() const override { return stop.load(); }
};

struct Script {
    std::vector<core::llm::Tool> tools{
        tool("read", "Read a file.", {{"path", "string", "File path.", true}}),
        tool("grep_search", "Search files.", {{"pattern", "string", "Pattern.", true}}),
    };
    RecordingHost host;
    core::tools::ToolScriptStore store;

    core::tools::ToolScriptOutcome run(std::string_view source) {
        return core::tools::run_tool_script(source, tools, host, store);
    }
};

} // namespace

#ifdef FILO_ENABLE_JAVASCRIPT

namespace {

/// The text items after the "Output:" header.
std::string output_of(const core::tools::ToolScriptOutcome& outcome) {
    const auto at = outcome.output.find("Output:\n");
    REQUIRE(at != std::string::npos);
    return outcome.output.substr(at + 8);
}

} // namespace

TEST_CASE("A script reports with text() under Codex's result header", "[exec]") {
    Script script;
    const auto outcome = script.run(R"(text("hello"); text({a: 1}); text(42);)");
    REQUIRE(outcome.completed);
    CHECK_THAT(outcome.output, StartsWith("Script completed\nWall time "));
    CHECK_THAT(outcome.output, ContainsSubstring(" seconds\nOutput:\n"));
    CHECK(output_of(outcome) == "hello\n{\"a\":1}\n42");
}

TEST_CASE("The value of the last expression is not output", "[exec]") {
    Script script;
    const auto outcome = script.run("const x = 1 + 1;\nx");
    REQUIRE(outcome.completed);
    CHECK(output_of(outcome).empty());
}

TEST_CASE("Calls issued together reach the host as one batch", "[exec]") {
    Script script;
    const auto outcome = script.run(R"(
const [a, b] = await Promise.all([
  tools.read({path: "a.txt"}),
  tools.grep_search({pattern: "x"}),
]);
const c = await tools.read({path: "c.txt"});
text(a.echo.path + b.echo.pattern + c.echo.path);
)");
    REQUIRE(outcome.completed);
    CHECK(output_of(outcome) == "a.txtxc.txt");
    REQUIRE(script.host.batches.size() == 2);
    REQUIRE(script.host.batches[0].size() == 2);
    CHECK(script.host.batches[0][0].tool_name == "read");
    CHECK(script.host.batches[0][0].arguments == R"({"path":"a.txt"})");
    CHECK(script.host.batches[0][1].tool_name == "grep_search");
    CHECK(script.host.batches[1].size() == 1);
}

TEST_CASE("A tool result that is not JSON arrives as a string", "[exec]") {
    Script script;
    script.host.reply = [](const ToolScriptCall&) {
        return ToolScriptReply{.ran = true, .value = "plain text"};
    };
    const auto outcome = script.run(R"(const r = await tools.read({path: "a"}); text(typeof r + ":" + r);)");
    REQUIRE(outcome.completed);
    CHECK(output_of(outcome) == "string:plain text");
}

TEST_CASE("A call the host refuses rejects with its reason", "[exec]") {
    Script script;
    script.host.reply = [](const ToolScriptCall& call) {
        if (call.tool_name == "grep_search") return ToolScriptReply{.ran = false, .value = "denied by the user"};
        return ToolScriptReply{.ran = true, .value = R"({"ok":true})"};
    };
    const auto outcome = script.run(R"(
const results = await Promise.allSettled([tools.read({path: "a"}), tools.grep_search({pattern: "b"})]);
text(results.map(r => r.status === "fulfilled" ? "ok" : r.reason.message).join("|"));
)");
    REQUIRE(outcome.completed);
    CHECK(output_of(outcome) == "ok|denied by the user");
}

TEST_CASE("Only the offered tools exist, under normalized identifiers", "[exec]") {
    Script script;
    script.tools.push_back(tool("mcp__docs-server.lookup", "Look up docs."));
    const auto outcome = script.run(R"(
text(typeof tools.apply_patch);
text(ALL_TOOLS.map(t => t.name).join(","));
await tools.mcp__docs_server_lookup({q: 1});
)");
    REQUIRE(outcome.completed);
    CHECK(output_of(outcome) == "undefined\nread,grep_search,mcp__docs_server_lookup");
    REQUIRE(script.host.batches.size() == 1);
    // The host is told the tool's real name, not the identifier.
    CHECK(script.host.batches[0][0].tool_name == "mcp__docs-server.lookup");
}

TEST_CASE("A thrown error fails the script and keeps the earlier output", "[exec]") {
    Script script;
    const auto outcome = script.run("text('before');\nthrow new Error('boom');");
    CHECK_FALSE(outcome.completed);
    CHECK_THAT(outcome.output, StartsWith("Script failed\n"));
    CHECK_THAT(output_of(outcome), StartsWith("before\nScript error:\nError: boom"));
}

TEST_CASE("exit() ends the script successfully, even inside try", "[exec]") {
    Script script;
    const auto outcome = script.run(R"(
text("first");
try { exit(); } catch (e) { text("caught"); }
text("after");
)");
    REQUIRE(outcome.completed);
    CHECK(output_of(outcome) == "first");
}

TEST_CASE("store() keeps a value for a later script", "[exec]") {
    Script script;
    REQUIRE(script.run(R"(store("k", {n: 3}); text(String(load("missing"))))").completed);
    const auto outcome = script.run(R"(text(load("k").n))");
    REQUIRE(outcome.completed);
    CHECK(output_of(outcome) == "3");
}

TEST_CASE("store() refuses a value past the session's budget", "[exec]") {
    Script script;
    const auto outcome = script.run(R"(
store("small", 1);
try { store("big", "x".repeat(17 * 1024 * 1024)); } catch (e) { text(e.message); }
text(load("small") + ":" + String(load("big")));
)");
    REQUIRE(outcome.completed);
    CHECK_THAT(output_of(outcome), StartsWith("store() is full"));
    CHECK_THAT(output_of(outcome), ContainsSubstring("\n1:undefined"));
    script.store.clear();
    CHECK_FALSE(script.store.load("small").has_value());
}

TEST_CASE("An awaited timer fires; a pending one does not keep the script alive", "[exec]") {
    Script script;
    const auto started = std::chrono::steady_clock::now();
    const auto outcome = script.run(R"(
setTimeout(() => text("never"), 30000);
await new Promise(resolve => setTimeout(resolve, 20));
const id = setTimeout(() => text("cleared"), 0);
clearTimeout(id);
text("done");
)");
    REQUIRE(outcome.completed);
    CHECK(output_of(outcome) == "done");
    CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds(5));
}

TEST_CASE("A script awaiting a promise nothing settles fails instead of hanging", "[exec]") {
    Script script;
    const auto outcome = script.run("await new Promise(() => {});");
    CHECK_FALSE(outcome.completed);
    CHECK_THAT(outcome.output, ContainsSubstring("nothing settles"));
}

TEST_CASE("Cancelling the turn stops a running script", "[exec]") {
    Script script;
    // The user stops the turn while the script's first tool call runs.
    script.host.reply = [&script](const ToolScriptCall&) {
        script.host.stop = true;
        return ToolScriptReply{.ran = true, .value = "{}"};
    };
    const auto outcome = script.run("await tools.read({path: 'a'});\nwhile (true) {}");
    CHECK_FALSE(outcome.completed);
    CHECK_THAT(outcome.output, StartsWith("Script terminated\n"));
    CHECK_THAT(outcome.output, ContainsSubstring("cancelled"));
}

TEST_CASE("The pragma's max_output_tokens bounds the result", "[exec]") {
    Script script;
    const auto outcome = script.run(
        "// @exec: {\"max_output_tokens\": 10}\n"
        "text('a'.repeat(1000)); text('END');");
    REQUIRE(outcome.completed);
    const auto output = output_of(outcome);
    CHECK(output.size() < 100);
    CHECK_THAT(output, ContainsSubstring("bytes truncated"));
    CHECK_THAT(output, ContainsSubstring("END"));
}

TEST_CASE("A script that outlives its yield stays a cell wait can resume", "[exec]") {
    core::tools::ToolScriptSession session;
    RecordingHost host;
    const std::vector<core::llm::Tool> tools{
        tool("read", "Read a file.", {{"path", "string", "File path.", true}}),
    };
    const auto started = std::chrono::steady_clock::now();
    const auto yielded = session.execute(
        "// @exec: {\"yield_time_ms\": 40}\n"
        "text('started');\n"
        "await new Promise(resolve => setTimeout(resolve, 250));\n"
        "text('finished');\n",
        tools, host);
    CHECK(yielded.running);
    CHECK_FALSE(yielded.completed);
    CHECK_THAT(yielded.output, StartsWith("Script running with cell ID c1\n"));
    CHECK(output_of(yielded) == "started");
    CHECK(std::chrono::steady_clock::now() - started < std::chrono::milliseconds(200));

    const auto done = session.resume(R"({"cell_id":"c1","yield_time_ms":1000})", host);
    CHECK(done.completed);
    CHECK_FALSE(done.running);
    CHECK_THAT(done.output, StartsWith("Script completed\n"));
    CHECK(output_of(done) == "finished");

    const auto missing = session.resume(R"({"cell_id":"c1"})", host);
    CHECK_FALSE(missing.running);
    CHECK_THAT(missing.output, ContainsSubstring("No running exec cell"));
}

TEST_CASE("wait can terminate a running cell", "[exec]") {
    core::tools::ToolScriptSession session;
    RecordingHost host;
    const auto yielded = session.execute(
        "// @exec: {\"yield_time_ms\": 30}\n"
        "text('live');\n"
        "await new Promise(resolve => setTimeout(resolve, 5000));\n",
        {}, host);
    REQUIRE(yielded.running);
    const auto stopped = session.resume(R"({"cell_id":"c1","terminate":true})", host);
    CHECK_FALSE(stopped.running);
    CHECK_THAT(stopped.output, StartsWith("Script terminated\n"));
    CHECK(output_of(stopped).empty());
    CHECK(output_of(yielded) == "live");
}

#else

TEST_CASE("A build without JavaScript fails a script instead of running it", "[exec]") {
    Script script;
    const auto outcome = script.run("text('hello')");
    CHECK_FALSE(outcome.completed);
    CHECK_THAT(outcome.output, ContainsSubstring("no JavaScript runtime"));
    CHECK(script.host.batches.empty());
}

#endif // FILO_ENABLE_JAVASCRIPT

TEST_CASE("A bare string becomes a tool's only required string parameter", "[exec]") {
    const core::tools::ToolDefinition patch{
        .name = "apply_patch",
        .parameters = {{"patch", "string", "Patch.", true}, {"dry_run", "boolean", "", false}},
    };
    CHECK(core::tools::nested_tool_arguments(patch, R"("*** Begin Patch")")
          == R"({"patch":"*** Begin Patch"})");
    CHECK(core::tools::nested_tool_arguments(patch, R"({"patch":"x"})") == R"({"patch":"x"})");

    const core::tools::ToolDefinition two{
        .name = "move_file",
        .parameters = {{"source", "string", "", true}, {"destination", "string", "", true}},
    };
    CHECK(core::tools::nested_tool_arguments(two, R"("a")") == R"("a")");
}

TEST_CASE("An exec call carries its script as a source argument", "[exec]") {
    const std::string source = "text(\"é\\n\" + 'quote\"');\n";
    const auto arguments = core::tools::arguments_for_source(source);
    CHECK(core::tools::source_from_arguments(arguments) == source);
    CHECK_FALSE(core::tools::source_from_arguments(R"({"script":"x"})").has_value());
    // A script may write, so AUTO takes the writer lease before one runs.
    CHECK(core::tools::exec_argument_definition().annotations.destructive_hint);
}

TEST_CASE("The exec description documents every offered tool in TypeScript", "[exec]") {
    const std::vector<core::llm::Tool> tools{
        tool("read", "Read a file.", {
            {"path", "string", "File path.", true},
            {"limit_lines", "integer", "How many lines.", false},
        }),
        tool("mcp__docs-server.lookup", "Look up docs."),
    };
    const std::string description = core::tools::tool_script_description(tools);
    CHECK_THAT(description, StartsWith("Run JavaScript code to orchestrate tool calls."));
    CHECK_THAT(description, ContainsSubstring("yield_time_ms"));
    CHECK_THAT(description, ContainsSubstring("Script running with cell ID"));
    CHECK_THAT(core::tools::wait_tool_description(),
               ContainsSubstring("Script running with cell ID"));
    CHECK_THAT(description, ContainsSubstring("### `read`\nRead a file."));
    CHECK_THAT(description, ContainsSubstring("declare const tools: { read(args: {\n"));
    CHECK_THAT(description, ContainsSubstring("  // File path.\n  path: string;\n"));
    CHECK_THAT(description, ContainsSubstring("  limit_lines?: number;\n"));
    CHECK_THAT(description, ContainsSubstring("### `mcp__docs_server_lookup` (`mcp__docs-server.lookup`)"));
    CHECK_THAT(description, ContainsSubstring("`text(value"));
}

TEST_CASE("A tool's output schema becomes the type its call resolves to", "[exec]") {
    auto described = tool("grep_search", "Search files.", {{"pattern", "string", "Pattern.", true}});
    described.function.output_schema =
        R"({"type":"object","properties":{"matches":{"type":"array","items":{"type":"string"}}},"required":["matches"]})";
    const std::string description =
        core::tools::tool_script_description({described, tool("read", "Read a file.")});
    CHECK_THAT(description, ContainsSubstring("): Promise<{ matches: Array<string>; }>; };"));
    CHECK_THAT(description, ContainsSubstring("read(args: {\n}): Promise<unknown>; };"));
}

TEST_CASE("Scripts reach run_terminal_command as Codex's exec_command", "[exec]") {
    const std::vector<core::llm::Tool> offered{
        tool("run_terminal_command", "Run a command.", {{"command", "string", "Bash.", true}}),
        tool("read", "Read a file."),
    };
    const auto surface = core::tools::script_tool_surface(offered);
    REQUIRE(surface.size() == 2);
    CHECK(surface[0].function.name == "exec_command");
    CHECK(surface[1].function.name == "read");

    const std::string description = core::tools::tool_script_description(offered);
    CHECK_THAT(description, ContainsSubstring("### `exec_command`"));
    CHECK_THAT(description, ContainsSubstring("  cmd: string;\n"));
    CHECK_THAT(description, ContainsSubstring("wall_time_seconds: number;"));
    CHECK_FALSE(description.contains("run_terminal_command"));

    SECTION("an object call keeps its directory, timeout, and budget") {
        const auto alias = core::tools::resolve_script_alias(
            "exec_command",
            R"({"cmd":"rg -n \"x\"","workdir":"/repo/sub","timeout_ms":1500,"max_output_tokens":20})",
            "/repo");
        REQUIRE(alias.has_value());
        CHECK(alias->tool_name == "run_terminal_command");
        CHECK(alias->arguments
              == R"({"command":"rg -n \"x\"","working_dir":"/repo/sub","timeout_seconds":2})");
        CHECK(alias->max_output_tokens == 20);
    }
    SECTION("a bare string runs in the turn cwd with the default timeout") {
        const auto alias = core::tools::resolve_script_alias("exec_command", R"("ls")", "/repo");
        REQUIRE(alias.has_value());
        CHECK(alias->arguments
              == R"({"command":"ls","working_dir":"/repo","timeout_seconds":600})");
        CHECK(alias->max_output_tokens == 10000);
    }
    SECTION("a call without cmd is left for validation to reject") {
        const auto alias = core::tools::resolve_script_alias("exec_command", R"({"workdir":"/x"})", "");
        REQUIRE(alias.has_value());
        CHECK(alias->arguments == "{}");
    }
    CHECK_FALSE(core::tools::resolve_script_alias("read", R"({"path":"a"})", "/repo").has_value());
}

TEST_CASE("An exec_command result has Codex's shape and token truncation", "[exec]") {
    core::tools::ScriptToolAlias alias{.tool_name = "run_terminal_command", .max_output_tokens = 10000};
    CHECK(core::tools::script_alias_result(alias, R"({"output":"ok\n","exit_code":0})", 0.25)
          == R"({"wall_time_seconds":0.250,"exit_code":0,"output":"ok\n"})");
    CHECK(core::tools::script_alias_result(alias, R"({"error":"bad working_dir"})", 0.1)
          == R"({"error":"bad working_dir"})");

    alias.max_output_tokens = 2;
    const auto truncated = core::tools::script_alias_result(
        alias, R"({"output":"abcdefghijklmnopqrst","exit_code":3})", 1.0);
    CHECK(truncated
          == R"({"wall_time_seconds":1.000,"exit_code":3,"original_token_count":5,"output":"abcd…3 tokens truncated…qrst"})");
}

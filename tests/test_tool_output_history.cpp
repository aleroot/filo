#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/Agent.hpp"
#include "core/agent/ToolOutputHistory.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/tools/Tool.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/utils/JsonUtils.hpp"
#include "core/tools/ToolNames.hpp"
#include "TestSessionContext.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kLargeOutputToolName = "test_large_output_history_tool";
constexpr std::string_view kLargeErrorToolName = "test_large_error_history_tool";

class LargeOutputTool final : public core::tools::Tool {
public:
    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = std::string(kLargeOutputToolName),
            .title = "Large Output Test Tool",
            .description = "Returns an intentionally large payload for history truncation tests.",
            .parameters = {},
            .annotations = {
                .read_only_hint = true,
                .idempotent_hint = true,
            },
        };
    }

    [[nodiscard]] std::string execute(
        const std::string&,
        const core::context::SessionContext&) override {
        return std::string(R"({"output":")") + std::string(120 * 1024, 'x') + R"("})";
    }
};

class LargeErrorTool final : public core::tools::Tool {
public:
    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = std::string(kLargeErrorToolName),
            .title = "Large Error Test Tool",
            .description = "Returns an intentionally large error payload for truncation tests.",
            .parameters = {},
            .annotations = {
                .read_only_hint = true,
                .idempotent_hint = true,
            },
        };
    }

    [[nodiscard]] std::string execute(
        const std::string&,
        const core::context::SessionContext&) override {
        return std::string(R"({"error":")") + std::string(120 * 1024, 'e') + R"("})";
    }
};

class ToolThenFinalProvider final : public core::llm::LLMProvider {
public:
    explicit ToolThenFinalProvider(std::string tool_name)
        : tool_name_(std::move(tool_name)) {}

    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        ++calls_;
        if (calls_ == 1) {
            core::llm::ToolCall tc;
            tc.index = 0;
            tc.id = "tc-1";
            tc.type = "function";
            tc.function.name = tool_name_;
            tc.function.arguments = "{}";

            core::llm::StreamChunk chunk;
            chunk.tools = {tc};
            chunk.is_final = true;
            callback(chunk);
            return;
        }

        callback(core::llm::StreamChunk::make_content("final"));
        callback(core::llm::StreamChunk::make_final());
    }

private:
    std::string tool_name_;
    int calls_ = 0;
};

void send_and_wait(const std::shared_ptr<core::agent::Agent>& agent,
                   std::string_view prompt) {
    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;

    agent->send_message(
        std::string(prompt),
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&]() {
            {
                std::lock_guard lock(done_mutex);
                done = true;
            }
            done_cv.notify_one();
        });

    std::unique_lock lock(done_mutex);
    REQUIRE(done_cv.wait_for(lock, std::chrono::seconds(5), [&]() { return done; }));
}

[[nodiscard]] std::size_t count_occurrences(std::string_view text, std::string_view needle) {
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string_view::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

} // namespace

TEST_CASE("ToolOutputHistory leaves compact outputs unchanged", "[agent][tool-history]") {
    const std::string small = R"({"output":"ok"})";
    const std::string clamped = core::agent::tool_output_history::clamp_for_history("read_file", small);
    CHECK(clamped == small);
}

TEST_CASE("ToolOutputHistory truncates oversized non-error outputs", "[agent][tool-history]") {
    const std::string large = std::string(R"({"output":")") + std::string(80 * 1024, 'a') + R"("})";
    const std::string clamped = core::agent::tool_output_history::clamp_for_history("write_file", large);

    CHECK(clamped.size() < large.size());
    CHECK_THAT(clamped, Catch::Matchers::ContainsSubstring(R"("truncated":true)"));
    CHECK_THAT(clamped, Catch::Matchers::ContainsSubstring(R"("original_chars":)"));
    CHECK_THAT(clamped, Catch::Matchers::ContainsSubstring(R"("digest_fnv1a64":)"));
    CHECK_THAT(clamped, Catch::Matchers::ContainsSubstring(R"("head":)"));
}

TEST_CASE("ToolOutputHistory light-compresses oversized read_file output", "[agent][tool-history]") {
    std::string source;
    for (int i = 0; i < 500; ++i) {
        source += "// filler line " + std::to_string(i) + "\n";
    }
    source += "#include <vector>\n";
    source += "class WidgetController {\n";
    source += "public:\n";
    source += "    void render() const;\n";
    source += "};\n";
    for (int i = 0; i < 500; ++i) {
        source += "int value_" + std::to_string(i) + " = " + std::to_string(i) + ";\n";
    }

    const std::string raw = std::string(R"({"content":")")
        + core::utils::escape_json_string(source)
        + R"("})";

    const std::string compressed = core::agent::tool_output_history::clamp_for_history(
        "read_file",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 4 * 1024,
            .head_chars = 2 * 1024,
            .tail_chars = 1024,
        },
        "light");

    CHECK(compressed.size() < raw.size());
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"("compressed":true)"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("[light read_file summary]"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("class WidgetController"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"("digest_fnv1a64":)"));
}

TEST_CASE("ToolOutputHistory light-compresses oversized shell output", "[agent][tool-history]") {
    std::string output;
    for (int i = 0; i < 700; ++i) {
        output += "note line " + std::to_string(i) + "\n";
    }
    output += "ERROR: test_auth_flow failed at assertion 12\n";
    output += "warning: retrying flaky setup\n";
    for (int i = 0; i < 700; ++i) {
        output += "more noise " + std::to_string(i) + "\n";
    }

    const std::string raw = std::string(R"({"output":")")
        + core::utils::escape_json_string(output)
        + R"(","exit_code":1})";

    const std::string compressed = core::agent::tool_output_history::clamp_for_history(
        "run_terminal_command",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 4 * 1024,
            .head_chars = 2 * 1024,
            .tail_chars = 1024,
        },
        "light");

    CHECK(compressed.size() < raw.size());
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("[light shell summary]"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("ERROR: test_auth_flow failed"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("Exit code: 1"));
}

TEST_CASE("ToolOutputHistory full mode returns cached stubs for unchanged read_file output",
          "[agent][tool-history]") {
    std::string source;
    for (int i = 0; i < 120; ++i) {
        source += "int helper_" + std::to_string(i) + "() { return " + std::to_string(i) + "; }\n";
    }
    const std::string raw = std::string(R"({"content":")")
        + core::utils::escape_json_string(source)
        + R"("})";
    const core::agent::tool_output_history::Context ctx{
        .tool_arguments = R"({"path":"src/example.cpp"})",
        .session_id = "full-cache-test-session",
    };

    const auto limits = core::agent::tool_output_history::Limits{
        .max_chars = 8 * 1024,
        .head_chars = 5 * 1024,
        .tail_chars = 3 * 1024,
    };

    const std::string first = core::agent::tool_output_history::clamp_for_history(
        "read_file",
        raw,
        limits,
        "full",
        ctx);
    const std::string second = core::agent::tool_output_history::clamp_for_history(
        "read_file",
        raw,
        limits,
        "full",
        ctx);

    CHECK(first == raw);
    CHECK(second.size() < raw.size());
    CHECK_THAT(second, Catch::Matchers::ContainsSubstring(R"("compression":"full")"));
    CHECK_THAT(second, Catch::Matchers::ContainsSubstring("[cached read]"));
    CHECK_THAT(second, Catch::Matchers::ContainsSubstring("src/example.cpp"));
    CHECK_THAT(second, Catch::Matchers::ContainsSubstring("unchanged"));
    CHECK_THAT(second, !Catch::Matchers::ContainsSubstring(R"("ref":)"));
    CHECK_THAT(second, !Catch::Matchers::ContainsSubstring("[cached F"));
}

TEST_CASE("ToolOutputHistory full mode summarizes oversized first read",
          "[agent][tool-history]") {
    std::string source;
    for (int i = 0; i < 1200; ++i) {
        source += "int oversized_helper_" + std::to_string(i)
            + "() { return " + std::to_string(i) + "; }\n";
    }
    const std::string raw = std::string(R"({"content":")")
        + core::utils::escape_json_string(source)
        + R"("})";

    const std::string first = core::agent::tool_output_history::clamp_for_history(
        "read_file",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 4 * 1024,
            .head_chars = 2 * 1024,
            .tail_chars = 1024,
        },
        "full",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"path":"src/oversized.cpp"})",
            .session_id = "full-oversized-first-read-session",
        });

    CHECK(first.size() < raw.size());
    CHECK_THAT(first, Catch::Matchers::ContainsSubstring(R"("compression":"full")"));
    CHECK_THAT(first, Catch::Matchers::ContainsSubstring("[full read_file summary]"));
    CHECK_THAT(first, Catch::Matchers::ContainsSubstring("Use read_file with offset_line/limit_lines"));
    CHECK_THAT(first, Catch::Matchers::ContainsSubstring("int oversized_helper_"));
    CHECK_THAT(first, Catch::Matchers::ContainsSubstring("Tail:"));
}

TEST_CASE("ToolOutputHistory full mode never cache-stubs instruction files",
          "[agent][tool-history]") {
    std::string instructions;
    for (int i = 0; i < 160; ++i) {
        instructions += "- Always preserve instruction line " + std::to_string(i) + "\n";
    }
    const std::string raw = std::string(R"({"content":")")
        + core::utils::escape_json_string(instructions)
        + R"("})";
    const core::agent::tool_output_history::Context ctx{
        .tool_arguments = R"({"path":"AGENTS.md"})",
        .session_id = "instruction-cache-test-session",
    };

    const auto limits = core::agent::tool_output_history::Limits{
        .max_chars = 1024,
        .head_chars = 512,
        .tail_chars = 256,
    };

    const std::string first = core::agent::tool_output_history::clamp_for_history(
        "read_file",
        raw,
        limits,
        "full",
        ctx);
    const std::string second = core::agent::tool_output_history::clamp_for_history(
        "read_file",
        raw,
        limits,
        "full",
        ctx);

    CHECK(first == raw);
    CHECK(second == raw);
    CHECK_THAT(second, !Catch::Matchers::ContainsSubstring("[cached read]"));
}

TEST_CASE("ToolOutputHistory light mode keeps instruction files exact",
          "[agent][tool-history]") {
    std::string instructions;
    for (int i = 0; i < 220; ++i) {
        instructions += "- Preserve light-mode instruction line " + std::to_string(i) + "\n";
    }
    const std::string raw = std::string(R"({"content":")")
        + core::utils::escape_json_string(instructions)
        + R"("})";

    const std::string result = core::agent::tool_output_history::clamp_for_history(
        "read_file",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 1024,
            .head_chars = 512,
            .tail_chars = 256,
        },
        "light",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"path":".filo/steering/AGENTS.md"})",
            .session_id = "light-instruction-test-session",
        });

    CHECK(result == raw);
}

TEST_CASE("ToolOutputHistory full mode compresses known shell command output",
          "[agent][tool-history]") {
    std::string output;
    output += "On branch main\n";
    output += "Your branch is up to date with 'origin/main'.\n\n";
    output += "Changes not staged for commit:\n";
    output += "  modified:   src/core/agent/ToolOutputHistory.cpp\n";
    for (int i = 0; i < 200; ++i) {
        output += "verbose ignored status detail " + std::to_string(i) + "\n";
    }
    output += "Untracked files:\n";
    output += "  tests/new_test.cpp\n";

    const std::string raw = std::string(R"({"output":")")
        + core::utils::escape_json_string(output)
        + R"(","exit_code":0})";

    const std::string compressed = core::agent::tool_output_history::clamp_for_history(
        "run_terminal_command",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 12 * 1024,
            .head_chars = 7 * 1024,
            .tail_chars = 3 * 1024,
        },
        "full",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"command":"git status --short"})",
            .session_id = "full-shell-test-session",
        });

    CHECK(compressed.size() < raw.size());
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"("compression":"full")"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("[full shell context pack]"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("Command family: git-status"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("modified:   src/core/agent/ToolOutputHistory.cpp"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("Untracked files:"));
}

TEST_CASE("ToolOutputHistory full mode preserves git status porcelain entries",
          "[agent][tool-history]") {
    std::string output;
    output += "## main...origin/main [ahead 1]\n";
    output += " M README.md\n";
    output += "M  src/core/agent/ToolOutputHistory.cpp\n";
    output += "AM tests/test_tool_output_history.cpp\n";
    output += "?? docs/new-note.md\n";
    for (int i = 0; i < 400; ++i) {
        output += "ignored verbose status line " + std::to_string(i) + "\n";
    }

    const std::string raw = std::string(R"({"output":")")
        + core::utils::escape_json_string(output)
        + R"(","exit_code":0})";

    const std::string compressed = core::agent::tool_output_history::clamp_for_history(
        "run_terminal_command",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 3 * 1024,
            .head_chars = 2 * 1024,
            .tail_chars = 512,
        },
        "full",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"command":"git status --short"})",
            .session_id = "full-status-test-session",
        });

    CHECK(compressed.size() < raw.size());
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("## main...origin/main [ahead 1]"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(" M README.md"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("M  src/core/agent/ToolOutputHistory.cpp"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("AM tests/test_tool_output_history.cpp"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("?? docs/new-note.md"));
}

TEST_CASE("ToolOutputHistory full mode preserves git diff structural lines verbatim",
          "[agent][tool-history]") {
    std::string output;
    output += "diff --git a/src/example.cpp b/src/example.cpp\n";
    output += "index 1111111..2222222 100644\n";
    output += "--- a/src/example.cpp\n";
    output += "+++ b/src/example.cpp\n";
    output += "@@ -10,8 +10,8 @@ bool enabled()\n";
    for (int i = 0; i < 300; ++i) {
        output += " context line " + std::to_string(i) + "\n";
    }
    output += "-    return false;\n";
    output += "+    return true;\n";
    output += "-    return false;\n";
    output += "+    return true;\n";

    const std::string raw = std::string(R"({"output":")")
        + core::utils::escape_json_string(output)
        + R"(","exit_code":0})";

    const std::string compressed = core::agent::tool_output_history::clamp_for_history(
        "run_terminal_command",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 2 * 1024,
            .head_chars = 1024,
            .tail_chars = 512,
        },
        "full",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"command":"git diff"})",
            .session_id = "full-diff-test-session",
        });

    CHECK(compressed.size() < raw.size());
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("[full shell context pack]"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("+    return true;"));
    CHECK(count_occurrences(compressed, "+    return true;") == 2);
    CHECK(count_occurrences(compressed, "-    return false;") == 2);
}

TEST_CASE("ToolOutputHistory full mode preserves git diff summary variants",
          "[agent][tool-history]") {
    std::string output;
    output += "M\tREADME.md\n";
    output += "A\tsrc/new_file.cpp\n";
    output += "D\tdocs/old_file.md\n";
    for (int i = 0; i < 400; ++i) {
        output += "ignored summary noise " + std::to_string(i) + "\n";
    }

    const std::string raw = std::string(R"({"output":")")
        + core::utils::escape_json_string(output)
        + R"(","exit_code":0})";

    const std::string compressed = core::agent::tool_output_history::clamp_for_history(
        "run_terminal_command",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 3 * 1024,
            .head_chars = 2 * 1024,
            .tail_chars = 512,
        },
        "full",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"command":"git diff --name-status"})",
            .session_id = "full-diff-summary-test-session",
        });

    CHECK(compressed.size() < raw.size());
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("Command family: git-diff-summary"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"(M\tREADME.md)"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"(A\tsrc/new_file.cpp)"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"(D\tdocs/old_file.md)"));
}

TEST_CASE("ToolOutputHistory full mode keeps build diagnostics exact when bounded",
          "[agent][tool-history]") {
    std::string output;
    output += "Compiling filo v0.1.0\n";
    output += "error[E0425]: cannot find value `missing_symbol` in this scope\n";
    output += "  --> src/main.rs:42:13\n";
    output += "   |\n";
    output += "42 |     println!(\"{}\", missing_symbol);\n";
    output += "   |                    ^^^^^^^^^^^^^^ not found in this scope\n";

    const std::string raw = std::string(R"({"output":")")
        + core::utils::escape_json_string(output)
        + R"(","exit_code":101})";

    const std::string result = core::agent::tool_output_history::clamp_for_history(
        "run_terminal_command",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 4 * 1024,
            .head_chars = 2 * 1024,
            .tail_chars = 1024,
        },
        "full",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"command":"cargo test"})",
            .session_id = "full-build-diagnostics-test-session",
        });

    CHECK(result == raw);
    CHECK_THAT(result, !Catch::Matchers::ContainsSubstring("[full shell context pack]"));
}

TEST_CASE("ToolOutputHistory full mode falls back to verbatim truncation for huge build diagnostics",
          "[agent][tool-history]") {
    std::string output;
    output += "error: failed to compile crate\n";
    output += "  --> src/lib.rs:1:1\n";
    for (int i = 0; i < 700; ++i) {
        output += "diagnostic context line " + std::to_string(i)
            + " with file span src/lib.rs:" + std::to_string(i + 2) + ":7\n";
    }
    output += "final diagnostic: could not compile filo\n";

    const std::string raw = std::string(R"({"output":")")
        + core::utils::escape_json_string(output)
        + R"(","exit_code":101})";

    const std::string result = core::agent::tool_output_history::clamp_for_history(
        "run_terminal_command",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 3 * 1024,
            .head_chars = 2 * 1024,
            .tail_chars = 512,
        },
        "full",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"command":"cargo test"})",
            .session_id = "full-build-diagnostics-huge-test-session",
        });

    CHECK(result.size() < raw.size());
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring(R"("truncated":true)"));
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring("error: failed to compile crate"));
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring("final diagnostic"));
    CHECK_THAT(result, !Catch::Matchers::ContainsSubstring("[full shell context pack]"));
}

TEST_CASE("ToolOutputHistory full mode truncates oversized read_file errors",
          "[agent][tool-history]") {
    const std::string raw = std::string(R"({"error":")")
        + std::string(16 * 1024, 'x')
        + R"("})";

    const std::string result = core::agent::tool_output_history::clamp_for_history(
        "read_file",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 1024,
            .head_chars = 640,
            .tail_chars = 256,
        },
        "full",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"path":"missing.cpp"})",
            .session_id = "full-read-file-error-test-session",
        });

    CHECK(result.size() < raw.size());
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring("Tool output truncated for history"));
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring(R"("tool":"read_file")"));
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring(R"("digest_fnv1a64":)"));
}

TEST_CASE("ToolOutputHistory preserves error semantics when truncating", "[agent][tool-history]") {
    const std::string large_error = std::string(R"({"error":")") + std::string(80 * 1024, 'z') + R"("})";
    const std::string clamped = core::agent::tool_output_history::clamp_for_history("replace", large_error);

    CHECK(clamped.size() < large_error.size());
    CHECK_THAT(clamped, Catch::Matchers::ContainsSubstring(R"("error":"Tool output truncated for history)"));
    CHECK_THAT(clamped, Catch::Matchers::ContainsSubstring(R"("original_chars":)"));
}

TEST_CASE("ToolOutputHistory preserves large activate_skill payloads", "[agent][tool-history]") {
    const std::string large_skill =
        std::string(R"({"content":")") + std::string(300 * 1024, 's') + R"("})";
    const std::string clamped = core::agent::tool_output_history::clamp_for_history(
        core::tools::names::kActivateSkill,
        large_skill);

    CHECK(clamped == large_skill);
}

TEST_CASE("Agent stores oversized tool output in compact history format", "[agent][tool-history]") {
    auto provider = std::make_shared<ToolThenFinalProvider>(std::string(kLargeOutputToolName));
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<LargeOutputTool>());

    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());
    send_and_wait(agent, "Run large output tool");

    const auto history = agent->get_history();
    bool found = false;
    for (const auto& msg : history) {
        if (msg.role == "tool" && msg.name == kLargeOutputToolName) {
            found = true;
            CHECK(msg.content.size() < 40 * 1024);
            CHECK_THAT(msg.content, Catch::Matchers::ContainsSubstring(R"("truncated":true)"));
            CHECK_THAT(msg.content, Catch::Matchers::ContainsSubstring(R"("original_chars":)"));
        }
    }
    REQUIRE(found);
}

TEST_CASE("Agent keeps error marker when compacting oversized error payloads", "[agent][tool-history]") {
    auto provider = std::make_shared<ToolThenFinalProvider>(std::string(kLargeErrorToolName));
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<LargeErrorTool>());

    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());
    send_and_wait(agent, "Run large error tool");

    const auto history = agent->get_history();
    bool found = false;
    for (const auto& msg : history) {
        if (msg.role == "tool" && msg.name == kLargeErrorToolName) {
            found = true;
            CHECK(msg.content.size() < 40 * 1024);
            CHECK_THAT(msg.content, Catch::Matchers::ContainsSubstring(R"("error":"Tool output truncated for history)"));
        }
    }
    REQUIRE(found);
}

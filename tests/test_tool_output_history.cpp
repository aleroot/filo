#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/Agent.hpp"
#include "core/agent/ToolOutputHistory.hpp"
#include "core/agent/ToolResultStore.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/tools/Tool.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/utils/JsonUtils.hpp"
#include "core/tools/ToolNames.hpp"
#include "TestSessionContext.hpp"

#include <simdjson.h>

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kLargeOutputToolName = "test_large_output_history_tool";
constexpr std::string_view kLargeErrorToolName = "test_large_error_history_tool";

class TempDir final {
public:
    TempDir()
        : path_(std::filesystem::temp_directory_path()
                / std::format(
                    "filo_tool_result_test_{}",
                    std::chrono::steady_clock::now().time_since_epoch().count())) {
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

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

[[nodiscard]] std::string large_grep_output() {
    std::string raw = R"({"matches":[)";
    bool first = true;
    for (int file = 0; file < 8; ++file) {
        for (int line = 0; line < 14; ++line) {
            if (!first) raw += ',';
            first = false;
            raw += R"({"path":"src/module_)";
            raw += std::to_string(file);
            raw += R"(/component.cpp","line":)";
            raw += std::to_string(line + 1);
            raw += R"(,"text":")";
            raw += core::utils::escape_json_string(
                "needle match with useful surrounding source context "
                + std::string(80, static_cast<char>('a' + file)));
            raw += R"("})";
        }
    }
    raw += "]}";
    return raw;
}

[[nodiscard]] std::string large_file_search_output() {
    std::string raw = R"({"files":[)";
    for (int i = 0; i < 180; ++i) {
        if (i > 0) raw += ',';
        raw += '"';
        raw += core::utils::escape_json_string(
            "/workspace/src/feature_" + std::to_string(i % 12)
            + "/very_long_component_name_" + std::to_string(i) + ".cpp");
        raw += '"';
    }
    raw += "]}";
    return raw;
}

[[nodiscard]] std::string large_list_directory_output() {
    std::string raw = R"({"entries":[)";
    bool first = true;
    for (int i = 0; i < 80; ++i) {
        if (!first) raw += ',';
        first = false;
        raw += R"({"type":"dir","name":")";
        raw += core::utils::escape_json_string("generated_directory_" + std::to_string(i));
        raw += R"("})";
    }
    for (int i = 0; i < 160; ++i) {
        if (!first) raw += ',';
        first = false;
        raw += R"({"type":"file","name":")";
        raw += core::utils::escape_json_string("generated_file_" + std::to_string(i) + ".hpp");
        raw += R"("})";
    }
    raw += "]}";
    return raw;
}

} // namespace

TEST_CASE("ToolOutputHistory leaves compact outputs unchanged", "[agent][tool-history]") {
    const std::string small = R"({"output":"ok"})";
    const std::string clamped = core::agent::tool_output_history::clamp_for_history("read_file", small);
    CHECK(clamped == small);
}

TEST_CASE("ToolOutputHistory scales limits from a regression-free token budget",
          "[agent][tool-history]") {
    using core::agent::tool_output_history::limits_for_tool;

    CHECK(limits_for_tool("read_file", 3072).max_chars == 12 * 1024);
    CHECK(limits_for_tool("run_terminal_command", 3072).max_chars == 10 * 1024);
    CHECK(limits_for_tool("read_file", 1536).max_chars == 6 * 1024);
    CHECK(limits_for_tool("run_terminal_command", 6144).max_chars == 20 * 1024);
    CHECK(limits_for_tool("activate_skill", 1536).max_chars == 2 * 1024 * 1024);
    CHECK(limits_for_tool("read_tool_result", 256).max_chars == 32 * 1024);
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

TEST_CASE("ToolOutputHistory keeps UTF-8 previews valid at byte boundaries",
          "[agent][tool-history]") {
    const std::string box = "\xE2\x94\x80"; // U+2500 BOX DRAWINGS LIGHT HORIZONTAL
    const std::string raw = std::string(R"({"output":")")
        + box + std::string(16, 'x') + box + R"("})";

    const std::string clamped = core::agent::tool_output_history::clamp_for_history(
        "replace", raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 16,
            .head_chars = 12,
            .tail_chars = 4,
        });

    simdjson::dom::parser parser;
    simdjson::padded_string padded{clamped};
    simdjson::dom::object document;
    REQUIRE(parser.parse(padded).get(document) == simdjson::SUCCESS);

    std::string_view head;
    std::string_view tail;
    REQUIRE(document["head"].get(head) == simdjson::SUCCESS);
    REQUIRE(document["tail"].get(tail) == simdjson::SUCCESS);
    CHECK(head == R"({"output":")");
    CHECK(tail == R"("})");
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
        "light",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"path":"src/widgets/WidgetController.cpp"})",
            .session_id = "light-read-metadata-session",
        });

    CHECK(compressed.size() < raw.size());
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"("compressed":true)"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"("path":"src/widgets/WidgetController.cpp")"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"("original_lines":)"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("[light read_file summary]"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("Original lines:"));
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
        "light",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"command":"pytest tests/test_auth.py"})",
            .session_id = "light-shell-metadata-session",
        });

    CHECK(compressed.size() < raw.size());
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"("command":"pytest tests/test_auth.py")"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"("original_lines":)"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("[light shell summary]"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("Original lines:"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("ERROR: test_auth_flow failed"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("Exit code: 1"));
}

TEST_CASE("ToolOutputHistory leaves compact search and list outputs unchanged",
          "[agent][tool-history]") {
    const std::string grep = R"({"matches":[{"path":"src/a.cpp","line":7,"text":"needle"}]})";
    const std::string files = R"({"files":["src/a.cpp","src/b.cpp"]})";
    const std::string entries =
        R"({"entries":[{"type":"dir","name":"src"},{"type":"file","name":"README.md"}]})";

    CHECK(core::agent::tool_output_history::clamp_for_history(
        core::tools::names::kGrepSearch,
        grep,
        "full") == grep);
    CHECK(core::agent::tool_output_history::clamp_for_history(
        core::tools::names::kFileSearch,
        files,
        "light") == files);
    CHECK(core::agent::tool_output_history::clamp_for_history(
        core::tools::names::kListDirectory,
        entries,
        "ultra") == entries);
}

TEST_CASE("ToolOutputHistory full mode summarizes oversized grep_search output",
          "[agent][tool-history]") {
    const std::string raw = large_grep_output();
    const std::string compressed = core::agent::tool_output_history::clamp_for_history(
        core::tools::names::kGrepSearch,
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 2 * 1024,
            .head_chars = 1200,
            .tail_chars = 512,
        },
        "full");

    CHECK(compressed.size() < raw.size());
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"("compression":"full")"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("[full grep_search summary]"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("Grouped matches:"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("src/module_0/component.cpp"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("needle match with useful"));
    CHECK_THAT(compressed, !Catch::Matchers::ContainsSubstring(R"("head":)"));
}

TEST_CASE("ToolOutputHistory light mode summarizes oversized file_search output",
          "[agent][tool-history]") {
    const std::string raw = large_file_search_output();
    const std::string compressed = core::agent::tool_output_history::clamp_for_history(
        core::tools::names::kFileSearch,
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 2 * 1024,
            .head_chars = 1200,
            .tail_chars = 512,
        },
        "light");

    CHECK(compressed.size() < raw.size());
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"("compression":"light")"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("[light file_search summary]"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("Top path groups:"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("feature_"));
}

TEST_CASE("ToolOutputHistory ultra mode summarizes oversized list_directory output",
          "[agent][tool-history]") {
    const std::string raw = large_list_directory_output();
    const std::string compressed = core::agent::tool_output_history::clamp_for_history(
        core::tools::names::kListDirectory,
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 2 * 1024,
            .head_chars = 1200,
            .tail_chars = 512,
        },
        "ultra");

    CHECK(compressed.size() < raw.size());
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"("compression":"ultra")"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("[ultra list_directory summary]"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("80 dirs, 160 files"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("Directories:"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("Files:"));
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

TEST_CASE("ToolOutputHistory ultra mode summarizes moderately sized first read",
          "[agent][tool-history]") {
    std::string source;
    for (int i = 0; i < 300; ++i) {
        source += "int ultra_helper_" + std::to_string(i)
            + "() { return " + std::to_string(i) + "; }\n";
    }
    const std::string raw = std::string(R"({"content":")")
        + core::utils::escape_json_string(source)
        + R"("})";

    const std::string result = core::agent::tool_output_history::clamp_for_history(
        "read_file",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 12 * 1024,
            .head_chars = 8 * 1024,
            .tail_chars = 4 * 1024,
        },
        "ultra",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"path":"src/ultra.cpp"})",
            .session_id = "ultra-read-file-test-session",
        });

    CHECK(result.size() < raw.size());
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring(R"("compression":"ultra")"));
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring("[full read_file summary]"));
    CHECK_THAT(result, Catch::Matchers::ContainsSubstring("src/ultra.cpp"));
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

TEST_CASE("ToolOutputHistory ultra mode uses tighter shell context packs",
          "[agent][tool-history]") {
    std::string output;
    for (int i = 0; i < 260; ++i) {
        output += "ERROR: repeated failing test shard " + std::to_string(i) + "\n";
    }
    for (int i = 0; i < 260; ++i) {
        output += "progress line " + std::to_string(i) + "\n";
    }

    const std::string raw = std::string(R"({"output":")")
        + core::utils::escape_json_string(output)
        + R"(","exit_code":0})";

    const std::string compressed = core::agent::tool_output_history::clamp_for_history(
        "run_terminal_command",
        raw,
        core::agent::tool_output_history::Limits{
            .max_chars = 10 * 1024,
            .head_chars = 7 * 1024,
            .tail_chars = 3 * 1024,
        },
        "ultra",
        core::agent::tool_output_history::Context{
            .tool_arguments = R"({"command":"custom-check"})",
            .session_id = "ultra-shell-test-session",
        });

    CHECK(compressed.size() < raw.size());
    CHECK(compressed.size() < 6 * 1024);
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring(R"("compression":"ultra")"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("[full shell context pack]"));
    CHECK_THAT(compressed, Catch::Matchers::ContainsSubstring("ERROR: repeated failing test shard 0"));
    CHECK_THAT(compressed, !Catch::Matchers::ContainsSubstring("Head:"));
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
    TempDir offload_root;
    auto provider = std::make_shared<ToolThenFinalProvider>(std::string(kLargeOutputToolName));
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<LargeOutputTool>());

    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context(),
        offload_root.path());
    send_and_wait(agent, "Run large output tool");

    const auto history = agent->get_history();
    bool found = false;
    for (const auto& msg : history) {
        if (msg.role == "tool" && msg.name == kLargeOutputToolName) {
            found = true;
            CHECK(msg.content.size() < 40 * 1024);
            CHECK_THAT(msg.content, Catch::Matchers::ContainsSubstring(R"("truncated":true)"));
            CHECK_THAT(msg.content, Catch::Matchers::ContainsSubstring(R"("original_chars":)"));
            CHECK_THAT(msg.content, Catch::Matchers::ContainsSubstring(R"("offload":)"));
            CHECK_THAT(msg.content, Catch::Matchers::ContainsSubstring(R"("reader":"read_tool_result")"));

            simdjson::dom::parser parser;
            simdjson::padded_string padded{msg.content};
            simdjson::dom::element document;
            REQUIRE(parser.parse(padded).get(document) == simdjson::SUCCESS);
            std::string_view reference;
            REQUIRE(document["offload"]["reference"].get(reference) == simdjson::SUCCESS);

            core::agent::ToolResultStore store(offload_root.path());
            std::string restored;
            std::uint64_t offset = 0;
            while (true) {
                const auto chunk = store.read({}, reference, offset, 4096);
                REQUIRE(chunk.has_value());
                restored += chunk->content;
                if (chunk->complete) break;
                REQUIRE(chunk->next_offset > offset);
                offset = chunk->next_offset;
            }
            const std::string expected =
                std::string(R"({"output":")") + std::string(120 * 1024, 'x') + R"("})";
            CHECK(restored == expected);
            CHECK_FALSE(store.read("another-session", reference, 0, 4096).has_value());
        }
    }
    REQUIRE(found);
}

TEST_CASE("Agent keeps error marker when compacting oversized error payloads", "[agent][tool-history]") {
    TempDir offload_root;
    auto provider = std::make_shared<ToolThenFinalProvider>(std::string(kLargeErrorToolName));
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<LargeErrorTool>());

    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context(),
        offload_root.path());
    send_and_wait(agent, "Run large error tool");

    const auto history = agent->get_history();
    bool found = false;
    for (const auto& msg : history) {
        if (msg.role == "tool" && msg.name == kLargeErrorToolName) {
            found = true;
            CHECK(msg.content.size() < 40 * 1024);
            CHECK_THAT(msg.content, Catch::Matchers::ContainsSubstring(R"("error":"Tool output truncated for history)"));
            CHECK_THAT(msg.content, Catch::Matchers::ContainsSubstring(R"("offload":)"));
        }
    }
    REQUIRE(found);
}

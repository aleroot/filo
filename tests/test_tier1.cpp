/**
 * @file test_tier1.cpp
 * @brief Unit tests for TIER 1 features:
 *   - DeleteFileTool / MoveFileTool
 *   - BudgetTracker (token accounting + cost estimation)
 *   - PermissionGate (promise-based approval)
 *   - Agent loop breaker (consecutive failure detection)
 *   - McpClientSession helpers (parse_tools_list, JSON-RPC builders)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/tools/DeleteFileTool.hpp"
#include "core/tools/MoveFileTool.hpp"
#include "core/budget/BudgetTracker.hpp"
#include "core/agent/PermissionGate.hpp"
#include "core/mcp/McpClientSession.hpp"
#include "core/llm/Models.hpp"

#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

using namespace core::tools;
using namespace core::budget;
using namespace core::agent;
using namespace core::mcp;

namespace fs = std::filesystem;

// ============================================================================
// Helpers
// ============================================================================

static void write_file(const std::string& path, const std::string& content) {
    std::ofstream ofs(path);
    ofs << content;
}

static bool file_contains(const std::string& path, const std::string& text) {
    std::ifstream ifs(path);
    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    return content.find(text) != std::string::npos;
}

// ============================================================================
// DeleteFileTool
// ============================================================================

TEST_CASE("DeleteFileTool — deletes an existing file", "[tier1][tools][delete]") {
    const std::string path = "tier1_del_test.txt";
    write_file(path, "temporary content");
    REQUIRE(fs::exists(path));

    DeleteFileTool tool;
    std::string result = tool.execute(R"({"file_path":")" + path + R"("})");

    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"success\":true"));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring(path));
    REQUIRE_FALSE(fs::exists(path));
}

TEST_CASE("DeleteFileTool — returns error for missing file", "[tier1][tools][delete]") {
    DeleteFileTool tool;
    std::string result = tool.execute(R"({"file_path":"/nonexistent/path/filo_test.txt"})");
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"error\""));
}

TEST_CASE("DeleteFileTool — returns error for missing argument", "[tier1][tools][delete]") {
    DeleteFileTool tool;
    std::string result = tool.execute(R"({})");
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"error\""));
}

TEST_CASE("DeleteFileTool — returns error for invalid JSON", "[tier1][tools][delete]") {
    DeleteFileTool tool;
    std::string result = tool.execute("not json");
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"error\""));
}

// ============================================================================
// MoveFileTool
// ============================================================================

TEST_CASE("MoveFileTool — renames a file", "[tier1][tools][move]") {
    const std::string src = "tier1_move_src.txt";
    const std::string dst = "tier1_move_dst.txt";
    write_file(src, "move me");
    REQUIRE(fs::exists(src));

    MoveFileTool tool;
    std::string result = tool.execute(
        R"({"source":")" + src + R"(","destination":")" + dst + R"("})");

    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"success\":true"));
    REQUIRE_FALSE(fs::exists(src));
    REQUIRE(fs::exists(dst));
    REQUIRE(file_contains(dst, "move me"));

    fs::remove(dst);
}

TEST_CASE("MoveFileTool — returns error for missing source", "[tier1][tools][move]") {
    MoveFileTool tool;
    std::string result = tool.execute(
        R"({"source":"/no/such/file.txt","destination":"/tmp/dst.txt"})");
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"error\""));
}

TEST_CASE("MoveFileTool — returns error for missing arguments", "[tier1][tools][move]") {
    MoveFileTool tool;
    REQUIRE_THAT(tool.execute(R"({"source":"x"})"),
                 Catch::Matchers::ContainsSubstring("\"error\""));
    REQUIRE_THAT(tool.execute(R"({})"),
                 Catch::Matchers::ContainsSubstring("\"error\""));
}

TEST_CASE("MoveFileTool — moves file into subdirectory", "[tier1][tools][move]") {
    const std::string src     = "tier1_move_into_subdir.txt";
    const std::string sub_dir = "tier1_subdir_move_test";
    const std::string dst     = sub_dir + "/moved.txt";

    write_file(src, "subdir target");
    fs::create_directory(sub_dir);

    MoveFileTool tool;
    std::string result = tool.execute(
        R"({"source":")" + src + R"(","destination":")" + dst + R"("})");

    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"success\":true"));
    REQUIRE(fs::exists(dst));
    REQUIRE_FALSE(fs::exists(src));

    fs::remove_all(sub_dir);
}

// ============================================================================
// BudgetTracker
// ============================================================================

TEST_CASE("BudgetTracker — records and accumulates usage", "[tier1][budget]") {
    // Use a fresh BudgetTracker state
    auto& bt = BudgetTracker::get_instance();
    bt.reset_session();

    core::llm::TokenUsage u1{ .prompt_tokens = 100, .completion_tokens = 50, .total_tokens = 150 };
    core::llm::TokenUsage u2{ .prompt_tokens = 200, .completion_tokens = 80, .total_tokens = 280 };

    bt.record(u1);
    bt.record(u2);

    auto total = bt.session_total();
    REQUIRE(total.prompt_tokens     == 300);
    REQUIRE(total.completion_tokens == 130);
    REQUIRE(total.total_tokens      == 430);
}

TEST_CASE("BudgetTracker — reset_session clears all counters", "[tier1][budget]") {
    auto& bt = BudgetTracker::get_instance();
    bt.record({ .prompt_tokens = 999, .completion_tokens = 999, .total_tokens = 1998 });
    bt.reset_session();

    auto total = bt.session_total();
    REQUIRE(total.prompt_tokens     == 0);
    REQUIRE(total.completion_tokens == 0);
    REQUIRE(total.total_tokens      == 0);
    REQUIRE(bt.session_cost_usd()   == 0.0);
}

TEST_CASE("BudgetTracker — cost estimation for known models", "[tier1][budget]") {
    auto& bt = BudgetTracker::get_instance();
    bt.reset_session();

    // 1M prompt tokens + 1M completion tokens with grok-code-fast-1 ($0.20 + $1.50)
    bt.record({ .prompt_tokens = 1'000'000, .completion_tokens = 1'000'000 }, "grok-code-fast-1");
    double cost = bt.session_cost_usd();
    REQUIRE(cost == Catch::Approx(1.70).epsilon(0.001));

    bt.reset_session();
}

TEST_CASE("BudgetTracker — status_string is human-readable", "[tier1][budget]") {
    auto& bt = BudgetTracker::get_instance();
    bt.reset_session();

    // Empty → empty string
    REQUIRE(bt.status_string().empty());

    bt.record({ .prompt_tokens = 1500, .completion_tokens = 750 });
    std::string s = bt.status_string();
    REQUIRE_THAT(s, Catch::Matchers::ContainsSubstring("↑"));
    REQUIRE_THAT(s, Catch::Matchers::ContainsSubstring("↓"));

    bt.reset_session();
}

TEST_CASE("BudgetTracker — thread-safe concurrent recording", "[tier1][budget]") {
    auto& bt = BudgetTracker::get_instance();
    bt.reset_session();

    constexpr int N = 100;
    std::vector<std::thread> threads;
    threads.reserve(N);
    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&bt]() {
            bt.record({ .prompt_tokens = 10, .completion_tokens = 5 });
        });
    }
    for (auto& t : threads) t.join();

    auto total = bt.session_total();
    REQUIRE(total.prompt_tokens     == N * 10);
    REQUIRE(total.completion_tokens == N * 5);

    bt.reset_session();
}

// ============================================================================
// TokenUsage operators
// ============================================================================

TEST_CASE("TokenUsage — addition operators", "[tier1][models]") {
    core::llm::TokenUsage a{ .prompt_tokens = 100, .completion_tokens = 50,  .total_tokens = 150 };
    core::llm::TokenUsage b{ .prompt_tokens = 200, .completion_tokens = 100, .total_tokens = 300 };

    auto c = a + b;
    REQUIRE(c.prompt_tokens     == 300);
    REQUIRE(c.completion_tokens == 150);
    REQUIRE(c.total_tokens      == 450);

    a += b;
    REQUIRE(a.prompt_tokens     == 300);
    REQUIRE(a.completion_tokens == 150);
}

TEST_CASE("TokenUsage — has_data() returns correct results", "[tier1][models]") {
    core::llm::TokenUsage zero{};
    REQUIRE_FALSE(zero.has_data());

    core::llm::TokenUsage with_prompt{ .prompt_tokens = 10 };
    REQUIRE(with_prompt.has_data());
}

// ============================================================================
// rates_for_model — cost table
// ============================================================================

TEST_CASE("rates_for_model — known models return correct rates", "[tier1][budget]") {
    auto r_gpt54 = rates_for_model("gpt-5.4");
    REQUIRE(r_gpt54.input_per_m  == Catch::Approx(2.50));
    REQUIRE(r_gpt54.output_per_m == Catch::Approx(10.00));

    auto r_code = rates_for_model("grok-code-fast-1");
    REQUIRE(r_code.input_per_m  == Catch::Approx(0.20));
    REQUIRE(r_code.output_per_m == Catch::Approx(1.50));

    auto r_mini = rates_for_model("grok-3-mini");
    REQUIRE(r_mini.input_per_m  == Catch::Approx(0.30));
    REQUIRE(r_mini.output_per_m == Catch::Approx(0.50));

    auto r_fast = rates_for_model("grok-4-fast-reasoning");
    REQUIRE(r_fast.input_per_m  == Catch::Approx(0.20));
    REQUIRE(r_fast.output_per_m == Catch::Approx(0.50));

    // Claude 4.6 Opus: $5.00/$25.00 per MTok (reduced from earlier 4.0 pricing)
    auto r_opus = rates_for_model("claude-opus-4-6");
    REQUIRE(r_opus.input_per_m  == Catch::Approx(5.00));
    REQUIRE(r_opus.output_per_m == Catch::Approx(25.00));

    auto r_sonnet = rates_for_model("claude-sonnet-4-6");
    REQUIRE(r_sonnet.input_per_m  == Catch::Approx(3.00));
    REQUIRE(r_sonnet.output_per_m == Catch::Approx(15.00));

    auto r_unknown = rates_for_model("some-unknown-model-xyz");
    REQUIRE(r_unknown.input_per_m  == Catch::Approx(2.00));
    REQUIRE(r_unknown.output_per_m == Catch::Approx(8.00));
}

// ============================================================================
// PermissionGate
// ============================================================================

TEST_CASE("PermissionGate — approve resolves to true", "[tier1][permission]") {
    auto& gate = PermissionGate::get_instance();

    bool result = false;
    std::thread worker([&]() {
        result = gate.request("write_file", R"({"file_path":"test.cpp"})");
    });

    // Give the worker a moment to block on the promise
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    REQUIRE(gate.has_pending());

    gate.resolve(true);
    worker.join();
    REQUIRE(result == true);
}

TEST_CASE("PermissionGate — deny resolves to false", "[tier1][permission]") {
    auto& gate = PermissionGate::get_instance();

    bool result = true;
    std::thread worker([&]() {
        result = gate.request("run_terminal_command", R"({"command":"rm -rf /"})");
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    REQUIRE(gate.has_pending());

    auto peeked = gate.peek_pending();
    REQUIRE(peeked.has_value());
    REQUIRE(peeked->tool_name == "run_terminal_command");

    gate.resolve(false);
    worker.join();
    REQUIRE(result == false);
}

TEST_CASE("PermissionGate — args_preview is truncated at 300 chars", "[tier1][permission]") {
    auto& gate = PermissionGate::get_instance();

    std::string long_args(500, 'x');
    std::thread worker([&]() {
        [[maybe_unused]] const bool ignored = gate.request("write_file", long_args);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    REQUIRE(gate.has_pending());

    auto peeked = gate.peek_pending();
    REQUIRE(peeked.has_value());
    REQUIRE(peeked->args_preview.size() <= 304);  // 300 + "…"

    gate.resolve(false);
    worker.join();
}

// ============================================================================
// make_allow_key / make_allow_label helpers
// ============================================================================

#include "tui/Conversation.hpp"

TEST_CASE("make_allow_key — run_terminal_command extracts first word", "[tier1][permission][allowkey]") {
    using tui::make_allow_key;
    CHECK(make_allow_key("run_terminal_command", R"({"command":"git status --porcelain"})")
          == "run_terminal_command:git");
    CHECK(make_allow_key("run_terminal_command", R"({"command":"npm run build"})")
          == "run_terminal_command:npm");
    CHECK(make_allow_key("run_terminal_command", R"({"command":"python3 script.py","working_dir":"/tmp"})")
          == "run_terminal_command:python3");
}

TEST_CASE("make_allow_key — run_terminal_command single-word command", "[tier1][permission][allowkey]") {
    using tui::make_allow_key;
    CHECK(make_allow_key("run_terminal_command", R"({"command":"ls"})")
          == "run_terminal_command:ls");
}

TEST_CASE("make_allow_key — file tools key on tool name", "[tier1][permission][allowkey]") {
    using tui::make_allow_key;
    CHECK(make_allow_key("write_file",    R"({"file_path":"a.txt","content":"x"})") == "write_file");
    CHECK(make_allow_key("apply_patch",   R"({"patch":"---"})") == "apply_patch");
    CHECK(make_allow_key("replace",       R"({"file_path":"a.txt"})") == "replace");
    CHECK(make_allow_key("delete_file",   R"({"path":"a.txt"})") == "delete_file");
    CHECK(make_allow_key("move_file",     R"({"source_path":"a","destination_path":"b"})") == "move_file");
}

TEST_CASE("make_allow_key — unknown tool passthrough", "[tier1][permission][allowkey]") {
    using tui::make_allow_key;
    CHECK(make_allow_key("custom_tool", R"({})") == "custom_tool");
}

TEST_CASE("make_allow_label — run_terminal_command shows program name", "[tier1][permission][allowlabel]") {
    using tui::make_allow_label;
    CHECK(make_allow_label("run_terminal_command", R"({"command":"git status"})") == "'git' commands");
    CHECK(make_allow_label("run_terminal_command", R"({"command":"make all"})") == "'make' commands");
}

TEST_CASE("make_allow_label — file tools have descriptive labels", "[tier1][permission][allowlabel]") {
    using tui::make_allow_label;
    CHECK(make_allow_label("write_file",     "{}") == "file modifications");
    CHECK(make_allow_label("apply_patch",    "{}") == "file modifications");
    CHECK(make_allow_label("replace",        "{}") == "file modifications");
    CHECK(make_allow_label("replace_in_file","{}") == "file modifications");
    CHECK(make_allow_label("delete_file",    "{}") == "file deletions");
    CHECK(make_allow_label("move_file",      "{}") == "file moves");
}

// ============================================================================
// needs_permission helper
// ============================================================================

TEST_CASE("needs_permission — dangerous tools identified correctly", "[tier1][permission]") {
    REQUIRE(needs_permission("write_file"));
    REQUIRE(needs_permission("apply_patch"));
    REQUIRE(needs_permission("run_terminal_command"));
    REQUIRE(needs_permission("replace"));
    REQUIRE(needs_permission("replace_in_file"));
    REQUIRE(needs_permission("delete_file"));
    REQUIRE(needs_permission("move_file"));

    REQUIRE_FALSE(needs_permission("read_file"));
    REQUIRE_FALSE(needs_permission("list_directory"));
    REQUIRE_FALSE(needs_permission("grep_search"));
    REQUIRE_FALSE(needs_permission("file_search"));
    REQUIRE_FALSE(needs_permission("get_time"));
}

// ============================================================================
// MCP: parse_tools_list
// ============================================================================

TEST_CASE("parse_tools_list — parses a simple tools/list response", "[tier1][mcp]") {
    constexpr std::string_view json = R"({
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file from the filesystem.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file path to read."
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "list_directory",
                "description": "List directory contents.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path":      {"type": "string", "description": "Dir path."},
                        "recursive": {"type": "boolean", "description": "Recurse?"}
                    },
                    "required": ["path"]
                }
            }
        ]
    })";

    auto tools = parse_tools_list(json);
    REQUIRE(tools.size() == 2);

    REQUIRE(tools[0].name == "read_file");
    REQUIRE_THAT(tools[0].description, Catch::Matchers::ContainsSubstring("Read a file"));
    REQUIRE(tools[0].parameters.size() == 1);
    REQUIRE(tools[0].parameters[0].name == "path");
    REQUIRE(tools[0].parameters[0].required == true);

    REQUIRE(tools[1].name == "list_directory");
    REQUIRE(tools[1].parameters.size() == 2);
    // "path" is required, "recursive" is not
    const auto& params = tools[1].parameters;
    auto path_it = std::ranges::find_if(params, [](const auto& p){ return p.name == "path"; });
    auto rec_it  = std::ranges::find_if(params, [](const auto& p){ return p.name == "recursive"; });
    REQUIRE(path_it != params.end());
    REQUIRE(rec_it  != params.end());
    REQUIRE(path_it->required == true);
    REQUIRE(rec_it->required  == false);
}

TEST_CASE("parse_tools_list — returns empty vector on malformed JSON", "[tier1][mcp]") {
    REQUIRE(parse_tools_list("not json").empty());
    REQUIRE(parse_tools_list(R"({})").empty());
    REQUIRE(parse_tools_list(R"({"tools":[]})").empty());
}

TEST_CASE("parse_tools_list — skips tools without a name", "[tier1][mcp]") {
    constexpr std::string_view json = R"({
        "tools": [
            {"description": "no name here", "inputSchema": {}},
            {"name": "valid_tool", "description": "has a name", "inputSchema": {}}
        ]
    })";
    auto tools = parse_tools_list(json);
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].name == "valid_tool");
}

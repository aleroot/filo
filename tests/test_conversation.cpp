/**
 * @file test_conversation.cpp
 * @brief Comprehensive unit tests for tui/Conversation (Gemini CLI Parity).
 *
 * Covers:
 *  - Message factory functions (User, Assistant, Info, Warning, Error, ToolGroup)
 *  - ToolActivity creation and status management
 *  - summarize_tool_arguments() — arg-summary extraction
 *  - summarize_tool_result() — JSON result classification
 *  - Tool status helpers (color, icon, label, spinner)
 *  - render_history_panel() — smoke tests for all message types
 *  - Animation frame cycling
 *  - ConversationState (history separation)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/Conversation.hpp"
#include "tui/TuiTheme.hpp"

#include <ftxui/screen/screen.hpp>

using namespace tui;
using Catch::Matchers::ContainsSubstring;

namespace {
std::string strip_ansi(std::string_view input) {
    std::string out;
    out.reserve(input.size());

    for (std::size_t i = 0; i < input.size();) {
        if (input[i] == '\x1b' && i + 1 < input.size() && input[i + 1] == '[') {
            i += 2;
            while (i < input.size()) {
                const char ch = input[i++];
                if (ch >= '@' && ch <= '~') {
                    break;
                }
            }
            continue;
        }

        out.push_back(input[i]);
        ++i;
    }

    return out;
}
} // namespace

// ============================================================================
// Message Factory Tests
// ============================================================================

TEST_CASE("make_user_message — basic creation", "[tui][conversation][factory]") {
    auto msg = make_user_message("Hello, world!", "12:00:00");
    REQUIRE(msg.type == MessageType::User);
    REQUIRE(msg.text == "Hello, world!");
    REQUIRE(msg.timestamp == "12:00:00");
    REQUIRE(!msg.id.empty());
}

TEST_CASE("make_assistant_message — pending state", "[tui][conversation][factory]") {
    auto msg = make_assistant_message("Response text", "12:01:00", true);
    REQUIRE(msg.type == MessageType::Assistant);
    REQUIRE(msg.text == "Response text");
    REQUIRE(msg.timestamp == "12:01:00");
    REQUIRE(msg.pending == true);
    REQUIRE(msg.thinking == true);
}

TEST_CASE("make_assistant_message — completed state", "[tui][conversation][factory]") {
    auto msg = make_assistant_message("Done", "12:02:00", false);
    REQUIRE(msg.pending == false);
    REQUIRE(msg.thinking == false);
}

TEST_CASE("make_info_message — basic creation", "[tui][conversation][factory]") {
    auto msg = make_info_message("File saved successfully");
    REQUIRE(msg.type == MessageType::Info);
    REQUIRE(msg.text == "File saved successfully");
    REQUIRE(msg.margin_top == 1);
}

TEST_CASE("make_info_message — with secondary text", "[tui][conversation][factory]") {
    auto msg = make_info_message("Auto-approved", "always-allow list");
    REQUIRE(msg.secondary_text == "always-allow list");
}

TEST_CASE("make_warning_message — basic creation", "[tui][conversation][factory]") {
    auto msg = make_warning_message("Context window nearly full");
    REQUIRE(msg.type == MessageType::Warning);
    REQUIRE(msg.text == "Context window nearly full");
}

TEST_CASE("make_error_message — basic creation", "[tui][conversation][factory]") {
    auto msg = make_error_message("Failed to save file");
    REQUIRE(msg.type == MessageType::Error);
    REQUIRE(msg.text == "Failed to save file");
}

TEST_CASE("make_system_message — basic creation", "[tui][conversation][factory]") {
    auto msg = make_system_message("System notification");
    REQUIRE(msg.type == MessageType::System);
    REQUIRE(msg.text == "System notification");
}

TEST_CASE("make_system_disclosure_message — basic creation", "[tui][conversation][factory]") {
    auto msg = make_system_disclosure_message(
        "Internal session rotated.",
        "Previous segment: old\nNew segment: new");
    REQUIRE(msg.type == MessageType::System);
    REQUIRE(msg.text == "Internal session rotated.");
    REQUIRE(msg.disclosure_text == "Previous segment: old\nNew segment: new");
}

TEST_CASE("append_ui_message — collapses repeated system disclosures",
          "[tui][conversation][factory]") {
    std::vector<UiMessage> messages;
    append_ui_message(messages, make_system_disclosure_message(
        "Internal session rotated.",
        "Previous segment: seg-a\nNew segment: seg-b\nReason: threshold-a"));
    append_ui_message(messages, make_system_disclosure_message(
        "Internal session rotated.",
        "Previous segment: seg-c\nNew segment: seg-d\nReason: threshold-b"));

    REQUIRE(messages.size() == 1);
    REQUIRE(messages[0].repeat_count == 2);
    REQUIRE(messages[0].disclosure_text == "Previous segment: seg-c\nNew segment: seg-d\nReason: threshold-b");
}

TEST_CASE("make_tool_group_message — with tools", "[tui][conversation][factory]") {
    std::vector<ToolActivity> tools;
    tools.push_back(make_tool_activity("id1", "read_file", "{}", "src/main.cpp"));
    tools.push_back(make_tool_activity("id2", "grep_search", "{}", "pattern in src/"));
    
    auto msg = make_tool_group_message(std::move(tools), true, true);
    REQUIRE(msg.type == MessageType::ToolGroup);
    REQUIRE(msg.tools.size() == 2);
    REQUIRE(msg.tool_group_border_top == true);
    REQUIRE(msg.tool_group_border_bottom == true);
}

// ============================================================================
// ToolActivity Factory and Status Tests
// ============================================================================

TEST_CASE("make_tool_activity — basic creation", "[tui][conversation][tool]") {
    auto tool = make_tool_activity("tc-123", "read_file", R"({"path":"main.cpp"})", "main.cpp");
    REQUIRE(tool.id == "tc-123");
    REQUIRE(tool.name == "read_file");
    REQUIRE(tool.description == "main.cpp");
    REQUIRE(tool.auto_approved == false);
    REQUIRE(tool.status == ToolActivity::Status::Pending);
}

TEST_CASE("ToolActivity status transitions", "[tui][conversation][tool]") {
    auto tool = make_tool_activity("id", "test_tool", "{}", "");
    REQUIRE(tool.status == ToolActivity::Status::Pending);
    
    tool.status = ToolActivity::Status::Executing;
    REQUIRE(tool.status == ToolActivity::Status::Executing);
    
    tool.status = ToolActivity::Status::Succeeded;
    REQUIRE(tool.status == ToolActivity::Status::Succeeded);
}

TEST_CASE("ToolActivity progress tracking", "[tui][conversation][tool]") {
    auto tool = make_tool_activity("id", "long_running", "{}", "");
    tool.progress = 50;
    tool.progress_total = 100;
    tool.progress_message = "Processing...";
    
    REQUIRE(tool.progress.value() == 50);
    REQUIRE(tool.progress_total.value() == 100);
    REQUIRE(tool.progress_message == "Processing...");
}

// ============================================================================
// Tool Status Helpers
// ============================================================================

TEST_CASE("tool_status_color — all states", "[tui][conversation][status]") {
    REQUIRE(tool_status_color(ToolActivity::Status::Pending) == ColorToolPending);
    REQUIRE(tool_status_color(ToolActivity::Status::Executing) == ColorYellowBright);
    REQUIRE(tool_status_color(ToolActivity::Status::Succeeded) == ColorToolDone);
    REQUIRE(tool_status_color(ToolActivity::Status::Failed) == ColorToolFail);
}

TEST_CASE("tool_status_icon — all states", "[tui][conversation][status]") {
    REQUIRE(!tool_status_icon(ToolActivity::Status::Pending).empty());
    REQUIRE(!tool_status_icon(ToolActivity::Status::Executing).empty());
    REQUIRE(!tool_status_icon(ToolActivity::Status::Succeeded).empty());
    REQUIRE(!tool_status_icon(ToolActivity::Status::Failed).empty());
    REQUIRE(!tool_status_icon(ToolActivity::Status::Denied).empty());
}

TEST_CASE("tool_status_label — all states", "[tui][conversation][status]") {
    REQUIRE(std::string(tool_status_label(ToolActivity::Status::Pending)) == "Pending");
    REQUIRE(std::string(tool_status_label(ToolActivity::Status::Executing)) == "Running");
    REQUIRE(std::string(tool_status_label(ToolActivity::Status::Succeeded)) == "Done");
    REQUIRE(std::string(tool_status_label(ToolActivity::Status::Failed)) == "Failed");
    REQUIRE(std::string(tool_status_label(ToolActivity::Status::Denied)) == "Denied");
}

TEST_CASE("tool_status_spinner — cycles correctly", "[tui][conversation][status]") {
    REQUIRE(tool_status_spinner(0) == tool_status_spinner(4));
    REQUIRE(tool_status_spinner(1) == tool_status_spinner(5));
    REQUIRE(tool_status_spinner(0) != tool_status_spinner(1));
}

// ============================================================================
// summarize_tool_arguments
// ============================================================================

TEST_CASE("summarize_tool_arguments — run_terminal_command", "[tui][conversation][args]") {
    const auto result = summarize_tool_arguments(
        "run_terminal_command", R"({"command":"ls -la","working_dir":"/tmp"})");
    REQUIRE_THAT(result, ContainsSubstring("ls -la"));
}

TEST_CASE("summarize_tool_arguments — move_file", "[tui][conversation][args]") {
    const auto result = summarize_tool_arguments(
        "move_file", R"({"source_path":"a.txt","destination_path":"b.txt"})");
    REQUIRE_THAT(result, ContainsSubstring("a.txt"));
    REQUIRE_THAT(result, ContainsSubstring("b.txt"));
}

TEST_CASE("summarize_tool_arguments — grep_search", "[tui][conversation][args]") {
    const auto result = summarize_tool_arguments(
        "grep_search", R"({"pattern":"foo","path":"/src"})");
    REQUIRE_THAT(result, ContainsSubstring("foo"));
    REQUIRE_THAT(result, ContainsSubstring("/src"));
}

TEST_CASE("summarize_tool_arguments — read_file", "[tui][conversation][args]") {
    const auto result = summarize_tool_arguments(
        "read_file", R"({"path":"/home/user/README.md"})");
    REQUIRE_THAT(result, ContainsSubstring("README.md"));
}

TEST_CASE("summarize_tool_arguments — apply_patch", "[tui][conversation][args]") {
    const auto result = summarize_tool_arguments(
        "apply_patch",
        R"({"patch":"--- src/main.cpp\n+++ src/main.cpp\n@@ -1,1 +1,1 @@\n-old\n+new"})");
    REQUIRE_FALSE(result.empty());
}

TEST_CASE("summarize_tool_arguments — invalid JSON", "[tui][conversation][args]") {
    const auto result = summarize_tool_arguments("any_tool", "not json at all");
    REQUIRE_FALSE(result.empty());
}

// ============================================================================
// summarize_tool_result
// ============================================================================

TEST_CASE("summarize_tool_result — output field → Succeeded", "[tui][conversation][result]") {
    const auto summary = summarize_tool_result(R"({"output":"hello world"})");
    REQUIRE(summary.state == ToolResultSummary::State::Succeeded);
    REQUIRE_THAT(summary.preview, ContainsSubstring("hello world"));
}

TEST_CASE("summarize_tool_result — error field → Failed", "[tui][conversation][result]") {
    const auto summary = summarize_tool_result(R"({"error":"something went wrong"})");
    REQUIRE(summary.state == ToolResultSummary::State::Failed);
    REQUIRE_FALSE(summary.preview.empty());
}

TEST_CASE("summarize_tool_result — success:true → done preview", "[tui][conversation][result]") {
    const auto summary = summarize_tool_result(R"({"success":true})");
    REQUIRE(summary.preview == "done");
}

TEST_CASE("summarize_tool_result — matches field", "[tui][conversation][result]") {
    const auto summary = summarize_tool_result(R"({"matches":"src/foo.cpp:12: match"})");
    REQUIRE_FALSE(summary.preview.empty());
}

TEST_CASE("summarize_tool_result — empty matches", "[tui][conversation][result]") {
    const auto summary = summarize_tool_result(R"({"matches":""})");
    REQUIRE(summary.preview == "no matches");
}

TEST_CASE("apply_tool_result — terminal command success captures exit code",
          "[tui][conversation][result]") {
    auto tool = make_tool_activity("t1", "run_terminal_command", "{}", "pwd");
    apply_tool_result(tool, R"({"output":"ok\n","exit_code":0})");
    REQUIRE(tool.status == ToolActivity::Status::Succeeded);
    REQUIRE(tool.result.exit_code.has_value());
    REQUIRE(*tool.result.exit_code == 0);
    REQUIRE(tool.result.summary == "ok\n");
    REQUIRE(tool.result.truncated == false);
}

TEST_CASE("apply_tool_result — terminal command failure is surfaced",
          "[tui][conversation][result]") {
    auto tool = make_tool_activity("t2", "run_terminal_command", "{}", "npm test");
    apply_tool_result(tool, R"({"output":"failed\n","exit_code":2})");
    REQUIRE(tool.status == ToolActivity::Status::Failed);
    REQUIRE(tool.result.exit_code.has_value());
    REQUIRE(*tool.result.exit_code == 2);
    REQUIRE(tool.result.summary == "failed\n");
}

TEST_CASE("apply_tool_result — output truncation marker is detected",
          "[tui][conversation][result]") {
    auto tool = make_tool_activity("t3", "run_terminal_command", "{}", "cat big.log");
    apply_tool_result(
        tool,
        R"({"output":"line 1\n... [OUTPUT TRUNCATED AT 4MB] ...\n","exit_code":-1})");
    REQUIRE(tool.status == ToolActivity::Status::Failed);
    REQUIRE(tool.result.truncated == true);
}

// ============================================================================
// Animation Helpers
// ============================================================================

TEST_CASE("spinner_frame — cycles through all frames", "[tui][conversation][animation]") {
    REQUIRE(spinner_frame(0) == spinner_frame(4));
    REQUIRE(spinner_frame(1) == spinner_frame(5));
    REQUIRE(spinner_frame(0) != spinner_frame(1));
    REQUIRE_FALSE(spinner_frame(2).empty());
}

TEST_CASE("thinking_pulse_frame — cycles through all frames", "[tui][conversation][animation]") {
    REQUIRE(thinking_pulse_frame(0) == thinking_pulse_frame(6));
    REQUIRE(thinking_pulse_frame(1) == thinking_pulse_frame(7));
    REQUIRE_FALSE(thinking_pulse_frame(3).empty());
}

TEST_CASE("message_uses_animation — assistant thinking requires spinner", "[tui][conversation][animation]") {
    auto message = make_assistant_message("", "", true);
    REQUIRE(message_uses_animation(message, true));
    REQUIRE_FALSE(message_uses_animation(message, false));
}

TEST_CASE("message_uses_animation — executing tool animates", "[tui][conversation][animation]") {
    UiMessage message = make_tool_group_message({});
    auto tool = make_tool_activity("id", "grep_search", "{}", "search");
    tool.status = ToolActivity::Status::Executing;
    message.tools.push_back(std::move(tool));
    REQUIRE(message_uses_animation(message, true));
}

TEST_CASE("message_uses_animation — completed tool without pending assistant is static", "[tui][conversation][animation]") {
    auto message = make_assistant_message("", "", false);
    auto tool = make_tool_activity("id", "read_file", "{}", "file");
    tool.status = ToolActivity::Status::Succeeded;
    message.tools.push_back(std::move(tool));
    REQUIRE_FALSE(message_uses_animation(message, true));
}

TEST_CASE("conversation_uses_animation — any animated message enables ticker", "[tui][conversation][animation]") {
    std::vector<UiMessage> messages;
    messages.push_back(make_system_message("idle"));
    auto assistant = make_assistant_message("", "", true);
    messages.push_back(std::move(assistant));
    REQUIRE(conversation_uses_animation(messages, true));
    REQUIRE_FALSE(conversation_uses_animation(messages, false));
}

// ============================================================================
// render_history_panel — Smoke Tests
// ============================================================================

TEST_CASE("render_history_panel — empty message list", "[tui][conversation][render]") {
    REQUIRE_NOTHROW(render_history_panel({}, 0));
}

TEST_CASE("render_history_panel — user message", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    messages.push_back(make_user_message("Hello, filo!", "12:00:00"));
    REQUIRE_NOTHROW(render_history_panel(messages, 0));
}

TEST_CASE("render_history_panel — assistant message", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    messages.push_back(make_assistant_message("Hello! How can I help?", "12:01:00", false));
    REQUIRE_NOTHROW(render_history_panel(messages, 0));
}

TEST_CASE("render_history_panel — info message", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    messages.push_back(make_info_message("File saved"));
    REQUIRE_NOTHROW(render_history_panel(messages, 0));
}

TEST_CASE("render_history_panel — warning message", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    messages.push_back(make_warning_message("Context nearly full"));
    REQUIRE_NOTHROW(render_history_panel(messages, 0));
}

TEST_CASE("render_history_panel — error message", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    messages.push_back(make_error_message("Failed to save"));
    REQUIRE_NOTHROW(render_history_panel(messages, 0));
}

TEST_CASE("render_history_panel — system message", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    messages.push_back(make_system_message("System notification"));
    REQUIRE_NOTHROW(render_history_panel(messages, 0));
}

TEST_CASE("render_history_panel — system disclosure is compact by default",
          "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    messages.push_back(make_system_disclosure_message(
        "Internal session rotated to keep the working set lean (context preserved).",
        "Previous segment: seg-a\nNew segment: seg-b\nReason: threshold"));

    auto panel = render_history_panel(messages, 0);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(160),
                                        ftxui::Dimension::Fit(panel));
    ftxui::Render(screen, panel);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, ContainsSubstring("Internal session rotated"));
    REQUIRE_THAT(output, ContainsSubstring("click or Ctrl+O for details"));
    REQUIRE_THAT(output, !ContainsSubstring("Previous segment: seg-a"));
}

TEST_CASE("render_history_panel — system disclosure expands with option",
          "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    messages.push_back(make_system_disclosure_message(
        "Internal session rotated to keep the working set lean (context preserved).",
        "Previous segment: seg-a\nNew segment: seg-b\nReason: threshold"));

    auto panel = render_history_panel(
        messages,
        0,
        ConversationRenderOptions{
            .expand_system_details = true,
        });
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(160),
                                        ftxui::Dimension::Fit(panel));
    ftxui::Render(screen, panel);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, ContainsSubstring("Internal session rotated"));
    REQUIRE_THAT(output, ContainsSubstring("Previous segment: seg-a"));
    REQUIRE_THAT(output, ContainsSubstring("New segment: seg-b"));
    REQUIRE_THAT(output, ContainsSubstring("Reason: threshold"));
}

TEST_CASE("render_history_panel — repeated system disclosure shows counter and latest hint",
          "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    auto repeated = make_system_disclosure_message(
        "Internal session rotated to keep the working set lean (context preserved).",
        "Previous segment: seg-c\nNew segment: seg-d\nReason: threshold-b");
    repeated.repeat_count = 3;
    messages.push_back(std::move(repeated));

    auto compact_panel = render_history_panel(messages, 0);
    auto compact_screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(160),
                                                ftxui::Dimension::Fit(compact_panel));
    ftxui::Render(compact_screen, compact_panel);
    const auto compact_output = strip_ansi(compact_screen.ToString());
    REQUIRE_THAT(compact_output, ContainsSubstring("(x3)"));
    REQUIRE_THAT(compact_output, ContainsSubstring("click or Ctrl+O for details"));

    auto expanded_panel = render_history_panel(
        messages,
        0,
        ConversationRenderOptions{
            .expand_system_details = true,
        });
    auto expanded_screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(160),
                                                 ftxui::Dimension::Fit(expanded_panel));
    ftxui::Render(expanded_screen, expanded_panel);
    const auto expanded_output = strip_ansi(expanded_screen.ToString());
    REQUIRE_THAT(expanded_output, ContainsSubstring("Collapsed 3 repeated events. Showing latest details."));
    REQUIRE_THAT(expanded_output, ContainsSubstring("Previous segment: seg-c"));
}

TEST_CASE("render_history_panel — tool group", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    std::vector<ToolActivity> tools;
    tools.push_back(make_tool_activity("t1", "read_file", "{}", "main.cpp"));
    tools.push_back(make_tool_activity("t2", "grep_search", "{}", "pattern"));
    messages.push_back(make_tool_group_message(std::move(tools)));
    REQUIRE_NOTHROW(render_history_panel(messages, 0));
}

TEST_CASE("render_history_panel — assistant with tools", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("Let me check...", "", true);
    msg.tools.push_back(make_tool_activity("t1", "list_directory", "{}", "."));
    messages.push_back(std::move(msg));
    REQUIRE_NOTHROW(render_history_panel(messages, 0));
}

TEST_CASE("render_history_panel — assistant tool with auto-approved badge",
          "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("Running tools", "", true);
    auto tool = make_tool_activity("t1", "run_terminal_command", R"({"command":"pwd"})", "cmd: pwd");
    tool.auto_approved = true;
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));
    REQUIRE_NOTHROW(render_history_panel(messages, 0));
}

TEST_CASE("render_history_panel — large tool output remains renderable",
          "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("Summarized answer", "", false);
    auto tool = make_tool_activity("t2", "run_terminal_command", R"({"command":"cat big.log"})", "cmd: cat big.log");
    tool.result.summary =
        "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\n"
        "line 9\nline 10\nline 11\nline 12\nline 13\nline 14\nline 15\n";
    tool.status = ToolActivity::Status::Succeeded;
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));
    REQUIRE_NOTHROW(render_history_panel(messages, 0));
}

TEST_CASE("render_history_panel — expanded tool output mode",
          "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("Expanded logs", "", false);
    auto tool = make_tool_activity("t3", "run_terminal_command", R"({"command":"tail -n 50 app.log"})", "cmd: tail -n 50 app.log");
    tool.result.summary = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\n";
    tool.status = ToolActivity::Status::Succeeded;
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    REQUIRE_NOTHROW(render_history_panel(
        messages,
        0,
        ConversationRenderOptions{
            .expand_tool_results = true,
            .tool_result_preview_max_lines = 3,
        }));
}

TEST_CASE("render_history_panel — timestamps hidden", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    messages.push_back(make_user_message("Hello", "12:00:00"));
    REQUIRE_NOTHROW(render_history_panel(
        messages, 0, ConversationRenderOptions{.show_timestamps = false}));
}

TEST_CASE("render_history_panel — spinner hidden", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", true);
    msg.thinking = true;
    messages.push_back(std::move(msg));
    REQUIRE_NOTHROW(render_history_panel(
        messages, 0, ConversationRenderOptions{.show_spinner = false}));
}

TEST_CASE("render_history_panel — scroll positions", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    for (int i = 0; i < 10; ++i) {
        messages.push_back(make_user_message("Message " + std::to_string(i), ""));
    }
    for (float pos : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
        REQUIRE_NOTHROW(render_history_panel(
            messages, 0, ConversationRenderOptions{.scroll_pos = pos}));
    }
}

TEST_CASE("render_history_panel — thinking animation", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", true);
    msg.thinking = true;
    messages.push_back(std::move(msg));
    
    for (std::size_t tick = 0; tick < 32; ++tick) {
        REQUIRE_NOTHROW(render_history_panel(messages, tick));
    }
}

TEST_CASE("render_history_panel — mixed conversation", "[tui][conversation][render]") {
    std::vector<UiMessage> messages;
    
    // User message
    messages.push_back(make_user_message("List files", "09:00:00"));
    
    // Assistant with tools and response
    auto asst = make_assistant_message("Here are the files:", "", false);
    asst.tools.push_back(make_tool_activity("t1", "list_directory", "{}", "."));
    asst.show_lightbulb = true;
    messages.push_back(std::move(asst));
    
    // Status messages
    messages.push_back(make_info_message("Auto-saved"));
    messages.push_back(make_warning_message("Low disk space"));
    
    REQUIRE_NOTHROW(render_history_panel(messages, 0));
    REQUIRE_NOTHROW(render_history_panel(messages, 31));
}

// ============================================================================
// ConversationState Tests (History Separation)
// ============================================================================

TEST_CASE("ConversationState — empty on creation", "[tui][conversation][state]") {
    ConversationState state;
    REQUIRE(state.history.empty());
    REQUIRE(state.pending.empty());
    REQUIRE(!state.has_active_turn());
}

TEST_CASE("ConversationState — all_messages combines history and pending", "[tui][conversation][state]") {
    ConversationState state;
    state.history.push_back(make_user_message("History msg", ""));
    state.pending.push_back(make_assistant_message("Pending msg", "", true));
    
    auto all = state.all_messages();
    REQUIRE(all.size() == 2);
}

TEST_CASE("ConversationState — commit_pending moves messages", "[tui][conversation][state]") {
    ConversationState state;
    state.pending.push_back(make_user_message("User", ""));
    state.pending.push_back(make_assistant_message("Assistant", "", false));
    
    state.commit_pending();
    
    REQUIRE(state.history.size() == 2);
    REQUIRE(state.pending.empty());
}

TEST_CASE("ConversationState — clear removes all", "[tui][conversation][state]") {
    ConversationState state;
    state.history.push_back(make_user_message("History", ""));
    state.pending.push_back(make_assistant_message("Pending", "", true));
    
    state.clear();
    
    REQUIRE(state.history.empty());
    REQUIRE(state.pending.empty());
}

TEST_CASE("ConversationState — has_active_turn detects pending assistant", "[tui][conversation][state]") {
    ConversationState state;
    REQUIRE(!state.has_active_turn());
    
    state.pending.push_back(make_assistant_message("", "", true));
    REQUIRE(state.has_active_turn());
    
    state.pending.clear();
    state.pending.push_back(make_assistant_message("", "", false));
    REQUIRE(!state.has_active_turn());
}

// ============================================================================
// find_tool_activity Tests
// ============================================================================

TEST_CASE("find_tool_activity — finds existing tool", "[tui][conversation][tool]") {
    UiMessage msg = make_assistant_message("", "", true);
    msg.tools.push_back(make_tool_activity("id1", "tool1", "{}", ""));
    msg.tools.push_back(make_tool_activity("id2", "tool2", "{}", ""));
    
    auto* tool = find_tool_activity(msg, "id2");
    REQUIRE(tool != nullptr);
    REQUIRE(tool->name == "tool2");
}

TEST_CASE("find_tool_activity — returns nullptr for missing", "[tui][conversation][tool]") {
    UiMessage msg = make_assistant_message("", "", true);
    msg.tools.push_back(make_tool_activity("id1", "tool1", "{}", ""));
    
    auto* tool = find_tool_activity(msg, "nonexistent");
    REQUIRE(tool == nullptr);
}

TEST_CASE("find_tool_activity — const version", "[tui][conversation][tool]") {
    UiMessage msg = make_assistant_message("", "", true);
    msg.tools.push_back(make_tool_activity("id1", "tool1", "{}", ""));
    
    const UiMessage& const_msg = msg;
    const auto* tool = find_tool_activity(const_msg, "id1");
    REQUIRE(tool != nullptr);
    REQUIRE(tool->name == "tool1");
}

// ============================================================================
// Allow-list Helpers
// ============================================================================

TEST_CASE("make_allow_key — run_terminal_command extracts program", "[tui][conversation][allow]") {
    auto key = make_allow_key("run_terminal_command", R"({"command":"git status"})");
    REQUIRE_THAT(key, ContainsSubstring("git"));
}

TEST_CASE("make_allow_key — other tools use name", "[tui][conversation][allow]") {
    auto key = make_allow_key("read_file", R"({"path":"test.cpp"})");
    REQUIRE(key == "read_file");
}

TEST_CASE("make_allow_label — creates readable label", "[tui][conversation][allow]") {
    auto label = make_allow_label("run_terminal_command", R"({"command":"npm install"})");
    REQUIRE_THAT(label, ContainsSubstring("npm"));
}

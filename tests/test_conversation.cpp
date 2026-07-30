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
#include "tui/RewindPicker.hpp"
#include "tui/TuiTheme.hpp"

#include <ftxui/dom/node.hpp>
#include <ftxui/dom/selection.hpp>
#include <ftxui/screen/screen.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <string>
#if !defined(_WIN32)
#include <ctime>
#endif

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

#if !defined(_WIN32)
class ScopedTimezone {
public:
    explicit ScopedTimezone(const char* timezone) {
        if (const char* current = std::getenv("TZ")) {
            previous_ = std::string(current);
        }
        setenv("TZ", timezone, 1);
        tzset();
    }

    ~ScopedTimezone() {
        if (previous_.has_value()) {
            setenv("TZ", previous_->c_str(), 1);
        } else {
            unsetenv("TZ");
        }
        tzset();
    }

private:
    std::optional<std::string> previous_;
};
#endif
} // namespace

// ============================================================================
// Message Factory Tests
// ============================================================================

TEST_CASE("local_time_str formats system clock using local timezone", "[tui][conversation][time]") {
#if defined(_WIN32)
    SUCCEED("Timezone override test is POSIX-only.");
#else
    ScopedTimezone timezone("CET-1CEST,M3.5.0/2,M10.5.0/3");

    using namespace std::chrono;
    const auto utc_time =
        sys_days{year{2026} / June / 6} + 16h + 57min + 53s;

    REQUIRE(local_time_str(utc_time) == "18:57:53");
#endif
}

TEST_CASE("make_user_message — basic creation", "[tui][conversation][factory]") {
    auto msg = make_user_message("Hello, world!", "12:00:00");
    REQUIRE(msg.type == MessageType::User);
    REQUIRE(msg.text == "Hello, world!");
    REQUIRE(msg.timestamp == "12:00:00");
    REQUIRE(!msg.id.empty());
}

TEST_CASE("make_shell_command_message — pending direct command", "[tui][conversation][factory]") {
    auto msg = make_shell_command_message("pwd", "12:00:01", true);
    REQUIRE(msg.type == MessageType::ShellCommand);
    REQUIRE(msg.text == "pwd");
    REQUIRE(msg.timestamp == "12:00:01");
    REQUIRE(msg.pending == true);
    REQUIRE(msg.finalized == false);
    REQUIRE(msg.secondary_text.empty());
}

TEST_CASE("remove_latest_ui_turn removes shell commands as visible turns",
          "[tui][conversation][rewind]") {
    std::vector<UiMessage> messages;
    messages.push_back(make_user_message("before"));
    messages.push_back(make_assistant_message("response", {}, false));
    messages.push_back(make_shell_command_message("pwd", {}, false));
    messages.back().secondary_text = "/tmp\n";

    REQUIRE(remove_latest_ui_turn(messages));
    REQUIRE(messages.size() == 2);
    CHECK(messages[0].type == MessageType::User);
    CHECK(messages[1].type == MessageType::Assistant);
}

TEST_CASE("truncate_ui_before_user_turn removes the selected turn and tail",
          "[tui][conversation][rewind]") {
    std::vector<UiMessage> messages{
        make_system_message("header"),
        make_user_message("first"),
        make_assistant_message("one", {}, false),
        make_user_message("second"),
        make_assistant_message("two", {}, false),
    };

    REQUIRE(truncate_ui_before_user_turn(messages, 1));
    REQUIRE(messages.size() == 3);
    CHECK(messages.back().text == "one");
    CHECK_FALSE(truncate_ui_before_user_turn(messages, 2));
}

TEST_CASE("rewind picker exposes recent user checkpoints",
          "[tui][conversation][rewind]") {
    std::vector<core::llm::Message> history{
        {.role = "user", .content = "first prompt"},
        {.role = "assistant", .content = "first answer"},
        {.role = "user", .content = "I ran a shell command", .synthetic = true},
        {.role = "user", .content = "Resume after output limit", .synthetic = true},
        {.role = "user", .content = "Expanded: /tmp/example.png", .input_text = "second\nprompt"},
        {.role = "assistant", .content = "second answer"},
    };
    RewindPickerState state;

    REQUIRE(open_rewind_picker(state, history));
    REQUIRE(state.active);
    REQUIRE(state.options.size() == 4);
    CHECK(state.selected == 1);
    CHECK(state.options[0].history_index == 0);
    CHECK(state.options[1].history_index == 4);
    CHECK(state.options[1].user_ordinal == 1);
    CHECK(state.options[1].label == "Expanded: /tmp/example.png");
    CHECK(state.options[1].action == RewindPickerOption::Action::RewindToMessage);

    const auto selected = handle_rewind_picker_event(
        state,
        ftxui::Event::Return,
        false);
    REQUIRE(selected.selection.has_value());
    CHECK(selected.selection->prompt == "second\nprompt");
    CHECK_FALSE(state.active);
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

TEST_CASE("summarize_tool_arguments — task shows subagent label", "[tui][conversation][args]") {
    const auto result = summarize_tool_arguments(
        "task",
        R"({"description":"verify root cause","prompt":"check the code","subagent_type":"explore"})");
    REQUIRE_THAT(result, ContainsSubstring("@explore"));
    REQUIRE_THAT(result, ContainsSubstring("verify root cause"));
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

TEST_CASE("message_uses_animation — executing subagent animates", "[tui][conversation][animation]") {
    auto message = make_assistant_message("", "", false);
    auto tool = make_tool_activity("parent", "task", "{}", "@explore");
    tool.status = ToolActivity::Status::Succeeded;
    ToolActivity::SubagentActivity subagent;
    subagent.id = "task_1";
    subagent.worker_name = "explore";
    subagent.status = ToolActivity::Status::Executing;
    tool.subagents.push_back(std::move(subagent));
    message.tools.push_back(std::move(tool));
    REQUIRE(message_uses_animation(message, true));
    REQUIRE_FALSE(message_uses_animation(message, false));
}

TEST_CASE("message_uses_animation — pending shell command animates", "[tui][conversation][animation]") {
    auto message = make_shell_command_message("sleep 1", "", true);
    REQUIRE(message_uses_animation(message, true));
    REQUIRE_FALSE(message_uses_animation(message, false));
}

TEST_CASE("find_subagent_activity — finds nested agent", "[tui][conversation][tool]") {
    auto tool = make_tool_activity("parent", "task", "{}", "@general");
    ToolActivity::SubagentActivity subagent;
    subagent.id = "task_00000001";
    subagent.worker_name = "general";
    tool.subagents.push_back(std::move(subagent));

    auto* found = find_subagent_activity(tool, "task_00000001");
    REQUIRE(found != nullptr);
    REQUIRE(found->worker_name == "general");
    REQUIRE(find_subagent_activity(tool, "missing") == nullptr);
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

TEST_CASE("animation cadence covers review and hidden elapsed indicators",
          "[tui][conversation][animation][policy]") {
    using namespace std::chrono_literals;

    const auto review = select_animation_cadence(true, false, true, false);
    REQUIRE(review.has_value());
    REQUIRE(review->period == 150ms);
    REQUIRE(review->advance_frame);

    const auto hidden_assistant = select_animation_cadence(false, true, false, true);
    REQUIRE(hidden_assistant.has_value());
    REQUIRE(hidden_assistant->period == 1s);
    REQUIRE_FALSE(hidden_assistant->advance_frame);

    const auto hidden_review = select_animation_cadence(false, false, true, false);
    REQUIRE(hidden_review.has_value());
    REQUIRE(hidden_review->period == 1s);
    REQUIRE_FALSE(hidden_review->advance_frame);

    REQUIRE_FALSE(select_animation_cadence(false, false, false, true).has_value());
    REQUIRE_FALSE(select_animation_cadence(true, false, false, false).has_value());
}

// ============================================================================
// LiveText reactive-leaf selection
// ============================================================================
//
// The animated "Working..." label is a custom LiveText node (its surrounding
// transcript tree is cached and reused across ticks). Selection must behave
// exactly like ftxui::text(): only the actually-selected cells are copied to
// the clipboard and only those cells receive the selection highlight.

TEST_CASE("animated working label supports per-cell selection",
          "[tui][conversation][render][selection]") {
    // thinking_pulse_frame(3) == "..." so the live label reads "Working...".
    std::atomic<std::size_t> tick{3};
    auto msg = make_assistant_message("", "", true);  // pending + thinking
    ConversationRenderOptions options;
    options.show_spinner = true;
    options.animation_tick = &tick;

    auto element = render_assistant_message(msg, 3, options);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(80),
                                        ftxui::Dimension::Fit(element));
    ftxui::Render(screen, element);

    // Locate the screen cell where the "Working" label begins by scanning the
    // rendered cells (robust against the animated prefix width).
    int label_row = -1;
    int label_col = -1;
    for (int y = 0; y < screen.dimy() && label_row < 0; ++y) {
        for (int x = 0; x + 6 < screen.dimx(); ++x) {
            if (screen.CellAt(x,     y).character == "W"
                && screen.CellAt(x + 1, y).character == "o"
                && screen.CellAt(x + 2, y).character == "r"
                && screen.CellAt(x + 3, y).character == "k"
                && screen.CellAt(x + 4, y).character == "i"
                && screen.CellAt(x + 5, y).character == "n"
                && screen.CellAt(x + 6, y).character == "g") {
                label_row = y;
                label_col = x;
                break;
            }
        }
    }
    REQUIRE(label_row >= 0);

    // A partial selection over just the "Worki" cells copies exactly those
    // glyphs — not the whole cached label (the previous whole-node behavior).
    ftxui::Selection partial(label_col, label_row, label_col + 4, label_row);
    ftxui::Render(screen, element.get(), partial);
    REQUIRE(partial.GetParts() == "Worki");

    // A full-row selection copies the entire live label exactly once.
    ftxui::Selection full_row(0, label_row, screen.dimx() - 1, label_row);
    ftxui::Render(screen, element.get(), full_row);
    REQUIRE_THAT(full_row.GetParts(), ContainsSubstring("Working..."));
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

// ============================================================================
// Reasoning (chain-of-thought) disclosure
// ============================================================================

namespace {
std::string render_panel_text(const std::vector<UiMessage>& messages,
                              ConversationRenderOptions options = {}) {
    auto panel = render_history_panel(messages, 0, std::move(options));
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(160),
                                        ftxui::Dimension::Fit(panel));
    ftxui::Render(screen, panel);
    return strip_ansi(screen.ToString());
}
} // namespace

TEST_CASE("tool presentation uses semantic labels and compact result metrics",
          "[tui][conversation][render][tool]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_tool_activity(
        "read-1",
        "read_file",
        R"({"path":"src/main.cpp","offset_line":10})",
        "src/main.cpp");
    apply_tool_result(tool, R"({"content":"alpha\nbeta\n"})");
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    const auto compact = render_panel_text(messages);
    REQUIRE_THAT(compact, ContainsSubstring("Read"));
    REQUIRE_THAT(compact, ContainsSubstring("2 lines"));
    REQUIRE(compact.find("read_file") == std::string::npos);
    REQUIRE(compact.find("alpha") == std::string::npos);

    const auto expanded = render_panel_text(
        messages,
        ConversationRenderOptions{.expand_tool_results = true});
    REQUIRE_THAT(expanded, ContainsSubstring("alpha"));
    REQUIRE_THAT(expanded, ContainsSubstring("beta"));
    REQUIRE_THAT(expanded, ContainsSubstring("10"));
    REQUIRE_THAT(expanded, ContainsSubstring("11"));
}

TEST_CASE("assistant narration renders before the tools from the same step",
          "[tui][conversation][render][tool][ordering]") {
    std::vector<UiMessage> messages;
    auto message = make_assistant_message("I will inspect the configuration.", "", false);
    auto tool = make_tool_activity(
        "read-config",
        "read_file",
        R"({"path":"config.toml"})",
        "config.toml");
    tool.status = ToolActivity::Status::Succeeded;
    message.tools.push_back(std::move(tool));
    messages.push_back(std::move(message));

    const std::string rendered = render_panel_text(messages);
    const auto narration = rendered.find("I will inspect the configuration.");
    const auto tool_card = rendered.find("Read");
    REQUIRE(narration != std::string::npos);
    REQUIRE(tool_card != std::string::npos);
    CHECK(narration < tool_card);
}

TEST_CASE("tool presentation groups grep results by file",
          "[tui][conversation][render][tool]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_tool_activity(
        "grep-1",
        "grep_search",
        R"({"pattern":"Widget","path":"src"})",
        "Widget in src");
    apply_tool_result(
        tool,
        R"({"matches":[{"path":"src/a.cpp","line":12,"text":"class Widget {};"},{"path":"src/a.cpp","line":31,"text":"Widget value;"}]})");
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    const auto compact = render_panel_text(messages);
    REQUIRE_THAT(compact, ContainsSubstring("Search"));
    REQUIRE_THAT(compact, ContainsSubstring("2 matches"));
    REQUIRE(compact.find("src/a.cpp") == std::string::npos);

    const auto expanded = render_panel_text(
        messages,
        ConversationRenderOptions{.expand_tool_results = true});
    REQUIRE_THAT(expanded, ContainsSubstring("src/a.cpp"));
    REQUIRE_THAT(expanded, ContainsSubstring("class Widget"));
    REQUIRE_THAT(expanded, ContainsSubstring("31"));
}

TEST_CASE("tool presentation renders todos as a checklist",
          "[tui][conversation][render][tool]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_tool_activity(
        "todo-1",
        "write_todos",
        R"({"todos":[{"content":"Inspect renderer","status":"completed"},{"content":"Add tests","status":"in_progress"}]})",
        "");
    apply_tool_result(
        tool,
        R"({"ok":true,"todos":[{"id":"1","content":"Inspect renderer","status":"completed"},{"id":"2","content":"Add tests","status":"in_progress"}]})");
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    const auto compact = render_panel_text(messages);
    REQUIRE_THAT(compact, ContainsSubstring("Plan"));
    REQUIRE_THAT(compact, ContainsSubstring("2 items"));

    const auto expanded = render_panel_text(
        messages,
        ConversationRenderOptions{.expand_tool_results = true});
    REQUIRE_THAT(expanded, ContainsSubstring("Inspect renderer"));
    REQUIRE_THAT(expanded, ContainsSubstring("Add tests"));
    REQUIRE_THAT(expanded, ContainsSubstring("✓"));
    REQUIRE_THAT(expanded, ContainsSubstring("●"));
}

TEST_CASE("tool header always labels a finished tool that produced no metric",
          "[tui][conversation][render][tool][regression]") {
    // A structured tool that fails has no JSON to count, so its metric is empty.
    // The header must fall back to the status label rather than render blank —
    // otherwise only the glyph colour distinguishes success from failure.
    const auto render_single_tool = [](ToolActivity tool) {
        std::vector<UiMessage> messages;
        auto msg = make_assistant_message("", "", false);
        msg.tools.push_back(std::move(tool));
        messages.push_back(std::move(msg));
        return render_panel_text(messages);
    };

    SECTION("failed search") {
        auto tool = make_tool_activity(
            "grep-fail", "grep_search", R"({"pattern":"["})", "bad pattern");
        apply_tool_result(tool, R"({"error":"regex compile failed"})");
        REQUIRE(tool.status == ToolActivity::Status::Failed);
        REQUIRE_THAT(render_single_tool(std::move(tool)),
                     ContainsSubstring(std::string(tool_status_label(
                         ToolActivity::Status::Failed))));
    }

    SECTION("denied file search") {
        auto tool = make_tool_activity(
            "files-denied", "file_search", R"({"pattern":"*.c"})", "find");
        tool.status = ToolActivity::Status::Denied;
        tool.result.summary = "Permission denied by user.";
        REQUIRE_THAT(render_single_tool(std::move(tool)),
                     ContainsSubstring(std::string(tool_status_label(
                         ToolActivity::Status::Denied))));
    }

    SECTION("failed read") {
        auto tool = make_tool_activity(
            "read-fail", "read_file", R"({"path":"missing.txt"})", "missing.txt");
        apply_tool_result(tool, R"({"error":"no such file"})");
        REQUIRE(tool.status == ToolActivity::Status::Failed);
        REQUIRE_THAT(render_single_tool(std::move(tool)),
                     ContainsSubstring(std::string(tool_status_label(
                         ToolActivity::Status::Failed))));
    }
}

TEST_CASE("cancelled shell reports cancellation rather than the killed exit code",
          "[tui][conversation][render][tool][regression]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_tool_activity(
        "shell-cancel",
        "run_terminal_command",
        R"({"command":"sleep 100"})",
        "sleep 100");
    // The process was killed, so the payload still carries a zero exit status.
    apply_tool_result(tool, R"({"exit_code":0,"output":"partial\n[INTERRUPTED: user]"})");
    REQUIRE(tool.status == ToolActivity::Status::Cancelled);
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    const auto rendered = render_panel_text(messages);
    REQUIRE_THAT(rendered, ContainsSubstring(std::string(tool_status_label(
                               ToolActivity::Status::Cancelled))));
    REQUIRE(rendered.find("exit 0") == std::string::npos);
}

TEST_CASE("successful shell still reports its exit code",
          "[tui][conversation][render][tool]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_tool_activity(
        "shell-fail", "run_terminal_command", R"({"command":"false"})", "false");
    apply_tool_result(tool, R"({"exit_code":1,"output":"boom"})");
    REQUIRE(tool.status == ToolActivity::Status::Failed);
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    REQUIRE_THAT(render_panel_text(messages), ContainsSubstring("exit 1"));
}

TEST_CASE("structured tools keep the payload their renderers need",
          "[tui][conversation][tool][regression]") {
    // These renderers used to rely on apply_tool_result happening to fall
    // through and leave the raw JSON in `summary`. Adding an "output" field
    // upstream would have silently emptied them, so the retention is explicit.
    const auto payload_visible_to_renderer = [](std::string_view name,
                                                std::string_view args,
                                                std::string_view result) {
        auto tool = make_tool_activity("id", std::string(name), std::string(args), "");
        apply_tool_result(tool, result);
        return tool.result.raw_payload.empty() ? tool.result.summary
                                               : tool.result.raw_payload;
    };

    CHECK_THAT(payload_visible_to_renderer(
                   "grep_search", R"({"pattern":"x"})",
                   R"({"output":"2 hits","matches":[{"path":"a.cpp","line":1,"text":"x"}]})"),
               ContainsSubstring("\"matches\""));
    CHECK_THAT(payload_visible_to_renderer(
                   "list_directory", R"({"path":"src"})",
                   R"({"output":"1 entry","entries":[{"type":"dir","name":"core"}]})"),
               ContainsSubstring("\"entries\""));
    CHECK_THAT(payload_visible_to_renderer(
                   "fetch_url", R"({"url":"https://example.com"})",
                   R"({"content":"body text","status_code":200})"),
               ContainsSubstring("\"status_code\""));

    // Tools with no structured renderer must not pay for a second copy.
    auto plain = make_tool_activity("id", "read_file", R"({"path":"a"})", "");
    apply_tool_result(plain, R"({"content":"line one\n"})");
    CHECK(plain.result.raw_payload.empty());
}

TEST_CASE("list and file results render their entries",
          "[tui][conversation][render][tool]") {
    const auto render_expanded = [](std::string_view name,
                                    std::string_view args,
                                    std::string_view result) {
        std::vector<UiMessage> messages;
        auto msg = make_assistant_message("", "", false);
        auto tool = make_tool_activity("id", std::string(name), std::string(args), "");
        apply_tool_result(tool, result);
        msg.tools.push_back(std::move(tool));
        messages.push_back(std::move(msg));
        return render_panel_text(
            messages, ConversationRenderOptions{.expand_tool_results = true});
    };

    const auto listing = render_expanded(
        "list_directory", R"({"path":"src"})",
        R"({"entries":[{"type":"dir","name":"core"},{"type":"file","name":"main.cpp"}]})");
    CHECK_THAT(listing, ContainsSubstring("core"));
    CHECK_THAT(listing, ContainsSubstring("main.cpp"));

    const auto files = render_expanded(
        "file_search", R"({"pattern":"*.cpp"})",
        R"({"files":["src/a.cpp","src/b.cpp"]})");
    CHECK_THAT(files, ContainsSubstring("src/a.cpp"));
    CHECK_THAT(files, ContainsSubstring("src/b.cpp"));
}

TEST_CASE("web search reports its result count",
          "[tui][conversation][render][tool]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_tool_activity("web-1", "web_search", R"({"query":"x"})", "x");
    apply_tool_result(
        tool,
        R"({"results":[{"title":"First","url":"https://example.com/1"},)"
        R"({"title":"Second","url":"https://example.com/2"}]})");
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    const auto rendered = render_panel_text(messages);
    CHECK_THAT(rendered, ContainsSubstring("Web search"));
    CHECK_THAT(rendered, ContainsSubstring("2 results"));
}

TEST_CASE("web search results honour the compact preview limit",
          "[tui][conversation][render][tool]") {
    constexpr int kResultCount = 25;
    std::string payload = R"({"results":[)";
    for (int i = 0; i < kResultCount; ++i) {
        if (i != 0) payload += ",";
        payload += std::format(
            R"({{"title":"Result {0}","url":"https://example.com/{0}"}})", i);
    }
    payload += "]}";

    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_tool_activity("web-1", "web_search", R"({"query":"x"})", "x");
    apply_tool_result(tool, payload);
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    // Open the disclosure the way a click does, without the global "show
    // everything" toggle, so the compact preview limit is the thing under test.
    std::unordered_map<std::string, bool> disclosure_state;
    disclosure_state[tool_disclosure_key(messages.front().tools.front(), 0)] = true;

    ConversationRenderOptions clicked_open;
    clicked_open.system_disclosure_expanded = &disclosure_state;

    // The previous implementation ignored the shared limit and hardcoded its own.
    const auto compact = render_panel_text(messages, clicked_open);
    CHECK_THAT(compact, ContainsSubstring("more results"));
    CHECK(compact.find("Result 24") == std::string::npos);

    const auto expanded = render_panel_text(
        messages, ConversationRenderOptions{.expand_tool_results = true});
    CHECK_THAT(expanded, ContainsSubstring("Result 24"));
}

TEST_CASE("tool disclosure defaults follow tool outcome",
          "[tui][conversation][tool]") {
    auto tool = make_tool_activity("d1", "read_file", R"({"path":"a"})", "a");
    apply_tool_result(tool, R"({"content":"x\n"})");
    CHECK_FALSE(tool_disclosure_defaults_expanded(tool));

    tool.status = ToolActivity::Status::Failed;
    CHECK(tool_disclosure_defaults_expanded(tool));

    auto edit = make_tool_activity(
        "d2", "write_file", R"({"path":"a.md","content":"hi\n"})", "a.md");
    CHECK(tool_disclosure_defaults_expanded(edit));
}

TEST_CASE("tool disclosure keys separate identical id-less calls",
          "[tui][conversation][tool][regression]") {
    auto first = make_tool_activity("", "read_file", R"({"path":"a"})", "a");
    auto second = make_tool_activity("", "read_file", R"({"path":"a"})", "a");
    CHECK(tool_disclosure_key(first, 0) != tool_disclosure_key(second, 1));

    auto identified = make_tool_activity("call-1", "read_file", R"({"path":"a"})", "a");
    CHECK(tool_disclosure_key(identified, 0) == tool_disclosure_key(identified, 7));
}

TEST_CASE("tool activity prepares its transcript diff preview",
          "[tui][conversation][tool]") {
    const auto tool = make_tool_activity(
        "write-1",
        "write_file",
        R"({"path":"notes.md","content":"first\nsecond\n"})",
        "notes.md");
    REQUIRE_FALSE(tool.diff_preview.empty());
    CHECK(tool.diff_preview.title == "notes.md");
}

namespace {

/// Renders the transcript content without the scroll viewport, so that a card
/// taller than the terminal can be asserted on in full.
std::string render_content_text(const std::vector<UiMessage>& messages,
                                ConversationRenderOptions options = {}) {
    auto content = render_history_content(messages, 0, std::move(options));
    // `extend_beyond_screen` — a deliberately expanded diff is taller than the
    // host terminal, and clamping here would hide exactly what is under test.
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(160),
                                        ftxui::Dimension::Fit(content, true));
    ftxui::Render(screen, content);
    return strip_ansi(screen.ToString());
}

/// A write_file call whose content is `line_count` numbered lines.
ToolActivity make_write_tool(std::string id, std::size_t line_count) {
    std::string content;
    for (std::size_t i = 0; i < line_count; ++i) {
        content += std::format("line {}\\n", i);
    }
    return make_tool_activity(
        std::move(id),
        "write_file",
        std::format(R"({{"path":"big.txt","content":"{}"}})", content),
        "big.txt");
}

} // namespace

TEST_CASE("diff preview keeps every line so expanding can reveal them",
          "[tui][conversation][tool][diff][regression]") {
    // Regression: the preview used to be clamped in make_tool_activity, which
    // deleted the remaining lines before any disclosure state existed. No click
    // could bring them back and "… N more lines" was a dead end.
    const auto tool = make_write_tool("write-long", 60);

    // +2 for the "+++ b/…" header and the "@@ file content @@" hunk line.
    CHECK(tool.diff_preview.total_line_count == 62);
    CHECK(tool.diff_preview.lines().size() == 62);
    CHECK(tool.diff_preview.hidden_line_count == 0);
    CHECK_FALSE(tool.diff_preview.truncated_at_source);
}

TEST_CASE("diff stats describe the whole change, not the visible fragment",
          "[tui][conversation][render][tool][diff][regression]") {
    // Regression: the header counted Add/Delete lines of the *clamped* preview,
    // so a long edit reported nonsense such as "+0 -7" for a 26-line change.
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_tool_activity(
        "replace-1",
        "replace",
        R"({"file_path":"a.txt","old_string":"a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn",)"
        R"("new_string":"1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16"})",
        "a.txt");
    CHECK(tool.diff_preview.deleted_count == 14);
    CHECK(tool.diff_preview.added_count == 16);

    apply_tool_result(tool, R"({"result":"Done"})");
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    CHECK_THAT(render_panel_text(messages), ContainsSubstring("+16 -14"));
}

TEST_CASE("long diffs start collapsed and advertise their size",
          "[tui][conversation][render][tool][diff]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_write_tool("write-long", 60);
    apply_tool_result(tool, R"({"result":"Done"})");
    const auto key = tool_disclosure_key(tool, 0);
    REQUIRE_FALSE(tool_disclosure_defaults_expanded(tool));
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    // Collapsed: no diff body, but the header states what is behind the chevron.
    const auto collapsed = render_content_text(messages);
    CHECK_THAT(collapsed, ContainsSubstring("▶"));
    CHECK_THAT(collapsed, ContainsSubstring("62 diff lines"));
    CHECK(collapsed.find("line 59") == std::string::npos);

    // Expanded by a click: the whole diff, including its very last line.
    std::unordered_map<std::string, bool> disclosure_state;
    disclosure_state[key] = true;
    ConversationRenderOptions clicked_open;
    clicked_open.system_disclosure_expanded = &disclosure_state;

    const auto expanded = render_content_text(messages, clicked_open);
    CHECK_THAT(expanded, ContainsSubstring("▼"));
    CHECK_THAT(expanded, ContainsSubstring("line 0"));
    CHECK_THAT(expanded, ContainsSubstring("line 59"));
    CHECK(expanded.find("more lines") == std::string::npos);
}

TEST_CASE("short diffs still open on their own",
          "[tui][conversation][render][tool][diff]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_write_tool("write-short", 4);
    apply_tool_result(tool, R"({"result":"Done"})");
    REQUIRE(tool_disclosure_defaults_expanded(tool));
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    const auto rendered = render_content_text(messages);
    CHECK_THAT(rendered, ContainsSubstring("line 3"));
    CHECK(rendered.find("diff lines") == std::string::npos);
}

TEST_CASE("edit metadata JSON is nested behind a collapsed disclosure",
          "[tui][conversation][render][tool][diff][regression]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_tool_activity(
        "replace-json",
        "replace",
        R"({"file_path":"a.txt","old_string":"before","new_string":"after"})",
        "a.txt");
    apply_tool_result(
        tool,
        R"({"truncated":true,"tool":"replace","digest_fnv1a64":"machine-only"})");
    REQUIRE(tool_disclosure_defaults_expanded(tool));
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    const auto collapsed = render_content_text(messages);
    CHECK_THAT(collapsed, ContainsSubstring("Diff: a.txt"));
    CHECK_THAT(collapsed, ContainsSubstring("after"));
    CHECK_THAT(collapsed, ContainsSubstring("▶ Raw result"));
    CHECK(collapsed.find("digest_fnv1a64") == std::string::npos);

    std::unordered_map<std::string, bool> disclosure_state;
    disclosure_state[tool_raw_result_disclosure_key(
        messages.front().tools.front(), 0)] = true;
    ConversationRenderOptions opened;
    opened.system_disclosure_expanded = &disclosure_state;

    const auto expanded = render_content_text(messages, opened);
    CHECK_THAT(expanded, ContainsSubstring("▼ Raw result"));
    CHECK_THAT(expanded, ContainsSubstring("digest_fnv1a64"));
    CHECK(expanded.find("Raw result") < expanded.find("Diff: a.txt"));
}

TEST_CASE("a diff beyond the model ceiling is cut once and says so",
          "[tui][conversation][render][tool][diff]") {
    // The only truncation the user cannot undo. It must be reported as such,
    // never as something another click would reveal.
    const std::size_t over_ceiling = kToolDiffModelMaxLines + 50;
    auto tool = make_write_tool("write-ceiling", over_ceiling);
    apply_tool_result(tool, R"({"result":"Done"})");

    // +2 for the file header and hunk marker.
    CHECK(tool.diff_preview.total_line_count == over_ceiling + 2);
    CHECK(tool.diff_preview.added_count == over_ceiling);
    CHECK(tool.diff_preview.lines().size() == kToolDiffModelMaxLines);
    CHECK(tool.diff_preview.truncated_at_source);
    // Too long to open unprompted.
    CHECK_FALSE(tool_disclosure_defaults_expanded(tool));

    const auto key = tool_disclosure_key(tool, 0);
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    std::unordered_map<std::string, bool> disclosure_state;
    disclosure_state[key] = true;
    ConversationRenderOptions options;
    options.system_disclosure_expanded = &disclosure_state;
    // Ctrl+O lifts the render budget, so the ceiling is the only limit left.
    options.expand_tool_results = true;

    CHECK_THAT(render_content_text(messages, options),
               ContainsSubstring("diff too large to display in full"));
}

TEST_CASE("clamp_diff_preview reports hidden lines against the true total",
          "[tui][conversation][tool][diff]") {
    const auto full = make_write_tool("clamp-src", 30).diff_preview;
    REQUIRE(full.total_line_count == 32);

    const auto clamped = clamp_diff_preview(full, 10);
    CHECK(clamped.lines().size() == 10);
    CHECK(clamped.hidden_line_count == 22);
    // Stats survive clamping — they describe the change, not the view.
    CHECK(clamped.added_count == full.added_count);
    CHECK(clamped.total_line_count == full.total_line_count);

    // A no-op clamp must not invent hidden lines...
    const auto unclamped = clamp_diff_preview(full, 0);
    CHECK(unclamped.lines().size() == full.lines().size());
    CHECK(unclamped.hidden_line_count == 0);

    // ...and clamping twice narrows rather than compounds.
    const auto twice = clamp_diff_preview(clamped, 4);
    CHECK(twice.lines().size() == 4);
    CHECK(twice.hidden_line_count == 28);
}

TEST_CASE("a tool without a diff is unaffected by the diff budget",
          "[tui][conversation][render][tool][diff]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_tool_activity("read-nodiff", "read_file", R"({"path":"a"})", "a");
    apply_tool_result(tool, R"({"content":"only line\n"})");
    CHECK(tool.diff_preview.empty());
    CHECK(tool.diff_preview.total_line_count == 0);
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    const auto rendered = render_content_text(
        messages, ConversationRenderOptions{.expand_tool_results = true});
    CHECK_THAT(rendered, ContainsSubstring("only line"));
    CHECK(rendered.find("diff lines") == std::string::npos);
    CHECK(rendered.find("more lines") == std::string::npos);
}

TEST_CASE("an expanded diff past the render budget says how much it shows",
          "[tui][conversation][render][tool][diff]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_write_tool("write-huge", 40);
    apply_tool_result(tool, R"({"result":"Done"})");
    const auto key = tool_disclosure_key(tool, 0);
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    std::unordered_map<std::string, bool> disclosure_state;
    disclosure_state[key] = true;
    ConversationRenderOptions options;
    options.system_disclosure_expanded = &disclosure_state;
    options.tool_diff_expanded_max_lines = 10;

    const auto rendered = render_content_text(messages, options);
    CHECK_THAT(rendered, ContainsSubstring("showing the first 10 of 42"));
    CHECK(rendered.find("line 39") == std::string::npos);

    // Ctrl+O means "everything you have", budget included.
    options.expand_tool_results = true;
    CHECK_THAT(render_content_text(messages, options), ContainsSubstring("line 39"));
}

TEST_CASE("render — active reasoning stays collapsed by default",
          "[tui][conversation][render][reasoning]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", true);
    msg.reasoning_active = true;
    msg.reasoning_text = "Consider the failure mode.\nThen verify the fix.";
    messages.push_back(std::move(msg));

    const auto output = render_panel_text(messages);
    // Collapsed by default even while streaming: the live "Thinking" header and
    // a collapsed chevron render, but the chain-of-thought body stays hidden
    // until the user expands the box.
    REQUIRE_THAT(output, ContainsSubstring("▶"));
    REQUIRE_THAT(output, ContainsSubstring("Thinking"));
    REQUIRE_THAT(output, !ContainsSubstring("Consider the failure mode."));
}

TEST_CASE("render — finished reasoning collapses to a Thought summary",
          "[tui][conversation][render][reasoning]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("Here is the answer.", "", false);
    msg.finalized = true;
    msg.reasoning_active = false;
    msg.reasoning_elapsed = "7s";
    msg.reasoning_text = "Private chain of thought that stays hidden.";
    messages.push_back(std::move(msg));

    const auto output = render_panel_text(messages);
    // Collapsed by default: summary + hint shown, body hidden, answer visible.
    REQUIRE_THAT(output, ContainsSubstring("▶"));
    REQUIRE_THAT(output, ContainsSubstring("Thought for 7s"));
    REQUIRE_THAT(output, ContainsSubstring("Here is the answer."));
    REQUIRE_THAT(output, !ContainsSubstring("Private chain of thought"));
}

TEST_CASE("render — finished reasoning expands with the global option",
          "[tui][conversation][render][reasoning]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("Answer.", "", false);
    msg.finalized = true;
    msg.reasoning_elapsed = "3s";
    msg.reasoning_text = "Revealed chain of thought.";
    messages.push_back(std::move(msg));

    const auto output = render_panel_text(
        messages, ConversationRenderOptions{.expand_system_details = true});
    REQUIRE_THAT(output, ContainsSubstring("▼"));
    REQUIRE_THAT(output, ContainsSubstring("Revealed chain of thought."));
}

TEST_CASE("render — reasoning is suppressed when show_reasoning is false",
          "[tui][conversation][render][reasoning]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("Answer.", "", false);
    msg.finalized = true;
    msg.reasoning_elapsed = "9s";
    msg.reasoning_text = "Suppressed thoughts.";
    messages.push_back(std::move(msg));

    const auto output = render_panel_text(
        messages, ConversationRenderOptions{.show_reasoning = false});
    REQUIRE_THAT(output, !ContainsSubstring("Thought for 9s"));
    REQUIRE_THAT(output, !ContainsSubstring("Suppressed thoughts."));
    REQUIRE_THAT(output, ContainsSubstring("Answer."));
}

TEST_CASE("render — generic work uses a neutral status indicator",
          "[tui][conversation][render][reasoning]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", true);  // pending + thinking, no reasoning
    messages.push_back(std::move(msg));

    const auto output = render_panel_text(messages);
    // No disclosure or lightbulb: this is progress, not provider reasoning.
    REQUIRE_THAT(output, ContainsSubstring("Working"));
    REQUIRE_THAT(output, !ContainsSubstring("💡"));
    REQUIRE_THAT(output, !ContainsSubstring("▼"));
    REQUIRE_THAT(output, !ContainsSubstring("Thought for"));
}

TEST_CASE("render — generic completed activity leaves no faux thought trace",
          "[tui][conversation][render][reasoning]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("Done.", "", false);
    msg.finalized = true;
    msg.activity_recorded = true;
    msg.reasoning_kind = UiMessage::ActivityKind::Analyzing;
    msg.reasoning_elapsed = "4s";
    // No reasoning_text — the phase produced no inner tokens.
    messages.push_back(std::move(msg));

    const auto output = render_panel_text(messages);
    REQUIRE_THAT(output, !ContainsSubstring("Analyzed for 4s"));
    REQUIRE_THAT(output, !ContainsSubstring("Thought for"));
    REQUIRE_THAT(output, !ContainsSubstring("💡"));
    REQUIRE_THAT(output, ContainsSubstring("Done."));
}

TEST_CASE("render — generic completed thinking leaves no faux thought trace",
          "[tui][conversation][render][reasoning]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("Answer.", "", false);
    msg.finalized = true;
    msg.activity_recorded = true;
    msg.reasoning_kind = UiMessage::ActivityKind::Thinking;
    msg.reasoning_elapsed = "2s";
    messages.push_back(std::move(msg));

    const auto output = render_panel_text(messages);
    REQUIRE_THAT(output, !ContainsSubstring("Thought for 2s"));
    REQUIRE_THAT(output, !ContainsSubstring("💡"));
}

TEST_CASE("render — live generic activity uses a neutral status indicator",
          "[tui][conversation][render][reasoning]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", true);
    msg.thinking = true;
    msg.activity_recorded = true;
    msg.reasoning_kind = UiMessage::ActivityKind::Analyzing;
    // reasoning_active stays false because no reasoning text streamed.
    messages.push_back(std::move(msg));

    const auto output = render_panel_text(messages);
    REQUIRE_THAT(output, ContainsSubstring("Working"));
    REQUIRE_THAT(output, !ContainsSubstring("💡"));
}

TEST_CASE("render — reasoning text always uses the Thought (not Analyzed) label",
          "[tui][conversation][render][reasoning]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("Answer.", "", false);
    msg.finalized = true;
    msg.activity_recorded = true;
    // Even if the kind was tagged Analyzing, presence of text means it is a
    // chain of thought — the past-tense label follows the kind, but the box is
    // expandable. Here we assert the Thinking kind path with text.
    msg.reasoning_kind = UiMessage::ActivityKind::Thinking;
    msg.reasoning_elapsed = "6s";
    msg.reasoning_text = "step by step.";
    messages.push_back(std::move(msg));

    const auto output = render_panel_text(
        messages, ConversationRenderOptions{.expand_system_details = true});
    REQUIRE_THAT(output, ContainsSubstring("Thought for 6s"));
    REQUIRE_THAT(output, ContainsSubstring("step by step."));
}

TEST_CASE("render — activity disclosure always leads with the lightbulb",
          "[tui][conversation][render][reasoning]") {
    // Live streaming reasoning: header must carry the lightbulb + present label.
    {
        std::vector<UiMessage> messages;
        auto msg = make_assistant_message("", "", true);
        msg.reasoning_active = true;
        msg.reasoning_text = "live thoughts.";
        messages.push_back(std::move(msg));
        const auto output = render_panel_text(messages);
        REQUIRE_THAT(output, ContainsSubstring("💡"));
        REQUIRE_THAT(output, ContainsSubstring("Thinking"));
    }
    // Finished collapsed trace: still leads with the lightbulb.
    {
        std::vector<UiMessage> messages;
        auto msg = make_assistant_message("Answer.", "", false);
        msg.finalized = true;
        msg.activity_recorded = true;
        msg.reasoning_elapsed = "5s";
        msg.reasoning_text = "hidden thoughts.";
        messages.push_back(std::move(msg));
        const auto output = render_panel_text(messages);
        REQUIRE_THAT(output, ContainsSubstring("💡"));
        REQUIRE_THAT(output, ContainsSubstring("Thought for 5s"));
    }
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
    asst.show_activity_status = true;
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

TEST_CASE("LiveAssistantTimeline does not resurrect a removed or finished turn",
          "[tui][conversation][live_timeline][lifecycle]") {
    std::vector<UiMessage> messages;
    messages.push_back(make_assistant_message("", "", true));
    LiveAssistantTimeline removed(messages.back().id);
    messages.clear();
    CHECK(removed.begin_step(messages) == nullptr);
    CHECK(messages.empty());

    messages.push_back(make_assistant_message("", "", true));
    LiveAssistantTimeline finished(messages.back().id);
    REQUIRE(finished.begin_step(messages) != nullptr);
    finished.finish(messages, "0s", true);
    CHECK(finished.begin_step(messages) == nullptr);
    REQUIRE(messages.size() == 1);
    CHECK(messages[0].stopped);
    CHECK(messages[0].finalized);
}

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


TEST_CASE("write diff stats count file lines, not the trailing newline",
          "[tui][conversation][render][tool][regression]") {
    std::vector<UiMessage> messages;
    auto msg = make_assistant_message("", "", false);
    auto tool = make_tool_activity(
        "write-stats",
        "write_file",
        R"({"path":"notes.md","content":"alpha\nbeta\n"})",
        "notes.md");
    apply_tool_result(tool, R"({"success":true})");
    msg.tools.push_back(std::move(tool));
    messages.push_back(std::move(msg));

    // A file whose content ends in a newline has two lines, not three.
    const auto rendered = render_panel_text(messages);
    REQUIRE_THAT(rendered, ContainsSubstring("+2 -0"));
}

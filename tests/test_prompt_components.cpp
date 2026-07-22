/**
 * @file test_prompt_components.cpp
 * @brief Unit tests for the picker panel rendering in tui/PromptComponents.
 *
 * Verifies that:
 *  - render_mention_prompt_panel and render_command_prompt_panel do not crash
 *    for any selected_index, including those beyond kPickerMaxDisplayRows
 *  - The panels render correctly with 0, 1, and many suggestions
 *  - search_mention_index respects kMaxAutocompleteSuggestions
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/PromptComponents.hpp"
#include "tui/Autocomplete.hpp"
#include "tui/Constants.hpp"
#include "tui/PickerState.hpp"

#include <ftxui/screen/screen.hpp>

#include <filesystem>
#include <stop_token>
#include <string>
#include <vector>

using namespace tui;

// ============================================================================
// Helpers
// ============================================================================

static std::vector<MentionSuggestion> make_mention_suggestions(int count) {
    std::vector<MentionSuggestion> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        out.push_back(MentionSuggestion{
            .display_path   = std::string("src/file_") + std::to_string(i) + ".cpp",
            .insertion_text = std::string("src/file_") + std::to_string(i) + ".cpp",
            .is_directory   = false,
        });
    }
    return out;
}

static std::vector<CommandSuggestion> make_command_suggestions(int count) {
    std::vector<CommandSuggestion> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        out.push_back(CommandSuggestion{
            .display_name      = std::string("/cmd") + std::to_string(i),
            .insertion_text    = std::string("/cmd") + std::to_string(i),
            .description       = "A test command",
            .aliases_label     = {},
            .accepts_arguments = false,
        });
    }
    return out;
}

static std::string strip_ansi(std::string_view input) {
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

TEST_CASE("render_startup_banner_panel — stays readable with provider metadata",
          "[tui][banner]") {
    auto panel = render_startup_banner_panel(
        "local-qwen",
        "qwen2.5-coder-7b-instruct-q4_k_m",
        0,
        "AGENTS.md, FILO.md",
        "Set XAI_API_KEY to start chatting with Grok.\nGet a key at: console.x.ai\n",
        "12:34:56");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(140),
                                        ftxui::Dimension::Fit(panel));
    ftxui::Render(screen, panel);

    const auto output = strip_ansi(screen.ToString());
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("Filo AI Agent"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("provider: local-qwen"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("model: qwen2.5-coder-7b-instruct-q4_k_m"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("MCP servers: 0"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("AGENTS.md, FILO.md"));
    // The live clock sits in the top-right corner of the banner.
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("12:34:56"));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("context:"));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("sandbox:"));

    const auto clock_end = output.find("12:34:56") + std::string_view("12:34:56").size();
    const auto clock_line_end = output.find('\n', clock_end);
    const auto context_start = output.find("AGENTS.md, FILO.md");
    const auto context_line_end = output.find('\n', context_start);
    const auto context_end = context_start + std::string_view("AGENTS.md, FILO.md").size();
    REQUIRE(clock_line_end != std::string::npos);
    REQUIRE(context_line_end != std::string::npos);
    REQUIRE(clock_line_end - clock_end == context_line_end - context_end);
}

TEST_CASE("runtime status summary has one compact canonical format",
          "[tui][banner]") {
    CHECK(format_runtime_status_summary("openai", "gpt-5", 2)
          == "provider: openai  —  model: gpt-5  —  MCP servers: 2");
    CHECK(format_runtime_status_summary("openai", "", 0)
          == "provider: openai  —  model: <provider default>  —  MCP servers: 0");
}

TEST_CASE("workspace status uses a lock only for an enabled sandbox",
          "[tui][status-bar][sandbox]") {
    CHECK(format_workspace_status_label("~/project", false) == " ~/project ");
    CHECK(format_workspace_status_label("~/project", true) == " 🔒 ~/project ");
    CHECK(format_workspace_status_label("", true).empty());
}

TEST_CASE("turn activity indicator spins only while a response is active",
          "[tui][status-bar][activity]") {
    const auto render_indicator = [](bool active, bool animate, std::size_t tick) {
        auto indicator = render_turn_activity_indicator(active, animate, tick);
        auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(2),
                                            ftxui::Dimension::Fixed(1));
        ftxui::Render(screen, indicator);
        return strip_ansi(screen.ToString());
    };

    const auto idle = render_indicator(false, true, 0);
    const auto active_frame_0 = render_indicator(true, true, 0);
    const auto active_frame_1 = render_indicator(true, true, 1);
    const auto static_active = render_indicator(true, false, 0);

    CHECK(idle.find_first_not_of(" \n") == std::string::npos);
    CHECK(active_frame_0 != active_frame_1);
    CHECK(active_frame_0.find_first_not_of(" \n") != std::string::npos);
    CHECK(static_active.find("●") != std::string::npos);
}

// ============================================================================
// render_mention_prompt_panel — smoke tests
// ============================================================================

TEST_CASE("render_mention_prompt_panel — empty suggestions list",
          "[tui][picker][mention]") {
    ftxui::Element input = ftxui::text("@foo");
    REQUIRE_NOTHROW(render_mention_prompt_panel({}, 0, std::move(input), "@foo"));
}

TEST_CASE("render_mention_prompt_panel — single suggestion, selected=0",
          "[tui][picker][mention]") {
    auto suggestions = make_mention_suggestions(1);
    ftxui::Element input = ftxui::text("@fi");
    REQUIRE_NOTHROW(render_mention_prompt_panel(suggestions, 0, std::move(input), "@fi"));
}

TEST_CASE("render_mention_prompt_panel — fewer than display rows",
          "[tui][picker][mention]") {
    auto suggestions = make_mention_suggestions(kPickerMaxDisplayRows / 2);
    for (int sel = 0; sel < static_cast<int>(suggestions.size()); ++sel) {
        ftxui::Element input = ftxui::text("@f");
        REQUIRE_NOTHROW(
            render_mention_prompt_panel(suggestions, sel, std::move(input), "@f"));
    }
}

TEST_CASE("render_mention_prompt_panel — exactly kPickerMaxDisplayRows suggestions",
          "[tui][picker][mention]") {
    auto suggestions = make_mention_suggestions(kPickerMaxDisplayRows);
    for (int sel = 0; sel < kPickerMaxDisplayRows; ++sel) {
        ftxui::Element input = ftxui::text("@src");
        REQUIRE_NOTHROW(
            render_mention_prompt_panel(suggestions, sel, std::move(input), "@src"));
    }
}

TEST_CASE("render_mention_prompt_panel — more than kPickerMaxDisplayRows (scroll region)",
          "[tui][picker][mention]") {
    // This is the key scenario: many results, selection at different scroll positions.
    const int count = kPickerMaxDisplayRows * 3;
    auto suggestions = make_mention_suggestions(count);

    // First visible window.
    for (int sel = 0; sel < kPickerMaxDisplayRows; ++sel) {
        ftxui::Element input = ftxui::text("@src");
        REQUIRE_NOTHROW(
            render_mention_prompt_panel(suggestions, sel, std::move(input), "@src"));
    }

    // Middle of the list (requires scrolling).
    for (int sel = count / 4; sel < count * 3 / 4; sel += count / 8) {
        ftxui::Element input = ftxui::text("@src");
        REQUIRE_NOTHROW(
            render_mention_prompt_panel(suggestions, sel, std::move(input), "@src"));
    }

    // Last item.
    {
        ftxui::Element input = ftxui::text("@src");
        REQUIRE_NOTHROW(
            render_mention_prompt_panel(suggestions, count - 1, std::move(input), "@src"));
    }
}

TEST_CASE("render_mention_prompt_panel — kMaxAutocompleteSuggestions items",
          "[tui][picker][mention]") {
    auto suggestions = make_mention_suggestions(static_cast<int>(kMaxAutocompleteSuggestions));
    // Selection at the bottom of the list should still render without crash.
    const int last = static_cast<int>(kMaxAutocompleteSuggestions) - 1;
    ftxui::Element input = ftxui::text("@");
    REQUIRE_NOTHROW(
        render_mention_prompt_panel(suggestions, last, std::move(input), "@"));
}

TEST_CASE("render_mention_prompt_panel — directory entry",
          "[tui][picker][mention]") {
    std::vector<MentionSuggestion> suggestions = {
        {.display_path = "src/", .insertion_text = "src/", .is_directory = true},
        {.display_path = "src/main.cpp", .insertion_text = "src/main.cpp", .is_directory = false},
    };
    ftxui::Element input = ftxui::text("@src");
    REQUIRE_NOTHROW(render_mention_prompt_panel(suggestions, 0, std::move(input), "@src"));
    ftxui::Element input2 = ftxui::text("@src");
    REQUIRE_NOTHROW(render_mention_prompt_panel(suggestions, 1, std::move(input2), "@src"));
}

// ============================================================================
// render_command_prompt_panel — smoke tests
// ============================================================================

TEST_CASE("render_command_prompt_panel — empty suggestions list",
          "[tui][picker][command]") {
    ftxui::Element input = ftxui::text("/");
    REQUIRE_NOTHROW(render_command_prompt_panel({}, 0, std::move(input), "/"));
}

TEST_CASE("render_command_prompt_panel — single suggestion",
          "[tui][picker][command]") {
    auto suggestions = make_command_suggestions(1);
    ftxui::Element input = ftxui::text("/c");
    REQUIRE_NOTHROW(render_command_prompt_panel(suggestions, 0, std::move(input), "/c"));
}

TEST_CASE("render_command_prompt_panel — many suggestions scroll coverage",
          "[tui][picker][command]") {
    const int count = kPickerMaxDisplayRows + 4;
    auto suggestions = make_command_suggestions(count);

    for (int sel = 0; sel < count; ++sel) {
        ftxui::Element input = ftxui::text("/cmd");
        REQUIRE_NOTHROW(
            render_command_prompt_panel(suggestions, sel, std::move(input), "/cmd"));
    }
}

TEST_CASE("render_command_prompt_panel — command with aliases label",
          "[tui][picker][command]") {
    std::vector<CommandSuggestion> suggestions = {{
        .display_name      = "/help",
        .insertion_text    = "/help",
        .description       = "Show help",
        .aliases_label     = "?, h",
        .accepts_arguments = false,
    }};
    ftxui::Element input = ftxui::text("/h");
    REQUIRE_NOTHROW(render_command_prompt_panel(suggestions, 0, std::move(input), "/h"));
}

TEST_CASE("render_command_prompt_panel — command that accepts arguments",
          "[tui][picker][command]") {
    std::vector<CommandSuggestion> suggestions = {{
        .display_name      = "/model",
        .insertion_text    = "/model",
        .description       = "Set the active model",
        .aliases_label     = {},
        .accepts_arguments = true,
    }};
    ftxui::Element input = ftxui::text("/model");
    REQUIRE_NOTHROW(render_command_prompt_panel(suggestions, 0, std::move(input), "/model"));
}

TEST_CASE("render_permission_prompt_panel — keeps file header visible with long diff",
          "[tui][picker][permission]") {
    ToolDiffPreview preview;
    preview.title = "src/tui/very/long/path/to/a_file_with_a_banner_that_should_stay_visible.cpp";
    preview.lines.push_back({.kind = DiffLineKind::Header, .content = "--- a/src/tui/file.cpp"});
    preview.lines.push_back({.kind = DiffLineKind::Header, .content = "+++ b/src/tui/file.cpp"});
    preview.lines.push_back({.kind = DiffLineKind::Hunk, .content = "@@ -1,3 +1,40 @@"});
    for (int i = 0; i < 40; ++i) {
        preview.lines.push_back({
            .kind = DiffLineKind::Context,
            .old_line = i + 1,
            .new_line = i + 1,
            .content = "context line " + std::to_string(i),
        });
    }

    auto panel = render_permission_prompt_panel(
        "apply_patch",
        R"({"patch":"*** Update File: src/tui/file.cpp"})",
        preview,
        "file modifications",
        0);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(140),
                                        ftxui::Dimension::Fixed(20));
    ftxui::Render(screen, panel);

    const auto output = strip_ansi(screen.ToString());
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Diff Preview"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring(
        "File: src/tui/very/long/path/to/a_file_with_a_banner_that_should_stay_visible.cpp"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("@@ -1,3 +1,40 @@"));
}

TEST_CASE("render_review_picker_panel — base branch mode renders selectable refs",
          "[tui][picker][review]") {
    std::vector<ReviewBaseRef> refs = {
        {.name = "main", .description = "tracks origin/main"},
        {.name = "release", .description = ""},
    };

    auto panel = render_review_picker_panel(
        ReviewPickerMode::EnterBaseBranch,
        1,
        "",
        refs,
        0);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(100),
                                        ftxui::Dimension::Fixed(18));
    ftxui::Render(screen, panel);

    const auto output = strip_ansi(screen.ToString());
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Base branch"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("main"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("tracks origin/main"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("release"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Enter selects the highlighted ref"));
}

TEST_CASE("render_settings_panel — inherited and scoped rows render without crashing",
          "[tui][picker][settings]") {
    std::vector<SettingsPanelRow> rows = {
        {
            .label = "General · Start Mode",
            .value = "DEBUG",
            .description = "The agent mode Filo should start in.",
            .inherited = false,
        },
        {
            .label = "Model · Manual Provider",
            .value = "grok (grok-code-fast-1)",
            .description = "Used when Model Selection is Manual.",
            .inherited = true,
        },
    };

    REQUIRE_NOTHROW(render_settings_panel(
        "Workspace",
        "/tmp/project/.filo/settings.json",
        rows,
        1,
        "Workspace setting saved."));
}

TEST_CASE("render_model_provider_picker_panel — renders providers and active marker",
          "[tui][picker][model]") {
    std::vector<ModelProviderPickerRow> rows = {
        {.name = "claude", .description = "Default: claude-opus-4-8", .active = true},
        {.name = "kimi", .description = "Default: kimi-k2.7-code", .active = false},
    };

    auto panel = render_model_provider_picker_panel(rows, 0, false);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(100),
                                        ftxui::Dimension::Fixed(16));
    ftxui::Render(screen, panel);

    const auto output = strip_ansi(screen.ToString());
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("MODEL PROVIDER"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("claude (active)"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Default: claude-opus-4-8"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("kimi"));
}

TEST_CASE("render_provider_model_picker_panel — renders model choices",
          "[tui][picker][model]") {
    std::vector<ModelPickerRow> rows = {
        {
            .id = "claude-opus-4-8",
            .selector = "claude-opus-4-8",
            .description = "Claude Opus 4.8 · aliases: opus",
            .active = true,
            .provider_default = true,
        },
        {
            .id = "claude-sonnet-5",
            .selector = "claude-sonnet-5",
            .description = "Claude Sonnet 5",
        },
    };

    auto panel = render_provider_model_picker_panel("claude", rows, 0);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(110),
                                        ftxui::Dimension::Fixed(16));
    ftxui::Render(screen, panel);

    const auto output = strip_ansi(screen.ToString());
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("MODEL"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("claude"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("claude-opus-4-8 (active)"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Claude Opus 4.8"));
}

TEST_CASE("render_option_selection_panel — renders generic command choices",
          "[tui][picker][option]") {
    std::vector<OptionPickerRow> rows = {
        {
            .value = "auto",
            .label = "Auto",
            .description = "Use provider default.",
            .active = false,
        },
        {
            .value = "medium",
            .label = "Medium",
            .description = "Balanced effort.",
            .active = true,
        },
    };

    auto panel = render_option_selection_panel(
        "EFFORT",
        rows,
        1,
        "medium",
        "Esc closes this panel.");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(90),
                                        ftxui::Dimension::Fixed(14));
    ftxui::Render(screen, panel);

    const auto output = strip_ansi(screen.ToString());
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("EFFORT"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Current: medium"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Medium (active)"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Balanced effort."));
}

TEST_CASE("render_command_prompt_panel — preserves command metadata in tall rows",
          "[tui][picker][command]") {
    std::vector<CommandSuggestion> suggestions = {{
        .display_name      = "/help",
        .insertion_text    = "/help",
        .description       = "Show help",
        .aliases_label     = "?, h",
        .accepts_arguments = true,
    }};

    auto panel = render_command_prompt_panel(suggestions, 0, ftxui::text("/h"), "/h");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(80),
                                        ftxui::Dimension::Fit(panel));
    ftxui::Render(screen, panel);

    const auto output = screen.ToString();
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("/help"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("aliases: ?, h"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Show help"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("[args]"));
}

TEST_CASE("render_conversation_search_panel — renders query and matches",
          "[tui][picker][search]") {
    std::vector<ConversationSearchHit> hits = {
        {.message_index = 3, .role = "User", .snippet = "Refactor the auth middleware to avoid regressions."},
        {.message_index = 8, .role = "Assistant", .snippet = "I found a regression in token refresh handling."},
    };

    auto panel = render_conversation_search_panel("regression", hits, 1);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(100),
                                        ftxui::Dimension::Fit(panel));
    ftxui::Render(screen, panel);

    const auto output = strip_ansi(screen.ToString());
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("SEARCH"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Query: regression"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Assistant"));
}

TEST_CASE("render_conversation_search_panel — handles empty query and no matches",
          "[tui][picker][search]") {
    REQUIRE_NOTHROW(render_conversation_search_panel("", {}, 0));
}

TEST_CASE("render_stderr_panel — renders dismissible error lines",
          "[tui][picker][stderr]") {
    auto panel = render_stderr_panel({
        "[2026-06-25 10:00:00] [ERROR] provider failed",
        "[2026-06-25 10:00:01] [ERROR] retry failed",
    });
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(100),
                                        ftxui::Dimension::Fit(panel));
    ftxui::Render(screen, panel);

    const auto output = strip_ansi(screen.ToString());
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("STDERR"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("Esc/c/x close"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("provider failed"));
}

// ============================================================================
// Viewport scrolling — render_mention_prompt_panel with many items
// ============================================================================

TEST_CASE("render_mention_prompt_panel — viewport slides to last item without crash",
          "[tui][picker][viewport]") {
    // Regression: with yframe+size(LESS_THAN) the panel only showed ~3 items
    // and never scrolled. The manual viewport fix slices rows so selected_index
    // is always in view.
    const int count = kPickerMaxDisplayRows * 4;  // well beyond one window
    auto suggestions = make_mention_suggestions(count);

    // Navigating from 0 to count-1 should never crash.
    for (int sel = 0; sel < count; ++sel) {
        ftxui::Element input = ftxui::text("@src");
        REQUIRE_NOTHROW(
            render_mention_prompt_panel(suggestions, sel, std::move(input), "@src"));
    }
}

TEST_CASE("render_command_prompt_panel — viewport slides to last item without crash",
          "[tui][picker][viewport]") {
    const int count = kPickerMaxDisplayRows * 4;
    auto suggestions = make_command_suggestions(count);

    for (int sel = 0; sel < count; ++sel) {
        ftxui::Element input = ftxui::text("/cmd");
        REQUIRE_NOTHROW(
            render_command_prompt_panel(suggestions, sel, std::move(input), "/cmd"));
    }
}

// ============================================================================
// search_mention_index — respects kMaxAutocompleteSuggestions cap
// ============================================================================

TEST_CASE("search_mention_index — returns at most kMaxAutocompleteSuggestions",
          "[tui][picker][autocomplete]") {
    // Build a synthetic index larger than the cap.
    std::vector<MentionSuggestion> large_index;
    const std::size_t N = kMaxAutocompleteSuggestions * 2;
    for (std::size_t i = 0; i < N; ++i) {
        large_index.push_back(MentionSuggestion{
            .display_path   = "src/file_" + std::to_string(i) + ".cpp",
            .insertion_text = "src/file_" + std::to_string(i) + ".cpp",
            .is_directory   = false,
        });
    }

    const auto results = search_mention_index(large_index, "src", kMaxAutocompleteSuggestions);
    REQUIRE(results.size() <= kMaxAutocompleteSuggestions);
}

TEST_CASE("search_mention_index — empty query returns up to max",
          "[tui][picker][autocomplete]") {
    std::vector<MentionSuggestion> index;
    for (std::size_t i = 0; i < 10; ++i) {
        index.push_back(MentionSuggestion{
            .display_path   = "file_" + std::to_string(i) + ".txt",
            .insertion_text = "file_" + std::to_string(i) + ".txt",
        });
    }
    const auto results = search_mention_index(index, "", 5);
    REQUIRE(results.size() == 5);
}

TEST_CASE("search_mention_index — query filters by basename",
          "[tui][picker][autocomplete]") {
    std::vector<MentionSuggestion> index = {
        {.display_path = "src/main.cpp",    .insertion_text = "src/main.cpp"},
        {.display_path = "src/helper.cpp",  .insertion_text = "src/helper.cpp"},
        {.display_path = "tests/main.cpp",  .insertion_text = "tests/main.cpp"},
    };
    const auto results = search_mention_index(index, "main", 10);
    REQUIRE(results.size() == 2);
}

TEST_CASE("search_mention_index — case-insensitive matching",
          "[tui][picker][autocomplete]") {
    std::vector<MentionSuggestion> index = {
        {.display_path = "src/MainApp.cpp", .insertion_text = "src/MainApp.cpp"},
        {.display_path = "src/helper.cpp",  .insertion_text = "src/helper.cpp"},
    };
    const auto upper = search_mention_index(index, "MAIN", 10);
    const auto lower = search_mention_index(index, "main", 10);
    REQUIRE(upper.size() == lower.size());
    REQUIRE(upper.size() == 1);
}

TEST_CASE("build_mention_index — cancellation returns without publishing partial results",
          "[tui][picker][autocomplete]") {
    std::stop_source stop_source;
    stop_source.request_stop();

    const auto results = build_mention_index(
        std::filesystem::current_path(),
        stop_source.get_token());

    REQUIRE(results.empty());
}

TEST_CASE("kMaxAutocompleteSuggestions — is large enough for big projects",
          "[tui][picker][constants]") {
    // Sanity-check that the constant was actually increased from the old value of 8.
    REQUIRE(kMaxAutocompleteSuggestions >= 32);
}

TEST_CASE("kPickerMaxDisplayRows — shows at least 8 items",
          "[tui][picker][constants]") {
    // Compact mention rows still need a generous viewport.
    REQUIRE(kPickerMaxDisplayRows >= 8);
}

TEST_CASE("kCommandPickerMaxDisplayRows — keeps tall command cards compact",
          "[tui][picker][constants]") {
    REQUIRE(kCommandPickerMaxDisplayRows >= 4);
    REQUIRE(kCommandPickerMaxDisplayRows < kPickerMaxDisplayRows);
}

// ============================================================================
// PickerState — suppress / ESC-dismiss behaviour
// ============================================================================

TEST_CASE("PickerState::suppress hides picker without clearing key",
          "[tui][picker][state]") {
    PickerState ps;
    ps.sync("/quit", 3);
    REQUIRE_FALSE(ps.suppressed);

    ps.suppress();
    CHECK(ps.suppressed == true);
    CHECK(ps.selected == 0);
    CHECK(ps.key == "/quit");  // Key preserved so sync with same text stays suppressed.
}

TEST_CASE("PickerState::suppress — same key keeps picker suppressed across syncs",
          "[tui][picker][state]") {
    PickerState ps;
    ps.sync("/quit", 3);
    ps.suppress();

    // Simulates repeated events while input text is still "/quit".
    ps.sync("/quit", 3);
    CHECK(ps.suppressed == true);
    ps.sync("/quit", 3);
    CHECK(ps.suppressed == true);
}

TEST_CASE("PickerState::suppress — different key un-suppresses picker",
          "[tui][picker][state]") {
    PickerState ps;
    ps.sync("/quit", 3);
    ps.suppress();
    CHECK(ps.suppressed == true);

    // History navigation loaded a different command.
    ps.sync("/provider", 2);
    CHECK(ps.suppressed == false);
    CHECK(ps.selected == 0);
}

TEST_CASE("PickerState::suppress — empty key (non-command entry) un-suppresses picker",
          "[tui][picker][state]") {
    PickerState ps;
    ps.sync("/quit", 3);
    ps.suppress();

    // History navigation loaded a plain message (no command).
    ps.sync("", 0);
    CHECK(ps.suppressed == false);
}

TEST_CASE("PickerState::clear resets suppression flag",
          "[tui][picker][state]") {
    PickerState ps;
    ps.sync("/quit", 3);
    ps.suppress();
    REQUIRE(ps.suppressed == true);

    ps.clear();
    CHECK(ps.suppressed == false);
    CHECK(ps.key.empty());
    CHECK(ps.selected == 0);
}

TEST_CASE("PickerState::sync resets selected on key change",
          "[tui][picker][state]") {
    PickerState ps;
    ps.sync("/quit", 3);
    ps.navigate_down(3);
    ps.navigate_down(3);
    REQUIRE(ps.selected == 2);

    ps.sync("/provider", 4);
    CHECK(ps.selected == 0);
}

TEST_CASE("PickerState navigation wraps around",
          "[tui][picker][state]") {
    PickerState ps;
    ps.sync("/q", 3);

    ps.navigate_up(3);   // 0 → 2
    CHECK(ps.selected == 2);
    ps.navigate_down(3); // 2 → 0
    CHECK(ps.selected == 0);
}

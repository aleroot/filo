#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/HistoryComponent.hpp"
#include "tui/Conversation.hpp"

#include <ftxui/dom/node.hpp>
#include <ftxui/dom/selection.hpp>
#include <ftxui/screen/screen.hpp>

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {
std::vector<tui::UiMessage> mock_messages() {
    std::vector<tui::UiMessage> msgs;
    msgs.push_back(tui::make_user_message("hello", ""));
    msgs.push_back(tui::make_assistant_message("world", "", false));
    return msgs;
}

tui::ConversationRenderOptions mock_options() {
    return tui::ConversationRenderOptions{};
}

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

std::string render_history_text(tui::HistoryComponent& history, int width = 160) {
    auto panel = history.OnRender();
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(width),
                                        ftxui::Dimension::Fit(panel));
    ftxui::Render(screen, panel);
    return strip_ansi(screen.ToString());
}

std::string render_history_viewport(tui::HistoryComponent& history,
                                    int width,
                                    int height) {
    auto panel = history.OnRender();
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(width),
                                        ftxui::Dimension::Fixed(height));
    ftxui::Render(screen, panel);
    return strip_ansi(screen.ToString());
}

std::string render_reference_viewport(const std::vector<tui::UiMessage>& messages,
                                      int width,
                                      int height) {
    auto panel = tui::render_history_panel(messages, 0, mock_options());
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(width),
                                        ftxui::Dimension::Fixed(height));
    ftxui::Render(screen, panel);
    return strip_ansi(screen.ToString());
}

std::string select_history_viewport(tui::HistoryComponent& history,
                                    int width,
                                    int height,
                                    ftxui::Selection selection) {
    auto panel = history.OnRender();
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(width),
                                        ftxui::Dimension::Fixed(height));
    ftxui::Render(screen, panel.get(), selection);
    return selection.GetParts();
}

int rendered_row_containing(std::string_view text, std::string_view marker) {
    const auto pos = text.find(marker);
    if (pos == std::string_view::npos) {
        return -1;
    }
    return static_cast<int>(std::count(text.begin(), text.begin() + pos, '\n'));
}

// FTXUI flexbox nodes recompute their required height several times per frame.
// This small node models a wrapped layout whose provisional height shrinks and
// then returns to its final value within the same Render() call.
class IterativeHeightNode final : public ftxui::Node {
public:
    void ComputeRequirement() override {
        static constexpr int kHeights[] = {100, 60, 100};
        requirement_ = ftxui::Requirement{};
        requirement_.min_x = 20;
        requirement_.min_y = kHeights[std::min(compute_index_, 2)];
        ++compute_index_;
    }

    void Check(Status* status) override {
        if (status->iteration == 0) {
            compute_index_ = 0;
        }
        status->need_iteration |= status->iteration < 3;
    }

private:
    int compute_index_ = 0;
};
}

TEST_CASE("HistoryComponent basic scrolling", "[tui][history_component]") {
    std::atomic<size_t> tick{0};
    tui::HistoryComponent history(mock_messages, tick, mock_options);

    static_cast<void>(render_history_text(history));

    history.ScrollDown(0.5f);
    history.ScrollUp(0.2f);
    // Internal state check if possible, or just ensuring no crash.
    
    history.JumpToMessage(0, 2);
    // scroll_pos should be 0.0f
    
    history.JumpToMessage(1, 2);
    // scroll_pos should be 1.0f
}

TEST_CASE("HistoryComponent scroll bounds", "[tui][history_component]") {
    std::atomic<size_t> tick{0};
    tui::HistoryComponent history(mock_messages, tick, mock_options);

    static_cast<void>(render_history_text(history));
    history.ScrollUp(10.0f); // Way past 0
    // Should be clamped to 0.0f
    
    history.ScrollDown(20.0f); // Way past 1
    // Should be clamped to 1.0f
}

// ============================================================================
// Auto-scroll intent tests
// ============================================================================

TEST_CASE("HistoryComponent default state is FollowBottom",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    tui::HistoryComponent history(mock_messages, tick, mock_options);

    // A freshly created component must follow the bottom and show no badge.
    REQUIRE(history.IsAutoScrollFollowing());
    REQUIRE(history.ScrollPosition() == 1.0f);
    REQUIRE_FALSE(history.HasNewContentIndicator());
}

TEST_CASE("HistoryComponent ScrollUp suspends auto-follow",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    tui::HistoryComponent history(mock_messages, tick, mock_options);

    static_cast<void>(render_history_text(history));
    history.ScrollUp(0.1f);

    REQUIRE_FALSE(history.IsAutoScrollFollowing());
    REQUIRE(history.ScrollPosition() < 1.0f);
}

TEST_CASE("HistoryComponent HandleWheel up suspends auto-follow",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    tui::HistoryComponent history(mock_messages, tick, mock_options);

    static_cast<void>(render_history_text(history));
    ftxui::Mouse mouse;
    mouse.button = ftxui::Mouse::WheelUp;
    mouse.motion = ftxui::Mouse::Pressed;
    history.HandleWheel(ftxui::Event::Mouse("", mouse));

    REQUIRE_FALSE(history.IsAutoScrollFollowing());
}

TEST_CASE("HistoryComponent ScrollDown to bottom resumes auto-follow",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    tui::HistoryComponent history(mock_messages, tick, mock_options);

    static_cast<void>(render_history_text(history));
    // Scroll up to suspend follow.
    history.ScrollUp(0.3f);
    REQUIRE_FALSE(history.IsAutoScrollFollowing());

    // Scroll back all the way down.
    history.ScrollDown(1.0f);

    REQUIRE(history.IsAutoScrollFollowing());
    REQUIRE(history.ScrollPosition() == 1.0f);
}

TEST_CASE("HistoryComponent new message does not yank while user is reading",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> messages = mock_messages();
    tui::HistoryComponent history(
        [&messages]() { return messages; },
        tick,
        mock_options);

    // Prime the actual FTXUI layout.
    static_cast<void>(render_history_text(history));

    // User scrolls up to read.
    history.ScrollUp(0.3f);
    REQUIRE_FALSE(history.IsAutoScrollFollowing());
    const float pos_before = history.ScrollPosition();

    // A new message arrives.
    messages.push_back(tui::make_info_message("New tool call"));
    static_cast<void>(render_history_text(history));

    // The absolute anchor stays put, so its relative position decreases as
    // the transcript grows. It must not be yanked back to the bottom.
    REQUIRE(history.ScrollPosition() < pos_before);
    // Badge must appear.
    REQUIRE(history.HasNewContentIndicator());
}

TEST_CASE("HistoryComponent preserves held anchor while streaming grows",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> messages;
    messages.push_back(tui::make_user_message("prompt", ""));
    messages.push_back(tui::make_assistant_message(
        "# First heading\n"
        "READ-01\n"
        "# Second heading\n"
        "READ-02\n"
        "# Third heading\n"
        "READ-03\n"
        "# Fourth heading\n"
        "READ-04\n"
        "# Fifth heading\n"
        "READ-05\n"
        "# Sixth heading\n"
        "READ-06\n"
        "# Seventh heading\n"
        "READ-07\n"
        "# Eighth heading\n"
        "READ-08\n"
        "# Ninth heading\n"
        "READ-09\n"
        "# Tenth heading\n"
        "READ-10",
        "",
        true));
    tui::HistoryComponent history(
        [&messages]() { return messages; },
        tick,
        mock_options);

    static_cast<void>(render_history_viewport(history, 40, 7));
    history.ScrollUp(0.35f);
    REQUIRE_FALSE(history.IsAutoScrollFollowing());
    const auto before = render_history_viewport(history, 40, 7);
    const int before_row = rendered_row_containing(before, "READ-07");
    REQUIRE(before_row >= 0);

    // Plain streamed text adds one rendered line, whereas the headings above
    // each occupy two. This is the layout mismatch that used to make the
    // estimate-based compensation twitch the viewport.
    messages.back().text += "\nSTREAMED-ONE\nSTREAMED-TWO";
    const auto after = render_history_viewport(history, 40, 7);

    REQUIRE_FALSE(history.IsAutoScrollFollowing());
    REQUIRE(history.HasNewContentIndicator());
    REQUIRE(rendered_row_containing(after, "READ-07") == before_row);
}

TEST_CASE("HistoryComponent new message snaps to bottom when following",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> messages = mock_messages();
    tui::HistoryComponent history(
        [&messages]() { return messages; },
        tick,
        mock_options);

    // Prime the component (default: FollowBottom).
    static_cast<void>(render_history_text(history));
    REQUIRE(history.IsAutoScrollFollowing());

    // Simulate a partial scroll down that keeps us in the "near bottom" zone.
    // (Do NOT scroll up, so intent stays FollowBottom.)
    messages.push_back(tui::make_info_message("Status update"));
    static_cast<void>(render_history_text(history));

    REQUIRE(history.ScrollPosition() == 1.0f);
    REQUIRE_FALSE(history.HasNewContentIndicator());
}

TEST_CASE("HistoryComponent ResetToBottom clears hold and indicator",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> messages = mock_messages();
    tui::HistoryComponent history(
        [&messages]() { return messages; },
        tick,
        mock_options);

    // Put the component into a held + indicator state.
    static_cast<void>(render_history_text(history));
    history.ScrollUp(0.4f);
    messages.push_back(tui::make_info_message("New content"));
    static_cast<void>(render_history_text(history));
    REQUIRE(history.HasNewContentIndicator());
    REQUIRE_FALSE(history.IsAutoScrollFollowing());

    // ResetToBottom() must clear everything.
    history.ResetToBottom();

    REQUIRE(history.IsAutoScrollFollowing());
    REQUIRE(history.ScrollPosition() == 1.0f);
    REQUIRE_FALSE(history.HasNewContentIndicator());
}

TEST_CASE("HistoryComponent transcript shrink resets stale hold state",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> messages = mock_messages();
    tui::HistoryComponent history(
        [&messages]() { return messages; },
        tick,
        mock_options);

    static_cast<void>(render_history_text(history));
    history.ScrollUp(0.4f);
    messages.push_back(tui::make_info_message("New content"));
    static_cast<void>(render_history_text(history));
    REQUIRE_FALSE(history.IsAutoScrollFollowing());
    REQUIRE(history.HasNewContentIndicator());

    messages.clear();
    messages.push_back(tui::make_system_message("Fresh transcript"));
    static_cast<void>(render_history_text(history));

    REQUIRE(history.IsAutoScrollFollowing());
    REQUIRE(history.ScrollPosition() == 1.0f);
    REQUIRE_FALSE(history.HasNewContentIndicator());
}

TEST_CASE("HistoryComponent JumpToMessage suspends follow when not at end",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    tui::HistoryComponent history(mock_messages, tick, mock_options);

    static_cast<void>(render_history_text(history));
    // Jump to the first of 5 messages => UserHeld.
    history.JumpToMessage(0, 5);
    REQUIRE_FALSE(history.IsAutoScrollFollowing());
    REQUIRE(history.ScrollPosition() == 0.0f);

    // Jump to the last of 5 messages => FollowBottom.
    history.JumpToMessage(4, 5);
    REQUIRE(history.IsAutoScrollFollowing());
    REQUIRE(history.ScrollPosition() == 1.0f);
}

TEST_CASE("HistoryComponent End key resumes auto-follow",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    tui::HistoryComponent history(mock_messages, tick, mock_options);

    static_cast<void>(render_history_text(history));
    history.ScrollUp(0.5f);
    REQUIRE_FALSE(history.IsAutoScrollFollowing());

    // Synthesise the End key event.
    history.OnEvent(ftxui::Event::End);

    REQUIRE(history.IsAutoScrollFollowing());
    REQUIRE(history.ScrollPosition() == 1.0f);
}

TEST_CASE("HistoryComponent toggles system disclosure by mouse click",
          "[tui][history_component]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> messages;
    messages.push_back(tui::make_system_disclosure_message(
        "Internal session rotated to keep the working set lean (context preserved).",
        "Previous segment: seg-a\nNew segment: seg-b\nReason: threshold"));

    tui::ConversationRenderOptions options;
    tui::HistoryComponent history(
        [&messages]() { return messages; },
        tick,
        [&options]() { return options; });

    const auto collapsed = render_history_text(history);
    REQUIRE_THAT(collapsed, Catch::Matchers::ContainsSubstring("click or Ctrl+O for details"));
    REQUIRE_THAT(collapsed, !Catch::Matchers::ContainsSubstring("Previous segment: seg-a"));

    ftxui::Mouse mouse;
    mouse.button = ftxui::Mouse::Left;
    mouse.motion = ftxui::Mouse::Pressed;
    mouse.x = 1;
    mouse.y = 0;
    REQUIRE(history.OnEvent(ftxui::Event::Mouse("", mouse)));

    const auto expanded = render_history_text(history);
    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("Previous segment: seg-a"));
    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("New segment: seg-b"));
}

TEST_CASE("HistoryComponent toggles a finished reasoning box by mouse click",
          "[tui][history_component][reasoning]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> messages;
    auto msg = tui::make_assistant_message("Final answer.", "", false);
    msg.finalized = true;
    msg.reasoning_elapsed = "5s";
    msg.reasoning_text = "Hidden chain of thought.";
    messages.push_back(std::move(msg));

    tui::ConversationRenderOptions options;
    tui::HistoryComponent history(
        [&messages]() { return messages; },
        tick,
        [&options]() { return options; });

    const auto collapsed = render_history_text(history);
    REQUIRE_THAT(collapsed, Catch::Matchers::ContainsSubstring("Thought for 5s"));
    REQUIRE_THAT(collapsed, !Catch::Matchers::ContainsSubstring("Hidden chain of thought."));

    // The reasoning header is not necessarily on row 0 (tools/answer text render
    // first), so click the row that actually carries the "Thought for" summary.
    const int summary_row = rendered_row_containing(collapsed, "Thought for 5s");
    REQUIRE(summary_row >= 0);

    ftxui::Mouse mouse;
    mouse.button = ftxui::Mouse::Left;
    mouse.motion = ftxui::Mouse::Pressed;
    mouse.x = 1;
    mouse.y = summary_row;
    REQUIRE(history.OnEvent(ftxui::Event::Mouse("", mouse)));

    const auto expanded = render_history_text(history);
    REQUIRE_THAT(expanded, Catch::Matchers::ContainsSubstring("Hidden chain of thought."));
}

// ============================================================================
// Render-cache regression tests
// ============================================================================
// The transcript Element tree is cached and reused across frames while only the
// scroll offset changes. These tests pin that behaviour down: scrolling must
// still move the viewport even though the underlying messages never change, and
// toggling a disclosure must rebuild the cached tree.

TEST_CASE("HistoryComponent wheel scroll moves viewport over cached content",
          "[tui][history_component][render_cache]") {
    std::atomic<size_t> tick{0};

    // A conversation tall enough that a short viewport cannot show both ends.
    std::vector<tui::UiMessage> messages;
    messages.push_back(tui::make_assistant_message("TOP_MARKER_LINE", "", false));
    for (int i = 0; i < 24; ++i) {
        messages.push_back(tui::make_assistant_message(
            "filler paragraph number " + std::to_string(i), "", false));
    }
    messages.push_back(tui::make_assistant_message("BOTTOM_MARKER_LINE", "", false));

    tui::HistoryComponent history(
        [&messages]() { return messages; },
        tick,
        mock_options);

    constexpr int kWidth = 80;
    constexpr int kHeight = 6;

    // First render primes the layout + cache and pins to the bottom.
    const auto at_bottom = render_history_viewport(history, kWidth, kHeight);
    REQUIRE_THAT(at_bottom, Catch::Matchers::ContainsSubstring("BOTTOM_MARKER_LINE"));
    REQUIRE_THAT(at_bottom, !Catch::Matchers::ContainsSubstring("TOP_MARKER_LINE"));

    // Wheel-scroll up several times. The messages vector is untouched, so the
    // *only* thing that can move the viewport is re-applying the scroll anchor
    // over the cached content tree.
    ftxui::Mouse wheel;
    wheel.motion = ftxui::Mouse::Pressed;
    wheel.button = ftxui::Mouse::WheelUp;
    for (int i = 0; i < 40; ++i) {
        REQUIRE(history.HandleWheel(ftxui::Event::Mouse("", wheel)));
    }

    const auto scrolled_up = render_history_viewport(history, kWidth, kHeight);
    REQUIRE_THAT(scrolled_up, Catch::Matchers::ContainsSubstring("TOP_MARKER_LINE"));
    REQUIRE_FALSE(history.IsAutoScrollFollowing());
    REQUIRE(history.ScrollPosition() < 1.0f);
}

TEST_CASE("scroll anchor survives provisional FTXUI layout height",
          "[tui][history_component][render_cache]") {
    auto content = std::make_shared<IterativeHeightNode>();
    auto anchor = std::make_shared<tui::ConversationScrollAnchor>();
    anchor->content_height = 100;
    anchor->focus_y = 90;
    anchor->follow_bottom = false;

    auto panel = tui::apply_scroll_viewport(content, 0.9f, anchor);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(40),
                                        ftxui::Dimension::Fixed(10));
    ftxui::Render(screen, panel);

    REQUIRE(anchor->content_height == 100);
    REQUIRE(anchor->focus_y == 90);
}

TEST_CASE("HistoryComponent render cache is stable across identical frames",
          "[tui][history_component][render_cache]") {
    std::atomic<size_t> tick{0};
    tui::HistoryComponent history(
        [&]() { return mock_messages(); },
        tick,
        mock_options);

    // Repeatedly rendering the same content with no state change must produce
    // identical output — i.e. the cache must not drift or corrupt layout.
    const auto first  = render_history_text(history);
    const auto second = render_history_text(history);
    const auto third  = render_history_text(history);
    REQUIRE(first == second);
    REQUIRE(second == third);
}

TEST_CASE("animation frames update without rebuilding transcript cache",
          "[tui][history_component][render_cache][animation]") {
    std::atomic<size_t> tick{0};
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(
        std::vector<tui::UiMessage>{tui::make_assistant_message("", "", true)});

    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [messages]() { return messages; }
        },
        tick,
        mock_options);

    const auto frame_0 = render_history_text(history);
    REQUIRE(history.CacheBuildCount() == 1);

    tick.store(1, std::memory_order_relaxed);
    const auto frame_1 = render_history_text(history);

    REQUIRE(history.CacheBuildCount() == 1);
    REQUIRE(frame_0 != frame_1);
    REQUIRE_THAT(frame_1, Catch::Matchers::ContainsSubstring("Working."));
}

TEST_CASE("shell pulse updates without rebuilding transcript cache",
          "[tui][history_component][render_cache][animation]") {
    std::atomic<size_t> tick{0};
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(
        std::vector<tui::UiMessage>{
            tui::make_shell_command_message("sleep 1", "", true)
        });
    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [messages]() { return messages; }
        },
        tick,
        mock_options);

    static_cast<void>(render_history_text(history));
    tick.store(3, std::memory_order_relaxed);
    const auto animated = render_history_text(history);

    REQUIRE(history.CacheBuildCount() == 1);
    REQUIRE_THAT(animated, Catch::Matchers::ContainsSubstring("Running..."));
}

TEST_CASE("elapsed activity text updates without rebuilding transcript cache",
          "[tui][history_component][render_cache][animation]") {
    std::atomic<size_t> tick{0};
    std::atomic<int> elapsed_seconds{1};
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(
        std::vector<tui::UiMessage>{tui::make_assistant_message("", "", true)});
    auto options = [&elapsed_seconds]() {
        auto value = mock_options();
        value.activity_elapsed = [&elapsed_seconds](std::string_view) {
            return std::to_string(elapsed_seconds.load(std::memory_order_relaxed)) + "s";
        };
        return value;
    };
    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [messages]() { return messages; }
        },
        tick,
        options);

    const auto first = render_history_text(history);
    elapsed_seconds.store(2, std::memory_order_relaxed);
    const auto second = render_history_text(history);

    REQUIRE(history.CacheBuildCount() == 1);
    REQUIRE_THAT(first, Catch::Matchers::ContainsSubstring("(1s)"));
    REQUIRE_THAT(second, Catch::Matchers::ContainsSubstring("(2s)"));
}

TEST_CASE("tool and subagent spinners update without rebuilding transcript cache",
          "[tui][history_component][render_cache][animation]") {
    std::atomic<size_t> tick{0};
    auto assistant = tui::make_assistant_message("", "", true);
    assistant.thinking = false;
    auto tool = tui::make_tool_activity("parent", "task", "{}", "delegate");
    tool.status = tui::ToolActivity::Status::Executing;
    tui::ToolActivity::SubagentActivity subagent;
    subagent.id = "child";
    subagent.worker_name = "worker";
    subagent.status = tui::ToolActivity::Status::Executing;
    tool.subagents.push_back(std::move(subagent));
    assistant.tools.push_back(std::move(tool));
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(
        std::vector<tui::UiMessage>{std::move(assistant)});

    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [messages]() { return messages; }
        },
        tick,
        mock_options);

    const auto frame_0 = render_history_text(history);
    tick.store(1, std::memory_order_relaxed);
    const auto frame_1 = render_history_text(history);

    REQUIRE(history.CacheBuildCount() == 1);
    REQUIRE(frame_0 != frame_1);
    REQUIRE_THAT(frame_0, Catch::Matchers::ContainsSubstring("○"));
    REQUIRE_THAT(frame_1, Catch::Matchers::ContainsSubstring("◔"));
}

TEST_CASE("large animated transcript retains one structural render",
          "[tui][history_component][render_cache][animation][performance]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> source;
    source.reserve(121);
    for (int i = 0; i < 120; ++i) {
        source.push_back(tui::make_assistant_message(
            "## Result " + std::to_string(i)
                + "\n\nA paragraph with **formatting**, `code`, and enough text to wrap "
                  "across a normal terminal viewport without changing between frames.",
            "",
            false));
    }
    source.push_back(tui::make_assistant_message("", "", true));
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(std::move(source));

    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [messages]() { return messages; }
        },
        tick,
        mock_options);

    for (std::size_t frame = 0; frame < 30; ++frame) {
        tick.store(frame, std::memory_order_relaxed);
        const auto rendered = render_history_viewport(history, 120, 40);
        REQUIRE_FALSE(rendered.empty());
    }

    REQUIRE(history.CacheBuildCount() == 1);
}

TEST_CASE("cached transcript frames perform no transcript-sized work",
          "[tui][history_component][render_cache][performance]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> source;
    source.reserve(200);
    for (int i = 0; i < 200; ++i) {
        source.push_back(tui::make_assistant_message(
            "## Result " + std::to_string(i)
                + "\n\nA detailed paragraph with **formatting**, `code`, and wrapping.",
            "",
            false));
    }
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(std::move(source));
    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [messages]() { return messages; }
        },
        tick,
        mock_options);

    static_cast<void>(render_history_viewport(history, 100, 20));
    const auto measured = history.MessageMeasureCount();
    const auto viewport_builds = history.ViewportBuildCount();
    REQUIRE(measured == messages->size());
    REQUIRE(viewport_builds == 1);

    for (int frame = 0; frame < 30; ++frame) {
        static_cast<void>(render_history_viewport(history, 100, 20));
    }

    REQUIRE(history.CacheBuildCount() == 1);
    REQUIRE(history.MessageMeasureCount() == measured);
    REQUIRE(history.ViewportBuildCount() == viewport_builds);
}

TEST_CASE("virtualized viewport matches the canonical FTXUI transcript layout",
          "[tui][history_component][render_cache][layout]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> source;
    source.push_back(tui::make_user_message(
        "Explain why a cached DOM can still be expensive to lay out.", "12:34"));
    source.push_back(tui::make_assistant_message(
        "## Layout\n\nThe **visible result** must match the canonical renderer, including "
        "wrapping, borders, spacing, and the bottom scroll anchor.\n\n"
        "- first item\n- second item\n- third item",
        "",
        false));
    for (int i = 0; i < 12; ++i) {
        source.push_back(tui::make_info_message("status line " + std::to_string(i)));
    }
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(source);
    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [messages]() { return messages; }
        },
        tick,
        mock_options);

    const auto reference = render_reference_viewport(source, 72, 14);
    const auto virtualized = render_history_viewport(history, 72, 14);
    REQUIRE(virtualized == reference);
}

TEST_CASE("viewport raster composes with outer FTXUI decorators",
          "[tui][history_component][render_cache][layout]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> source{
        tui::make_user_message("outer style remains visible", ""),
        tui::make_assistant_message("inner **styles** still win", "", false),
    };
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(source);
    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [messages]() { return messages; }
        },
        tick,
        mock_options);

    auto render = [](ftxui::Element panel) {
        panel = std::move(panel) | ftxui::color(ftxui::Color::Blue);
        auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(64),
                                            ftxui::Dimension::Fixed(10));
        ftxui::Render(screen, panel);
        return screen.ToString();
    };

    auto reference_panel = tui::render_history_panel(source, 0, mock_options());
    REQUIRE(render(history.OnRender()) == render(std::move(reference_panel)));
}

TEST_CASE("scrolling rebuilds only the bounded viewport",
          "[tui][history_component][render_cache][performance]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> source;
    source.reserve(150);
    for (int i = 0; i < 150; ++i) {
        source.push_back(tui::make_assistant_message(
            "message " + std::to_string(i) + " with enough content to wrap", "", false));
    }
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(std::move(source));
    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [messages]() { return messages; }
        },
        tick,
        mock_options);

    static_cast<void>(render_history_viewport(history, 80, 10));
    const auto measured = history.MessageMeasureCount();
    const auto viewport_builds = history.ViewportBuildCount();
    history.ScrollPageUp();
    static_cast<void>(render_history_viewport(history, 80, 10));

    REQUIRE(history.MessageMeasureCount() == measured);
    REQUIRE(history.ViewportBuildCount() == viewport_builds + 1);
}

TEST_CASE("terminal resize remeasures once and then returns to bounded frames",
          "[tui][history_component][render_cache][performance][layout]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> source;
    source.reserve(40);
    for (int i = 0; i < 40; ++i) {
        source.push_back(tui::make_assistant_message(
            "A wrapping message " + std::to_string(i)
                + " with enough words to have a width-dependent height.",
            "",
            false));
    }
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(source);
    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [messages]() { return messages; }
        },
        tick,
        mock_options);

    static_cast<void>(render_history_viewport(history, 100, 12));
    REQUIRE(history.MessageMeasureCount() == messages->size());

    const auto resized = render_history_viewport(history, 52, 12);
    REQUIRE(history.MessageMeasureCount() == 2 * messages->size());
    REQUIRE(resized == render_reference_viewport(source, 52, 12));

    const auto measured = history.MessageMeasureCount();
    static_cast<void>(render_history_viewport(history, 52, 12));
    REQUIRE(history.MessageMeasureCount() == measured);
}

TEST_CASE("streaming update remeasures only the changed message",
          "[tui][history_component][render_cache][performance]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> source;
    source.reserve(80);
    for (int i = 0; i < 80; ++i) {
        source.push_back(tui::make_assistant_message(
            "stable message " + std::to_string(i), "", false));
    }
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(source);
    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [&messages]() { return messages; }
        },
        tick,
        mock_options);

    static_cast<void>(render_history_viewport(history, 90, 12));
    const auto measured = history.MessageMeasureCount();

    source.back().text += "\nstreamed continuation";
    messages = std::make_shared<const std::vector<tui::UiMessage>>(source);
    static_cast<void>(render_history_viewport(history, 90, 12));

    REQUIRE(history.CacheBuildCount() == 2);
    REQUIRE(history.MessageMeasureCount() == measured + 1);
}

TEST_CASE("transcript selection keeps native visible-card semantics",
          "[tui][history_component][render_cache][selection]") {
    std::atomic<size_t> tick{0};
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(
        std::vector<tui::UiMessage>{
            tui::make_user_message("SELECT_THIS_EXACT_TEXT", "")
        });
    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [messages]() { return messages; }
        },
        tick,
        mock_options);

    static_cast<void>(render_history_viewport(history, 60, 6));
    const auto selected = select_history_viewport(
        history, 60, 6, ftxui::Selection(0, 0, 58, 5));
    REQUIRE_THAT(selected, Catch::Matchers::ContainsSubstring("SELECT_THIS_EXACT_TEXT"));
}

TEST_CASE("transcript selection cannot paint outside its viewport",
          "[tui][history_component][render_cache][selection][regression]") {
    std::atomic<size_t> tick{0};
    std::string long_message;
    for (int line = 0; line < 30; ++line) {
        long_message += "OFFSCREEN_TRANSCRIPT_LINE_" + std::to_string(line) + "\n";
    }
    auto messages = std::make_shared<const std::vector<tui::UiMessage>>(
        std::vector<tui::UiMessage>{
            tui::make_assistant_message(std::move(long_message), "", false)
        });
    tui::HistoryComponent history(
        std::function<tui::HistoryComponent::MessageSnapshot()>{
            [messages]() { return messages; }
        },
        tick,
        mock_options);

    constexpr int kWidth = 60;
    constexpr int kHeight = 8;
    auto panel = ftxui::vbox({
        ftxui::text("BANNER_MUST_STAY_VISIBLE"),
        history.OnRender() | ftxui::flex,
    });
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(kWidth),
                                        ftxui::Dimension::Fixed(kHeight));
    ftxui::Selection selection(0, 1, kWidth - 1, kHeight - 1);
    ftxui::Render(screen, panel.get(), selection);

    const auto rendered = strip_ansi(screen.ToString());
    REQUIRE(rendered.starts_with("BANNER_MUST_STAY_VISIBLE"));
    const auto first_transcript_line = rendered.find("OFFSCREEN_TRANSCRIPT_LINE_");
    REQUIRE(first_transcript_line != std::string::npos);
    REQUIRE(first_transcript_line > rendered.find('\n'));
}

TEST_CASE("HistoryComponent invalidates render cache when tool approval changes",
          "[tui][history_component][render_cache]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> messages;
    auto assistant = tui::make_assistant_message("", "", true);
    assistant.tools.push_back(tui::make_tool_activity(
        "tool-1", "read_file", R"({"path":"README.md"})", "Read README.md"));
    messages.push_back(std::move(assistant));

    tui::HistoryComponent history(
        [&messages]() { return messages; },
        tick,
        mock_options);

    const auto before_approval = render_history_text(history);
    REQUIRE_THAT(before_approval,
                 !Catch::Matchers::ContainsSubstring("auto-approved"));

    // YOLO approval changes a pending tool in place. Pending tools do not run
    // the animation ticker, so this field must invalidate the content cache on
    // its own when MainApp wakes the UI.
    messages.front().tools.front().auto_approved = true;

    const auto after_approval = render_history_text(history);
    REQUIRE_THAT(after_approval,
                 Catch::Matchers::ContainsSubstring("auto-approved"));
}

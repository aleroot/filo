#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/HistoryComponent.hpp"
#include "tui/Conversation.hpp"

#include <ftxui/screen/screen.hpp>

#include <atomic>
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
}

TEST_CASE("HistoryComponent layout fingerprinting", "[tui][history_component]") {
    std::atomic<size_t> tick{0};
    tui::HistoryComponent history(mock_messages, tick, mock_options);

    auto msgs1 = mock_messages();
    auto msgs2 = msgs1;
    
    // Fingerprints should be same for same messages
    // history_layout_fingerprint is private, but we can indirectly test it
    // via OnRender if we had more access, but let's just test that 
    // it handles basic operations.

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

    history.ScrollUp(0.1f);

    REQUIRE_FALSE(history.IsAutoScrollFollowing());
    REQUIRE(history.ScrollPosition() < 1.0f);
}

TEST_CASE("HistoryComponent HandleWheel up suspends auto-follow",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    tui::HistoryComponent history(mock_messages, tick, mock_options);

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

    // Prime last_message_count_ by rendering once.
    history.OnRender();

    // User scrolls up to read.
    history.ScrollUp(0.3f);
    REQUIRE_FALSE(history.IsAutoScrollFollowing());
    const float pos_before = history.ScrollPosition();

    // A new message arrives.
    messages.push_back(tui::make_info_message("New tool call"));
    history.OnRender();

    // Scroll position must NOT have been yanked to 1.0f. Because the content
    // became taller while the user was held, the equivalent visual anchor is
    // represented by a smaller ratio.
    REQUIRE(history.ScrollPosition() < pos_before);
    // Badge must appear.
    REQUIRE(history.HasNewContentIndicator());
}

TEST_CASE("HistoryComponent preserves held anchor while streaming grows",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    std::vector<tui::UiMessage> messages;
    messages.push_back(tui::make_user_message("prompt", ""));
    messages.push_back(tui::make_assistant_message(std::string(800, 'a'), "", true));
    tui::HistoryComponent history(
        [&messages]() { return messages; },
        tick,
        mock_options);

    history.OnRender();
    history.ScrollUp(0.02f);
    REQUIRE_FALSE(history.IsAutoScrollFollowing());
    const float pos_before = history.ScrollPosition();

    messages.back().text += std::string(2400, 'b');
    history.OnRender();

    REQUIRE_FALSE(history.IsAutoScrollFollowing());
    REQUIRE(history.HasNewContentIndicator());
    REQUIRE(history.ScrollPosition() < pos_before);
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
    history.OnRender();
    REQUIRE(history.IsAutoScrollFollowing());

    // Simulate a partial scroll down that keeps us in the "near bottom" zone.
    // (Do NOT scroll up, so intent stays FollowBottom.)
    messages.push_back(tui::make_info_message("Status update"));
    history.OnRender();

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
    history.OnRender();
    history.ScrollUp(0.4f);
    messages.push_back(tui::make_info_message("New content"));
    history.OnRender();
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

    history.OnRender();
    history.ScrollUp(0.4f);
    messages.push_back(tui::make_info_message("New content"));
    history.OnRender();
    REQUIRE_FALSE(history.IsAutoScrollFollowing());
    REQUIRE(history.HasNewContentIndicator());

    messages.clear();
    messages.push_back(tui::make_system_message("Fresh transcript"));
    history.OnRender();

    REQUIRE(history.IsAutoScrollFollowing());
    REQUIRE(history.ScrollPosition() == 1.0f);
    REQUIRE_FALSE(history.HasNewContentIndicator());
}

TEST_CASE("HistoryComponent JumpToMessage suspends follow when not at end",
          "[tui][history_component][auto_scroll]") {
    std::atomic<size_t> tick{0};
    tui::HistoryComponent history(mock_messages, tick, mock_options);

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

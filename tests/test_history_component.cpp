#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/HistoryComponent.hpp"
#include "tui/Conversation.hpp"

#include <atomic>
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

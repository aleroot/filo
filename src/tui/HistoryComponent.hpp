#pragma once

#include "Conversation.hpp"
#include "TranscriptViewport.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/screen/box.hpp>

namespace tui {

class HistoryComponent : public ftxui::ComponentBase {
public:
    using MessageSnapshot = std::shared_ptr<const std::vector<UiMessage>>;

    HistoryComponent(
        std::function<std::vector<UiMessage>()> get_messages,
        const std::atomic<size_t>& animation_tick,
        std::function<ConversationRenderOptions()> get_options);
    HistoryComponent(
        std::function<MessageSnapshot()> get_messages,
        const std::atomic<size_t>& animation_tick,
        std::function<ConversationRenderOptions()> get_options);

    ftxui::Element OnRender() override;
    bool Focusable() const override;
    bool OnEvent(ftxui::Event event) override;

    void ScrollUp(float amount);
    void ScrollDown(float amount);
    void ScrollPageUp();
    void ScrollPageDown();
    void JumpToMessage(std::size_t message_index, std::size_t message_count);

    // Unconditionally snap to bottom and resume auto-follow.
    // Call this when the user submits a new turn.
    void ResetToBottom();

    // Helper for external controllers (e.g. input forwarding wheel events).
    bool HandleWheel(ftxui::Event event);

    // Read-only introspection (used by unit tests and diagnostics).
    [[nodiscard]] bool  IsAutoScrollFollowing()  const noexcept;
    [[nodiscard]] bool  HasNewContentIndicator() const noexcept;
    [[nodiscard]] float ScrollPosition()         const noexcept;
    [[nodiscard]] std::size_t CacheBuildCount() const noexcept;
    [[nodiscard]] std::size_t ViewportBuildCount() const noexcept;
    [[nodiscard]] std::size_t MessageMeasureCount() const noexcept;

private:
    // ── Scroll-intent state machine ──────────────────────────────────────
    // FollowBottom: auto-scroll tracks new content (default).
    // UserHeld:     user explicitly scrolled away; auto-scroll is suspended.
    enum class ScrollIntent { FollowBottom, UserHeld };

    void SetScrollIntent(ScrollIntent intent);
    [[nodiscard]] bool  IsNearBottom()          const noexcept;
    [[nodiscard]] int CurrentFocusY() const noexcept;
    void ScrollByLines(int lines);

    bool OnMouseEvent(ftxui::Event event);

    // Folding hash of every option/disclosure signal that changes card layout.
    std::size_t compute_render_cache_key(const ConversationRenderOptions& options) const;
    static std::size_t combine_hash(std::size_t seed, std::size_t value);

    std::function<MessageSnapshot()> get_messages_;
    const std::atomic<size_t>& animation_tick_;
    std::function<ConversationRenderOptions()> get_options_;

    ScrollIntent   scroll_intent_           = ScrollIntent::FollowBottom;
    bool           new_content_while_held_  = false;
    TranscriptViewport transcript_viewport_;
    std::shared_ptr<ConversationScrollAnchor> scroll_anchor_ =
        std::make_shared<ConversationScrollAnchor>();
    std::unordered_map<std::string, bool>        disclosure_expanded_;
    std::unordered_map<std::string, ftxui::Box>  disclosure_hitboxes_;
};

} // namespace tui

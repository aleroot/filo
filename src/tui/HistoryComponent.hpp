#pragma once

#include "Conversation.hpp"

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

    // Folding hash of every signal that can change the built transcript tree.
    // Animation state is deliberately excluded: reactive leaf nodes read the
    // live tick without invalidating Markdown and tool-card structure.
    std::size_t compute_render_cache_key(std::size_t content_fingerprint,
                                         const ConversationRenderOptions& options) const;
    static std::size_t combine_hash(std::size_t seed, std::size_t value);
    static std::size_t history_content_fingerprint(const std::vector<UiMessage>& messages);

    // ── Render cache ────────────────────────────────────────────────────
    // The transcript Element tree is expensive to build (markdown parsing,
    // tool cards, borders). It only depends on content/options, NOT on the
    // scroll offset or terminal width, so we reuse it across frames while the
    // user scrolls or reads. This is what keeps scrolling responsive — even a
    // huge conversation costs one full rebuild per content change instead of
    // per frame.
    ftxui::Element cached_content_;
    std::size_t    cache_key_  = 0;
    bool           has_cache_  = false;
    std::size_t    cache_build_count_ = 0;

    std::function<MessageSnapshot()> get_messages_;
    const std::atomic<size_t>& animation_tick_;
    std::function<ConversationRenderOptions()> get_options_;

    ScrollIntent   scroll_intent_           = ScrollIntent::FollowBottom;
    bool           new_content_while_held_  = false;
    size_t         last_message_count_      = 0;
    std::size_t    last_content_fingerprint_ = 0;
    bool           has_content_snapshot_     = false;
    MessageSnapshot last_message_snapshot_;
    std::shared_ptr<ConversationScrollAnchor> scroll_anchor_ =
        std::make_shared<ConversationScrollAnchor>();
    std::unordered_map<std::string, bool>        disclosure_expanded_;
    std::unordered_map<std::string, ftxui::Box>  disclosure_hitboxes_;
};

} // namespace tui

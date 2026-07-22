#include "HistoryComponent.hpp"

#include "TuiTheme.hpp"

#include <algorithm>
#include <cmath>

namespace tui {
namespace {

constexpr int kWheelScrollLines = 3;
constexpr int kArrowScrollLines = 2;
constexpr int kPageScrollLines = 18;
constexpr int kBottomThresholdLines = 3;

} // namespace

HistoryComponent::HistoryComponent(
    std::function<std::vector<UiMessage>()> get_messages,
    const std::atomic<size_t>& animation_tick,
    std::function<ConversationRenderOptions()> get_options)
    : HistoryComponent(
          [get_messages = std::move(get_messages)]() {
              return std::make_shared<const std::vector<UiMessage>>(get_messages());
          },
          animation_tick,
          std::move(get_options)) {}

HistoryComponent::HistoryComponent(
    std::function<MessageSnapshot()> get_messages,
    const std::atomic<size_t>& animation_tick,
    std::function<ConversationRenderOptions()> get_options)
    : get_messages_(std::move(get_messages)),
      animation_tick_(animation_tick),
      get_options_(std::move(get_options)) {}

ftxui::Element HistoryComponent::OnRender() {
    const auto snapshot = get_messages_();
    auto options = get_options_();
    options.system_disclosure_expanded = &disclosure_expanded_;

    const auto sync = transcript_viewport_.Sync(
        snapshot,
        compute_render_cache_key(options));

    if (sync.current_count < sync.previous_count) {
        // The transcript was cleared or replaced; any prior user-held scroll
        // position belongs to the old content.
        ResetToBottom();
    } else if (sync.current_count > sync.previous_count) {
        // A new message card was added (user turn, tool card, assistant card, …).
        if (scroll_intent_ == ScrollIntent::FollowBottom) {
            scroll_anchor_->follow_bottom = true;  // keep pinned to the bottom
        } else {
            new_content_while_held_ = true;         // do not yank; show indicator
        }
    }

    // A streamed update can change an existing card without adding a message.
    // Surface it to a reader, but leave the anchor untouched: the scroll
    // decorator keeps the same FTXUI line in view.
    if (sync.content_changed && scroll_intent_ == ScrollIntent::UserHeld) {
        new_content_while_held_ = true;
    }

    const std::size_t tick = animation_tick_.load(std::memory_order_relaxed);
    options.animation_tick = &animation_tick_;
    auto panel = transcript_viewport_.Render(
        tick, options, scroll_anchor_, &disclosure_hitboxes_);

    // Overlay a "\u2193 new content" badge when new content arrived while the
    // user was scrolled up. It disappears as soon as they return to the bottom.
    if (!new_content_while_held_) {
        return panel;
    }

    auto badge = ftxui::text(" \u2193 new content ")
        | ftxui::bold
        | ftxui::color(ftxui::Color::Black)
        | ftxui::bgcolor(static_cast<ftxui::Color>(tui::ColorYellowBright));

    return ftxui::dbox({
        std::move(panel),
        ftxui::vbox({
            ftxui::filler(),
            ftxui::hbox({ftxui::filler(), std::move(badge)})
        })
    });
}

bool HistoryComponent::Focusable() const {
    return true;
}

bool HistoryComponent::OnEvent(ftxui::Event event) {
    if (event.is_mouse()) {
        return OnMouseEvent(event);
    }

    if (Focused()) {
        if (event == ftxui::Event::ArrowUp) {
            ScrollByLines(-kArrowScrollLines);
            return true;
        }
        if (event == ftxui::Event::ArrowDown) {
            ScrollByLines(kArrowScrollLines);
            return true;
        }
        if (event == ftxui::Event::PageUp) {
            ScrollByLines(-kPageScrollLines);
            return true;
        }
        if (event == ftxui::Event::PageDown) {
            ScrollByLines(kPageScrollLines);
            return true;
        }
        if (event == ftxui::Event::Home) {
            scroll_anchor_->focus_y = 0;
            SetScrollIntent(ScrollIntent::UserHeld);
            return true;
        }
        if (event == ftxui::Event::End) {
            scroll_anchor_->focus_y = scroll_anchor_->content_height;
            SetScrollIntent(ScrollIntent::FollowBottom);
            return true;
        }
        // Escape or Enter returns focus to the input.
        if (event == ftxui::Event::Escape || event == ftxui::Event::Return) {
            return false;
        }
    }
    return false;
}

void HistoryComponent::ScrollUp(float amount) {
    const int lines = std::max(1, static_cast<int>(std::ceil(
        amount * static_cast<float>(std::max(scroll_anchor_->content_height, 1)))));
    ScrollByLines(-lines);
}

void HistoryComponent::ScrollDown(float amount) {
    const int lines = std::max(1, static_cast<int>(std::ceil(
        amount * static_cast<float>(std::max(scroll_anchor_->content_height, 1)))));
    ScrollByLines(lines);
}

void HistoryComponent::ScrollPageUp() {
    ScrollByLines(-kPageScrollLines);
}

void HistoryComponent::ScrollPageDown() {
    ScrollByLines(kPageScrollLines);
}

void HistoryComponent::JumpToMessage(std::size_t message_index, std::size_t message_count) {
    if (message_count <= 1) {
        scroll_anchor_->focus_y = scroll_anchor_->content_height;
        SetScrollIntent(ScrollIntent::FollowBottom);
        return;
    }
    const float ratio = static_cast<float>(message_index)
        / static_cast<float>(message_count - 1);
    scroll_anchor_->focus_y = static_cast<int>(std::lround(
        std::clamp(ratio, 0.0f, 1.0f)
        * static_cast<float>(scroll_anchor_->content_height)));
    // Navigating to the last message resumes auto-follow; anything else is a
    // deliberate user jump that suspends it.
    SetScrollIntent(IsNearBottom() ? ScrollIntent::FollowBottom
                                   : ScrollIntent::UserHeld);
}

// ── Scroll-intent state machine ───────────────────────────────────────────

void HistoryComponent::ResetToBottom() {
    scroll_anchor_->focus_y = scroll_anchor_->content_height;
    SetScrollIntent(ScrollIntent::FollowBottom);
}

bool HistoryComponent::IsAutoScrollFollowing() const noexcept {
    return scroll_intent_ == ScrollIntent::FollowBottom;
}

bool HistoryComponent::HasNewContentIndicator() const noexcept {
    return new_content_while_held_;
}

float HistoryComponent::ScrollPosition() const noexcept {
    if (scroll_intent_ == ScrollIntent::FollowBottom) {
        return 1.0f;
    }
    return static_cast<float>(CurrentFocusY())
        / static_cast<float>(std::max(scroll_anchor_->content_height, 1));
}

std::size_t HistoryComponent::CacheBuildCount() const noexcept {
    return transcript_viewport_.CacheBuildCount();
}

std::size_t HistoryComponent::ViewportBuildCount() const noexcept {
    return transcript_viewport_.ViewportBuildCount();
}

std::size_t HistoryComponent::MessageMeasureCount() const noexcept {
    return transcript_viewport_.MessageMeasureCount();
}

void HistoryComponent::SetScrollIntent(ScrollIntent intent) {
    scroll_intent_ = intent;
    scroll_anchor_->follow_bottom = intent == ScrollIntent::FollowBottom;
    // Returning to FollowBottom clears the indicator badge — the user
    // has scrolled back down and will see the new content naturally.
    if (intent == ScrollIntent::FollowBottom) {
        new_content_while_held_ = false;
    }
}

bool HistoryComponent::IsNearBottom() const noexcept {
    return CurrentFocusY()
        >= std::max(0, scroll_anchor_->content_height - kBottomThresholdLines);
}

int HistoryComponent::CurrentFocusY() const noexcept {
    return std::clamp(scroll_anchor_->focus_y, 0, scroll_anchor_->content_height);
}

void HistoryComponent::ScrollByLines(int lines) {
    const int content_height = std::max(scroll_anchor_->content_height, 1);
    scroll_anchor_->focus_y = std::clamp(CurrentFocusY() + lines, 0, content_height);
    if (lines < 0) {
        SetScrollIntent(ScrollIntent::UserHeld);
    } else if (IsNearBottom()) {
        SetScrollIntent(ScrollIntent::FollowBottom);
    } else {
        SetScrollIntent(ScrollIntent::UserHeld);
    }
}

bool HistoryComponent::HandleWheel(ftxui::Event event) {
    // Single source of truth for mouse-wheel scrolling.
    //
    // Direction follows the universal terminal convention: WheelUp moves toward
    // the top of the transcript, WheelDown toward the bottom. The terminal
    // emulator is responsible for mapping the user's physical gesture —
    // including the macOS "natural scrolling" preference — onto these two
    // button codes, so no OS-specific inversion belongs here. (Verified against
    // FTXUI's Frame layout: raising focus_y scrolls the viewport toward later
    // content, so WheelDown → focus_y↑ is correct.)
    if (!event.is_mouse()) {
        return false;
    }
    if (event.mouse().button == ftxui::Mouse::WheelUp) {
        ScrollByLines(-kWheelScrollLines);
        return true;
    }
    if (event.mouse().button == ftxui::Mouse::WheelDown) {
        ScrollByLines(kWheelScrollLines);
        return true;
    }
    return false;
}

bool HistoryComponent::OnMouseEvent(ftxui::Event event) {
    // Clicking a system-message disclosure chevron toggles its details.
    if (event.mouse().button == ftxui::Mouse::Left
        && event.mouse().motion == ftxui::Mouse::Pressed) {
        for (const auto& [message_id, box] : disclosure_hitboxes_) {
            if (box.Contain(event.mouse().x, event.mouse().y)) {
                disclosure_expanded_[message_id] = !disclosure_expanded_[message_id];
                return true;
            }
        }
    }

    // Everything else of interest is the mouse wheel — route it through the
    // single shared handler. Returning false for plain clicks/drags leaves them
    // to FTXUI's text-selection machinery and never steals focus.
    return HandleWheel(event);
}

std::size_t HistoryComponent::compute_render_cache_key(
    const ConversationRenderOptions& options) const {
    std::size_t seed = 0;
    seed = combine_hash(seed, options.show_timestamps ? 1 : 0);
    seed = combine_hash(seed, options.show_spinner ? 1 : 0);
    seed = combine_hash(seed, options.show_reasoning ? 1 : 0);
    seed = combine_hash(seed, options.expand_system_details ? 1 : 0);
    seed = combine_hash(seed, options.expand_tool_results ? 1 : 0);
    seed = combine_hash(seed, options.tool_result_preview_max_lines);
    // Per-message disclosure (▶/▼) toggle state — toggling a card must rebuild.
    for (const auto& [id, expanded] : disclosure_expanded_) {
        seed = combine_hash(seed, std::hash<std::string>{}(id));
        seed = combine_hash(seed, expanded ? 1 : 0);
    }
    return seed;
}

std::size_t HistoryComponent::combine_hash(std::size_t seed, std::size_t value) {
    constexpr std::size_t kMix = static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    seed ^= value + kMix + (seed << 6) + (seed >> 2);
    return seed;
}

} // namespace tui

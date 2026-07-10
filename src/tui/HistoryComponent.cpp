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
    : get_messages_(std::move(get_messages)),
      animation_tick_(animation_tick),
      get_options_(std::move(get_options)) {}

ftxui::Element HistoryComponent::OnRender() {
    // Snapshot messages and build render options.
    const auto messages = get_messages_();
    auto options = get_options_();
    options.scroll_pos = ScrollPosition();
    options.scroll_anchor = scroll_anchor_;
    options.system_disclosure_expanded = &disclosure_expanded_;
    disclosure_hitboxes_.clear();
    options.system_disclosure_hitboxes = &disclosure_hitboxes_;

    // ── Intent-aware auto-scroll ─────────────────────────────────────────────
    const std::size_t content_fingerprint = history_content_fingerprint(messages);
    const bool content_changed = has_content_snapshot_
        && content_fingerprint != last_content_fingerprint_;
    last_content_fingerprint_ = content_fingerprint;
    has_content_snapshot_ = true;

    const std::size_t prev_message_count = last_message_count_;
    last_message_count_ = messages.size();

    if (messages.size() < prev_message_count) {
        // The transcript was cleared or replaced; any prior user-held scroll
        // position belongs to the old content.
        ResetToBottom();
    } else if (messages.size() > prev_message_count) {
        // A new message card was added (user turn, tool card, assistant card, etc.).
        if (scroll_intent_ == ScrollIntent::FollowBottom) {
            // User is watching — keep pinned to the bottom.
            scroll_anchor_->follow_bottom = true;
        } else {
            // User is reading — do NOT yank; show indicator instead.
            new_content_while_held_ = true;
        }
    }

    // A streamed update can change an existing assistant card without adding a
    // message. Surface that update to a reader, but leave the actual anchor
    // untouched: the scroll decorator keeps the same FTXUI line in view.
    if (content_changed && scroll_intent_ == ScrollIntent::UserHeld) {
        new_content_while_held_ = true;
    }
    // ───────────────────────────────────────────────────────────────────

    // Update scroll_pos for the render options snapshot.
    options.scroll_pos = ScrollPosition();
    options.scroll_anchor = scroll_anchor_;

    auto panel = render_history_panel(
        messages,
        animation_tick_.load(std::memory_order_relaxed),
        options);

    // Overlay a "\u2193 new content" badge when new content arrived while the user
    // was scrolled up. The badge disappears as soon as they return to the bottom.
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
    if (event.mouse().button == ftxui::Mouse::Left
        && event.mouse().motion == ftxui::Mouse::Pressed) {
        for (const auto& [message_id, box] : disclosure_hitboxes_) {
            if (box.Contain(event.mouse().x, event.mouse().y)) {
                disclosure_expanded_[message_id] = !disclosure_expanded_[message_id];
                return true;
            }
        }
    }

    // Only handle wheel events; never steal focus from clicks.
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

std::size_t HistoryComponent::combine_hash(std::size_t seed, std::size_t value) {
    constexpr std::size_t kMix = static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    seed ^= value + kMix + (seed << 6) + (seed >> 2);
    return seed;
}

std::size_t HistoryComponent::history_content_fingerprint(
    const std::vector<UiMessage>& messages) {
    std::size_t seed = messages.size();
    const auto hash_text = [](std::string_view value) {
        return std::hash<std::string_view>{}(value);
    };

    for (const auto& msg : messages) {
        seed = combine_hash(seed, hash_text(msg.id));
        seed = combine_hash(seed, static_cast<std::size_t>(msg.type));
        seed = combine_hash(seed, hash_text(msg.text));
        seed = combine_hash(seed, hash_text(msg.secondary_text));
        seed = combine_hash(seed, hash_text(msg.disclosure_text));
        seed = combine_hash(seed, hash_text(msg.timestamp));
        seed = combine_hash(seed, msg.repeat_count);
        seed = combine_hash(seed, static_cast<std::size_t>(msg.pending));
        seed = combine_hash(seed, static_cast<std::size_t>(msg.thinking));
        seed = combine_hash(seed, static_cast<std::size_t>(msg.show_lightbulb));
        seed = combine_hash(seed, static_cast<std::size_t>(msg.stopped));
        seed = combine_hash(seed, hash_text(msg.activity_elapsed));
        for (const auto& tool : msg.tools) {
            seed = combine_hash(seed, hash_text(tool.id));
            seed = combine_hash(seed, hash_text(tool.name));
            seed = combine_hash(seed, hash_text(tool.description));
            seed = combine_hash(seed, hash_text(tool.result.summary));
            seed = combine_hash(seed, static_cast<std::size_t>(tool.status));
            seed = combine_hash(seed, static_cast<std::size_t>(tool.result.truncated));
            seed = combine_hash(seed, tool.diff_preview.lines.size());
            seed = combine_hash(
                seed,
                static_cast<std::size_t>(tool.diff_preview.hidden_line_count));
            for (const auto& subagent : tool.subagents) {
                seed = combine_hash(seed, hash_text(subagent.id));
                seed = combine_hash(seed, hash_text(subagent.latest_text));
                seed = combine_hash(seed, hash_text(subagent.summary));
                seed = combine_hash(seed, static_cast<std::size_t>(subagent.status));
            }
        }
    }
    return seed;
}

} // namespace tui

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
    const auto messages = get_messages_();
    auto options = get_options_();
    options.system_disclosure_expanded = &disclosure_expanded_;

    // ── Intent-aware auto-scroll ─────────────────────────────────────────────
    // This state machine is cheap (bookkeeping only) and must run every frame,
    // independent of whether the content tree is rebuilt or served from cache.
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
    if (content_changed && scroll_intent_ == ScrollIntent::UserHeld) {
        new_content_while_held_ = true;
    }

    // ── Content Element cache ────────────────────────────────────────────────
    // Rebuilding the transcript tree (markdown parsing, tool cards, borders) is
    // by far the most expensive part of a frame. While scrolling or reading,
    // nothing about the *content* changes — only the scroll offset does — so we
    // build the tree once per content change and reuse it on every other frame,
    // re-applying only the cheap scroll decorator below.
    //
    // The cache key folds together every signal that can alter the built tree.
    // The animation tick is included *only* while something is animating, so an
    // idle reading/scrolling session never invalidates the cache.
    const std::size_t tick = animation_tick_.load(std::memory_order_relaxed);
    const bool animating = conversation_uses_animation(messages, options.show_spinner);
    const std::size_t key = compute_render_cache_key(
        content_fingerprint, options, animating ? tick : 0);

    if (!has_cache_ || key != cache_key_) {
        cache_key_ = key;
        has_cache_ = true;
        // Drop the old tree first so its reflect() nodes release their Box
        // references before we clear the hitbox map and rebuild.
        cached_content_.reset();
        disclosure_hitboxes_.clear();
        options.system_disclosure_hitboxes = &disclosure_hitboxes_;
        cached_content_ = render_history_content(messages, tick, options);
    }
    // On a cache hit we intentionally leave disclosure_hitboxes_ alone: the
    // cached reflect() nodes still reference its entries and refresh their
    // coordinates on every frame during layout (SetBox).

    // ── Scroll viewport (cheap; recomputed every frame) ─────────────────────
    auto panel = apply_scroll_viewport(
        cached_content_, ScrollPosition(), scroll_anchor_);

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
    std::size_t content_fingerprint,
    const ConversationRenderOptions& options,
    std::size_t tick_or_zero) const {
    std::size_t seed = content_fingerprint;
    seed = combine_hash(seed, options.show_timestamps ? 1 : 0);
    seed = combine_hash(seed, options.show_spinner ? 1 : 0);
    seed = combine_hash(seed, options.expand_system_details ? 1 : 0);
    seed = combine_hash(seed, options.expand_tool_results ? 1 : 0);
    seed = combine_hash(seed, options.tool_result_preview_max_lines);
    // Per-message disclosure (▶/▼) toggle state — toggling a card must rebuild.
    for (const auto& [id, expanded] : disclosure_expanded_) {
        seed = combine_hash(seed, std::hash<std::string>{}(id));
        seed = combine_hash(seed, expanded ? 1 : 0);
    }
    // Animation frame, but only folded in by the caller while animating.
    seed = combine_hash(seed, tick_or_zero);
    return seed;
}

std::size_t HistoryComponent::combine_hash(std::size_t seed, std::size_t value) {
    constexpr std::size_t kMix = static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    seed ^= value + kMix + (seed << 6) + (seed >> 2);
    return seed;
}

std::size_t HistoryComponent::history_content_fingerprint(
    const std::vector<UiMessage>& messages) {
    std::size_t seed = messages.size();
    const auto add_value = [&seed](std::size_t value) {
        seed = combine_hash(seed, value);
    };
    const auto add_text = [&add_value](std::string_view value) {
        add_value(std::hash<std::string_view>{}(value));
    };
    const auto add_optional_int = [&add_value](const std::optional<int>& value) {
        add_value(value.has_value() ? 1U : 0U);
        if (value.has_value()) {
            add_value(static_cast<std::size_t>(*value));
        }
    };

    for (const auto& msg : messages) {
        add_text(msg.id);
        add_value(static_cast<std::size_t>(msg.type));
        add_text(msg.text);
        add_text(msg.secondary_text);
        add_text(msg.disclosure_text);
        add_text(msg.icon);
        add_value(msg.repeat_count);
        add_value(msg.custom_color.has_value() ? 1U : 0U);
        if (msg.custom_color.has_value()) {
            add_text(msg.custom_color->Print(false));
        }
        add_value(static_cast<std::size_t>(msg.margin_top));
        add_value(static_cast<std::size_t>(msg.pending));
        add_value(static_cast<std::size_t>(msg.thinking));
        add_value(static_cast<std::size_t>(msg.show_lightbulb));
        add_value(static_cast<std::size_t>(msg.stopped));
        add_text(msg.activity_elapsed);
        add_text(msg.timestamp);
        add_value(msg.tools.size());

        for (const auto& tool : msg.tools) {
            add_text(tool.id);
            add_text(tool.name);
            add_text(tool.description);
            add_text(tool.result.summary);
            add_optional_int(tool.result.exit_code);
            add_value(static_cast<std::size_t>(tool.result.truncated));
            add_value(static_cast<std::size_t>(tool.auto_approved));
            add_value(static_cast<std::size_t>(tool.status));
            add_optional_int(tool.progress);
            add_optional_int(tool.progress_total);
            add_text(tool.progress_message);

            add_text(tool.diff_preview.title);
            add_value(tool.diff_preview.lines.size());
            for (const auto& line : tool.diff_preview.lines) {
                add_value(static_cast<std::size_t>(line.kind));
                add_text(line.content);
            }
            add_value(tool.diff_preview.hidden_line_count);

            add_value(tool.subagents.size());
            for (const auto& subagent : tool.subagents) {
                add_text(subagent.id);
                add_text(subagent.worker_name);
                add_text(subagent.description);
                add_text(subagent.provider);
                add_text(subagent.model);
                add_text(subagent.latest_text);
                add_text(subagent.summary);
                add_value(static_cast<std::size_t>(subagent.steps));
                add_value(static_cast<std::size_t>(subagent.tool_calls));
                add_value(static_cast<std::size_t>(subagent.status));

                add_value(subagent.recent_tools.size());
                for (const auto& child_tool : subagent.recent_tools) {
                    add_text(child_tool.id);
                    add_text(child_tool.name);
                    add_text(child_tool.description);
                    add_value(static_cast<std::size_t>(child_tool.status));
                }
            }
        }
    }
    return seed;
}

} // namespace tui

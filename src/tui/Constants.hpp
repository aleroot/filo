#pragma once

#include <chrono>
#include <compare>
#include <cstddef>
#include <optional>

namespace tui {

// ── Application identity ──────────────────────────────────────────────────────
inline constexpr const char* kAppVersion = "Filo v0.1.0";

// ── Autocomplete ──────────────────────────────────────────────────────────────

/// Maximum number of suggestions shown in the mention (@ file) or command (/)
/// picker dropdown.  Kept large so that the scrollable picker can surface many
/// results; the visible window is controlled by kPickerMaxDisplayRows.
inline constexpr std::size_t kMaxAutocompleteSuggestions = 64;

// ── Permission overlay ────────────────────────────────────────────────────────

/// Maximum number of characters shown in the tool-argument preview inside the
/// permission overlay.  Arguments longer than this are truncated with "…".
inline constexpr std::size_t kPermissionArgsPreviewMaxLen = 300;

// ── Tool-call display ─────────────────────────────────────────────────────────

/// Maximum length (in characters) of a tool-argument summary line shown next
/// to the tool-call activity in the conversation history.
inline constexpr std::size_t kToolPreviewMaxLen = 88;

/// Maximum number of diff lines rendered in the conversation tool activity.
inline constexpr std::size_t kToolDiffPreviewMaxLines = 10;

/// Maximum number of tool-result lines shown inline in conversation cards.
/// Additional lines are collapsed to keep long transcripts scrollable.
inline constexpr std::size_t kToolResultPreviewMaxLines = 12;

/// Maximum number of diff lines rendered in the permission overlay.
inline constexpr std::size_t kPermissionDiffPreviewMaxLines = 28;

// ── Prompt layout ─────────────────────────────────────────────────────────────

/// Estimated terminal column width used for wrapping-line-count estimation in
/// the prompt box height calculation.  This is intentionally conservative so
/// that the box does not grow too aggressively on narrower terminals.
inline constexpr int kEstimatedLineWidthChars = 56;

/// Maximum number of suggestion items visible in the picker viewport at once
/// for compact pickers such as file mentions. When the list is longer,
/// overflow is shown as "↑ N more above" / "↓ N more below" indicators and
/// the viewport slides as the user navigates.
inline constexpr int kPickerMaxDisplayRows = 12;

/// Slash-command suggestions are rendered as taller two-line cards, so keep a
/// smaller visible window to avoid the bottom panel taking over the screen.
inline constexpr int kCommandPickerMaxDisplayRows = 6;

/// Hard cap on the number of input rows visible inside the prompt box.  The
/// PromptInput element reports its full (virtualized) line window as its
/// minimum height requirement; without this cap a long paste enters a
/// feedback loop (bigger box → bigger virtualization window → bigger
/// requirement → bigger box) until the prompt swallows the whole screen and
/// clips the banner/status bar.  When content exceeds this, the inner
/// `frame` scrolls to keep the cursor visible.
inline constexpr int kPromptInputMaxVisibleLines = 10;

// ── Animation ─────────────────────────────────────────────────────────────────

/// Interval between animation frames in milliseconds (controls the spinner and
/// thinking-pulse animations in the conversation history).
/// 150 ms → thinking dots cycle ≈ 900 ms; status spinners cycle smoothly without layout jitter.
inline constexpr int kAnimationIntervalMs = 150;

struct AnimationCadence {
    std::chrono::milliseconds period;
    bool advance_frame = false;
    auto operator<=>(const AnimationCadence&) const = default;
};

// Select the cheapest cadence that preserves every visible behavior.
// Spinner frames run at the established 150 ms rhythm. When spinners are
// hidden, assistant/review elapsed labels need only a one-second refresh.
[[nodiscard]] constexpr std::optional<AnimationCadence> select_animation_cadence(
    bool show_spinner,
    bool assistant_active,
    bool review_active,
    bool conversation_animation_active) noexcept {
    if (show_spinner
        && (assistant_active || review_active || conversation_animation_active)) {
        return AnimationCadence{
            .period = std::chrono::milliseconds(kAnimationIntervalMs),
            .advance_frame = true,
        };
    }
    if (assistant_active || review_active) {
        return AnimationCadence{
            .period = std::chrono::seconds(1),
            .advance_frame = false,
        };
    }
    return std::nullopt;
}

} // namespace tui

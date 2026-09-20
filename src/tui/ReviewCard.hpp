#pragma once

#include <ftxui/dom/elements.hpp>

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

namespace tui {

/// One review unit (a file or a bundle of related files) as the transcript
/// shows it. The engine owns the review; this is purely what the user sees.
struct ReviewGroupRow {
    enum class State {
        Running,   ///< Model turn in flight for this unit.
        RiskPass,  ///< Extra risk-analysis turn before the review turn.
        Done,      ///< Unit reviewed; findings counted.
        Skipped,   ///< Never reviewed (diff over budget, or review stopped).
        Failed,    ///< Sent to the model, but no usable review came back.
    };

    std::string label;
    State state = State::Running;
    int findings = 0;
    int blocking = 0;
    std::string elapsed;
    /// Why this unit was skipped or failed. Shown verbatim, so it must stay a
    /// short single line: "diff too large", "not reviewed", a provider error.
    std::string note;
    /// Stable identity for concurrent review progress events. Zero is reserved
    /// for synthetic skipped rows that have no engine group.
    std::size_t group_index = 0;
    std::chrono::steady_clock::time_point started_at =
        std::chrono::steady_clock::time_point::min();
};

/// Snapshot of a running or finished /review, rendered as one transcript card.
/// A value type with no behaviour: the renderer is pure, so the card can be
/// re-rendered every frame and unit-tested without a TUI.
struct ReviewProgressView {
    std::string hint;
    std::size_t total_groups = 0;
    int files = 0;
    int changed_lines = 0;
    int skipped_files = 0;
    int risk_passes = 0;
    std::vector<ReviewGroupRow> rows;
    bool planned = false;
    bool finished = false;
    bool interrupted = false;
    std::string failure;
    std::string elapsed;

    [[nodiscard]] std::size_t completed_rows() const noexcept;
    /// Done + Failed: how far the campaign has moved, including units that
    /// were sent to the model and came back unusable.
    [[nodiscard]] std::size_t processed_rows() const noexcept;
    [[nodiscard]] std::size_t failed_rows() const noexcept;
    [[nodiscard]] int total_findings() const noexcept;
    [[nodiscard]] int total_blocking() const noexcept;
};

/// Human label for one unit's outcome ("clean", "2 findings · 1 blocking").
[[nodiscard]] std::string review_row_outcome(const ReviewGroupRow& row);

/// Trailing summary line ("9 files · 412 lines · 3 comments · 41.0s").
[[nodiscard]] std::string review_summary_line(const ReviewProgressView& view);

/// Status-bar pill text: "review" before planning, then "review (3/17)".
[[nodiscard]] std::string review_status_pill_label(const ReviewProgressView& view);

/// Renders the static review card. Activity animation belongs to the footer
/// review pill; @p tick and @p show_spinner remain API-compatible no-ops.
[[nodiscard]] ftxui::Element render_review_card(const ReviewProgressView& view,
                                                std::size_t tick,
                                                bool show_spinner);

/// Footer popover for the status-bar review pill. The pill shows review (N/M);
/// file names, elapsed time and totals live here.
[[nodiscard]] ftxui::Element render_review_details_panel(const ReviewProgressView& view,
                                                         std::size_t tick,
                                                         bool show_spinner,
                                                         int max_rows);

} // namespace tui

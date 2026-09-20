#include "ReviewCard.hpp"

#include "Constants.hpp"
#include "TuiTheme.hpp"

#include <algorithm>
#include <format>
#include <string_view>
#include <utility>

namespace tui {

using ftxui::Color;
using ftxui::Element;
using ftxui::Elements;
using ftxui::dim;
using ftxui::filler;
using ftxui::hbox;
using ftxui::text;
using ftxui::vbox;
using ftxui::xflex;

namespace {

[[nodiscard]] std::string_view row_glyph(ReviewGroupRow::State state) noexcept {
    switch (state) {
        case ReviewGroupRow::State::Running:  return "•";
        case ReviewGroupRow::State::RiskPass: return "◆";
        case ReviewGroupRow::State::Done:     return "✓";
        case ReviewGroupRow::State::Skipped:  return "⊘";
        case ReviewGroupRow::State::Failed:   return "✗";
    }
    return "○";
}

[[nodiscard]] Color row_color(const ReviewGroupRow& row) noexcept {
    switch (row.state) {
        case ReviewGroupRow::State::Running:  return ColorYellowBright;
        case ReviewGroupRow::State::RiskPass: return ColorQuestionCyan;
        case ReviewGroupRow::State::Done:
            return row.blocking > 0 ? ColorToolFail : ColorToolDone;
        case ReviewGroupRow::State::Skipped:  return Color::GrayLight;
        case ReviewGroupRow::State::Failed:   return ColorToolFail;
    }
    return Color::White;
}

[[nodiscard]] Element row_marker(const ReviewGroupRow& row) {
    // The footer review pill owns the live activity spinner. Review cards stay
    // static so the same review does not present two competing spinners.
    Element glyph = text(std::string(row_glyph(row.state)));
    return std::move(glyph) | ftxui::color(row_color(row)) | ftxui::bold;
}

[[nodiscard]] std::string join_metrics(const std::vector<std::string>& parts) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) out += " · ";
        out += parts[i];
    }
    return out;
}

} // namespace

std::size_t ReviewProgressView::completed_rows() const noexcept {
    return static_cast<std::size_t>(std::ranges::count_if(
        rows,
        [](const ReviewGroupRow& row) {
            return row.state == ReviewGroupRow::State::Done;
        }));
}

std::size_t ReviewProgressView::processed_rows() const noexcept {
    return static_cast<std::size_t>(std::ranges::count_if(
        rows,
        [](const ReviewGroupRow& row) {
            return row.state == ReviewGroupRow::State::Done
                || row.state == ReviewGroupRow::State::Failed;
        }));
}

std::size_t ReviewProgressView::failed_rows() const noexcept {
    return static_cast<std::size_t>(std::ranges::count_if(
        rows,
        [](const ReviewGroupRow& row) {
            return row.state == ReviewGroupRow::State::Failed;
        }));
}

int ReviewProgressView::total_findings() const noexcept {
    int total = 0;
    for (const auto& row : rows) total += row.findings;
    return total;
}

int ReviewProgressView::total_blocking() const noexcept {
    int total = 0;
    for (const auto& row : rows) total += row.blocking;
    return total;
}

std::string review_row_outcome(const ReviewGroupRow& row) {
    switch (row.state) {
        case ReviewGroupRow::State::Running:
            return "reviewing";
        case ReviewGroupRow::State::RiskPass:
            return "risk pass";
        case ReviewGroupRow::State::Skipped:
            // The reason travels with the row: a unit can be skipped because
            // its diff was too large, or because the review stopped before
            // reaching it. Reporting one cause for both was a lie.
            return row.note.empty() ? "skipped" : std::format("skipped · {}", row.note);
        case ReviewGroupRow::State::Failed:
            return row.note.empty() ? "failed" : std::format("failed · {}", row.note);
        case ReviewGroupRow::State::Done:
            break;
    }
    if (row.findings == 0) {
        return "clean";
    }
    if (row.blocking > 0) {
        return std::format("{} finding(s) · {} blocking", row.findings, row.blocking);
    }
    return std::format("{} finding(s)", row.findings);
}

std::string review_summary_line(const ReviewProgressView& view) {
    std::vector<std::string> parts;
    parts.push_back(std::format("{} file(s)", view.files));
    if (view.changed_lines > 0) {
        parts.push_back(std::format("{} changed line(s)", view.changed_lines));
    }
    parts.push_back(std::format("{} comment(s)", view.total_findings()));
    if (view.total_blocking() > 0) {
        parts.push_back(std::format("{} blocking", view.total_blocking()));
    }
    if (view.skipped_files > 0) {
        parts.push_back(std::format("{} skipped", view.skipped_files));
    }
    if (const auto failed = view.failed_rows(); failed > 0) {
        parts.push_back(std::format("{} failed", failed));
    }
    if (!view.elapsed.empty()) {
        parts.push_back(view.elapsed);
    }
    return join_metrics(parts);
}

std::string review_status_pill_label(const ReviewProgressView& view) {
    if (view.total_groups == 0) {
        return "review";
    }
    return std::format("review ({}/{})", view.processed_rows(), view.total_groups);
}

Element render_review_card(const ReviewProgressView& view,
                           std::size_t tick,
                           bool show_spinner) {
    static_cast<void>(tick);
    static_cast<void>(show_spinner);
    Elements lines;

    // ── Header: what is being reviewed, and how far along it is. ──
    const bool running = !view.finished && !view.interrupted && view.failure.empty();
    Element header_marker = text(view.failure.empty()
                                     ? (view.interrupted ? "⊘" : running ? "•" : "✓")
                                     : "✗");
    const Color header_color = !view.failure.empty()
        ? Color(ColorToolFail)
        : view.interrupted ? Color(Color::GrayLight)
        : running          ? Color(ColorYellowBright)
                           : Color(ColorToolDone);

    Elements header{
        text(" "),
        std::move(header_marker) | ftxui::color(header_color) | ftxui::bold,
        text(" "),
        text("review") | ftxui::bold | ftxui::color(ColorYellowBright),
    };
    if (!view.hint.empty()) {
        header.push_back(text("  "));
        header.push_back(text(view.hint) | ftxui::color(Color::GrayDark) | xflex);
    }

    std::vector<std::string> header_metrics;
    if (view.total_groups > 0) {
        header_metrics.push_back(
            std::format("{}/{}", view.processed_rows(), view.total_groups));
    }
    if (!view.elapsed.empty()) {
        header_metrics.push_back(view.elapsed);
    }
    if (!header_metrics.empty()) {
        header.push_back(filler());
        header.push_back(text(join_metrics(header_metrics))
                         | ftxui::color(header_color)
                         | dim);
    }
    lines.push_back(hbox(std::move(header)) | xflex);

    // ── One row per review unit, in completion order. ──
    for (const auto& row : view.rows) {
        Elements cells{
            text("   "),
            row_marker(row),
            text(" "),
            text(row.label) | ftxui::color(Color::GrayLight),
        };

        std::vector<std::string> metrics;
        if (!row.elapsed.empty()) {
            metrics.push_back(row.elapsed);
        }
        metrics.push_back(review_row_outcome(row));

        cells.push_back(filler());
        cells.push_back(text(join_metrics(metrics))
                        | ftxui::color(row_color(row))
                        | dim);
        lines.push_back(hbox(std::move(cells)) | xflex);
    }

    // ── Footer: totals, or why the review stopped early. ──
    if (!view.failure.empty()) {
        lines.push_back(hbox({
            text("   "),
            text(view.failure) | ftxui::color(ColorToolFail) | xflex,
        }) | xflex);
    } else if (view.interrupted) {
        const auto processed = view.processed_rows();
        const std::string message = processed > 0 && view.total_groups > 0
            ? std::format("Interrupted after {} of {} unit(s).",
                          processed, view.total_groups)
            : std::string("Interrupted before every file was reviewed.");
        lines.push_back(hbox({
            text("   "),
            text(message)
                | ftxui::color(Color::GrayLight)
                | dim
                | xflex,
        }) | xflex);
    } else if (view.finished) {
        lines.push_back(hbox({
            text("   "),
            text(review_summary_line(view)) | ftxui::color(Color::GrayDark) | dim | xflex,
        }) | xflex);
    }

    return vbox(std::move(lines));
}

Element render_review_details_panel(const ReviewProgressView& view,
                                    std::size_t tick,
                                    bool show_spinner,
                                    int max_rows) {
    static_cast<void>(tick);
    static_cast<void>(show_spinner);
    Elements rows;

    const bool running = !view.finished && !view.interrupted && view.failure.empty();
    std::vector<std::string> status_parts;
    status_parts.push_back(view.hint.empty() ? std::string("current changes") : view.hint);
    if (view.total_groups > 0) {
        status_parts.push_back(std::format(
            "{}/{} unit(s) reviewed", view.processed_rows(), view.total_groups));
    }
    if (view.files > 0) {
        status_parts.push_back(std::format("{} file(s)", view.files));
    }
    if (view.changed_lines > 0) {
        status_parts.push_back(std::format("{} changed line(s)", view.changed_lines));
    }
    if (!view.elapsed.empty()) {
        status_parts.push_back(view.elapsed);
    }
    rows.push_back(hbox({
        text("  "),
        text(join_metrics(status_parts)) | ftxui::color(Color::White) | xflex,
    }));

    std::vector<std::string> totals;
    totals.push_back(std::format("{} comment(s)", view.total_findings()));
    if (view.total_blocking() > 0) {
        totals.push_back(std::format("{} blocking", view.total_blocking()));
    }
    if (view.skipped_files > 0) {
        totals.push_back(std::format("{} skipped", view.skipped_files));
    }
    if (const auto failed = view.failed_rows(); failed > 0) {
        totals.push_back(std::format("{} failed", failed));
    }
    if (view.risk_passes > 0) {
        totals.push_back(std::format("{} risk pass(es)", view.risk_passes));
    }
    rows.push_back(hbox({
        text("  "),
        text(join_metrics(totals)) | ftxui::color(Color::GrayLight) | dim | xflex,
    }));
    rows.push_back(text(""));

    if (view.rows.empty()) {
        rows.push_back(hbox({
            text("  "),
            text(running ? "Planning review units…" : "No review units.")
                | ftxui::color(Color::GrayLight)
                | dim,
        }));
    }

    // Running units matter more than the long tail of finished ones, so the
    // window keeps the tail of the list when it does not fit.
    const auto visible = static_cast<std::size_t>(std::max(1, max_rows));
    const std::size_t first = view.rows.size() > visible ? view.rows.size() - visible : 0;
    if (first > 0) {
        rows.push_back(hbox({
            text("  "),
            text(std::format("… {} earlier unit(s)", first))
                | ftxui::color(Color::GrayDark)
                | dim,
        }));
    }
    for (std::size_t i = first; i < view.rows.size(); ++i) {
        const auto& row = view.rows[i];
        Elements cells{
            text("  "),
            row_marker(row),
            text(" "),
            text(row.label) | ftxui::color(Color::GrayLight),
            filler(),
        };
        std::vector<std::string> metrics;
        if (!row.elapsed.empty()) metrics.push_back(row.elapsed);
        metrics.push_back(review_row_outcome(row));
        cells.push_back(text(join_metrics(metrics) + " ")
                        | ftxui::color(row_color(row))
                        | dim);
        rows.push_back(hbox(std::move(cells)) | xflex);
    }

    if (!view.failure.empty()) {
        rows.push_back(text(""));
        rows.push_back(hbox({
            text("  "),
            text(view.failure) | ftxui::color(ColorToolFail) | xflex,
        }));
    } else if (view.interrupted) {
        const auto processed = view.processed_rows();
        const std::string message = processed > 0 && view.total_groups > 0
            ? std::format("Interrupted after {} of {} unit(s).",
                          processed, view.total_groups)
            : std::string("Interrupted before every file was reviewed.");
        rows.push_back(text(""));
        rows.push_back(hbox({
            text("  "),
            text(message)
                | ftxui::color(Color::GrayLight)
                | dim,
        }));
    }

    rows.push_back(text(""));
    rows.push_back(
        text("  Esc, q, or click the review pill to close")
        | ftxui::color(Color::GrayDark)
        | dim);

    return UiWindow(
        text(" Code review ") | ftxui::color(ColorYellowBright) | ftxui::bold,
        vbox(std::move(rows)) | xflex);
}

} // namespace tui

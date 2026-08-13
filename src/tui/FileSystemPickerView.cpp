#include "FileSystemPickerView.hpp"

#include "TextLayout.hpp"
#include "TuiTheme.hpp"

#include "core/utils/PathUtils.hpp"

#include <algorithm>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ftxui;

namespace tui {
namespace {

/// Widths chosen so the breadcrumb and the detail column stay readable on an
/// 80-column terminal while still using the space on a wide one.
constexpr int kBreadcrumbWidth = 96;
constexpr int kDetailWidth = 26;
/// Keep a few rows of context around the cursor instead of scrolling only when
/// the selection reaches the very edge.
constexpr int kScrollMargin = 3;

/// Teaches the Tab accelerator on the focused row only. Showing it on every
/// folder would be pure noise: it carries no information that distinguishes
/// one row from another.
[[nodiscard]] std::string choose_marker(const FileSystemRow& row, bool focused) {
    return focused && row.selectable && row.navigable
        ? std::string("   \xe2\x87\xa5 use")
        : std::string{};
}

/// Leading glyph, sized to a fixed two-cell gutter so labels stay aligned.
[[nodiscard]] std::string_view role_glyph(const FileSystemRow& row) {
    switch (row.role) {
        case FileSystemRowRole::ConfirmCurrent: return "\xe2\x9c\x93";  // ✓
        case FileSystemRowRole::Parent:         return "\xe2\x86\x91";  // ↑
        case FileSystemRowRole::QuickRoot:      return "\xe2\x97\x86";  // ◆
        case FileSystemRowRole::Directory:      return "\xe2\x96\xb8";  // ▸
        case FileSystemRowRole::File:           return " ";
    }
    return " ";
}

[[nodiscard]] Color role_color(const FileSystemRow& row) {
    switch (row.role) {
        case FileSystemRowRole::ConfirmCurrent:
        case FileSystemRowRole::QuickRoot:
        case FileSystemRowRole::Parent:
            return ColorYellowBright;
        case FileSystemRowRole::Directory:
            return ColorYellowDark;
        case FileSystemRowRole::File:
            return row.selectable ? Color{Color::White} : Color{Color::GrayDark};
    }
    return Color::White;
}

/// Left-truncate a path to @p width *without* padding it out. `fit_path_column`
/// always pads to the column width, which is wrong for inline text that shares
/// its line with other cells.
[[nodiscard]] std::string shorten_path(std::string_view path, int width) {
    std::string shortened = fit_path_column(path, width);
    while (!shortened.empty() && shortened.back() == ' ') {
        shortened.pop_back();
    }
    return shortened;
}

/// Right-align a detail cell, shortening long paths from the left first.
[[nodiscard]] std::string right_align_detail(std::string_view detail, int width) {
    return fit_column_right(shorten_path(detail, width), width);
}

[[nodiscard]] Element render_row(const FileSystemRow& row, bool focused) {
    const Color accent = focused ? Color{Color::Black} : role_color(row);
    const Color label_color = focused
        ? Color{Color::Black}
        : (row.selectable || row.navigable ? Color{Color::White}
                                           : Color{Color::GrayDark});

    Element label = text(row.label) | color(label_color);
    if (!row.selectable && !row.navigable && !focused) {
        label = std::move(label) | dim;
    }

    // The marker sits directly after the label so it reads as an annotation on
    // this row rather than floating in the gap before the detail column.
    auto line = hbox({
        text(std::format(" {} ", role_glyph(row))) | color(accent),
        std::move(label),
        text(choose_marker(row, focused)) | color(accent) | ftxui::bold,
        filler(),
        text(right_align_detail(row.detail, kDetailWidth))
            | color(focused ? Color{Color::Black} : Color{Color::GrayDark}),
    });
    return focused ? (std::move(line) | bgcolor(ColorYellowDark)) : line;
}

/// Fixed-height scrolling window with "N more" affordances at both ends. The
/// selection is kept `kScrollMargin` rows away from the edges when possible.
[[nodiscard]] std::vector<Element> render_viewport(
    const std::vector<FileSystemRow>& rows,
    int selected) {
    const int total = static_cast<int>(rows.size());
    if (total == 0) {
        return {text("  Nothing here matches.") | dim};
    }

    const int first = std::clamp(selected - kScrollMargin,
                                 0,
                                 std::max(0, total - kFileSystemPickerViewportRows));
    const int last = std::min(total, first + kFileSystemPickerViewportRows);

    std::vector<Element> elements;
    elements.reserve(static_cast<std::size_t>(last - first) + 2);
    if (first > 0) {
        elements.push_back(hbox({
            text(" \xe2\x86\x91 ") | color(ColorYellowDark),
            text(std::format("{} more above", first)) | color(Color::GrayDark) | dim | xflex,
        }));
    }
    for (int index = first; index < last; ++index) {
        elements.push_back(render_row(rows[static_cast<std::size_t>(index)],
                                      index == selected));
    }
    if (const int below = total - last; below > 0) {
        elements.push_back(hbox({
            text(" \xe2\x86\x93 ") | color(ColorYellowDark),
            text(std::format("{} more below", below)) | color(Color::GrayDark) | dim | xflex,
        }));
    }
    return elements;
}

/// Header hints carry only the *decisions*, so the line stays short enough to
/// survive a long title on a narrow terminal. Mechanics live in the footer.
[[nodiscard]] std::string_view decision_hints(const FileSystemPickerState& state) {
    if (state.mode == FileSystemPickerMode::PathEntry) {
        return "Tab complete   Enter go   Esc back ";
    }
    // Naming Enter and Tab separately is what teaches the drill-down model:
    // Enter goes deeper, Tab commits the folder under the cursor.
    return state.request.target == FileSystemPickerTarget::Directory
        ? "Enter open   Tab use   Esc cancel "
        : "Enter open/select   Esc cancel ";
}

[[nodiscard]] std::string_view mechanic_hints(const FileSystemPickerState& state) {
    return state.mode == FileSystemPickerMode::PathEntry
        ? ""
        : "\xe2\x86\x91\xe2\x86\x93 move \xc2\xb7 \xe2\x8c\xab up \xc2\xb7 "
          "type to filter \xc2\xb7 ^E path \xc2\xb7 ^A hidden";
}

/// Second line of the panel: where we are, or what is being typed.
[[nodiscard]] Element render_locator(const FileSystemPickerState& state) {
    if (state.mode == FileSystemPickerMode::PathEntry) {
        return hbox({
            text("  path ") | color(Color::GrayDark),
            text(state.path_buffer) | color(ColorYellowBright),
            text("\xe2\x96\x8f") | color(ColorYellowBright),  // ▏ cursor
        });
    }

    // The filter is what the user is actively manipulating, so it gets its
    // width first and the breadcrumb absorbs whatever is left.
    const std::string match_count =
        state.filter.empty()
            ? std::string{}
            : std::format("  ({} match{})",
                          state.visible.size(),
                          state.visible.size() == 1 ? "" : "es");
    const int filter_width = state.filter.empty()
        ? 0
        : static_cast<int>(state.filter.size() + match_count.size() + 10);

    std::vector<Element> cells{
        text("  "),
        text(shorten_path(core::utils::path::abbreviate_user_path(state.directory),
                          std::max(16, kBreadcrumbWidth - filter_width)))
            | color(Color::GrayLight),
    };
    if (!state.filter.empty()) {
        cells.push_back(text("   filter ") | color(Color::GrayDark));
        cells.push_back(text(state.filter) | color(ColorYellowBright) | ftxui::bold);
        cells.push_back(text(match_count) | color(Color::GrayDark) | dim);
    }
    cells.push_back(filler());
    return hbox(std::move(cells));
}

/// Two dedicated lines rather than one shared one: the message wraps freely
/// (an error the user cannot finish reading is worse than an extra line) while
/// the mechanics reminder keeps a stable position instead of being shoved
/// around by however long the message happens to be.
///
/// The transient status outranks the caller's static hint, because a failed
/// navigation is what the user just asked about.
[[nodiscard]] Element render_footer(const FileSystemPickerState& state) {
    Element message = text("");
    if (!state.status.empty()) {
        message = paragraph(state.status) | color(ColorWarn);
    } else {
        std::string hint = state.request.hint;
        if (state.truncated) {
            hint += hint.empty() ? "" : "   ";
            hint += std::format(
                "Listing capped at {} entries \xe2\x80\x94 narrow it with a filter.",
                kFileSystemPickerMaxEntries);
        }
        if (!hint.empty()) {
            message = paragraph(hint) | color(Color::GrayDark) | dim;
        }
    }

    std::vector<Element> lines{hbox({text("  "), std::move(message)})};
    if (const auto mechanics = mechanic_hints(state); !mechanics.empty()) {
        lines.push_back(hbox({
            text(std::format("  {}", mechanics)) | color(Color::GrayDark) | dim,
            filler(),
        }));
    }
    return vbox(std::move(lines));
}

} // namespace

Element render_file_system_picker_panel(const FileSystemPickerState& state) {
    return vbox({
               hbox({
                   text(std::format(" {} ", state.request.title))
                       | ftxui::bold | color(Color::Black) | bgcolor(ColorYellowBright),
                   filler(),
                   text(decision_hints(state)) | color(Color::GrayDark),
               }),
               separator(),
               render_locator(state),
               separator(),
               vbox(render_viewport(state.visible, state.selected)),
               filler(),
               render_footer(state),
           })
        | UiBorder(ColorYellowBright)
        | size(HEIGHT, GREATER_THAN, kFileSystemPickerViewportRows);
}

} // namespace tui

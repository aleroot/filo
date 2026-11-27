#pragma once

#include "MarkdownRenderer.hpp"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/box.hpp>

namespace tui {

// The pieces every document reader in the TUI is made of: a window that
// scrolls by drawn rows (so a table or a code block is never cut in the
// source and drawn wrong), the keys that scroll it, and how a file is drawn.

/// What a scroll window drew in the last frame.
struct ScrollMetrics {
    /// Rows the whole content takes at the window's width; -1 before the
    /// first frame.
    int content_rows = -1;
    int viewport_rows = 0;
    /// The first content row shown, after clamping to the content.
    int offset = 0;
    /// Where the window is on screen.
    ftxui::Box box{0, -1, 0, -1};
    /// The offset was placed by the window's anchor, not the one asked for.
    bool anchored = false;
    /// Written by a frame and not yet taken by an event handler. A handler
    /// takes the settled offset once (take_frame_offset), so two keys that
    /// arrive before the next frame do not both start from the old one.
    bool fresh = false;

    [[nodiscard]] bool drawn() const noexcept { return content_rows >= 0; }
    [[nodiscard]] int rows_below() const noexcept {
        return std::max(0, content_rows - offset - viewport_rows);
    }
};

/// The row a window should show near its top, computed once the window knows
/// its width (Markdown wraps, so a line's row depends on it).
using ScrollAnchor = std::function<int(int width)>;

/// Shows rows [offset, offset + height) of @p content laid out at the
/// window's width. The window takes the rows its parent gives it.
/// @p metrics, when given, receives what was drawn. With @p anchor, the
/// window places the anchor's row a third of the way down instead.
[[nodiscard]] ftxui::Element scroll_window(ftxui::Element content,
                                           int offset,
                                           ScrollMetrics* metrics = nullptr,
                                           ScrollAnchor anchor = {});

/// A scroll window over @p total_rows rows of one row each, of which only the
/// visible ones are built: @p rows(first, count) returns them.
[[nodiscard]] ftxui::Element scroll_window(int total_rows,
                                           std::function<ftxui::Element(int first, int count)> rows,
                                           int offset,
                                           ScrollMetrics* metrics = nullptr,
                                           ScrollAnchor anchor = {});

/// One row saying where a window is ("rows 12–40 of 300"), from what the
/// window drew in the same frame. Draw it after the window.
[[nodiscard]] ftxui::Element scroll_status(const ScrollMetrics& metrics);

/// Columns a scroll_status row is given: enough for "rows 1000–2000 of 9999".
inline constexpr int kScrollStatusColumns = 22;

/// Sets @p offset to where the last frame settled, if it has not been taken
/// yet. True when it was taken.
bool take_frame_offset(ScrollMetrics& metrics, int& offset);

/// Moves @p offset for ↑/↓ (and j/k), PgUp/PgDn, Home/End and the mouse
/// wheel, within a document of @p content_rows rows shown @p page_rows at a
/// time. True when the event was one of those.
bool handle_scroll_event(int& offset, const ftxui::Event& event, int page_rows, int content_rows);

/// How a file is drawn.
enum class DocumentFormat {
    /// Rendered Markdown.
    Markdown,
    /// Numbered lines, as written.
    Text,
};

[[nodiscard]] DocumentFormat document_format(const std::filesystem::path& path);

/// What render_document_content draws besides the text.
struct DocumentRenderOptions {
    /// Where Markdown links are recorded (ignored for text).
    MarkdownLinkTargets links = {};
    /// A one-based line to mark (a link's target, a find match).
    std::optional<int> marked_line;
};

/// @p content drawn as @p format (no scroll window).
[[nodiscard]] ftxui::Element render_document_body(std::string_view content,
                                                  DocumentFormat format,
                                                  const DocumentRenderOptions& options);

/// @p body (from render_document_body; for text, built here instead, row by
/// visible row) in a scroll window at @p offset. The window keeps views into
/// @p content: it must outlive the frame the window is drawn in. When @p line is set, the
/// window brings that one-based line into view and ignores @p offset.
[[nodiscard]] ftxui::Element render_document_content(std::string_view content,
                                                     DocumentFormat format,
                                                     const DocumentRenderOptions& options,
                                                     int offset,
                                                     ScrollMetrics* metrics,
                                                     std::optional<int> line = std::nullopt,
                                                     ftxui::Element markdown_body = nullptr);

/// The drawn row that one-based @p line of @p content starts at, when drawn
/// as @p format @p width columns wide.
[[nodiscard]] int document_line_row(std::string_view content,
                                    DocumentFormat format,
                                    int line,
                                    int width);

/// One-based lines of @p content that contain @p query, ignoring ASCII case.
[[nodiscard]] std::vector<int> find_document_lines(std::string_view content,
                                                   std::string_view query);

/// Lines in @p content (a final line without a newline counts).
[[nodiscard]] int document_line_count(std::string_view content);

} // namespace tui

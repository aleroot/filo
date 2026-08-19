#pragma once

// Column layout primitives shared by every TUI panel.
//
// Terminal columns only line up when each cell occupies the same number of
// display cells. `std::format`'s width specifier counts bytes, which breaks on
// UTF-8 and on strings that need truncating, so pad and trim by display width
// on glyph boundaries instead.

#include <string>
#include <string_view>

#include <ftxui/dom/elements.hpp>

namespace tui {

/// Pad or truncate @p text to exactly @p width display cells, appending an
/// ellipsis when characters had to be dropped.
[[nodiscard]] std::string fit_column(std::string_view text, int width);

/// `fit_column` with the padding on the left, for numeric columns.
[[nodiscard]] std::string fit_column_right(std::string_view text, int width);

/// Fit a path into @p width, truncating from the *left*: the distinguishing
/// part of a path is its tail, so `…/Projects/filo` beats `~/Documents/Deve…`.
[[nodiscard]] std::string fit_path_column(std::string_view path, int width);

/// A single-line text element that uses all of the width assigned by its
/// parent and adds an ellipsis only when that actual width is insufficient.
/// Unlike pre-truncating a string before layout, this remains responsive when
/// the terminal is resized.
[[nodiscard]] ftxui::Element elided_text(std::string text);

} // namespace tui

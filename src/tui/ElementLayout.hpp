#pragma once

#include <ftxui/dom/elements.hpp>

namespace tui {

/// Lays @p element out in a column @p width cells wide whose top-left cell is
/// (@p x, @p y), and returns its height (at least 1).
///
/// FTXUI only knows how tall wrapping content (paragraphs, flexboxes) is once
/// it has been given a width, so this runs the same requirement/box passes
/// FTXUI's own Render runs until the layout settles. The element is then
/// ready to Render at that position.
int layout_element(ftxui::Element& element, int width, int x = 0, int y = 0);

} // namespace tui

#pragma once

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>
#include <string_view>

namespace tui {

// Render a markdown string as a FTXUI element tree.
//
// Supports:
//   Block:  headings (#-######), fenced/indented code, unordered/ordered lists,
//           blockquotes, tables (|col|col|), horizontal rules, paragraphs
//   Inline: **bold**, *italic*, ***bold-italic***, `code`, ~~strikethrough~~,
//           [link text](url), backslash escapes
//
// Plain paragraphs use paragraph() and wrap; styled/mixed lines use hbox()
// (no wrapping) as a trade-off for inline decoration support.
ftxui::Element render_markdown(std::string_view text,
                               ftxui::Color base_color = ftxui::Color::White);

} // namespace tui

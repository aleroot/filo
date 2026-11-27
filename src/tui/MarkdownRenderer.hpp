#pragma once

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/box.hpp>
#include <ftxui/screen/color.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "TuiTheme.hpp"

namespace tui {

/// Where a render records the links it draws, so a click or a key can reach
/// them. Links are numbered from 0 in document order.
struct MarkdownLinkTargets {
    /// Receives the box of every piece of every link, keyed by
    /// markdown_link_key(scope, ...). Left alone when null.
    std::unordered_map<std::string, ftxui::Box>* hitboxes = nullptr;
    /// Tells the links of this render from those of others in a shared map.
    std::string scope;
    /// The link drawn as holding the keyboard, if any.
    std::optional<std::size_t> selected;
};

struct MarkdownRenderOptions {
    ftxui::Color base_color = ColorText;
    MarkdownLinkTargets links = {};
};

// Render a markdown string as a FTXUI element tree.
//
// Supports:
//   Block:  headings (#-######), fenced/indented code, unordered/ordered lists,
//           blockquotes, tables (|col|col|), horizontal rules, paragraphs
//   Inline: **bold**, *italic*, ***bold-italic***, `code`, ~~strikethrough~~,
//           [link text](url), backslash escapes; nested emphasis/links compose
//           (e.g. *a **b** c*, [**label**](url))
//
// A link to a web page shows its URL after the label; a link to a local file
// (see parse_document_link) shows only the label, since opening it is what
// the link is for.
//
// Plain paragraphs and mixed inline spans both wrap to the viewport width.
ftxui::Element render_markdown(std::string_view text,
                               ftxui::Color base_color = ColorText);
ftxui::Element render_markdown(std::string_view text, const MarkdownRenderOptions& options);

/// One recorded piece of a link, decoded from its hitbox key.
struct MarkdownLinkRef {
    std::string scope;
    std::size_t ordinal = 0;
    std::string href;
};

/// The hitbox key of piece @p part of link @p ordinal. A link that wraps is
/// drawn in several pieces, each with its own box.
[[nodiscard]] std::string markdown_link_key(std::string_view scope,
                                            std::size_t ordinal,
                                            std::size_t part,
                                            std::string_view href);

/// The link a hitbox key names, or nothing when @p key is not a link's.
[[nodiscard]] std::optional<MarkdownLinkRef> parse_markdown_link_key(std::string_view key);

} // namespace tui

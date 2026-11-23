#pragma once

#include <cstddef>
#include <functional>
#include <string_view>
#include <vector>

#include <ftxui/component/component.hpp>
#include <ftxui/component/component_options.hpp>
#include <ftxui/dom/elements.hpp>

namespace tui {

/// A stretch of the input drawn with @p style. Styles may change colours and
/// attributes but never the text, so every glyph keeps its cell and the cursor
/// and mouse mapping stay those of the plain text.
struct PromptInputSpan {
    std::size_t begin = 0;  ///< byte offset into the content
    std::size_t end = 0;    ///< one past the last byte
    ftxui::Decorator style;
};

/// Supplies the spans for @p content with the cursor at @p cursor. Called on
/// every frame, so it should memoize. Spans may come in any order; one that
/// overlaps an earlier span is dropped.
using PromptInputStyler =
    std::function<std::vector<PromptInputSpan>(std::string_view content, std::size_t cursor)>;

ftxui::Component PromptInput(ftxui::StringRef content,
                             ftxui::StringRef placeholder,
                             ftxui::InputOption option = {},
                             PromptInputStyler styler = {});

} // namespace tui

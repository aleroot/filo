#include "TextLayout.hpp"

#include <ftxui/screen/string.hpp>

#include <cstddef>

namespace tui {
namespace {

constexpr std::string_view kEllipsis = "\xe2\x80\xa6";  // U+2026

} // namespace

std::string fit_column(std::string_view text, int width) {
    if (width <= 0) {
        return {};
    }

    const int text_width = ftxui::string_width(text);
    if (text_width == width) {
        return std::string{text};
    }
    if (text_width < width) {
        return std::string{text}
            + std::string(static_cast<std::size_t>(width - text_width), ' ');
    }

    std::string out;
    int used = 0;
    for (const auto& glyph : ftxui::Utf8ToGlyphs(text)) {
        if (glyph.empty()) {
            continue;
        }
        const int glyph_width = ftxui::string_width(glyph);
        if (used + glyph_width > width - 1) {
            break;
        }
        out += glyph;
        used += glyph_width;
    }
    out += kEllipsis;
    used += 1;
    if (used < width) {
        out += std::string(static_cast<std::size_t>(width - used), ' ');
    }
    return out;
}

std::string fit_column_right(std::string_view text, int width) {
    const int text_width = ftxui::string_width(text);
    if (text_width >= width) {
        return fit_column(text, width);
    }
    return std::string(static_cast<std::size_t>(width - text_width), ' ')
        + std::string{text};
}

std::string fit_path_column(std::string_view path, int width) {
    if (width <= 0) {
        return {};
    }
    if (ftxui::string_width(path) <= width) {
        return fit_column(path, width);
    }

    // Walk glyphs right-to-left, keeping as many trailing glyphs as fit, then
    // prepend an ellipsis.
    const auto glyphs = ftxui::Utf8ToGlyphs(path);
    std::string tail;
    int used = 1;  // Reserve one cell for the leading ellipsis.
    for (auto it = glyphs.rbegin(); it != glyphs.rend(); ++it) {
        if (it->empty()) {
            continue;
        }
        const int glyph_width = ftxui::string_width(*it);
        if (used + glyph_width > width) {
            break;
        }
        tail.insert(tail.begin(), it->begin(), it->end());
        used += glyph_width;
    }
    tail.insert(0, kEllipsis);
    if (used < width) {
        tail += std::string(static_cast<std::size_t>(width - used), ' ');
    }
    return tail;
}

} // namespace tui

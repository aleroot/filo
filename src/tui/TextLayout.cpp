#include "TextLayout.hpp"

#include <ftxui/dom/node.hpp>
#include <ftxui/screen/string.hpp>

#include <cstddef>
#include <memory>
#include <utility>

namespace tui {
namespace {

constexpr std::string_view kEllipsis = "\xe2\x80\xa6";  // U+2026

class ElidedText final : public ftxui::Node {
public:
    explicit ElidedText(std::string value)
        : value_(std::move(value)) {}

    void ComputeRequirement() override {
        requirement_ = {};
        requirement_.min_x = ftxui::string_width(value_);
        requirement_.min_y = 1;
    }

    void SetBox(ftxui::Box box) override {
        Node::SetBox(box);
        const int width = box.x_max - box.x_min + 1;
        rendered_ = ftxui::text(fit_column(value_, width));
        rendered_->ComputeRequirement();
        rendered_->SetBox(box);
    }

    void Select(ftxui::Selection& selection) override {
        if (rendered_) rendered_->Select(selection);
    }

    void Render(ftxui::Screen& screen) override {
        if (rendered_) rendered_->Render(screen);
    }

private:
    std::string value_;
    ftxui::Element rendered_;
};

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

ftxui::Element elided_text(std::string text) {
    return std::make_shared<ElidedText>(std::move(text));
}

} // namespace tui

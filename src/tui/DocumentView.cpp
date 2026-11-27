#include "DocumentView.hpp"

#include "ElementLayout.hpp"
#include "StringUtils.hpp"
#include "TuiTheme.hpp"

#include "core/utils/AsciiUtils.hpp"

#include <algorithm>
#include <format>
#include <string>
#include <utility>

#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/util/autoreset.hpp>

namespace tui {
namespace {

using namespace ftxui;

constexpr int kWheelRows = 3;
constexpr int kTabWidth = 4;

/// Common to both windows: takes the rows its parent gives, clips what it
/// draws to them, and reports what it showed.
class ScrollWindowBase : public Node {
public:
    ScrollWindowBase(Elements children, int offset, ScrollMetrics* metrics, ScrollAnchor anchor)
        : Node(std::move(children)),
          requested_offset_(offset),
          metrics_(metrics),
          anchor_(std::move(anchor)) {}

    void Render(Screen& screen) override {
        const AutoReset<Box> stencil(&screen.stencil, Box::Intersection(box_, screen.stencil));
        render_content(screen);
    }

protected:
    /// Clamps the requested offset to @p content_rows and reports the frame.
    int place(const Box& box, int content_rows) {
        const int viewport = std::max(0, box.y_max - box.y_min + 1);
        int wanted = requested_offset_;
        if (anchor_) wanted = anchor_(box.x_max - box.x_min + 1) - viewport / 3;
        const int offset = std::clamp(wanted, 0, std::max(0, content_rows - viewport));
        if (metrics_ != nullptr) {
            *metrics_ = ScrollMetrics{
                .content_rows = content_rows,
                .viewport_rows = viewport,
                .offset = offset,
                .box = box,
                .anchored = static_cast<bool>(anchor_),
                .fresh = true,
            };
        }
        return offset;
    }

    void expand_requirement(Requirement child) {
        requirement_ = child;
        requirement_.min_y = 0;
        requirement_.flex_grow_y = 1;
        requirement_.flex_shrink_y = 1;
        // What scrolls is this window's own offset, not an outer frame.
        requirement_.focused.enabled = false;
    }

    virtual void render_content(Screen& screen) = 0;

private:
    int requested_offset_;
    ScrollMetrics* metrics_;
    ScrollAnchor anchor_;
};

/// A window over a fully built element.
class ElementScrollWindow final : public ScrollWindowBase {
public:
    ElementScrollWindow(Element content, int offset, ScrollMetrics* metrics, ScrollAnchor anchor)
        : ScrollWindowBase({std::move(content)}, offset, metrics, std::move(anchor)) {}

    void ComputeRequirement() override {
        Node::ComputeRequirement();
        expand_requirement(children_[0]->requirement());
    }

    void SetBox(Box box) override {
        Node::SetBox(box);
        // Final once FTXUI's layout passes settle: wrapping content reports
        // its height for the width it was last given.
        const int content_rows = std::max(children_[0]->requirement().min_y, 0);
        const int offset = place(box, content_rows);
        const int height = std::max(content_rows, box.y_max - box.y_min + 1);
        children_[0]->SetBox(Box{box.x_min, box.x_max, box.y_min - offset,
                                 box.y_min - offset + height - 1});
    }

private:
    void render_content(Screen& screen) override { children_[0]->Render(screen); }
};

/// A window over one-row lines, building only the visible ones.
class RowScrollWindow final : public ScrollWindowBase {
public:
    RowScrollWindow(int total_rows,
                    std::function<Element(int, int)> rows,
                    int offset,
                    ScrollMetrics* metrics,
                    ScrollAnchor anchor)
        : ScrollWindowBase({}, offset, metrics, std::move(anchor)),
          total_rows_(std::max(total_rows, 0)),
          rows_(std::move(rows)) {}

    void ComputeRequirement() override {
        expand_requirement(Requirement{});
        requirement_.flex_grow_x = 1;
        requirement_.flex_shrink_x = 1;
    }

    void SetBox(Box box) override {
        Node::SetBox(box);
        const int offset = place(box, total_rows_);
        const int count = std::min(box.y_max - box.y_min + 1, total_rows_ - offset);
        visible_ = count > 0 ? rows_(offset, count) : nullptr;
        if (visible_) {
            layout_element(visible_, box.x_max - box.x_min + 1, box.x_min, box.y_min);
        }
    }

private:
    void render_content(Screen& screen) override {
        if (visible_) visible_->Render(screen);
    }

    int total_rows_;
    std::function<Element(int, int)> rows_;
    Element visible_;
};

/// Reads the metrics of a window drawn earlier in the same frame.
class ScrollStatusNode final : public Node {
public:
    explicit ScrollStatusNode(const ScrollMetrics& metrics) : metrics_(metrics) {}

    void ComputeRequirement() override {
        requirement_ = Requirement{};
        requirement_.min_y = 1;
        requirement_.flex_grow_x = 1;
        requirement_.flex_shrink_x = 1;
    }

    void Render(Screen& screen) override {
        if (!metrics_.drawn() || metrics_.content_rows == 0) return;
        const int first = metrics_.offset + 1;
        const int last = std::min(metrics_.content_rows, metrics_.offset + metrics_.viewport_rows);
        const bool all = first == 1 && last == metrics_.content_rows;
        auto label = text(all ? std::format("{} rows", metrics_.content_rows)
                              : std::format("rows {}–{} of {}", first, last,
                                            metrics_.content_rows))
            | color(ColorTextMuted);
        label->ComputeRequirement();
        label->SetBox(box_);
        label->Render(screen);
    }

private:
    const ScrollMetrics& metrics_;
};

[[nodiscard]] std::string expand_tabs(std::string_view line) {
    std::string out;
    out.reserve(line.size());
    for (const char ch : line) {
        if (ch == '\t') {
            out.append(static_cast<std::size_t>(kTabWidth - static_cast<int>(out.size()) % kTabWidth),
                       ' ');
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

[[nodiscard]] Element render_text_rows(const std::vector<std::string_view>& lines,
                                       int first,
                                       int count,
                                       std::optional<int> marked_line) {
    const int gutter = static_cast<int>(std::to_string(lines.size()).size());
    Elements rows;
    rows.reserve(static_cast<std::size_t>(count));
    for (int i = first; i < first + count; ++i) {
        const bool marked = marked_line == i + 1;
        auto row = hbox({
            text(std::format("{:>{}} ", i + 1, gutter))
                | color(marked ? Color(ColorYellowBright) : Color(ColorTextMuted)),
            text(expand_tabs(lines[static_cast<std::size_t>(i)])) | color(ColorText),
            filler(),
        });
        if (marked) row = std::move(row) | bgcolor(FillYellowDark) | color(Color::Black);
        rows.push_back(std::move(row));
    }
    return vbox(std::move(rows));
}

/// The first @p lines lines of @p content.
[[nodiscard]] std::string_view leading_lines(std::string_view content, int lines) {
    std::size_t end = 0;
    for (int i = 0; i < lines && end < content.size(); ++i) {
        const auto newline = content.find('\n', end);
        if (newline == std::string_view::npos) return content;
        end = newline + 1;
    }
    return content.substr(0, end);
}

} // namespace

Element scroll_window(Element content, int offset, ScrollMetrics* metrics, ScrollAnchor anchor) {
    return std::make_shared<ElementScrollWindow>(std::move(content), offset, metrics,
                                                 std::move(anchor));
}

Element scroll_window(int total_rows,
                      std::function<Element(int, int)> rows,
                      int offset,
                      ScrollMetrics* metrics,
                      ScrollAnchor anchor) {
    return std::make_shared<RowScrollWindow>(total_rows, std::move(rows), offset, metrics,
                                             std::move(anchor));
}

Element scroll_status(const ScrollMetrics& metrics) {
    return std::make_shared<ScrollStatusNode>(metrics);
}

bool take_frame_offset(ScrollMetrics& metrics, int& offset) {
    if (!metrics.fresh) return false;
    metrics.fresh = false;
    offset = metrics.offset;
    return true;
}

bool handle_scroll_event(int& offset, const Event& event, int page_rows, int content_rows) {
    Event copy = event;  // Event::mouse() is not const.
    const int page = std::max(1, page_rows);
    const int max_offset = std::max(0, content_rows - page);
    const auto scroll_to = [&](int target) {
        offset = std::clamp(target, 0, max_offset);
        return true;
    };
    if (event.is_mouse()) {
        const auto button = copy.mouse().button;
        if (button == Mouse::WheelUp) return scroll_to(offset - kWheelRows);
        if (button == Mouse::WheelDown) return scroll_to(offset + kWheelRows);
        return false;
    }
    if (event == Event::ArrowUp || event == Event::Character('k') || event == Event::Character('K')) {
        return scroll_to(offset - 1);
    }
    if (event == Event::ArrowDown || event == Event::Character('j')
        || event == Event::Character('J')) {
        return scroll_to(offset + 1);
    }
    if (event == Event::PageUp) return scroll_to(offset - page);
    if (event == Event::PageDown || event == Event::Character(' ')) return scroll_to(offset + page);
    if (event == Event::Home || event == Event::Character('g')) return scroll_to(0);
    if (event == Event::End || event == Event::Character('G')) return scroll_to(max_offset);
    return false;
}

DocumentFormat document_format(const std::filesystem::path& path) {
    const auto extension = path.extension().string();
    for (const std::string_view markdown : {".md", ".markdown", ".mdown", ".mkd", ".mdx"}) {
        if (core::utils::ascii::iequals(extension, markdown)) return DocumentFormat::Markdown;
    }
    return DocumentFormat::Text;
}

Element render_document_body(std::string_view content,
                             DocumentFormat format,
                             const DocumentRenderOptions& options) {
    if (format == DocumentFormat::Markdown) {
        return render_markdown(content, MarkdownRenderOptions{.links = options.links});
    }
    const auto lines = split_lines_view(content);
    return render_text_rows(lines, 0, static_cast<int>(lines.size()), options.marked_line);
}

Element render_document_content(std::string_view content,
                                 DocumentFormat format,
                                 const DocumentRenderOptions& options,
                                 int offset,
                                 ScrollMetrics* metrics,
                                 std::optional<int> line,
                                 Element markdown_body) {
    ScrollAnchor anchor;
    if (line) {
        anchor = [content, format, target = *line](int width) {
            return document_line_row(content, format, target, width);
        };
    }
    if (format == DocumentFormat::Markdown) {
        if (!markdown_body) markdown_body = render_document_body(content, format, options);
        return scroll_window(std::move(markdown_body), offset, metrics, std::move(anchor));
    }
    // Views into @p content, which outlives the frame it is drawn in.
    auto lines = std::make_shared<std::vector<std::string_view>>(split_lines_view(content));
    const int total = static_cast<int>(lines->size());
    return scroll_window(
        total,
        [lines, marked = options.marked_line](int first, int count) {
            return render_text_rows(*lines, first, count, marked);
        },
        offset, metrics, std::move(anchor));
}

int document_line_row(std::string_view content, DocumentFormat format, int line, int width) {
    if (line <= 1) return 0;
    if (format == DocumentFormat::Text) return line - 1;
    const auto before = leading_lines(content, line - 1);
    if (before.empty()) return 0;
    auto element = render_markdown(before);
    return layout_element(element, width);
}

std::vector<int> find_document_lines(std::string_view content, std::string_view query) {
    std::vector<int> matches;
    if (query.empty()) return matches;
    const auto lines = split_lines_view(content);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const auto line = lines[i];
        const auto found = std::ranges::search(line, query, [](char a, char b) {
            return core::utils::ascii::to_lower(a) == core::utils::ascii::to_lower(b);
        });
        if (!found.empty()) matches.push_back(static_cast<int>(i) + 1);
    }
    return matches;
}

int document_line_count(std::string_view content) {
    if (content.empty()) return 0;
    const auto newlines = static_cast<int>(std::ranges::count(content, '\n'));
    return newlines + (content.ends_with('\n') ? 0 : 1);
}

} // namespace tui

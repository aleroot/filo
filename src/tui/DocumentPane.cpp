#include "DocumentPane.hpp"

#include "KeyInput.hpp"
#include "TextLayout.hpp"
#include "TuiTheme.hpp"

#include "core/utils/PathUtils.hpp"

#include <algorithm>
#include <format>
#include <map>
#include <utility>

namespace tui {
namespace {

using namespace ftxui;

constexpr int kMinSplitColumns = 120;
constexpr int kMinPaneColumns = 50;
constexpr int kMaxPaneColumns = 110;

[[nodiscard]] std::string link_scope(const DocumentTab& tab) {
    return tab.path.string();
}

/// The boxes each link of @p tab was drawn in last frame, by ordinal.
[[nodiscard]] std::map<std::size_t, std::pair<std::string, std::vector<Box>>>
links_of(const DocumentPaneState& state, const DocumentTab& tab) {
    std::map<std::size_t, std::pair<std::string, std::vector<Box>>> links;
    const auto scope = link_scope(tab);
    for (const auto& [key, box] : state.link_boxes) {
        const auto ref = parse_markdown_link_key(key);
        if (!ref || ref->scope != scope) continue;
        auto& entry = links[ref->ordinal];
        entry.first = ref->href;
        entry.second.push_back(box);
    }
    return links;
}

[[nodiscard]] bool intersects(const Box& a, const Box& b) {
    return a.x_min <= b.x_max && b.x_min <= a.x_max && a.y_min <= b.y_max && b.y_min <= a.y_max;
}

/// Takes what the last frame settled on: the clamped offset, and whether a
/// line asked for has been brought into view.
void sync_from_frame(DocumentPaneState& state, DocumentTab& tab) {
    if (state.metrics_tab != tab.path) return;
    if (take_frame_offset(state.metrics, tab.offset) && state.metrics.anchored) {
        tab.pending_line.reset();
    }
}

/// Brings link @p ordinal into view when the last frame drew it outside.
void reveal_link(DocumentPaneState& state, DocumentTab& tab, std::size_t ordinal) {
    const auto links = links_of(state, tab);
    const auto found = links.find(ordinal);
    if (found == links.end() || found->second.second.empty()) return;
    const Box& box = found->second.second.front();
    const Box& view = state.metrics.box;
    if (box.y_min >= view.y_min && box.y_max <= view.y_max) return;
    tab.offset += box.y_min - (view.y_min + state.metrics.viewport_rows / 3);
    tab.offset = std::max(0, tab.offset);
}

void step_link(DocumentPaneState& state, DocumentTab& tab, int step) {
    const auto links = links_of(state, tab);
    if (links.empty()) return;
    const auto count = static_cast<long long>(links.rbegin()->first + 1);
    const long long current = tab.selected_link
        ? static_cast<long long>(*tab.selected_link)
        : (step > 0 ? -1 : 0);
    const auto next = static_cast<std::size_t>(((current + step) % count + count) % count);
    tab.selected_link = next;
    reveal_link(state, tab, next);
}

/// Opens @p href, written in @p tab: a local file in the pane, anything
/// else through the host.
DocumentPaneAction follow_link(DocumentPaneState& state, const DocumentTab& tab, std::string href) {
    if (auto target = parse_document_link(href)) {
        state.reader.open(resolve_document_target(std::move(*target), tab.path.parent_path()));
        state.find = {};
        return {.kind = DocumentPaneAction::Kind::handled};
    }
    return {.kind = DocumentPaneAction::Kind::open_url, .url = std::move(href)};
}

DocumentPaneAction handle_find_typing(DocumentPaneState& state, DocumentTab& tab, const Event& event) {
    auto& find = state.find;
    const auto first_line = [&] { return std::max(1, tab.offset + 1); };
    if (event == Event::Escape) {
        find = {};
        tab.marked_line.reset();
    } else if (event == Event::Return) {
        find.editing = false;
    } else if (event == Event::Backspace) {
        if (!find.query.empty()) {
            find.query.pop_back();
            find_in_document(find, tab, first_line());
        }
    } else if (event.is_character()) {
        find.query += event.character();
        find_in_document(find, tab, first_line());
    }
    return {.kind = DocumentPaneAction::Kind::handled};
}

DocumentPaneAction handle_mouse(DocumentPaneState& state, Event event) {
    using Kind = DocumentPaneAction::Kind;
    auto& mouse = event.mouse();
    const bool press = mouse.button == Mouse::Left && mouse.motion == Mouse::Pressed;
    if (!state.pane_box.Contain(mouse.x, mouse.y)) {
        // A press elsewhere takes the keyboard back to where it lands.
        if (press) state.focused = false;
        return {};
    }
    auto* tab = state.reader.selected();
    if (tab) sync_from_frame(state, *tab);
    if (tab && handle_scroll_event(tab->offset, event, state.metrics.viewport_rows,
                                   state.metrics.content_rows)) {
        return {.kind = Kind::handled};
    }
    if (!press) return {.kind = Kind::handled};
    state.focused = true;
    for (std::size_t i = 0; i < state.close_boxes.size(); ++i) {
        if (!state.close_boxes[i].Contain(mouse.x, mouse.y)) continue;
        state.reader.select(i);
        state.reader.close_selected();
        state.find = {};
        if (state.reader.empty()) state.close();
        return {.kind = Kind::handled};
    }
    for (std::size_t i = 0; i < state.tab_boxes.size(); ++i) {
        if (state.tab_boxes[i].Contain(mouse.x, mouse.y)) {
            state.reader.select(i);
            state.find = {};
            return {.kind = Kind::handled};
        }
    }
    if (tab && state.metrics.box.Contain(mouse.x, mouse.y)) {
        for (auto& [ordinal, link] : links_of(state, *tab)) {
            for (const auto& box : link.second) {
                if (intersects(box, state.metrics.box) && box.Contain(mouse.x, mouse.y)) {
                    tab->selected_link = ordinal;
                    return follow_link(state, *tab, link.first);
                }
            }
        }
    }
    return {.kind = Kind::handled};
}

DocumentPaneAction handle_key(DocumentPaneState& state, const Event& event) {
    using Kind = DocumentPaneAction::Kind;
    auto* tab = state.reader.selected();
    if (tab == nullptr) {
        state.close();
        return {};
    }
    // Ctrl+C keeps its meaning everywhere, the find prompt included.
    if (is_ctrl_c_event(event)) return {};
    sync_from_frame(state, *tab);
    if (state.find.editing) return handle_find_typing(state, *tab, event);

    const auto is_any = [&event](const auto&... keys) { return ((event == keys) || ...); };
    if (handle_scroll_event(tab->offset, event, state.metrics.viewport_rows,
                            state.metrics.content_rows)) {
        tab->pending_line.reset();
    } else if (is_any(Event::Escape, Event::F6)) {
        if (event == Event::Escape && tab->selected_link) {
            tab->selected_link.reset();
        } else if (event == Event::Escape && state.find.has_matches()) {
            state.find = {};
            tab->marked_line.reset();
        } else {
            state.focused = false;
        }
    } else if (is_any(Event::ArrowLeft, Event::Character('['))) {
        state.reader.select_relative(-1);
        state.find = {};
    } else if (is_any(Event::ArrowRight, Event::Character(']'))) {
        state.reader.select_relative(1);
        state.find = {};
    } else if (event == Event::Tab) {
        step_link(state, *tab, 1);
    } else if (event == Event::TabReverse) {
        step_link(state, *tab, -1);
    } else if (event == Event::Return) {
        if (tab->selected_link) {
            const auto links = links_of(state, *tab);
            if (const auto found = links.find(*tab->selected_link); found != links.end()) {
                return follow_link(state, *tab, found->second.first);
            }
        }
    } else if (is_any(Event::Character('/'), Event::Character('f'), Event::F3)) {
        state.find = {.editing = true};
    } else if (event == Event::Character('n')) {
        step_document_find(state.find, *tab, 1);
    } else if (event == Event::Character('N')) {
        step_document_find(state.find, *tab, -1);
    } else if (event == Event::Character('x')) {
        state.reader.close_selected();
        state.find = {};
        if (state.reader.empty()) state.close();
    } else if (is_any(Event::Character('q'), Event::Character('Q'))) {
        state.close();
    }
    // Other keys stay out of the prompt while the pane holds the keyboard.
    return {.kind = Kind::handled};
}

[[nodiscard]] Element render_tab_strip(DocumentPaneState& state) {
    const auto& tabs = state.reader.tabs();
    state.tab_boxes.assign(tabs.size(), Box{0, -1, 0, -1});
    state.close_boxes.assign(tabs.size(), Box{0, -1, 0, -1});
    Elements items;
    for (std::size_t i = 0; i < tabs.size(); ++i) {
        const bool active = i == state.reader.selected_index();
        auto name = text(" " + tabs[i].path.filename().string() + " ");
        auto close = text("× ");
        if (active) {
            const auto fill = state.focused ? Color(FillYellowDark) : Color(ColorTextMuted);
            name = std::move(name) | bold | bgcolor(fill) | color(Color::Black);
            close = std::move(close) | bgcolor(fill) | color(Color::Black);
        } else {
            name = std::move(name) | color(ColorTextSecondary);
            close = std::move(close) | color(ColorTextMuted);
        }
        items.push_back(std::move(name) | reflect(state.tab_boxes[i]));
        items.push_back(std::move(close) | reflect(state.close_boxes[i]));
        items.push_back(text(" "));
    }
    return hflow(std::move(items));
}

[[nodiscard]] Element render_footer(const DocumentPaneState& state) {
    const auto& find = state.find;
    if (find.editing || find.has_matches() || !find.query.empty()) {
        std::string result;
        if (!find.query.empty()) {
            result = find.has_matches()
                ? std::format("  {} of {}", find.current + 1, find.matches.size())
                : std::string("  no match");
        }
        return hbox({
            text("/ ") | color(ColorYellowBright) | bold,
            text(find.query) | color(ColorText),
            text(find.editing ? "▏" : "") | color(ColorYellowBright),
            text(result) | color(ColorTextSecondary),
            filler(),
            text(find.editing ? "Enter keep · Esc cancel" : "n/N match · Esc clear")
                | color(ColorTextMuted),
        });
    }
    // Most useful first: a narrow pane cuts the tail.
    std::string hints = "F6 or click to read";
    if (state.focused) {
        hints = "↑↓ read · / find · Tab link · Esc back · x close";
        if (state.reader.tabs().size() > 1) hints += " · ←/→ tab";
        hints += " · q hide";
    }
    return hbox({
        scroll_status(state.metrics) | size(WIDTH, EQUAL, kScrollStatusColumns),
        elided_text(hints) | color(ColorTextMuted) | xflex_shrink,
    });
}

[[nodiscard]] Element render_body(DocumentPaneState& state, DocumentTab& tab) {
    state.metrics_tab = tab.path;
    if (!tab.problem.empty()) {
        state.metrics = {};
        return vbox({text(""), text("  " + tab.problem) | color(ColorTextSecondary)}) | flex;
    }
    // A Markdown document is parsed once per version and selection, not once
    // per frame. Its link boxes belong to that tree, so they are cleared only
    // when the tree is rebuilt.
    auto& cache = state.cache;
    if (tab.format == DocumentFormat::Markdown
        && (!cache.element || cache.tab != tab.path || cache.generation != tab.generation
            || cache.selected_link != tab.selected_link)) {
        state.link_boxes.clear();
        cache = {
            .tab = tab.path,
            .generation = tab.generation,
            .selected_link = tab.selected_link,
            .element = render_document_body(
                tab.content, tab.format,
                DocumentRenderOptions{
                    .links = {.hitboxes = &state.link_boxes,
                              .scope = link_scope(tab),
                              .selected = tab.selected_link},
                }),
        };
    }
    return render_document_content(
        tab.content, tab.format, DocumentRenderOptions{.marked_line = tab.marked_line},
        tab.offset, &state.metrics, tab.pending_line,
        tab.format == DocumentFormat::Markdown ? cache.element : nullptr);
}

} // namespace

void DocumentPaneState::show(const DocumentTarget& target) {
    reader.open(target);
    find = {};
    open = true;
    focused = true;
}

DocumentPaneState::Placement DocumentPaneState::show_for_agent(const DocumentTarget& target,
                                                             bool can_split) {
    const auto* reading = reader.selected();
    const bool in_front = reading != nullptr && reading->path == target.path;
    if (open && focused && reading != nullptr && !in_front) {
        reader.open(target, /*select=*/false);
        return Placement::behind;
    }
    reader.open(target);
    if (!in_front) find = {};
    if (!open && !can_split) return Placement::hidden;
    open = true;
    return Placement::front;
}

void DocumentPaneState::close() {
    open = false;
    focused = false;
    find = {};
}

DocumentPaneAction handle_document_pane_event(DocumentPaneState& state, const Event& event) {
    if (state.reader.empty()) return {};
    if (!state.open) {
        // A hidden pane keeps its tabs; F6 brings it back.
        if (event != Event::F6) return {};
        state.open = true;
        state.focused = true;
        return {.kind = DocumentPaneAction::Kind::handled};
    }
    if (event.is_mouse()) return handle_mouse(state, event);
    if (!state.focused) {
        if (event == Event::F6) {
            state.focused = true;
            return {.kind = DocumentPaneAction::Kind::handled};
        }
        return {};
    }
    return handle_key(state, event);
}

Element render_document_pane(DocumentPaneState& state) {
    auto* tab = state.reader.selected();
    if (tab == nullptr) return emptyElement();
    sync_from_frame(state, *tab);

    std::string location = core::utils::path::abbreviate_user_path(tab->path);
    if (tab->truncated) location += "  (first 2 MiB)";
    auto body = render_body(state, *tab);
    return vbox({
        render_tab_strip(state),
        elided_path_text(std::move(location)) | color(ColorTextMuted),
        text(""),
        std::move(body) | flex,
        render_footer(state),
    }) | reflect(state.pane_box);
}

std::optional<int> document_pane_split_width(int total) {
    if (total < kMinSplitColumns) return std::nullopt;
    return std::clamp(total * 45 / 100, kMinPaneColumns, kMaxPaneColumns);
}

} // namespace tui

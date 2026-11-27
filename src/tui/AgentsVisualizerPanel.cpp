#include "AgentsVisualizerPanel.hpp"

#include "KeyInput.hpp"
#include "MarkdownRenderer.hpp"
#include "StringUtils.hpp"
#include "TuiTheme.hpp"

#include <algorithm>
#include <format>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace tui {

namespace {

using namespace ftxui;

[[nodiscard]] std::string format_file_size(std::size_t bytes) {
    if (bytes < 1024) {
        return std::format("{} B", bytes);
    }
    if (bytes < 1024 * 1024) {
        return std::format("{:.1f} KB", static_cast<double>(bytes) / 1024.0);
    }
    return std::format("{:.2f} MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
}

} // namespace

int agents_visualizer_page_lines(std::size_t file_count, int available_rows) {
    // Window frame (2), source/status/stats + gap (4), gap + position and
    // hints (2), and the file tabs + gap (2) when there are several.
    const int chrome = 8 + (file_count > 1 ? 2 : 0);
    return std::max(3, available_rows - chrome);
}

AgentsVisualizerEventResult handle_agents_visualizer_panel_event(
    AgentsVisualizerPanelState& state,
    ftxui::Event event,
    const std::vector<core::context::SteeringFile>& files,
    const Box& popover_box,
    const std::vector<Box>& tab_hitboxes) {
    using Result = AgentsVisualizerEventResult;
    if (!state.active) return Result::ignored;

    const std::size_t count = files.size();
    state.selected_file = count == 0 ? 0 : std::min(state.selected_file, count - 1);
    // Scroll from where the last frame settled, within what it drew.
    std::ignore = take_frame_offset(state.metrics, state.scroll_offset);
    const auto scroll = [&](const Event& scroll_event) {
        return handle_scroll_event(state.scroll_offset, scroll_event,
                                   state.metrics.viewport_rows, state.metrics.content_rows);
    };
    const auto select_file = [&](std::size_t index) {
        state.selected_file = index;
        state.scroll_offset = 0;
        state.metrics = {};
    };
    const auto is_any = [&event](const auto&... keys) { return ((event == keys) || ...); };

    if (event.is_mouse()) {
        auto& mouse = event.mouse();
        if (!popover_box.Contain(mouse.x, mouse.y)) {
            // A press elsewhere dismisses the popover and still lands where it
            // was aimed; the wheel keeps scrolling whatever is under it.
            if (mouse.button == Mouse::Left && mouse.motion == Mouse::Pressed) {
                state.active = false;
            }
            return Result::ignored;
        }
        if (mouse.button == Mouse::Left && mouse.motion == Mouse::Pressed) {
            const auto tabs = tab_hitboxes | std::views::take(count);
            const auto hit = std::ranges::find_if(
                tabs, [&](const Box& tab) { return tab.Contain(mouse.x, mouse.y); });
            if (hit != tabs.end()) select_file(static_cast<std::size_t>(hit - tabs.begin()));
        }
        std::ignore = scroll(event);
        return Result::handled;
    }

    if (is_panel_dismiss_event(event)) {
        state.active = false;
    } else if (is_any(Event::ArrowLeft, Event::TabReverse)) {
        if (count > 0) select_file((state.selected_file + count - 1) % count);
    } else if (is_any(Event::ArrowRight, Event::Tab)) {
        if (count > 0) select_file((state.selected_file + 1) % count);
    } else if (is_any(Event::Character('u'), Event::Character('U'), Event::Character(' '))) {
        return count > 0 ? Result::toggle_selected_file : Result::handled;
    } else if (scroll(event)) {
        // Scrolled.
    } else if (event == Event::Custom || is_ctrl_c_event(event)) {
        // Preserve Ctrl+C and app events, but keep keystrokes out of the
        // prompt while the popover holds the keyboard.
        return Result::ignored;
    }
    return Result::handled;
}

Element render_agents_visualizer_panel(
    const std::vector<core::context::SteeringFile>& files,
    std::size_t selected_file_index,
    int scroll_offset,
    std::vector<Box>* tab_hitboxes,
    int max_visible_lines,
    ScrollMetrics* metrics) {
    Elements rows;

    if (files.empty()) {
        if (tab_hitboxes) {
            tab_hitboxes->clear();
        }
        rows.push_back(hbox({
            text("  ℹ ") | color(ColorYellowBright) | bold,
            text("No agent instruction files active in this workspace.") | color(ColorText),
        }));
        rows.push_back(text(""));
        rows.push_back(text("  Create an AGENTS.md file in your workspace root or .filo/steering/*.md")
                       | color(ColorTextSecondary));
        rows.push_back(text("  to provide project-specific rules, steering, and instructions.")
                       | color(ColorTextSecondary));
        rows.push_back(text(""));
        rows.push_back(text("  Esc, q, or click outside to close") | color(ColorTextMuted) | dim);

        return UiWindow(
            text(" 🤖 Agent Instructions ") | color(ColorYellowBright) | bold,
            vbox(std::move(rows)) | xflex);
    }

    const std::size_t active_index = std::min(selected_file_index, files.size() - 1);

    if (tab_hitboxes) {
        tab_hitboxes->clear();
        if (files.size() > 1) {
            tab_hitboxes->resize(files.size());
        }
    }

    // ── Multi-File Tab Bar ───────────────────────────────────────────────────
    if (files.size() > 1) {
        Elements tabs;
        for (std::size_t i = 0; i < files.size(); ++i) {
            const auto& file = files[i];
            const bool is_active = (i == active_index);
            const std::string mark = file.enabled ? "" : "[OFF] ";
            auto tab_el = text(std::format(" {}{} ", mark, file.label))
                | (is_active ? (file.enabled ? (bgcolor(FillYellowDark) | color(Color::Black) | bold)
                                             : (bgcolor(Color::Red) | color(Color::White) | bold))
                             : (file.enabled ? color(ColorYellowBright)
                                             : (color(Color::Red) | dim)));
            if (tab_hitboxes) {
                tab_el = std::move(tab_el) | reflect((*tab_hitboxes)[i]);
            }
            tabs.push_back(std::move(tab_el));
            tabs.push_back(text(" "));
        }
        rows.push_back(hbox({
            text("  Files: ") | color(ColorTextSecondary),
            hbox(std::move(tabs)),
        }));
        rows.push_back(text(""));
    }

    // ── Active File Metadata & Stats ─────────────────────────────────────────
    const auto& cur_file = files[active_index];
    const int total_lines = document_line_count(cur_file.content);
    const int section_count = static_cast<int>(std::ranges::count_if(
        split_lines_view(cur_file.content),
        [](std::string_view line) { return line.starts_with('#'); }));

    if (max_visible_lines <= 0) {
        const auto term_size = Terminal::Size();
        // Dynamically use all remaining height between banner and status bar
        const int reserved_lines = (files.size() > 1 ? 19 : 17);
        max_visible_lines = std::max(12, term_size.dimy - reserved_lines);
    }

    rows.push_back(hbox({
        text("  Source: ") | color(ColorTextSecondary),
        text(cur_file.path.string()) | bold | color(ColorText) | xflex,
    }));

    rows.push_back(hbox({
        text("  Status: ") | color(ColorTextSecondary),
        cur_file.enabled
            ? (text("LOADED (active in prompt context)") | bold | color(Color::Green))
            : (text("UNLOADED (excluded from prompt context)") | bold | color(Color::Red)),
    }));

    Elements stats_elements = {
        text("  Stats:  ") | color(ColorTextSecondary),
        text(std::format("{} lines", total_lines)) | bold | color(ColorYellowBright),
        text(" · ") | color(ColorTextMuted),
        text(format_file_size(cur_file.content.size())) | color(ColorText),
    };
    if (section_count > 0) {
        stats_elements.push_back(text(" · ") | color(ColorTextMuted));
        stats_elements.push_back(text(std::format("{} sections", section_count)) | color(ColorTextSecondary));
    }
    rows.push_back(hbox(std::move(stats_elements)));
    rows.push_back(text(""));

    // ── The file, scrolled by drawn rows ─────────────────────────────────────
    // A table or code block is laid out whole and then scrolled, so it is
    // never cut in the source and drawn as something else.
    if (cur_file.content.empty()) {
        rows.push_back(text("  (empty file)") | color(ColorTextMuted) | dim);
    } else {
        rows.push_back(hbox({
            text("  "),
            render_document_content(cur_file.content, document_format(cur_file.path), {},
                                    scroll_offset, metrics) | xflex,
        }) | size(HEIGHT, EQUAL, max_visible_lines));
    }

    // ── Footer: position and navigation hints ────────────────────────────────
    rows.push_back(metrics != nullptr
        ? hbox({text("  "), scroll_status(*metrics) | size(WIDTH, EQUAL, kScrollStatusColumns)})
        : text(""));
    std::string help_str = "  ↑/↓ scroll · PgUp/PgDn page";
    if (files.size() > 1) {
        help_str += " · ←/→ file";
    }
    help_str += " · u load/unload · Esc or click outside to close";
    rows.push_back(text(std::move(help_str)) | color(ColorTextMuted) | dim);

    return UiWindow(
        text(" 🤖 Agent Instructions ") | color(ColorYellowBright) | bold,
        vbox(std::move(rows)) | xflex);
}

} // namespace tui

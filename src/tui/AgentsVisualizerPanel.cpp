#include "AgentsVisualizerPanel.hpp"

#include "MarkdownRenderer.hpp"
#include "StringUtils.hpp"
#include "TuiTheme.hpp"

#include <algorithm>
#include <format>
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

Element render_agents_visualizer_panel(
    const std::vector<core::context::SteeringFile>& files,
    std::size_t selected_file_index,
    int scroll_offset,
    std::vector<Box>* tab_hitboxes,
    int max_visible_lines) {
    Elements rows;

    if (files.empty()) {
        if (tab_hitboxes) {
            tab_hitboxes->clear();
        }
        rows.push_back(hbox({
            text("  ℹ ") | color(ColorYellowBright) | bold,
            text("No agent instruction files active in this workspace.") | color(Color::White),
        }));
        rows.push_back(text(""));
        rows.push_back(text("  Create an AGENTS.md file in your workspace root or .filo/steering/*.md")
                       | color(Color::GrayLight));
        rows.push_back(text("  to provide project-specific rules, steering, and instructions.")
                       | color(Color::GrayLight));
        rows.push_back(text(""));
        rows.push_back(text("  Esc, q, or click header to close") | color(Color::GrayDark) | dim);

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
            auto tab_el = text(std::format(" {} ", file.label))
                | (is_active ? (bgcolor(ColorYellowDark) | color(Color::Black) | bold)
                             : (color(ColorYellowBright)));
            if (tab_hitboxes) {
                tab_el = std::move(tab_el) | reflect((*tab_hitboxes)[i]);
            }
            tabs.push_back(std::move(tab_el));
            tabs.push_back(text(" "));
        }
        rows.push_back(hbox({
            text("  Files: ") | color(Color::GrayLight),
            hbox(std::move(tabs)),
        }));
        rows.push_back(text(""));
    }

    // ── Active File Metadata & Stats ─────────────────────────────────────────
    const auto& cur_file = files[active_index];
    const auto lines = split_lines_view(cur_file.content);
    const int total_lines = static_cast<int>(lines.size());

    int section_count = 0;
    for (const auto& line : lines) {
        if (!line.empty() && line[0] == '#') {
            section_count++;
        }
    }

    if (max_visible_lines <= 0) {
        const auto term_size = Terminal::Size();
        // Dynamically use all remaining height between banner and status bar
        const int reserved_lines = (files.size() > 1 ? 19 : 17);
        max_visible_lines = std::max(12, term_size.dimy - reserved_lines);
    }

    const int max_scroll = std::max(0, total_lines - max_visible_lines);
    const int clamped_scroll = std::clamp(scroll_offset, 0, max_scroll);
    const int view_end = std::min(total_lines, clamped_scroll + max_visible_lines);
    const float view_pct = total_lines == 0 ? 100.0f
        : std::min(100.0f, (static_cast<float>(view_end) / static_cast<float>(total_lines)) * 100.0f);

    rows.push_back(hbox({
        text("  Source: ") | color(Color::GrayLight),
        text(cur_file.path.string()) | bold | color(Color::White) | xflex,
    }));

    Elements stats_elements = {
        text("  Stats:  ") | color(Color::GrayLight),
        text(std::format("{} lines", total_lines)) | bold | color(ColorYellowBright),
        text(" · ") | color(Color::GrayDark),
        text(format_file_size(cur_file.content.size())) | color(Color::White),
    };
    if (section_count > 0) {
        stats_elements.push_back(text(" · ") | color(Color::GrayDark));
        stats_elements.push_back(text(std::format("{} sections", section_count)) | color(Color::GrayLight));
    }
    stats_elements.push_back(text(" · ") | color(Color::GrayDark));
    stats_elements.push_back(
        text(std::format("showing lines {}-{} of {} ({:.0f}%)",
                         total_lines == 0 ? 0 : clamped_scroll + 1,
                         view_end,
                         total_lines,
                         view_pct))
        | color(Color::GrayLight));

    rows.push_back(hbox(std::move(stats_elements)));
    rows.push_back(text(""));

    // ── Viewport Slicing & Markdown Rendering ────────────────────────────────
    if (clamped_scroll > 0) {
        rows.push_back(hbox({
            text("  ▲ ") | color(ColorYellowDark),
            text(std::format("{} more lines above", clamped_scroll))
                | color(Color::GrayDark) | dim,
        }));
    }

    std::string visible_content;
    for (int i = clamped_scroll; i < view_end; ++i) {
        visible_content.append(lines[static_cast<std::size_t>(i)]);
        visible_content.push_back('\n');
    }

    if (!visible_content.empty()) {
        rows.push_back(render_markdown(visible_content) | xflex);
    } else {
        rows.push_back(text("  (empty file)") | color(Color::GrayDark) | dim);
    }

    if (view_end < total_lines) {
        rows.push_back(hbox({
            text("  ▼ ") | color(ColorYellowDark),
            text(std::format("{} more lines below", total_lines - view_end))
                | color(Color::GrayDark) | dim,
        }));
    }

    // ── Footer Navigation Hints ──────────────────────────────────────────────
    rows.push_back(text(""));
    std::string help_str = "  Esc, q, or click header to close   ↑/↓/Wheel to scroll";
    if (files.size() > 1) {
        help_str += "   ←/→/Tab to switch file";
    }
    rows.push_back(text(std::move(help_str)) | color(Color::GrayDark) | dim);

    return UiWindow(
        text(" 🤖 Agent Instructions ") | color(ColorYellowBright) | bold,
        vbox(std::move(rows)) | xflex);
}

} // namespace tui

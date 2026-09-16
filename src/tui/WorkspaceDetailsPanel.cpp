#include "WorkspaceDetailsPanel.hpp"

#include "KeyInput.hpp"
#include "TuiTheme.hpp"
#include "core/workspace/SessionWorkspace.hpp"

#include <ftxui/dom/flexbox_config.hpp>
#include <ftxui/screen/string.hpp>
#include <algorithm>
#include <format>
#include <utility>

namespace tui {

bool handle_workspace_details_event(
    WorkspaceDetailsPanelState& state,
    ftxui::Event event,
    std::size_t root_count) {
    using namespace ftxui;
    if (!state.active) return false;

    const auto last = root_count == 0 ? 0 : root_count - 1;
    state.selected = std::min(state.selected, last);
    if (is_panel_dismiss_event(event)) {
        state.active = false;
    } else if (event == Event::Home) {
        state.selected = 0;
    } else if (event == Event::End) {
        state.selected = last;
    } else if (event == Event::ArrowUp || event == Event::PageUp
               || (event.is_mouse() && event.mouse().button == Mouse::WheelUp)) {
        const std::size_t step = event == Event::PageUp ? 5 : 1;
        state.selected -= std::min(state.selected, step);
    } else if (event == Event::ArrowDown || event == Event::PageDown
               || (event.is_mouse() && event.mouse().button == Mouse::WheelDown)) {
        const std::size_t step = event == Event::PageDown ? 5 : 1;
        state.selected += std::min(last - state.selected, step);
    } else {
        // Preserve Ctrl+C and app/mouse events, but never edit or submit the
        // prompt while this read-only panel replaces it.
        return !event.is_mouse() && event != Event::Custom && !is_ctrl_c_event(event);
    }
    return true;
}

ftxui::Element render_workspace_details_panel(
    const core::workspace::WorkspaceSnapshot& workspace,
    std::size_t selected,
    int max_visible_lines) {
    using namespace ftxui;
    const auto roots = core::workspace::ordered_roots(workspace);
    const auto active_index = roots.empty() ? 0 : std::min(selected, roots.size() - 1);
    Elements entries;
    for (std::size_t i = 0; i < roots.size(); ++i) {
        const auto& root = roots[i];
        const bool primary = !workspace.primary.empty() && i == 0;
        auto name = root.filename().string();
        if (name.empty()) name = root.string();

        // Paths have no word boundaries: wrap at glyph boundaries so even a
        // long directory name remains readable on narrow terminals.
        Elements glyphs;
        for (const auto& glyph : Utf8ToGlyphs(root.string())) {
            glyphs.push_back(text(glyph));
        }
        auto entry = vbox({
            text(std::format(" {}. {} · {}", i + 1, name,
                             primary ? "Primary" : "Additional"))
                | bold | color(i == active_index ? Color(ColorYellowBright) : Color::GrayLight),
            flexbox(std::move(glyphs), FlexboxConfig().SetGap(0, 0))
                | color(Color::White) | xflex,
        });
        if (i == active_index) entry = std::move(entry) | focus;
        entries.push_back(std::move(entry));
    }
    if (entries.empty()) {
        entries.push_back(text(" No workspaces loaded.") | color(Color::GrayLight));
    }

    return UiWindow(
        text(std::format(" Workspaces ({}) ", roots.size())) | color(ColorYellowBright) | bold,
        vbox({
            vbox(std::move(entries)) | vscroll_indicator | yframe
                | size(HEIGHT, LESS_THAN, std::max(1, max_visible_lines)),
            text(""),
            paragraph("↑/↓/Wheel: browse · Esc, q, or click workspace to close")
                | color(Color::GrayDark) | dim,
        }) | xflex);
}

} // namespace tui

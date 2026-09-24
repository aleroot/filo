#include "WorkspaceDetailsPanel.hpp"

#include "KeyInput.hpp"
#include "TuiTheme.hpp"
#include "core/workspace/SessionWorkspace.hpp"

#include <ftxui/dom/flexbox_config.hpp>
#include <ftxui/screen/string.hpp>
#include <algorithm>
#include <filesystem>
#include <format>
#include <string_view>
#include <utility>
#include <vector>

namespace tui {
namespace {

struct WorkspaceDetailsEntries {
    std::vector<std::filesystem::path> workspaces;
    std::vector<std::filesystem::path> attachments;
};

WorkspaceDetailsEntries split_workspace_details_entries(
    const core::workspace::WorkspaceSnapshot& workspace) {
    WorkspaceDetailsEntries entries;
    if (!workspace.primary.empty()) {
        entries.workspaces.push_back(workspace.primary);
    }

    for (const auto& path : workspace.additional) {
        if (path.empty()) continue;
        if (core::workspace::is_attached_file(workspace, path)) {
            entries.attachments.push_back(path);
        } else {
            entries.workspaces.push_back(path);
        }
    }
    return entries;
}

ftxui::Element render_path_entry(
    const std::filesystem::path& path,
    std::size_t index,
    std::string_view kind,
    bool selected,
    ftxui::Color label_color) {
    using namespace ftxui;
    auto name = path.filename().string();
    if (name.empty()) name = path.string();

    // Paths have no word boundaries: wrap at glyph boundaries so even a long
    // path remains readable on narrow terminals.
    Elements glyphs;
    for (const auto& glyph : Utf8ToGlyphs(path.string())) {
        glyphs.push_back(text(glyph));
    }
    auto entry = vbox({
        text(std::format(" {}. {} · {}", index + 1, name, kind))
            | bold | color(selected ? Color(ColorYellowBright) : label_color),
        flexbox(std::move(glyphs), FlexboxConfig().SetGap(0, 0))
            | color(Color::White) | xflex,
    });
    if (selected) entry = std::move(entry) | focus;
    return entry;
}

ftxui::Element render_attachment_entry(
    const std::filesystem::path& path,
    bool selected) {
    using namespace ftxui;
    auto entry = text(path.string())
        | bold | color(selected ? Color(ColorYellowBright)
                                : static_cast<Color>(ColorQuestionCyan));
    if (selected) entry = std::move(entry) | focus;
    return entry;
}

} // namespace

bool handle_workspace_details_event(
    WorkspaceDetailsPanelState& state,
    ftxui::Event event,
    std::size_t entry_count) {
    using namespace ftxui;
    if (!state.active) return false;

    const auto last = entry_count == 0 ? 0 : entry_count - 1;
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

std::size_t workspace_details_workspace_count(
    const core::workspace::WorkspaceSnapshot& workspace) {
    return split_workspace_details_entries(workspace).workspaces.size();
}

std::size_t workspace_details_entry_count(
    const core::workspace::WorkspaceSnapshot& workspace) {
    const auto entries = split_workspace_details_entries(workspace);
    return entries.workspaces.size() + entries.attachments.size();
}

ftxui::Element render_workspace_details_panel(
    const core::workspace::WorkspaceSnapshot& workspace,
    std::size_t selected,
    int max_visible_lines) {
    using namespace ftxui;
    const auto sections = split_workspace_details_entries(workspace);
    const auto entry_count = sections.workspaces.size() + sections.attachments.size();
    const auto active_index = entry_count == 0 ? 0 : std::min(selected, entry_count - 1);

    Elements workspace_entries;
    for (std::size_t i = 0; i < sections.workspaces.size(); ++i) {
        const bool primary = !workspace.primary.empty() && i == 0;
        workspace_entries.push_back(render_path_entry(
            sections.workspaces[i], i, primary ? "Primary" : "Additional",
            i == active_index, Color::GrayLight));
    }
    if (workspace_entries.empty()) {
        workspace_entries.push_back(
            text(" No workspaces loaded") | color(Color::GrayLight));
    }

    Elements groups;
    groups.push_back(UiWindow(
        text(std::format(" Workspaces ({}) ", sections.workspaces.size()))
            | color(ColorYellowBright) | bold,
        vbox(std::move(workspace_entries)) | xflex) | xflex);

    if (!sections.attachments.empty()) {
        Elements attachment_entries;
        for (std::size_t i = 0; i < sections.attachments.size(); ++i) {
            const auto selected_index = sections.workspaces.size() + i;
            attachment_entries.push_back(render_attachment_entry(
                sections.attachments[i], selected_index == active_index));
        }
        groups.push_back(UiWindow(
            text(std::format(" Attachments ({}) ", sections.attachments.size()))
                | color(ColorQuestionCyan) | bold,
            vbox(std::move(attachment_entries)) | xflex) | xflex);
    }

    return UiWindow(
        text(" Workspace details ") | color(ColorYellowBright) | bold,
        vbox({
            vbox(std::move(groups)) | vscroll_indicator | yframe
                | size(HEIGHT, LESS_THAN, std::max(1, max_visible_lines)),
            text(""),
            paragraph("↑/↓/Wheel: browse · Esc, q, or click workspace label to close")
                | color(Color::GrayDark) | dim,
        }) | xflex);
}

} // namespace tui

#pragma once

#include "core/workspace/Workspace.hpp"

#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>
#include <cstddef>

namespace tui {

struct WorkspaceDetailsPanelState {
    bool active = false;
    std::size_t selected = 0;
};

/// Handles navigation and dismissal without letting typing reach the hidden prompt.
[[nodiscard]] bool handle_workspace_details_event(
    WorkspaceDetailsPanelState& state,
    ftxui::Event event,
    std::size_t root_count);

[[nodiscard]] ftxui::Element render_workspace_details_panel(
    const core::workspace::WorkspaceSnapshot& workspace,
    std::size_t selected = 0,
    int max_visible_lines = 12);

} // namespace tui

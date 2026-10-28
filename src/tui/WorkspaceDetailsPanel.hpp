#pragma once

#include "core/workspace/Workspace.hpp"
#include "core/landrun/LandrunMode.hpp"

#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace core::policy { class PolicySnapshot; }

namespace tui {

struct WorkspaceDetailsPanelState {
    bool active = false;
    std::size_t selected = 0;
    bool security = false;
};

/// Handles navigation and dismissal without letting typing reach the hidden prompt.
[[nodiscard]] bool handle_workspace_details_event(
    WorkspaceDetailsPanelState& state,
    ftxui::Event event,
    std::size_t entry_count);

/// Counts folder roots for the workspace status label, excluding attached files.
[[nodiscard]] std::size_t workspace_details_workspace_count(
    const core::workspace::WorkspaceSnapshot& workspace);

/// Counts selectable workspace and attachment entries in the details panel.
[[nodiscard]] std::size_t workspace_details_entry_count(
    const core::workspace::WorkspaceSnapshot& workspace);

[[nodiscard]] ftxui::Element render_workspace_details_panel(
    const core::workspace::WorkspaceSnapshot& workspace,
    std::size_t selected = 0,
    int max_visible_lines = 12);

[[nodiscard]] std::vector<std::string> security_detail_entries(
    core::landrun::LandrunMode mode,
    const std::vector<std::filesystem::path>& excluded_paths,
    const core::policy::PolicySnapshot* policy);

[[nodiscard]] ftxui::Element render_security_details_panel(
    const std::vector<std::string>& entries,
    std::size_t selected = 0,
    int max_visible_lines = 12);

} // namespace tui

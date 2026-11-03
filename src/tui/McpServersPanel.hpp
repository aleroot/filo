#pragma once

#include "core/mcp/McpServerStatus.hpp"

#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/box.hpp>
#include <chrono>
#include <cstddef>
#include <vector>

namespace tui {

struct McpServersPanelState {
    bool active = false;
    std::size_t selected = 0;
};

/// Keyboard and mouse handling while the popover is open: moves the selection
/// between servers, closes on Esc/q or a press outside `popover_box` (that
/// press is not consumed), and keeps keystrokes out of the prompt.
[[nodiscard]] bool handle_mcp_servers_panel_event(
    McpServersPanelState& state,
    ftxui::Event event,
    std::size_t server_count,
    const ftxui::Box& popover_box);

/// Server list plus the details of the selected server: endpoint, auth,
/// capabilities, tools and recent tool-call activity.
[[nodiscard]] ftxui::Element render_mcp_servers_panel(
    const std::vector<core::mcp::McpServerStatus>& servers,
    std::size_t selected,
    std::chrono::system_clock::time_point now,
    int max_visible_lines = 14);

} // namespace tui

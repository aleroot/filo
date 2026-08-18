#pragma once

#include "core/context/SteeringLoader.hpp"
#include <ftxui/dom/elements.hpp>
#include <cstddef>
#include <vector>

namespace tui {

/**
 * @brief Renders the popover panel displaying active agent instructions and project steering files.
 * @param files The loaded steering files.
 * @param selected_file_index Index of the currently active file tab.
 * @param scroll_offset Number of lines scrolled from the top of the active file.
 * @param tab_hitboxes Optional vector to receive hitbox bounds for mouse interaction on file tabs.
 * @return FTXUI Element representing the styled window.
 */
[[nodiscard]] ftxui::Element render_agents_visualizer_panel(
    const std::vector<core::context::SteeringFile>& files,
    std::size_t selected_file_index = 0,
    int scroll_offset = 0,
    std::vector<ftxui::Box>* tab_hitboxes = nullptr,
    int max_visible_lines = 0);

} // namespace tui

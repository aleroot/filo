#pragma once

#include "DocumentView.hpp"
#include "core/context/SteeringLoader.hpp"
#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/box.hpp>
#include <cstddef>
#include <vector>

namespace tui {

struct AgentsVisualizerPanelState {
    bool active = false;
    std::size_t selected_file = 0;
    /// First drawn row of the selected file in view.
    int scroll_offset = 0;
    /// What the last frame drew; the scroll keys are bounded by it.
    ScrollMetrics metrics;
};

enum class AgentsVisualizerEventResult {
    /// Not for the popover; let it reach the rest of the UI.
    ignored,
    handled,
    /// Handled; the caller should load or unload `files[state.selected_file]`.
    toggle_selected_file,
};

/// Rows of a file the popover can show when `available_rows` rows are free
/// for the whole popover, after its frame, metadata and hints.
[[nodiscard]] int agents_visualizer_page_lines(std::size_t file_count, int available_rows);

/// Keyboard and mouse handling while the popover is open: scrolls the selected
/// file within what the last frame drew (`state.metrics`), switches files by
/// key or tab click, and closes on Esc/q or a press outside `popover_box`
/// (that press is not consumed). Other keys are kept out of the prompt.
[[nodiscard]] AgentsVisualizerEventResult handle_agents_visualizer_panel_event(
    AgentsVisualizerPanelState& state,
    ftxui::Event event,
    const std::vector<core::context::SteeringFile>& files,
    const ftxui::Box& popover_box,
    const std::vector<ftxui::Box>& tab_hitboxes);

/**
 * @brief Renders the popover panel displaying active agent instructions and project steering files.
 * @param files The loaded steering files.
 * @param selected_file_index Index of the currently active file tab.
 * @param scroll_offset Drawn rows scrolled from the top of the active file.
 * @param tab_hitboxes Optional vector to receive hitbox bounds for mouse interaction on file tabs.
 * @param max_visible_lines Rows of the file shown at once; <= 0 derives it from the terminal height.
 * @param metrics Optional: receives what the file's window drew.
 * @return FTXUI Element representing the styled window.
 */
[[nodiscard]] ftxui::Element render_agents_visualizer_panel(
    const std::vector<core::context::SteeringFile>& files,
    std::size_t selected_file_index = 0,
    int scroll_offset = 0,
    std::vector<ftxui::Box>* tab_hitboxes = nullptr,
    int max_visible_lines = 0,
    ScrollMetrics* metrics = nullptr);

} // namespace tui

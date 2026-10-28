#pragma once

#include "core/mcp/RemoteActivity.hpp"

#include <chrono>
#include <cstddef>
#include <string>

#include <ftxui/dom/elements.hpp>

namespace tui {

enum class RemoteFooterTone {
    neutral,
    ready,
    running,
    success,
    error,
};

struct RemoteFooterStatus {
    std::string label;
    RemoteFooterTone tone{RemoteFooterTone::neutral};
    bool animated = false;
};

[[nodiscard]] RemoteFooterStatus format_remote_footer_status(
    const core::mcp::RemoteActivitySnapshot& snapshot,
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

/// Render the compact footer status pill. It is hosted in its own centered
/// slot in the status bar, so it carries no leading gap: padding would offset
/// it from the middle. Tone is communicated with foreground color only so the
/// persistent status remains visually lightweight.
[[nodiscard]] ftxui::Element render_remote_footer_status(
    const RemoteFooterStatus& status);

/// Center the middle item in the footer, yielding to the side items when
/// space is too narrow to keep it centered without overlap.
[[nodiscard]] ftxui::Element render_centered_footer(
    ftxui::Element left, ftxui::Element middle, ftxui::Element right);

[[nodiscard]] ftxui::Element render_remote_activity_panel(
    const core::mcp::RemoteActivitySnapshot& snapshot,
    std::size_t selected_activity);

} // namespace tui

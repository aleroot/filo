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

[[nodiscard]] ftxui::Element render_remote_activity_panel(
    const core::mcp::RemoteActivitySnapshot& snapshot,
    std::size_t selected_activity);

} // namespace tui

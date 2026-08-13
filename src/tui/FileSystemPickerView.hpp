#pragma once

// Rendering for the reusable filesystem browser. Kept apart from the model so
// the keyboard contract in `FileSystemPicker.hpp` stays testable without a
// terminal, and so the panel's look can evolve without touching behaviour.

#include "FileSystemPicker.hpp"

#include <ftxui/dom/elements.hpp>

namespace tui {

/// Rows shown at once before the viewport starts scrolling.
inline constexpr int kFileSystemPickerViewportRows = 12;

[[nodiscard]] ftxui::Element render_file_system_picker_panel(
    const FileSystemPickerState& state);

} // namespace tui

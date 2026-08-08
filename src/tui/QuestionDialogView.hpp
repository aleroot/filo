#pragma once

#include "QuestionDialog.hpp"

#include <ftxui/dom/elements.hpp>

namespace tui {

/// @p origin_label identifies the thread that asked the question; empty for
/// the current thread, rendered when a hidden thread needs input.
[[nodiscard]] ftxui::Element render_question_dialog_panel(
    const QuestionDialogState& state,
    ftxui::Element other_input_editor = {},
    std::string_view origin_label = {});

} // namespace tui

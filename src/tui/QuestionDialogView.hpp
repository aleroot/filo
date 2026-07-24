#pragma once

#include "QuestionDialog.hpp"

#include <ftxui/dom/elements.hpp>

namespace tui {

[[nodiscard]] ftxui::Element render_question_dialog_panel(
    const QuestionDialogState& state,
    ftxui::Element other_input_editor = {});

} // namespace tui

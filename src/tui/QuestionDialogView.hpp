#pragma once

#include "QuestionDialog.hpp"

#include <ftxui/dom/elements.hpp>

namespace tui {

/// @p origin_label is optional decoration. Isolation is owned by
/// ThreadModalHost: a dialog only paints on its origin thread.
[[nodiscard]] ftxui::Element render_question_dialog_panel(
    const QuestionDialogState& state,
    ftxui::Element other_input_editor = {},
    std::string_view origin_label = {});

} // namespace tui

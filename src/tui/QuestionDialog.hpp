#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tui {

inline constexpr std::string_view kQuestionDialogOtherLabel = "Other";

struct QuestionDialogOption {
    std::string label;
    std::string description;
    bool accepts_free_text = false;
};

struct QuestionDialogItem {
    std::string question;
    std::string header;
    std::vector<QuestionDialogOption> options;
    bool multi_select = false;
    std::string body;
};

/// Dialog model. The synthetic free-text option has no label: the view renders
/// it as an embedded input whose placeholder is "Other". The selected option
/// remains the single source of truth for whether that editor owns focus.
struct QuestionDialogState {
    bool active = false;
    std::vector<QuestionDialogItem> questions;
    int current_question_index = 0;
    int selected_option = 0;
    std::vector<std::pair<std::string, std::string>> answers;
    std::vector<int> multi_selected;
    bool other_input_error = false;
    std::string other_input_text;
    int other_input_cursor_position = 0;
};

enum class QuestionDialogAnswerProgress {
    Ignored,
    Advanced,
    Completed,
    EmptyOther,
};

void activate_question_dialog(QuestionDialogState& state,
                              std::vector<QuestionDialogItem> questions);

[[nodiscard]] bool question_dialog_option_accepts_free_text(
    const QuestionDialogOption& option);

/// True while the highlighted option is the synthetic "Other" entry.
[[nodiscard]] bool question_dialog_selected_option_is_other(
    const QuestionDialogState& state);

/// Moves the highlight by `delta` options, wrapping around.
void move_question_dialog_selection(QuestionDialogState& state, int delta);

/// Highlights `index`; returns false when the index is out of range.
[[nodiscard]] bool select_question_dialog_option(QuestionDialogState& state,
                                                 int index);

/// Toggles the highlighted option for multi-select questions.
void toggle_question_dialog_multi_selection(QuestionDialogState& state);

/// Clears a pending "Other" draft. Returns true when a draft was discarded,
/// letting callers treat Escape as "clear the draft" before "close dialog".
[[nodiscard]] bool clear_question_dialog_other_input(QuestionDialogState& state);

/// Records the answer for the current question (option label, joined
/// multi-selection, or the "Other" free text) and advances the dialog.
[[nodiscard]] QuestionDialogAnswerProgress
accept_question_dialog_answer(QuestionDialogState& state);

} // namespace tui

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
};

struct QuestionDialogItem {
    std::string question;
    std::string header;
    std::vector<QuestionDialogOption> options;
    bool multi_select = false;
    std::string body;
};

struct QuestionDialogState {
    bool active = false;
    std::vector<QuestionDialogItem> questions;
    int current_question_index = 0;
    int selected_option = 0;
    std::vector<std::pair<std::string, std::string>> answers;
    std::vector<int> multi_selected;
    bool show_other_input = false;
    bool other_input_error = false;
    std::string other_input_text;
    int other_input_cursor_position = 0;
};

enum class QuestionDialogAnswerProgress {
    Ignored,
    EditingOther,
    Advanced,
    Completed,
    EmptyOther,
};

void activate_question_dialog(QuestionDialogState& state,
                              std::vector<QuestionDialogItem> questions);

[[nodiscard]] bool question_dialog_selected_option_is_other(
    const QuestionDialogState& state);

[[nodiscard]] QuestionDialogAnswerProgress
accept_question_dialog_selected_answer(QuestionDialogState& state);

[[nodiscard]] QuestionDialogAnswerProgress
accept_question_dialog_other_answer(QuestionDialogState& state);

void cancel_question_dialog_other_input(QuestionDialogState& state);

} // namespace tui

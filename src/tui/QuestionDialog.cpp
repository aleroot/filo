#include "QuestionDialog.hpp"

#include "core/utils/StringUtils.hpp"

namespace tui {

namespace {

const QuestionDialogItem* current_question(const QuestionDialogState& state) {
    if (!state.active
        || state.current_question_index < 0
        || state.current_question_index >= static_cast<int>(state.questions.size())) {
        return nullptr;
    }
    return &state.questions[static_cast<std::size_t>(state.current_question_index)];
}

QuestionDialogAnswerProgress record_answer(QuestionDialogState& state,
                                           std::string answer) {
    const auto* question = current_question(state);
    if (question == nullptr) {
        return QuestionDialogAnswerProgress::Ignored;
    }

    state.answers.emplace_back(question->question, std::move(answer));
    state.show_other_input = false;
    state.other_input_error = false;
    state.other_input_text.clear();
    state.other_input_cursor_position = 0;
    state.multi_selected.clear();

    if (state.current_question_index + 1 < static_cast<int>(state.questions.size())) {
        ++state.current_question_index;
        state.selected_option = 0;
        return QuestionDialogAnswerProgress::Advanced;
    }

    state.active = false;
    return QuestionDialogAnswerProgress::Completed;
}

} // namespace

void activate_question_dialog(QuestionDialogState& state,
                              std::vector<QuestionDialogItem> questions) {
    state = QuestionDialogState{
        .active = !questions.empty(),
        .questions = std::move(questions),
    };
}

bool question_dialog_selected_option_is_other(const QuestionDialogState& state) {
    const auto* question = current_question(state);
    if (question == nullptr
        || state.selected_option < 0
        || state.selected_option >= static_cast<int>(question->options.size())) {
        return false;
    }
    return question->options[static_cast<std::size_t>(state.selected_option)].label
        == kQuestionDialogOtherLabel;
}

QuestionDialogAnswerProgress
accept_question_dialog_selected_answer(QuestionDialogState& state) {
    const auto* question = current_question(state);
    if (question == nullptr
        || state.selected_option < 0
        || state.selected_option >= static_cast<int>(question->options.size())) {
        return QuestionDialogAnswerProgress::Ignored;
    }

    const auto& selected =
        question->options[static_cast<std::size_t>(state.selected_option)];
    if (selected.label == kQuestionDialogOtherLabel) {
        state.show_other_input = true;
        state.other_input_error = false;
        state.other_input_cursor_position =
            static_cast<int>(state.other_input_text.size());
        return QuestionDialogAnswerProgress::EditingOther;
    }

    return record_answer(state, selected.label);
}

QuestionDialogAnswerProgress
accept_question_dialog_other_answer(QuestionDialogState& state) {
    if (!state.show_other_input || current_question(state) == nullptr) {
        return QuestionDialogAnswerProgress::Ignored;
    }

    if (core::utils::str::trim_ascii_copy(state.other_input_text).empty()) {
        state.other_input_error = true;
        return QuestionDialogAnswerProgress::EmptyOther;
    }

    return record_answer(state, state.other_input_text);
}

void cancel_question_dialog_other_input(QuestionDialogState& state) {
    state.show_other_input = false;
    state.other_input_error = false;
}

} // namespace tui

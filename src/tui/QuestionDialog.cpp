#include "QuestionDialog.hpp"

#include "core/utils/StringUtils.hpp"

#include <algorithm>
#include <optional>

namespace tui {

namespace {

constexpr std::string_view kAnswerSeparator = ", ";

const QuestionDialogItem* current_question(const QuestionDialogState& state) {
    if (!state.active
        || state.current_question_index < 0
        || state.current_question_index >= static_cast<int>(state.questions.size())) {
        return nullptr;
    }
    return &state.questions[static_cast<std::size_t>(state.current_question_index)];
}

const QuestionDialogOption* selected_option(const QuestionDialogState& state,
                                            const QuestionDialogItem& question) {
    if (state.selected_option < 0
        || state.selected_option >= static_cast<int>(question.options.size())) {
        return nullptr;
    }
    return &question.options[static_cast<std::size_t>(state.selected_option)];
}

bool has_other_draft(const QuestionDialogState& state) {
    return !core::utils::str::trim_ascii_copy(state.other_input_text).empty();
}

bool is_checked(const QuestionDialogState& state, int index) {
    return std::ranges::find(state.multi_selected, index) != state.multi_selected.end();
}

/// Answer contributed by a single option: its label, or the typed free text
/// for the synthetic "Other" entry (nullopt when that text is still blank).
std::optional<std::string> option_answer(const QuestionDialogState& state,
                                         const QuestionDialogOption& option) {
    if (!question_dialog_option_accepts_free_text(option)) {
        return option.label;
    }
    if (!has_other_draft(state)) {
        return std::nullopt;
    }
    return state.other_input_text;
}

/// Builds the answer string for the current question. Multi-select questions
/// join every checked option; an "Other" entry contributes its free text as
/// soon as the user typed something, so it needs no separate checkbox toggle.
std::optional<std::string> compose_answer(const QuestionDialogState& state,
                                          const QuestionDialogItem& question) {
    if (question.multi_select) {
        std::string joined;
        for (std::size_t i = 0; i < question.options.size(); ++i) {
            const auto& option = question.options[i];
            const bool other_with_draft =
                question_dialog_option_accepts_free_text(option)
                && has_other_draft(state);
            if (!is_checked(state, static_cast<int>(i)) && !other_with_draft) {
                continue;
            }
            auto answer = option_answer(state, option);
            if (!answer.has_value()) {
                return std::nullopt;
            }
            if (!joined.empty()) {
                joined += kAnswerSeparator;
            }
            joined += *answer;
        }
        if (!joined.empty()) {
            return joined;
        }
        // Nothing checked: fall back to the highlighted option.
    }

    const auto* option = selected_option(state, question);
    if (option == nullptr) {
        return std::nullopt;
    }
    return option_answer(state, *option);
}

QuestionDialogAnswerProgress record_answer(QuestionDialogState& state,
                                           std::string answer) {
    const auto* question = current_question(state);
    if (question == nullptr) {
        return QuestionDialogAnswerProgress::Ignored;
    }

    state.answers.emplace_back(question->question, std::move(answer));
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

bool question_dialog_option_accepts_free_text(
    const QuestionDialogOption& option) {
    return option.accepts_free_text;
}

bool question_dialog_selected_option_is_other(const QuestionDialogState& state) {
    const auto* question = current_question(state);
    if (question == nullptr) {
        return false;
    }
    const auto* option = selected_option(state, *question);
    return option != nullptr
        && question_dialog_option_accepts_free_text(*option);
}

void move_question_dialog_selection(QuestionDialogState& state, int delta) {
    const auto* question = current_question(state);
    if (question == nullptr || question->options.empty()) {
        return;
    }

    const int count = static_cast<int>(question->options.size());
    state.selected_option =
        ((state.selected_option + delta) % count + count) % count;
    state.other_input_error = false;
}

bool select_question_dialog_option(QuestionDialogState& state, int index) {
    const auto* question = current_question(state);
    if (question == nullptr
        || index < 0
        || index >= static_cast<int>(question->options.size())) {
        return false;
    }

    state.selected_option = index;
    state.other_input_error = false;
    return true;
}

void toggle_question_dialog_multi_selection(QuestionDialogState& state) {
    const auto* question = current_question(state);
    if (question == nullptr || !question->multi_select) {
        return;
    }
    if (selected_option(state, *question) == nullptr) {
        return;
    }

    const auto checked = std::ranges::find(
        state.multi_selected,
        state.selected_option);
    if (checked == state.multi_selected.end()) {
        state.multi_selected.push_back(state.selected_option);
    } else {
        state.multi_selected.erase(checked);
    }
}

bool clear_question_dialog_other_input(QuestionDialogState& state) {
    if (!question_dialog_selected_option_is_other(state)
        || state.other_input_text.empty()) {
        return false;
    }

    state.other_input_text.clear();
    state.other_input_cursor_position = 0;
    state.other_input_error = false;
    return true;
}

QuestionDialogAnswerProgress
accept_question_dialog_answer(QuestionDialogState& state) {
    const auto* question = current_question(state);
    if (question == nullptr || selected_option(state, *question) == nullptr) {
        return QuestionDialogAnswerProgress::Ignored;
    }

    auto answer = compose_answer(state, *question);
    if (!answer.has_value()) {
        state.other_input_error = true;
        return QuestionDialogAnswerProgress::EmptyOther;
    }

    return record_answer(state, std::move(*answer));
}

} // namespace tui

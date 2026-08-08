#include "QuestionDialogController.hpp"

#include "PromptInput.hpp"
#include "QuestionDialogView.hpp"
#include "TuiTheme.hpp"

#include <ftxui/component/component_options.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>

using namespace ftxui;

namespace tui {

namespace {

constexpr int kMaxQuickSelectOptions = 5;

} // namespace

void QuestionDialogEventResult::resolve() {
    if (promise) {
        promise->set_value(std::move(answers));
        promise.reset();
    }
}

QuestionDialogController::QuestionDialogController() {
    auto option = InputOption();
    option.transform = [](InputState input_state) {
        if (input_state.is_placeholder) {
            return input_state.element
                | color(input_state.focused
                            ? static_cast<Color>(ColorQuestionCyan)
                            : Color{Color::GrayLight});
        }
        return input_state.element
            | color(input_state.focused
                        ? static_cast<Color>(ColorQuestionCyan)
                        : Color{Color::White});
    };
    option.multiline = false;
    option.cursor_position = &state_.other_input_cursor_position;
    option.on_change = [this]() {
        state_.other_input_error = false;
    };
    option.on_enter = [this]() {
        submit_requested_ = true;
    };
    editor_component_ = PromptInput(
        &state_.other_input_text,
        std::string(kQuestionDialogOtherLabel),
        std::move(option));
}

QuestionDialogPromise QuestionDialogController::open(
    core::tools::QuestionRequest request,
    std::string origin_label) {
    std::vector<QuestionDialogItem> questions;
    questions.reserve(request.questions.size());
    for (auto& question : request.questions) {
        QuestionDialogItem item{
            .question = std::move(question.question),
            .header = std::move(question.header),
            .multi_select = question.multi_select,
            .body = std::move(question.body),
        };
        item.options.reserve(question.options.size());
        for (auto& option : question.options) {
            item.options.push_back(QuestionDialogOption{
                .label = std::move(option.label),
                .description = std::move(option.description),
                .accepts_free_text = option.accepts_free_text,
            });
        }
        questions.push_back(std::move(item));
    }

    std::lock_guard lock(mutex_);
    auto displaced = std::move(promise_);
    activate_question_dialog(state_, std::move(questions));
    promise_ = std::move(request.promise);
    origin_label_ = std::move(origin_label);
    origin_session_id_ = std::move(request.session_id);
    submit_requested_ = false;
    focus_other_editor_if_selected();
    return displaced;
}

std::string QuestionDialogController::origin_label() const {
    std::lock_guard lock(mutex_);
    return origin_label_;
}

QuestionDialogEventResult QuestionDialogController::force_interrupt() {
    return handle_event(ftxui::Event::Escape, /*is_interrupt=*/true);
}

bool QuestionDialogController::active() const {
    std::lock_guard lock(mutex_);
    return state_.active;
}

Component QuestionDialogController::editor_component() const {
    return editor_component_;
}

Element QuestionDialogController::render() {
    std::lock_guard lock(mutex_);
    // Only label hidden threads: an empty label keeps the dialog exactly as
    // the single-thread experience has always looked.
    return render_question_dialog_panel(
        state_,
        editor_component_->Render(),
        origin_label_);
}

void QuestionDialogController::focus_other_editor_if_selected() {
    if (question_dialog_selected_option_is_other(state_)) {
        editor_component_->TakeFocus();
    }
}

void QuestionDialogController::sync_selection_focus(
    QuestionDialogEventResult& result) {
    if (question_dialog_selected_option_is_other(state_)) {
        editor_component_->TakeFocus();
    } else {
        result.restore_main_input_focus = true;
    }
}

void QuestionDialogController::apply_answer_progress(
    QuestionDialogAnswerProgress progress,
    QuestionDialogEventResult& result) {
    switch (progress) {
    case QuestionDialogAnswerProgress::Advanced:
        sync_selection_focus(result);
        break;
    case QuestionDialogAnswerProgress::Completed:
        result.restore_main_input_focus = true;
        result.answers = state_.answers;
        result.promise = std::move(promise_);
        result.origin_session_id = std::move(origin_session_id_);
        origin_label_.clear();
        break;
    case QuestionDialogAnswerProgress::Ignored:
    case QuestionDialogAnswerProgress::EmptyOther:
        break;
    }
}

void QuestionDialogController::dismiss(
    bool is_interrupt,
    QuestionDialogEventResult& result) {
    state_.active = false;
    result.origin_session_id = std::move(origin_session_id_);
    origin_label_.clear();
    result.stop_agent = is_interrupt;
    result.restore_main_input_focus = true;
    result.promise = std::move(promise_);
}

QuestionDialogEventResult QuestionDialogController::handle_event(
    const Event& event,
    bool is_interrupt) {
    std::lock_guard lock(mutex_);

    QuestionDialogEventResult result;
    if (!state_.active) {
        return result;
    }
    result.handled = true;

    if (is_interrupt) {
        dismiss(true, result);
        return result;
    }

    // Navigation and dismissal stay owned by the dialog while the embedded
    // "Other" editor is selected, so the highlight never gets trapped in it.
    if (event == Event::ArrowUp) {
        move_question_dialog_selection(state_, -1);
        sync_selection_focus(result);
        return result;
    }
    if (event == Event::ArrowDown) {
        move_question_dialog_selection(state_, 1);
        sync_selection_focus(result);
        return result;
    }
    if (event == Event::Escape) {
        if (!clear_question_dialog_other_input(state_)) {
            dismiss(false, result);
        }
        return result;
    }

    // "Other" highlighted: every remaining key edits the free text, so digits
    // and spaces are typed instead of being swallowed by option shortcuts.
    if (question_dialog_selected_option_is_other(state_)) {
        submit_requested_ = false;
        (void)editor_component_->OnEvent(event);
        if (submit_requested_) {
            apply_answer_progress(accept_question_dialog_answer(state_), result);
        }
        return result;
    }

    const auto& question = state_.questions[
        static_cast<std::size_t>(state_.current_question_index)];
    const int option_count = static_cast<int>(question.options.size());

    if (event == Event::Character(' ') && question.multi_select) {
        toggle_question_dialog_multi_selection(state_);
        return result;
    }
    if (event == Event::Return) {
        apply_answer_progress(accept_question_dialog_answer(state_), result);
        return result;
    }

    for (int option_number = 1;
         option_number <= std::min(option_count, kMaxQuickSelectOptions);
         ++option_number) {
        if (event != Event::Character(
                         static_cast<char>('0' + option_number))) {
            continue;
        }
        if (!select_question_dialog_option(state_, option_number - 1)) {
            break;
        }
        // Selecting "Other" opens its editor instead of answering right away.
        if (question_dialog_selected_option_is_other(state_)) {
            editor_component_->TakeFocus();
        } else if (!question.multi_select) {
            apply_answer_progress(accept_question_dialog_answer(state_), result);
        }
        break;
    }

    return result;
}

} // namespace tui

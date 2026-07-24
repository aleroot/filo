#include "QuestionDialogController.hpp"

#include "PromptInput.hpp"
#include "QuestionDialogView.hpp"
#include "TuiTheme.hpp"

#include <ftxui/component/component_options.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>

using namespace ftxui;

namespace tui {

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
            return input_state.element | color(Color::GrayDark);
        }
        return input_state.element | color(ColorQuestionCyan);
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
        "Type your own instruction...",
        std::move(option));
}

QuestionDialogPromise QuestionDialogController::open(
    core::tools::QuestionRequest request) {
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
            });
        }
        questions.push_back(std::move(item));
    }

    std::lock_guard lock(mutex_);
    auto displaced = std::move(promise_);
    activate_question_dialog(state_, std::move(questions));
    promise_ = std::move(request.promise);
    submit_requested_ = false;
    return displaced;
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
    return render_question_dialog_panel(
        state_,
        state_.show_other_input ? editor_component_->Render() : Element{});
}

void QuestionDialogController::apply_answer_progress(
    QuestionDialogAnswerProgress progress,
    QuestionDialogEventResult& result) {
    switch (progress) {
    case QuestionDialogAnswerProgress::EditingOther:
        editor_component_->TakeFocus();
        break;
    case QuestionDialogAnswerProgress::Advanced:
        result.restore_main_input_focus = true;
        break;
    case QuestionDialogAnswerProgress::Completed:
        result.restore_main_input_focus = true;
        result.answers = state_.answers;
        result.promise = std::move(promise_);
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

    auto& question = state_.questions[
        static_cast<std::size_t>(state_.current_question_index)];
    const int option_count = static_cast<int>(question.options.size());

    if (state_.show_other_input) {
        if (event == Event::Escape) {
            cancel_question_dialog_other_input(state_);
            result.restore_main_input_focus = true;
            return result;
        }
        if (is_interrupt) {
            dismiss(true, result);
            return result;
        }

        submit_requested_ = false;
        (void)editor_component_->OnEvent(event);
        if (submit_requested_) {
            apply_answer_progress(
                accept_question_dialog_other_answer(state_),
                result);
        }
        return result;
    }

    if (event == Event::ArrowUp && option_count > 0) {
        state_.selected_option =
            (state_.selected_option + option_count - 1) % option_count;
        return result;
    }
    if (event == Event::ArrowDown && option_count > 0) {
        state_.selected_option =
            (state_.selected_option + 1) % option_count;
        return result;
    }
    if (event == Event::Character(' ') && question.multi_select) {
        const auto selected = std::ranges::find(
            state_.multi_selected,
            state_.selected_option);
        if (selected == state_.multi_selected.end()) {
            state_.multi_selected.push_back(state_.selected_option);
        } else {
            state_.multi_selected.erase(selected);
        }
        return result;
    }
    if (event == Event::Return) {
        apply_answer_progress(
            accept_question_dialog_selected_answer(state_),
            result);
        return result;
    }
    if (event == Event::Escape || is_interrupt) {
        dismiss(is_interrupt, result);
        return result;
    }

    for (int option_number = 1;
         option_number <= std::min(option_count, 5);
         ++option_number) {
        if (event != Event::Character(
                         static_cast<char>('0' + option_number))) {
            continue;
        }
        state_.selected_option = option_number - 1;
        if (!question.multi_select) {
            apply_answer_progress(
                accept_question_dialog_selected_answer(state_),
                result);
        }
        break;
    }

    return result;
}

} // namespace tui

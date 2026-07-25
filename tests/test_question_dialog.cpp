#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/context/SessionContext.hpp"
#include "tui/QuestionDialogController.hpp"
#include "tui/QuestionDialogView.hpp"

#include <ftxui/screen/screen.hpp>

#include <chrono>
#include <string>
#include <string_view>
#include <thread>

using namespace tui;

namespace {

std::string strip_ansi(std::string_view input) {
    std::string output;
    output.reserve(input.size());

    for (std::size_t i = 0; i < input.size();) {
        if (input[i] == '\x1b'
            && i + 1 < input.size()
            && input[i + 1] == '[') {
            i += 2;
            while (i < input.size()) {
                const char character = input[i++];
                if (character >= '@' && character <= '~') {
                    break;
                }
            }
            continue;
        }
        output.push_back(input[i++]);
    }

    return output;
}

core::tools::QuestionRequest make_request() {
    core::tools::QuestionRequest request;
    request.questions.push_back(core::tools::QuestionItem{
        .question = "How should the model proceed?",
        .header = "Approach",
        .options = {
            {"Use defaults", "Apply the standard behavior."},
            {"", "", true},
        },
    });
    request.promise = std::make_shared<
        std::promise<std::optional<QuestionDialogAnswers>>>();
    return request;
}

std::string render(QuestionDialogController& controller) {
    auto element = controller.render();
    auto screen = ftxui::Screen::Create(
        ftxui::Dimension::Fixed(120),
        ftxui::Dimension::Fit(element));
    ftxui::Render(screen, element);
    return strip_ansi(screen.ToString());
}

std::string render_panel(const QuestionDialogState& state,
                         ftxui::Element editor = ftxui::Element{}) {
    auto panel = render_question_dialog_panel(state, std::move(editor));
    auto screen = ftxui::Screen::Create(
        ftxui::Dimension::Fixed(120),
        ftxui::Dimension::Fit(panel));
    ftxui::Render(screen, panel);
    return strip_ansi(screen.ToString());
}

/// PromptInput treats a Return that lands within a few milliseconds of a
/// keystroke as a pasted newline, so tests pause before submitting to emulate
/// a real key press.
void settle_before_return() {
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
}

QuestionDialogState make_state() {
    QuestionDialogState state;
    activate_question_dialog(state, {
        QuestionDialogItem{
            .question = "Which deployment strategy should be used?",
            .header = "Deploy",
            .options = {
                {"Blue-green", "Switch traffic after verification."},
                {"Canary", "Roll out gradually."},
                {"", "", true},
            },
        },
        QuestionDialogItem{
            .question = "Enable monitoring?",
            .header = "Monitor",
            .options = {
                {"Yes", ""},
                {"No", ""},
                {"", "", true},
            },
        },
    });
    return state;
}

} // namespace

TEST_CASE("ask question emits one label-free Other input row",
          "[tui][question-dialog][ask-user-question]") {
    core::tools::AskUserQuestionTool tool;
    std::vector<core::tools::QuestionItem> captured_questions;
    tool.setQuestionCallback(
        [&](core::tools::QuestionRequest request) {
            captured_questions = request.questions;
            request.promise->set_value(std::nullopt);
        });

    const auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{});
    (void)tool.execute(
        R"({"questions":[{"question":"How?","options":[{"label":"Use defaults"},{"label":"Other"}]}]})",
        context);

    REQUIRE(captured_questions.size() == 1);
    const auto& options = captured_questions.front().options;
    REQUIRE(options.size() == 2);
    CHECK(options[0].label == "Use defaults");
    CHECK_FALSE(options[0].accepts_free_text);
    CHECK(options[1].label.empty());
    CHECK(options[1].accepts_free_text);
}

TEST_CASE("question dialog identifies when the embedded Other input is selected",
          "[tui][question-dialog][other]") {
    auto state = make_state();

    CHECK_FALSE(question_dialog_selected_option_is_other(state));

    move_question_dialog_selection(state, -1);  // wraps onto "Other"
    CHECK(state.selected_option == 2);
    CHECK(question_dialog_selected_option_is_other(state));

    move_question_dialog_selection(state, 1);  // wraps back to the first option
    CHECK(state.selected_option == 0);
    CHECK_FALSE(question_dialog_selected_option_is_other(state));

    REQUIRE(select_question_dialog_option(state, 2));
    CHECK(question_dialog_selected_option_is_other(state));
    CHECK_FALSE(select_question_dialog_option(state, 7));
}

TEST_CASE("question dialog turns Other into a validated free-text answer",
          "[tui][question-dialog][other]") {
    auto state = make_state();

    REQUIRE(select_question_dialog_option(state, 2));
    REQUIRE(question_dialog_selected_option_is_other(state));

    state.other_input_text = "   ";
    state.other_input_cursor_position = 3;
    CHECK(accept_question_dialog_answer(state)
          == QuestionDialogAnswerProgress::EmptyOther);
    CHECK(state.other_input_error);
    CHECK(state.current_question_index == 0);

    state.other_input_text = "Deploy to the EU region first";
    state.other_input_cursor_position =
        static_cast<int>(state.other_input_text.size());
    CHECK(accept_question_dialog_answer(state)
          == QuestionDialogAnswerProgress::Advanced);
    REQUIRE(state.answers.size() == 1);
    CHECK(state.answers.front().second == "Deploy to the EU region first");
    CHECK(state.current_question_index == 1);
    CHECK_FALSE(question_dialog_selected_option_is_other(state));
    CHECK(state.other_input_text.empty());
    CHECK(state.other_input_cursor_position == 0);

    CHECK(accept_question_dialog_answer(state)
          == QuestionDialogAnswerProgress::Completed);
    CHECK_FALSE(state.active);
    REQUIRE(state.answers.size() == 2);
    CHECK(state.answers.back().second == "Yes");
}

TEST_CASE("question dialog clears an Other draft before closing",
          "[tui][question-dialog][other]") {
    auto state = make_state();

    CHECK_FALSE(clear_question_dialog_other_input(state));

    REQUIRE(select_question_dialog_option(state, 2));
    state.other_input_text = "Custom plan";
    state.other_input_cursor_position = 11;
    state.other_input_error = true;

    CHECK(clear_question_dialog_other_input(state));
    CHECK(state.other_input_text.empty());
    CHECK(state.other_input_cursor_position == 0);
    CHECK_FALSE(state.other_input_error);
    CHECK_FALSE(clear_question_dialog_other_input(state));
}

TEST_CASE("question dialog joins multi-select answers including Other free text",
          "[tui][question-dialog][multi-select]") {
    QuestionDialogState state;
    activate_question_dialog(state, {
        QuestionDialogItem{
            .question = "Which checks should run?",
            .header = "Checks",
            .options = {
                {"Unit tests", ""},
                {"Lint", ""},
                {"", "", true},
            },
            .multi_select = true,
        },
    });

    REQUIRE(select_question_dialog_option(state, 0));
    toggle_question_dialog_multi_selection(state);
    REQUIRE(select_question_dialog_option(state, 1));
    toggle_question_dialog_multi_selection(state);
    toggle_question_dialog_multi_selection(state);  // untoggles "Lint"

    REQUIRE(select_question_dialog_option(state, 2));
    state.other_input_text = "Fuzz the parser";
    state.other_input_cursor_position =
        static_cast<int>(state.other_input_text.size());

    CHECK(accept_question_dialog_answer(state)
          == QuestionDialogAnswerProgress::Completed);
    REQUIRE(state.answers.size() == 1);
    CHECK(state.answers.front().second == "Unit tests, Fuzz the parser");
}

TEST_CASE("question dialog renders Other as an embedded input row",
          "[tui][question-dialog][other][render]") {
    QuestionDialogState state;
    activate_question_dialog(state, {
        QuestionDialogItem{
            .question = "How should the model proceed?",
            .header = "Approach",
            .options = {
                {"Use defaults", "Apply the standard behavior."},
                {"", "", true},
            },
        },
    });

    const auto unselected =
        render_panel(state, ftxui::text("Other"));
    REQUIRE_THAT(
        unselected,
        Catch::Matchers::ContainsSubstring("  [2] Other"));

    REQUIRE(select_question_dialog_option(state, 1));
    REQUIRE(accept_question_dialog_answer(state)
            == QuestionDialogAnswerProgress::EmptyOther);

    const auto output =
        render_panel(state, ftxui::text("Explain the custom behavior"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring(
        "→ [2] Explain the custom behavior"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring(
        "Enter another answer before submitting."));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring(
        "Shift+Enter: new line"));
}

TEST_CASE("QuestionDialogController owns selection and promise resolution",
          "[tui][question-dialog][controller]") {
    QuestionDialogController controller;
    auto request = make_request();
    auto future = request.promise->get_future();

    CHECK_FALSE(controller.open(std::move(request)));
    CHECK(controller.active());

    auto result = controller.handle_event(ftxui::Event::Return, false);
    REQUIRE(result.handled);
    REQUIRE(result.has_resolution());
    CHECK(result.restore_main_input_focus);
    CHECK_FALSE(result.stop_agent);
    result.resolve();

    const auto answers = future.get();
    REQUIRE(answers.has_value());
    REQUIRE(answers->size() == 1);
    CHECK(answers->front().first == "How should the model proceed?");
    CHECK(answers->front().second == "Use defaults");
    CHECK_FALSE(controller.active());
}

TEST_CASE("QuestionDialogController types into Other while it is highlighted",
          "[tui][question-dialog][controller][other]") {
    QuestionDialogController controller;
    auto request = make_request();
    auto future = request.promise->get_future();
    CHECK_FALSE(controller.open(std::move(request)));

    // The embedded field is visible before selection with "Other" as its
    // placeholder.
    REQUIRE_THAT(
        render(controller),
        Catch::Matchers::ContainsSubstring("[2] Other"));

    // Highlighting "Other" with the arrows focuses that same input row.
    auto hover_other = controller.handle_event(ftxui::Event::ArrowDown, false);
    REQUIRE(hover_other.handled);
    CHECK_FALSE(hover_other.has_resolution());
    REQUIRE_THAT(
        render(controller),
        Catch::Matchers::ContainsSubstring("→ [2] Other"));

    // Digits and spaces are typed instead of triggering option shortcuts.
    REQUIRE(
        controller.handle_event(ftxui::Event::Character("Plan 1"), false).handled);
    REQUIRE(controller.handle_event(ftxui::Event::Character('1'), false).handled);
    REQUIRE_THAT(
        render(controller),
        Catch::Matchers::ContainsSubstring("Plan 11"));

    // The draft remains in the embedded row while another option is selected.
    auto leave_with_draft =
        controller.handle_event(ftxui::Event::ArrowUp, false);
    REQUIRE(leave_with_draft.handled);
    CHECK(leave_with_draft.restore_main_input_focus);
    REQUIRE_THAT(
        render(controller),
        Catch::Matchers::ContainsSubstring("  [2] Plan 11"));

    auto return_to_draft =
        controller.handle_event(ftxui::Event::ArrowDown, false);
    REQUIRE(return_to_draft.handled);
    REQUIRE_THAT(
        render(controller),
        Catch::Matchers::ContainsSubstring("→ [2] Plan 11"));

    // Escape discards the draft but keeps the dialog open.
    auto clear = controller.handle_event(ftxui::Event::Escape, false);
    REQUIRE(clear.handled);
    CHECK_FALSE(clear.has_resolution());
    CHECK(controller.active());
    const auto cleared = render(controller);
    REQUIRE_THAT(cleared, !Catch::Matchers::ContainsSubstring("Plan 11"));
    REQUIRE_THAT(
        cleared,
        Catch::Matchers::ContainsSubstring("→ [2] Other"));

    // Moving away keeps the embedded field visible but removes its focus.
    auto leave_other = controller.handle_event(ftxui::Event::ArrowUp, false);
    REQUIRE(leave_other.handled);
    CHECK(leave_other.restore_main_input_focus);
    REQUIRE_THAT(
        render(controller),
        Catch::Matchers::ContainsSubstring("  [2] Other"));

    // Quick-selecting "Other" opens the editor instead of answering.
    auto quick_select = controller.handle_event(
        ftxui::Event::Character('2'),
        false);
    REQUIRE(quick_select.handled);
    CHECK_FALSE(quick_select.has_resolution());

    REQUIRE(
        controller.handle_event(ftxui::Event::Character("Ship it"), false).handled);
    settle_before_return();
    auto submit = controller.handle_event(ftxui::Event::Return, false);
    REQUIRE(submit.handled);
    REQUIRE(submit.has_resolution());
    submit.resolve();

    const auto answers = future.get();
    REQUIRE(answers.has_value());
    REQUIRE(answers->size() == 1);
    CHECK(answers->front().second == "Ship it");
    CHECK_FALSE(controller.active());
}

TEST_CASE("QuestionDialogController dismisses an empty Other on escape",
          "[tui][question-dialog][controller][other]") {
    QuestionDialogController controller;
    auto request = make_request();
    auto future = request.promise->get_future();
    CHECK_FALSE(controller.open(std::move(request)));

    REQUIRE(controller.handle_event(ftxui::Event::ArrowDown, false).handled);

    auto dismissed = controller.handle_event(ftxui::Event::Escape, false);
    REQUIRE(dismissed.handled);
    REQUIRE(dismissed.has_resolution());
    CHECK(dismissed.restore_main_input_focus);
    CHECK_FALSE(dismissed.stop_agent);
    dismissed.resolve();

    CHECK_FALSE(future.get().has_value());
    CHECK_FALSE(controller.active());
}

TEST_CASE("QuestionDialogController stops the agent when interrupted",
          "[tui][question-dialog][controller][other]") {
    QuestionDialogController controller;
    auto request = make_request();
    auto future = request.promise->get_future();
    CHECK_FALSE(controller.open(std::move(request)));

    REQUIRE(controller.handle_event(ftxui::Event::ArrowDown, false).handled);
    REQUIRE(controller.handle_event(ftxui::Event::Character("Custom"), false).handled);

    auto cancel = controller.handle_event(ftxui::Event::CtrlC, true);
    REQUIRE(cancel.handled);
    REQUIRE(cancel.has_resolution());
    CHECK(cancel.stop_agent);
    cancel.resolve();

    CHECK_FALSE(future.get().has_value());
    CHECK_FALSE(controller.active());
}

TEST_CASE("QuestionDialogController returns a displaced request for safe resolution",
          "[tui][question-dialog][controller][lifecycle]") {
    QuestionDialogController controller;
    auto first = make_request();
    auto first_future = first.promise->get_future();
    CHECK_FALSE(controller.open(std::move(first)));

    auto second = make_request();
    auto displaced = controller.open(std::move(second));
    REQUIRE(displaced);
    displaced->set_value(std::nullopt);

    CHECK_FALSE(first_future.get().has_value());
    CHECK(controller.active());
}

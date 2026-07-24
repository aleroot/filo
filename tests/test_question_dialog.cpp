#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/QuestionDialogController.hpp"
#include "tui/QuestionDialogView.hpp"

#include <ftxui/screen/screen.hpp>

#include <string>
#include <string_view>

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
            {"Other", ""},
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

} // namespace

TEST_CASE("question dialog turns Other into a validated free-text answer",
          "[tui][question-dialog][other]") {
    QuestionDialogState state;
    activate_question_dialog(state, {
        QuestionDialogItem{
            .question = "Which deployment strategy should be used?",
            .header = "Deploy",
            .options = {
                {"Blue-green", "Switch traffic after verification."},
                {"Canary", "Roll out gradually."},
                {"Other", ""},
            },
        },
        QuestionDialogItem{
            .question = "Enable monitoring?",
            .header = "Monitor",
            .options = {
                {"Yes", ""},
                {"No", ""},
                {"Other", ""},
            },
        },
    });

    state.selected_option = 2;
    REQUIRE(question_dialog_selected_option_is_other(state));
    CHECK(accept_question_dialog_selected_answer(state)
          == QuestionDialogAnswerProgress::EditingOther);
    CHECK(state.show_other_input);

    state.other_input_text = "   ";
    state.other_input_cursor_position = 3;
    CHECK(accept_question_dialog_other_answer(state)
          == QuestionDialogAnswerProgress::EmptyOther);
    CHECK(state.other_input_error);
    CHECK(state.current_question_index == 0);

    state.other_input_text = "Deploy to the EU region first";
    state.other_input_cursor_position =
        static_cast<int>(state.other_input_text.size());
    CHECK(accept_question_dialog_other_answer(state)
          == QuestionDialogAnswerProgress::Advanced);
    REQUIRE(state.answers.size() == 1);
    CHECK(state.answers.front().second == "Deploy to the EU region first");
    CHECK(state.current_question_index == 1);
    CHECK_FALSE(state.show_other_input);
    CHECK(state.other_input_text.empty());
    CHECK(state.other_input_cursor_position == 0);

    CHECK(accept_question_dialog_selected_answer(state)
          == QuestionDialogAnswerProgress::Completed);
    CHECK_FALSE(state.active);
    REQUIRE(state.answers.size() == 2);
    CHECK(state.answers.back().second == "Yes");
}

TEST_CASE("question dialog renders Other with the shared prompt-box treatment",
          "[tui][question-dialog][other][render]") {
    QuestionDialogState state;
    activate_question_dialog(state, {
        QuestionDialogItem{
            .question = "How should the model proceed?",
            .header = "Approach",
            .options = {
                {"Use defaults", "Apply the standard behavior."},
                {"Other", ""},
            },
        },
    });
    state.selected_option = 1;
    REQUIRE(accept_question_dialog_selected_answer(state)
            == QuestionDialogAnswerProgress::EditingOther);
    REQUIRE(accept_question_dialog_other_answer(state)
            == QuestionDialogAnswerProgress::EmptyOther);

    auto panel = render_question_dialog_panel(
        state,
        ftxui::text("Explain the custom behavior"));
    auto screen = ftxui::Screen::Create(
        ftxui::Dimension::Fixed(120),
        ftxui::Dimension::Fit(panel));
    ftxui::Render(screen, panel);

    const auto output = strip_ansi(screen.ToString());
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("→ [2] Other"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring(
        "> Explain the custom behavior"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring(
        "Enter an instruction before submitting."));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring(
        "Shift+Enter: new line"));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("Other:"));
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

TEST_CASE("QuestionDialogController edits Other and preserves its draft on back",
          "[tui][question-dialog][controller][other]") {
    QuestionDialogController controller;
    auto request = make_request();
    auto future = request.promise->get_future();
    CHECK_FALSE(controller.open(std::move(request)));

    auto choose_other = controller.handle_event(
        ftxui::Event::Character('2'),
        false);
    REQUIRE(choose_other.handled);
    CHECK_FALSE(choose_other.has_resolution());

    const auto editing = render(controller);
    REQUIRE_THAT(
        editing,
        Catch::Matchers::ContainsSubstring("Type your own instruction..."));

    auto type = controller.handle_event(ftxui::Event::Character("Custom"), false);
    REQUIRE(type.handled);
    CHECK_FALSE(type.has_resolution());
    REQUIRE_THAT(
        render(controller),
        Catch::Matchers::ContainsSubstring("Custom"));

    auto back = controller.handle_event(ftxui::Event::Escape, false);
    REQUIRE(back.handled);
    CHECK(back.restore_main_input_focus);
    CHECK_FALSE(back.has_resolution());
    REQUIRE_THAT(
        render(controller),
        !Catch::Matchers::ContainsSubstring("Custom"));

    const auto reopen = controller.handle_event(
        ftxui::Event::Character('2'),
        false);
    REQUIRE(reopen.handled);
    REQUIRE_THAT(
        render(controller),
        Catch::Matchers::ContainsSubstring("Custom"));

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

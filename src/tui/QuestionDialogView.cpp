#include "QuestionDialogView.hpp"

#include "PromptComponents.hpp"
#include "TuiTheme.hpp"

#include <ftxui/dom/elements.hpp>

#include <algorithm>
#include <format>
#include <utility>

using namespace ftxui;

namespace tui {

namespace {

inline constexpr RgbColor ColorQuestionYellow{255, 200, 80};

Element render_question_tabs(const QuestionDialogState& state) {
    if (state.questions.size() <= 1) {
        return emptyElement();
    }

    Elements tabs;
    for (std::size_t i = 0; i < state.questions.size(); ++i) {
        const auto& question = state.questions[i];
        const std::string label = question.header.empty()
            ? std::format("Q{}", i + 1)
            : question.header;

        std::string icon;
        ftxui::Color style;
        if (static_cast<int>(i) == state.current_question_index) {
            icon = "\xe2\x97\x8f";  // ●
            style = ColorQuestionCyan;
        } else if (i < state.answers.size()) {
            icon = "\xe2\x9c\x93";  // ✓
            style = Color::Green;
        } else {
            icon = "\xe2\x97\x8b";  // ○
            style = Color::GrayDark;
        }

        if (i > 0) {
            tabs.push_back(text("  ") | color(Color::GrayDark));
        }
        tabs.push_back(
            text(std::format("{} {}", icon, label)) | color(style));
    }

    return hbox(std::move(tabs));
}

Element render_question_option(
    const QuestionDialogOption& option,
    int index,
    int selected_index,
    bool multi_select,
    const std::vector<int>& multi_selected,
    bool show_other_input,
    bool other_input_error,
    Element other_input_editor) {
    const int number = index + 1;
    const bool is_selected = index == selected_index;
    const bool is_other = option.label == kQuestionDialogOtherLabel;

    Elements lines;
    Element option_line;
    if (multi_select) {
        const bool checked =
            std::ranges::find(multi_selected, index) != multi_selected.end();
        const std::string checkbox = checked ? "[\xe2\x9c\x93]" : "[ ]";
        const Color option_color = is_selected
            ? static_cast<Color>(ColorQuestionCyan)
            : Color::GrayLight;
        option_line = text(std::format("{} {}", checkbox, option.label))
            | color(option_color);
    } else {
        option_line = is_selected
            ? text(std::format("\xe2\x86\x92 [{}] {}", number, option.label))
                | color(ColorQuestionCyan)
            : text(std::format("  [{}] {}", number, option.label))
                | color(Color::GrayLight);
    }
    lines.push_back(std::move(option_line));

    if (!option.description.empty() && !(is_other && show_other_input)) {
        lines.push_back(
            text(std::format("      {}", option.description)) | dim);
    }

    if (is_other && is_selected && show_other_input) {
        if (!other_input_editor) {
            other_input_editor = text("");
        }
        lines.push_back(
            render_prompt_box(
                std::move(other_input_editor),
                ColorQuestionCyan)
            | xflex);
        if (other_input_error) {
            lines.push_back(
                text("      Enter an instruction before submitting.")
                | color(Color::Red));
        }
    }

    return vbox(std::move(lines));
}

} // namespace

Element render_question_dialog_panel(
    const QuestionDialogState& state,
    Element other_input_editor) {
    if (state.questions.empty()
        || state.current_question_index < 0
        || state.current_question_index >= static_cast<int>(state.questions.size())) {
        return emptyElement();
    }

    const auto& current_question =
        state.questions[static_cast<std::size_t>(state.current_question_index)];
    Elements children;

    auto tabs = render_question_tabs(state);
    if (tabs != emptyElement()) {
        children.push_back(std::move(tabs));
        children.push_back(text(""));
    }

    children.push_back(
        text(std::format("? {}", current_question.question))
        | color(ColorQuestionYellow));

    if (current_question.multi_select) {
        children.push_back(
            text("  (SPACE to toggle, ENTER to submit)") | dim);
    }
    children.push_back(text(""));

    if (!current_question.body.empty()) {
        children.push_back(
            text("  \xe2\x96\xb6 Press ctrl-e to view full content")
            | color(ColorQuestionCyan));
        children.push_back(text(""));
    }

    for (std::size_t i = 0; i < current_question.options.size(); ++i) {
        const bool use_other_editor =
            static_cast<int>(i) == state.selected_option
            && current_question.options[i].label == kQuestionDialogOtherLabel;
        children.push_back(render_question_option(
            current_question.options[i],
            static_cast<int>(i),
            state.selected_option,
            current_question.multi_select,
            state.multi_selected,
            state.show_other_input,
            state.other_input_error,
            use_other_editor ? std::move(other_input_editor) : Element{}));
    }

    if (state.show_other_input) {
        children.push_back(text(""));
        children.push_back(
            text("  Type your instruction  Enter: submit  "
                 "Shift+Enter: new line  Esc: back")
            | dim);
    } else if (state.questions.size() > 1) {
        children.push_back(text(""));
        children.push_back(
            text("  \xe2\x97\x84/\xe2\x96\xba switch  "
                 "\xe2\x96\xb2/\xe2\x96\xbc select  "
                 "Enter: submit  Esc: exit")
            | dim);
    }

    auto content = vbox(std::move(children));
    const std::string help = state.show_other_input
        ? "Enter: submit  Shift+Enter: new line  Esc: back"
        : std::format(
              "Up/Down: select  Enter: confirm  1-{}: quick select  Esc: exit",
              std::min<std::size_t>(current_question.options.size(), 5));

    return vbox({
        hbox({
            text(" ? QUESTION ")
                | ftxui::bold
                | color(ColorQuestionCyan),
            filler(),
            text(help) | color(Color::GrayDark),
        }),
        separator(),
        std::move(content),
        filler(),
    }) | UiBorder(ColorQuestionCyan)
       | size(HEIGHT, GREATER_THAN, 12);
}

} // namespace tui

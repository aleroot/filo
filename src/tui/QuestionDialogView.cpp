#include "QuestionDialogView.hpp"

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

/// Renders one option row. The synthetic free-text option is the input itself;
/// "Other" is supplied only by the editor placeholder.
Element render_question_option(
    const QuestionDialogState& state,
    const QuestionDialogItem& question,
    int index,
    Element other_input_editor) {
    const auto& option = question.options[static_cast<std::size_t>(index)];
    const int number = index + 1;
    const bool is_selected = index == state.selected_option;
    const bool accepts_free_text =
        question_dialog_option_accepts_free_text(option);

    if (accepts_free_text) {
        if (!other_input_editor) {
            other_input_editor = text(std::string(kQuestionDialogOtherLabel));
        }

        const Color prefix_color = is_selected
            ? static_cast<Color>(ColorQuestionCyan)
            : Color::GrayLight;
        auto input_row = hbox({
            text(std::format(
                "{} [{}] ",
                is_selected ? "\xe2\x86\x92" : " ",
                number))
                | color(prefix_color),
            std::move(other_input_editor) | xflex,
        }) | xflex;

        Elements lines;
        lines.push_back(std::move(input_row));
        if (state.other_input_error && is_selected) {
            lines.push_back(
                text("      Enter another answer before submitting.")
                | color(Color::Red));
        }
        return vbox(std::move(lines));
    }

    Elements lines;
    Element option_line;
    if (question.multi_select) {
        const bool checked =
            std::ranges::find(state.multi_selected, index) != state.multi_selected.end();
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

    if (!option.description.empty()) {
        lines.push_back(
            text(std::format("      {}", option.description)) | dim);
    }

    return vbox(std::move(lines));
}

} // namespace

Element render_question_dialog_panel(
    const QuestionDialogState& state,
    Element other_input_editor,
    std::string_view origin_label) {
    if (state.questions.empty()
        || state.current_question_index < 0
        || state.current_question_index >= static_cast<int>(state.questions.size())) {
        return emptyElement();
    }

    const auto& current_question =
        state.questions[static_cast<std::size_t>(state.current_question_index)];
    const bool editing_other =
        question_dialog_selected_option_is_other(state);
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
        const bool row_owns_editor =
            question_dialog_option_accepts_free_text(
                current_question.options[i]);
        children.push_back(render_question_option(
            state,
            current_question,
            static_cast<int>(i),
            row_owns_editor ? std::move(other_input_editor) : Element{}));
    }

    children.push_back(text(""));
    children.push_back(
        text(editing_other
                 ? "  Type your instruction  Enter: submit  "
                   "Shift+Enter: new line  Esc: clear"
                 : "  \xe2\x96\xb2/\xe2\x96\xbc select  "
                   "Enter: confirm  Esc: exit")
        | dim);

    auto content = vbox(std::move(children));
    const std::string help = editing_other
        ? "Up/Down: select  Enter: submit  Esc: clear draft"
        : std::format(
              "Up/Down: select  Enter: confirm  1-{}: quick select  Esc: exit",
              std::min<std::size_t>(current_question.options.size(), 5));

    return vbox({
        hbox({
            text(" ? QUESTION ")
                | ftxui::bold
                | color(ColorQuestionCyan),
            !origin_label.empty()
                ? text(std::format("thread: {}", origin_label))
                      | ftxui::bold | color(ColorQuestionCyan)
                : text(""),
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

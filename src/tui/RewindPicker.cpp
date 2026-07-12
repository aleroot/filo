#include "RewindPicker.hpp"

#include <algorithm>
#include <cctype>
#include <format>
#include <string_view>

namespace tui {

namespace {

constexpr auto kDoubleEscapeWindow = std::chrono::milliseconds(900);
constexpr std::size_t kMaxPromptPreview = 72;

[[nodiscard]] std::string prompt_preview(std::string_view prompt) {
    std::string preview;
    preview.reserve(std::min(prompt.size(), kMaxPromptPreview));
    bool previous_was_space = false;
    for (const unsigned char ch : prompt) {
        const bool is_space = std::isspace(ch) != 0;
        if (is_space) {
            if (!preview.empty() && !previous_was_space) {
                preview.push_back(' ');
            }
        } else {
            preview.push_back(static_cast<char>(ch));
        }
        previous_was_space = is_space;
        if (preview.size() >= kMaxPromptPreview) {
            preview.resize(kMaxPromptPreview - 3);
            preview += "...";
            break;
        }
    }
    while (!preview.empty() && preview.back() == ' ') preview.pop_back();
    return preview.empty() ? std::string("<attachment or empty prompt>") : preview;
}

} // namespace

bool open_rewind_picker(
    RewindPickerState& state,
    const std::vector<core::llm::Message>& messages) {

    struct UserTurn {
        std::size_t history_index;
        std::size_t user_ordinal;
        std::string prompt;
        std::string display;
    };

    std::vector<UserTurn> turns;
    std::size_t ordinal = 0;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].role != "user" || messages[i].synthetic) continue;
        turns.push_back(UserTurn{
            .history_index = i,
            .user_ordinal = ordinal++,
            .prompt = messages[i].input_text.empty()
                ? messages[i].content
                : messages[i].input_text,
            .display = core::llm::message_text_for_display(messages[i]),
        });
    }

    if (turns.empty()) {
        state = {};
        return false;
    }

    state.active = true;
    state.options.clear();

    state.options.reserve(turns.size() + 2);
    for (std::size_t i = 0; i < turns.size(); ++i) {
        state.options.push_back(RewindPickerOption{
            .label = prompt_preview(turns[i].display),
            .description = std::format("before prompt {}", turns[i].user_ordinal + 1),
            .action = RewindPickerOption::Action::RewindToMessage,
            .history_index = turns[i].history_index,
            .user_ordinal = turns[i].user_ordinal,
            .prompt = std::move(turns[i].prompt),
        });
    }
    state.selected = static_cast<int>(state.options.size()) - 1;
    state.options.push_back(RewindPickerOption{
        .label = "Summarize and compact",
        .description = "Keep the conversation, reduce context",
        .action = RewindPickerOption::Action::SummarizeAndCompact,
    });
    state.options.push_back(RewindPickerOption{
        .label = "Cancel",
        .description = "Return to the prompt",
        .action = RewindPickerOption::Action::Cancel,
    });
    return true;
}

RewindPickerEventResult handle_rewind_picker_event(
    RewindPickerState& state,
    const ftxui::Event& event,
    bool cancel_event) {

    if (!state.active) {
        return {};
    }

    RewindPickerEventResult result{.handled = true};
    const int count = static_cast<int>(state.options.size());

    if (event == ftxui::Event::ArrowUp) {
        if (count > 0) {
            state.selected = (state.selected + count - 1) % count;
        }
        return result;
    }

    if (event == ftxui::Event::ArrowDown) {
        if (count > 0) {
            state.selected = (state.selected + 1) % count;
        }
        return result;
    }

    if (event == ftxui::Event::Return) {
        if (state.selected >= 0 && state.selected < count) {
            result.selection = state.options[static_cast<std::size_t>(state.selected)];
        }
        state.active = false;
        return result;
    }

    if (event == ftxui::Event::Escape || cancel_event) {
        result.selection = RewindPickerOption{
            .label = "Cancel",
            .action = RewindPickerOption::Action::Cancel,
        };
        state.active = false;
        return result;
    }

    for (int n = 1; n <= std::min(count, 9); ++n) {
        if (event == ftxui::Event::Character(static_cast<char>('0' + n))) {
            result.selection = state.options[static_cast<std::size_t>(n - 1)];
            state.active = false;
            return result;
        }
    }

    return result;
}

void reset_double_escape_on_non_escape(DoubleEscapeState& state,
                                       const ftxui::Event& event) {
    if (event != ftxui::Event::Escape && event != ftxui::Event::Custom) {
        state.last_press = std::chrono::steady_clock::time_point::min();
    }
}

bool record_escape_press(DoubleEscapeState& state) {
    const auto now = std::chrono::steady_clock::now();
    const bool is_double_escape =
        state.last_press != std::chrono::steady_clock::time_point::min()
        && now - state.last_press <= kDoubleEscapeWindow;
    state.last_press = is_double_escape
        ? std::chrono::steady_clock::time_point::min()
        : now;
    return is_double_escape;
}

} // namespace tui

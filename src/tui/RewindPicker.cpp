#include "RewindPicker.hpp"

#include <algorithm>

namespace tui {

namespace {

constexpr auto kDoubleEscapeWindow = std::chrono::milliseconds(900);

[[nodiscard]] std::optional<RewindPickerAction> action_for_index(int index) {
    switch (index) {
    case 0: return RewindPickerAction::RewindLastTurn;
    case 1: return RewindPickerAction::SummarizeAndCompact;
    case 2: return RewindPickerAction::Cancel;
    default: return std::nullopt;
    }
}

} // namespace

void open_rewind_picker(RewindPickerState& state) {
    state.active = true;
    state.selected = 0;
    state.options = {
        RewindPickerOption{
            .label = "Rewind last turn",
            .description = "Remove the latest user/assistant exchange",
        },
        RewindPickerOption{
            .label = "Summarize and compact",
            .description = "Replace earlier context with a summary",
        },
        RewindPickerOption{
            .label = "Cancel",
            .description = "Return to the prompt",
        },
    };
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
        result.action = action_for_index(state.selected);
        state.active = false;
        return result;
    }

    if (event == ftxui::Event::Escape || cancel_event) {
        result.action = RewindPickerAction::Cancel;
        state.active = false;
        return result;
    }

    for (int n = 1; n <= std::min(count, 3); ++n) {
        if (event == ftxui::Event::Character(static_cast<char>('0' + n))) {
            result.action = action_for_index(n - 1);
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

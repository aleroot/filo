#pragma once

#include <ftxui/component/event.hpp>

#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace tui {

struct RewindPickerOption {
    std::string label;
    std::string description;
};

struct RewindPickerState {
    bool active = false;
    int selected = 0;
    std::vector<RewindPickerOption> options;
};

enum class RewindPickerAction {
    RewindLastTurn,
    SummarizeAndCompact,
    Cancel,
};

struct RewindPickerEventResult {
    bool handled = false;
    std::optional<RewindPickerAction> action;
};

struct DoubleEscapeState {
    std::chrono::steady_clock::time_point last_press =
        std::chrono::steady_clock::time_point::min();
};

void open_rewind_picker(RewindPickerState& state);

[[nodiscard]] RewindPickerEventResult handle_rewind_picker_event(
    RewindPickerState& state,
    const ftxui::Event& event,
    bool cancel_event);

void reset_double_escape_on_non_escape(DoubleEscapeState& state,
                                       const ftxui::Event& event);

[[nodiscard]] bool record_escape_press(DoubleEscapeState& state);

} // namespace tui

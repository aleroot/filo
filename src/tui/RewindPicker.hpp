#pragma once

#include <ftxui/component/event.hpp>

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "core/llm/Models.hpp"

namespace tui {

struct RewindPickerOption {
    std::string label;
    std::string description;
    enum class Action {
        RewindToMessage,
        SummarizeAndCompact,
        Cancel,
    } action = Action::Cancel;
    std::size_t history_index = 0;
    std::size_t user_ordinal = 0;
    std::string prompt;
};

struct RewindPickerState {
    bool active = false;
    int selected = 0;
    std::vector<RewindPickerOption> options;
};

struct RewindPickerEventResult {
    bool handled = false;
    std::optional<RewindPickerOption> selection{};
};

struct DoubleEscapeState {
    std::chrono::steady_clock::time_point last_press =
        std::chrono::steady_clock::time_point::min();
};

[[nodiscard]] bool open_rewind_picker(
    RewindPickerState& state,
    const std::vector<core::llm::Message>& messages);

[[nodiscard]] RewindPickerEventResult handle_rewind_picker_event(
    RewindPickerState& state,
    const ftxui::Event& event,
    bool cancel_event);

void reset_double_escape_on_non_escape(DoubleEscapeState& state,
                                       const ftxui::Event& event);

[[nodiscard]] bool record_escape_press(DoubleEscapeState& state);

} // namespace tui

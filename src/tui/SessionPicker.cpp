#include "SessionPicker.hpp"

#include "KeyInput.hpp"

#include <algorithm>
#include <format>

namespace tui {
namespace {

[[nodiscard]] bool is_printable_char(const ftxui::Event& event) {
    if (!event.is_character()) {
        return false;
    }
    const std::string ch = event.character();
    if (ch.size() != 1) {
        return false;
    }
    const unsigned char c = static_cast<unsigned char>(ch[0]);
    // Exclude control characters and DEL. Letter bindings (n/d/r) are only
    // active in Browse mode, so filter/rename can accept those characters.
    return c >= 32 && c != 127;
}

void clamp_selection(SessionPickerState& state) {
    const int count = static_cast<int>(state.filtered.size());
    if (count <= 0) {
        state.selected = 0;
        return;
    }
    state.selected = std::clamp(state.selected, 0, count - 1);
}

} // namespace

void refresh_session_picker_filter(SessionPickerState& state) {
    state.filtered = core::session::filter_threads(state.sessions, state.query);
    clamp_selection(state);
}

namespace {

void open_picker(SessionPickerState& state,
                 std::vector<core::session::SessionInfo> entries,
                 std::string_view current_session_id,
                 SessionPickerResource resource) {
    state = SessionPickerState{};
    state.active = true;
    state.resource = resource;
    state.mode = SessionPickerMode::Browse;
    state.sessions = std::move(entries);
    state.current_session_id = std::string{current_session_id};
    refresh_session_picker_filter(state);

    // Prefer focusing the entry owned by the current thread when present.
    if (!state.current_session_id.empty()) {
        for (int i = 0; i < static_cast<int>(state.filtered.size()); ++i) {
            if (state.filtered[static_cast<std::size_t>(i)].session_id
                == state.current_session_id) {
                state.selected = i;
                break;
            }
        }
    }
}

} // namespace

void open_session_picker(SessionPickerState& state,
                         std::vector<core::session::SessionInfo> sessions,
                         std::string_view current_session_id) {
    open_picker(state,
                std::move(sessions),
                current_session_id,
                SessionPickerResource::SavedSessions);
}

void open_thread_picker(SessionPickerState& state,
                        std::vector<core::session::SessionInfo> threads,
                        std::string_view current_session_id) {
    open_picker(state,
                std::move(threads),
                current_session_id,
                SessionPickerResource::ActiveThreads);
}

std::optional<core::session::SessionInfo>
selected_session(const SessionPickerState& state) {
    if (state.filtered.empty()) {
        return std::nullopt;
    }
    const int idx = std::clamp(state.selected, 0,
                               static_cast<int>(state.filtered.size()) - 1);
    return state.filtered[static_cast<std::size_t>(idx)];
}

SessionPickerEventResult handle_session_picker_event(
    SessionPickerState& state,
    const ftxui::Event& event) {
    if (!state.active) {
        return {};
    }

    SessionPickerEventResult result{.handled = true};

    // ── Rename mode ──────────────────────────────────────────────────────────
    if (state.mode == SessionPickerMode::Rename) {
        if (event == ftxui::Event::Escape) {
            state.mode = SessionPickerMode::Browse;
            state.rename_buffer.clear();
            state.status_message.clear();
            return result;
        }
        if (event == ftxui::Event::Return) {
            const auto focused = selected_session(state);
            if (!focused) {
                state.mode = SessionPickerMode::Browse;
                return result;
            }
            result.action = SessionPickerAction::RenameCommit;
            result.filtered_index = state.selected;
            result.rename_name = state.rename_buffer;
            state.mode = SessionPickerMode::Browse;
            state.rename_buffer.clear();
            return result;
        }
        if (event == ftxui::Event::Backspace) {
            if (!state.rename_buffer.empty()) {
                state.rename_buffer.pop_back();
            }
            return result;
        }
        if (is_printable_char(event)) {
            state.rename_buffer += event.character();
            return result;
        }
        return result;
    }

    // ── Filter mode ──────────────────────────────────────────────────────────
    if (state.mode == SessionPickerMode::Filter) {
        if (event == ftxui::Event::Escape) {
            if (!state.query.empty()) {
                state.query.clear();
                refresh_session_picker_filter(state);
            }
            state.mode = SessionPickerMode::Browse;
            return result;
        }
        if (event == ftxui::Event::Return) {
            state.mode = SessionPickerMode::Browse;
            return result;
        }
        if (event == ftxui::Event::Backspace) {
            if (!state.query.empty()) {
                state.query.pop_back();
                refresh_session_picker_filter(state);
            } else {
                state.mode = SessionPickerMode::Browse;
            }
            return result;
        }
        if (event == ftxui::Event::ArrowUp || event == ftxui::Event::ArrowDown) {
            const int count = static_cast<int>(state.filtered.size());
            if (count > 0) {
                if (event == ftxui::Event::ArrowUp) {
                    state.selected = (state.selected + count - 1) % count;
                } else {
                    state.selected = (state.selected + 1) % count;
                }
            }
            return result;
        }
        if (is_printable_char(event)) {
            state.query += event.character();
            refresh_session_picker_filter(state);
            return result;
        }
        return result;
    }

    // ── Browse mode ──────────────────────────────────────────────────────────
    const int count = static_cast<int>(state.filtered.size());

    if (event == ftxui::Event::Escape) {
        state.active = false;
        result.action = SessionPickerAction::Close;
        return result;
    }
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
        if (count > 0) {
            result.action = SessionPickerAction::Open;
            result.filtered_index = state.selected;
            state.active = false;
        }
        return result;
    }
    if (event == ftxui::Event::Backspace || event == ftxui::Event::Delete) {
        if (state.resource == SessionPickerResource::SavedSessions && count > 0) {
            // Preserve the original /sessions interaction: Delete/Backspace
            // removes the selected persisted session in one action.
            result.action = SessionPickerAction::Delete;
            result.filtered_index = state.selected;
        }
        return result;
    }
    if (event == ftxui::Event::Character('c')
        || event == ftxui::Event::Character('C')) {
        if (state.resource == SessionPickerResource::ActiveThreads && count > 0) {
            // Closing a live runtime is deliberately distinct from deleting
            // its persisted session. MainApp archives before dropping it.
            result.action = SessionPickerAction::CloseThread;
            result.filtered_index = state.selected;
        }
        return result;
    }
    if (event == ftxui::Event::Character('n')
        || event == ftxui::Event::Character('N')) {
        if (state.resource == SessionPickerResource::ActiveThreads) {
            result.action = SessionPickerAction::NewThread;
            state.active = false;
        }
        return result;
    }
    if (event == ftxui::Event::Character('r')
        || event == ftxui::Event::Character('R')) {
        if (state.resource == SessionPickerResource::ActiveThreads && count > 0) {
            const auto focused = selected_session(state);
            state.mode = SessionPickerMode::Rename;
            state.rename_buffer = focused ? focused->name : std::string{};
            state.status_message = "Rename thread — Enter to save, Esc to cancel";
        }
        return result;
    }
    if (event == ftxui::Event::Character('/')) {
        state.mode = SessionPickerMode::Filter;
        state.status_message = state.resource == SessionPickerResource::ActiveThreads
            ? "Filter active threads…"
            : "Filter saved sessions…";
        return result;
    }
    // Agentty-style: Ctrl+J while open closes the browser (toggle).
    // Use is_ctrl_j_event (not raw LF) so Enter is never misclassified.
    if (is_ctrl_j_event(event) || is_ctrl_h_event(event)) {
        state.active = false;
        result.action = SessionPickerAction::Close;
        return result;
    }

    return result;
}

} // namespace tui

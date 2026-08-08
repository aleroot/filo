#pragma once

// Shared picker mechanics for two deliberately distinct resources:
// active in-process threads and persisted sessions.
//
// Owns picker state + pure event handling so MainApp stays a thin wiring layer.
// Rendering lives in PromptComponents; catalogue logic lives in ThreadCatalog.

#include "core/session/SessionStore.hpp"
#include "core/session/ThreadCatalog.hpp"

#include <ftxui/component/event.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tui {

enum class SessionPickerMode {
    Browse,
    Filter,
    Rename,
};

enum class SessionPickerResource {
    ActiveThreads,
    SavedSessions,
};

struct SessionPickerState {
    bool active = false;
    SessionPickerResource resource = SessionPickerResource::SavedSessions;
    SessionPickerMode mode = SessionPickerMode::Browse;
    int selected = 0;
    std::vector<core::session::SessionInfo> sessions;   ///< full catalogue
    std::vector<core::session::SessionInfo> filtered;   ///< after query
    std::string query;
    std::string rename_buffer;
    std::string status_message;
    std::string current_session_id;
};

enum class SessionPickerAction {
    None,
    Open,
    Delete,
    CloseThread,
    NewThread,
    RenameCommit,
    Close,
};

struct SessionPickerEventResult {
    bool handled = false;
    SessionPickerAction action = SessionPickerAction::None;
    /// Index into state.filtered for Open/Delete/CloseThread/RenameCommit.
    std::optional<int> filtered_index;
    std::string rename_name;
};

/// Populate the persisted-session browser.
void open_session_picker(SessionPickerState& state,
                         std::vector<core::session::SessionInfo> sessions,
                         std::string_view current_session_id);

/// Populate the active-thread browser. Always opens; `N` can create another
/// thread even if the registry catalogue is unexpectedly empty.
void open_thread_picker(SessionPickerState& state,
                        std::vector<core::session::SessionInfo> threads,
                        std::string_view current_session_id);

/// Recompute filtered list and clamp selection after query / catalogue changes.
void refresh_session_picker_filter(SessionPickerState& state);

/// Handle a key event while the picker is active.
[[nodiscard]] SessionPickerEventResult handle_session_picker_event(
    SessionPickerState& state,
    const ftxui::Event& event);

/// Snapshot of the focused filtered entry, if any.
[[nodiscard]] std::optional<core::session::SessionInfo>
selected_session(const SessionPickerState& state);

} // namespace tui

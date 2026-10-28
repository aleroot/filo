#pragma once

#include <cstddef>

namespace tui {

/// What the visible tab and the rest of the app are doing when Ctrl+C lands.
struct CtrlCState {
    /// The visible tab runs a tool call or a direct shell command.
    bool tool_running = false;
    /// The visible tab runs a turn, a review, or a terminal command.
    bool session_running = false;
    /// Other tabs that run a turn, a review, or a direct shell command.
    std::size_t other_threads_working = 0;
};

enum class CtrlCAction {
    InterruptTool,  ///< Kill the visible tab's running tool; its session continues.
    StopSession,    ///< Stop the visible tab's session; other tabs keep working.
    ConfirmExit,    ///< Quit only on a second press: other tabs would be stopped.
    Exit,           ///< Nothing is running anywhere: quit now.
};

/// Ctrl+C acts at the narrowest level that is busy, and only on the visible
/// tab: the running tool first, then the session, then the app. Quitting
/// while another tab works needs a confirmation, because it stops that work.
[[nodiscard]] constexpr CtrlCAction decide_ctrl_c(const CtrlCState& state) noexcept {
    if (state.tool_running) return CtrlCAction::InterruptTool;
    if (state.session_running) return CtrlCAction::StopSession;
    if (state.other_threads_working > 0) return CtrlCAction::ConfirmExit;
    return CtrlCAction::Exit;
}

} // namespace tui

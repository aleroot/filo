#pragma once

#include <ftxui/component/event.hpp>

namespace tui {

/// Shared dismissal keys for read-only details panels.
bool is_panel_dismiss_event(const ftxui::Event& event);

/// True when @p event is a redraw notification instead of user input: Filo's own
/// wake_ui() signal and the notification FTXUI posts after every selection
/// change are the same `Event::Custom`.
///
/// A handler must answer it by returning false, never true. Reporting it as
/// handled breaks two things:
///  - FTXUI cancels an in-progress mouse selection whenever the component tree
///    consumes an event, and it posts Custom right after every selection
///    change, so a left-drag would wipe itself between two mouse moves and no
///    text could be selected out of the transcript.
///  - A modal that claims it and calls wake_ui() requeues Custom from inside the
///    same FTXUI task drain (since 7.0.3 posted tasks run before the frame is
///    drawn), which spins forever and freezes the UI.
bool is_refresh_notification(const ftxui::Event& event);

bool is_ctrl_letter_event(const ftxui::Event& event, char letter);
bool is_ctrl_x_event(const ftxui::Event& event);
bool is_ctrl_o_event(const ftxui::Event& event);
bool is_ctrl_y_event(const ftxui::Event& event);
bool is_ctrl_d_event(const ftxui::Event& event);
bool is_ctrl_f_event(const ftxui::Event& event);
bool is_ctrl_g_event(const ftxui::Event& event);
bool is_ctrl_v_event(const ftxui::Event& event);
bool is_ctrl_l_event(const ftxui::Event& event);
bool is_ctrl_c_event(const ftxui::Event& event);
bool is_ctrl_p_event(const ftxui::Event& event);
bool is_ctrl_r_event(const ftxui::Event& event);
bool is_ctrl_t_event(const ftxui::Event& event);
bool is_ctrl_n_event(const ftxui::Event& event);
bool is_ctrl_j_event(const ftxui::Event& event);
bool is_ctrl_h_event(const ftxui::Event& event);

/// True when @p event is a Ctrl+Enter keypress (modifyOtherKeys CSI sequence
/// ESC[27;5;13~). Used by list pickers that distinguish Enter (activate) from
/// Ctrl+Enter (copy-to-clipboard).
bool is_ctrl_enter_event(const ftxui::Event& event);

} // namespace tui

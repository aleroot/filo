#pragma once

#include <ftxui/component/event.hpp>

namespace tui {

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

/// True when @p event is a Ctrl+Enter keypress (modifyOtherKeys CSI sequence
/// ESC[27;5;13~). Used by list pickers that distinguish Enter (activate) from
/// Ctrl+Enter (copy-to-clipboard).
bool is_ctrl_enter_event(const ftxui::Event& event);

} // namespace tui

#include <catch2/catch_test_macros.hpp>

#include "tui/CtrlCPolicy.hpp"

using tui::CtrlCAction;
using tui::decide_ctrl_c;

TEST_CASE("Ctrl+C interrupts the running tool before anything else", "[tui][ctrl-c]") {
    CHECK(decide_ctrl_c({.tool_running = true}) == CtrlCAction::InterruptTool);
    CHECK(decide_ctrl_c({.tool_running = true, .session_running = true})
          == CtrlCAction::InterruptTool);
    CHECK(decide_ctrl_c({.tool_running = true, .session_running = true,
                         .other_threads_working = 2})
          == CtrlCAction::InterruptTool);
}

TEST_CASE("Ctrl+C stops only the visible session, whatever other tabs do", "[tui][ctrl-c]") {
    CHECK(decide_ctrl_c({.session_running = true}) == CtrlCAction::StopSession);
    CHECK(decide_ctrl_c({.session_running = true, .other_threads_working = 1})
          == CtrlCAction::StopSession);
}

TEST_CASE("Ctrl+C on an idle tab quits, confirming while other tabs work", "[tui][ctrl-c]") {
    CHECK(decide_ctrl_c({}) == CtrlCAction::Exit);
    CHECK(decide_ctrl_c({.other_threads_working = 1}) == CtrlCAction::ConfirmExit);
}

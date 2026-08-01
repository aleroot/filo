#include <catch2/catch_test_macros.hpp>

#include "tui/KeyInput.hpp"

#include <ftxui/component/event.hpp>

TEST_CASE("KeyInput detects raw Ctrl byte events", "[tui][key_input]") {
    REQUIRE(tui::is_ctrl_x_event(ftxui::Event::Special({24}))); // Ctrl+X
    REQUIRE(tui::is_ctrl_d_event(ftxui::Event::Special({4})));  // Ctrl+D
    REQUIRE(tui::is_ctrl_c_event(ftxui::Event::Special({3})));  // Ctrl+C
    REQUIRE(tui::is_ctrl_l_event(ftxui::Event::Special({12}))); // Ctrl+L
    REQUIRE_FALSE(tui::is_ctrl_x_event(ftxui::Event::Character('x')));
}

TEST_CASE("KeyInput detects all supported Ctrl events", "[tui][key_input]") {
    // We'll use the raw byte representation for simplicity in this test.
    // 'a' is 1, 'b' is 2, ..., 'z' is 26.
    REQUIRE(tui::is_ctrl_o_event(ftxui::Event::Special({15})));
    REQUIRE(tui::is_ctrl_y_event(ftxui::Event::Special({25})));
    REQUIRE(tui::is_ctrl_f_event(ftxui::Event::Special({6})));
    REQUIRE(tui::is_ctrl_g_event(ftxui::Event::Special({7})));
    REQUIRE(tui::is_ctrl_v_event(ftxui::Event::Special({22})));
    REQUIRE(tui::is_ctrl_p_event(ftxui::Event::Special({16})));
    REQUIRE(tui::is_ctrl_r_event(ftxui::Event::Special({18})));
}

TEST_CASE("KeyInput keeps the prompt-editor and run-code bindings distinct",
          "[tui][key_input]") {
    // Ctrl+G opens the prompt editor (Claude Code parity), Ctrl+X is the
    // Gemini CLI-compatible alias, and Ctrl+R runs fenced code.
    const auto ctrl_r = ftxui::Event::Special({18});
    REQUIRE(tui::is_ctrl_r_event(ctrl_r));
    REQUIRE_FALSE(tui::is_ctrl_g_event(ctrl_r));
    REQUIRE_FALSE(tui::is_ctrl_x_event(ctrl_r));

    const auto ctrl_g = ftxui::Event::Special({7});
    REQUIRE_FALSE(tui::is_ctrl_r_event(ctrl_g));
    REQUIRE(tui::is_ctrl_r_event(ftxui::Event::Special("\x1B[114;5u")));
}

TEST_CASE("KeyInput detects kitty keyboard protocol events", "[tui][key_input]") {
    REQUIRE(tui::is_ctrl_letter_event(ftxui::Event::Special("\x1B[120;5u"), 'x'));
    REQUIRE(tui::is_ctrl_g_event(ftxui::Event::Special("\x1B[103;5u")));
    REQUIRE(tui::is_ctrl_letter_event(ftxui::Event::Special("\x1B[68;5u"), 'd'));
    REQUIRE(tui::is_ctrl_p_event(ftxui::Event::Special("\x1B[112;5u")));
    REQUIRE_FALSE(tui::is_ctrl_letter_event(ftxui::Event::Special("\x1B[120;1u"), 'x'));
}

TEST_CASE("KeyInput detects modifyOtherKeys events", "[tui][key_input]") {
    REQUIRE(tui::is_ctrl_letter_event(ftxui::Event::Special("\x1B[27;5;121~"), 'y'));
    REQUIRE(tui::is_ctrl_y_event(ftxui::Event::Special("\x1B[27;5;121~")));
    REQUIRE_FALSE(tui::is_ctrl_y_event(ftxui::Event::Special("\x1B[27;3;121~")));
}

TEST_CASE("KeyInput detects Alt+P model picker events", "[tui][key_input]") {
    REQUIRE(tui::is_alt_p_event(ftxui::Event::Special("\x1B" "p")));
    REQUIRE(tui::is_alt_p_event(ftxui::Event::Special("\x1B[112;3u")));
    REQUIRE(tui::is_alt_p_event(ftxui::Event::Special("\x1B[27;3;112~")));

    REQUIRE_FALSE(tui::is_alt_p_event(ftxui::Event::Special({16}))); // Ctrl+P
    REQUIRE_FALSE(tui::is_alt_p_event(ftxui::Event::Character('p')));
    REQUIRE_FALSE(tui::is_alt_p_event(ftxui::Event::Special("\x1B" "P")));
    REQUIRE_FALSE(tui::is_alt_p_event(ftxui::Event::Special("\x1B[112;6u")));
}

TEST_CASE("KeyInput detects Ctrl+Enter events", "[tui][key_input]") {
    // modifyOtherKeys CSI: ESC[27;5;13~  (modifier 5 = Ctrl, key 13 = Enter)
    REQUIRE(tui::is_ctrl_enter_event(ftxui::Event::Special("\x1B[27;5;13~")));
    // Shift+Enter (modifier 2) must not match.
    REQUIRE_FALSE(tui::is_ctrl_enter_event(ftxui::Event::Special("\x1B[27;2;13~")));
    // Plain Enter must not match.
    REQUIRE_FALSE(tui::is_ctrl_enter_event(ftxui::Event::Return));
    // Ctrl+Y (same modifier, different key) must not match.
    REQUIRE_FALSE(tui::is_ctrl_enter_event(ftxui::Event::Special("\x1B[27;5;121~")));
    // Garbage must not match.
    REQUIRE_FALSE(tui::is_ctrl_enter_event(ftxui::Event::Special("xyz")));
}

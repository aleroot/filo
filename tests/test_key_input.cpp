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
    REQUIRE(tui::is_ctrl_v_event(ftxui::Event::Special({22})));
}

TEST_CASE("KeyInput detects kitty keyboard protocol events", "[tui][key_input]") {
    REQUIRE(tui::is_ctrl_letter_event(ftxui::Event::Special("\x1B[120;5u"), 'x'));
    REQUIRE(tui::is_ctrl_letter_event(ftxui::Event::Special("\x1B[68;5u"), 'd'));
    REQUIRE_FALSE(tui::is_ctrl_letter_event(ftxui::Event::Special("\x1B[120;1u"), 'x'));
}

TEST_CASE("KeyInput detects modifyOtherKeys events", "[tui][key_input]") {
    REQUIRE(tui::is_ctrl_letter_event(ftxui::Event::Special("\x1B[27;5;121~"), 'y'));
    REQUIRE(tui::is_ctrl_y_event(ftxui::Event::Special("\x1B[27;5;121~")));
    REQUIRE_FALSE(tui::is_ctrl_y_event(ftxui::Event::Special("\x1B[27;3;121~")));
}

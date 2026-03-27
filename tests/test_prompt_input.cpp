#include <catch2/catch_test_macros.hpp>

#include "tui/PromptInput.hpp"

#include <ftxui/dom/elements.hpp>
#include <ftxui/component/component_options.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/screen/screen.hpp>
#include <string>
#include <thread>
#include <chrono>

using namespace ftxui;

// Helper: build a PromptInput component with a shared content string and cursor
static ftxui::Component make_input(std::string& content,
                                   int& cursor,
                                   bool multiline = false) {
    InputOption opt;
    opt.content = &content;
    opt.cursor_position = &cursor;
    opt.multiline = multiline;
    return tui::PromptInput(&content, "", opt);
}

// ── Cursor movement ───────────────────────────────────────────────────────────

TEST_CASE("PromptInput - Ctrl+A moves to start of line", "[prompt_input][cursor]") {
    std::string content = "hello world";
    int cursor = static_cast<int>(content.size());

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special({1}));  // Ctrl+A

    REQUIRE(cursor == 0);
    REQUIRE(content == "hello world");
}

TEST_CASE("PromptInput - Ctrl+E moves to end of line", "[prompt_input][cursor]") {
    std::string content = "hello world";
    int cursor = 0;

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special({5}));  // Ctrl+E

    REQUIRE(cursor == static_cast<int>(content.size()));
    REQUIRE(content == "hello world");
}

TEST_CASE("PromptInput - Ctrl+F moves right one character", "[prompt_input][cursor]") {
    std::string content = "hello";
    int cursor = 0;

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special({6}));  // Ctrl+F

    REQUIRE(cursor == 1);
    REQUIRE(content == "hello");
}

TEST_CASE("PromptInput - Alt+Left moves one word left", "[prompt_input][cursor]") {
    std::string content = "hello world";
    int cursor = static_cast<int>(content.size());  // at end

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special("\x1B[1;3D"));  // Alt+Left

    REQUIRE(cursor == 6);  // start of "world"
}

TEST_CASE("PromptInput - Alt+B moves one word left", "[prompt_input][cursor]") {
    std::string content = "hello world";
    int cursor = static_cast<int>(content.size());  // at end

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special("\x1B" "b"));  // Alt+B

    REQUIRE(cursor == 6);  // start of "world"
}

TEST_CASE("PromptInput - Alt+Right moves one word right", "[prompt_input][cursor]") {
    std::string content = "hello world";
    int cursor = 0;

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special("\x1B[1;3C"));  // Alt+Right

    REQUIRE(cursor == 5);  // end of "hello"
}

TEST_CASE("PromptInput - Alt+F moves one word right", "[prompt_input][cursor]") {
    std::string content = "hello world";
    int cursor = 0;

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special("\x1B" "f"));  // Alt+F

    REQUIRE(cursor == 5);  // end of "hello"
}

// ── Editing ───────────────────────────────────────────────────────────────────

TEST_CASE("PromptInput - Ctrl+K deletes to end of line", "[prompt_input][edit]") {
    std::string content = "hello world";
    int cursor = 5;  // after "hello"

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special({11}));  // Ctrl+K

    REQUIRE(content == "hello");
    REQUIRE(cursor == 5);
}

TEST_CASE("PromptInput - Ctrl+K at end of content is no-op", "[prompt_input][edit]") {
    std::string content = "hello";
    int cursor = static_cast<int>(content.size());

    auto comp = make_input(content, cursor);
    bool handled = comp->OnEvent(Event::Special({11}));  // Ctrl+K

    REQUIRE(!handled);
    REQUIRE(content == "hello");
}

TEST_CASE("PromptInput - Ctrl+U deletes to start of line", "[prompt_input][edit]") {
    std::string content = "hello world";
    int cursor = 5;  // after "hello"

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special({21}));  // Ctrl+U

    REQUIRE(content == " world");
    REQUIRE(cursor == 0);
}

TEST_CASE("PromptInput - Ctrl+U at start of content is no-op", "[prompt_input][edit]") {
    std::string content = "hello";
    int cursor = 0;

    auto comp = make_input(content, cursor);
    bool handled = comp->OnEvent(Event::Special({21}));  // Ctrl+U

    REQUIRE(!handled);
    REQUIRE(content == "hello");
}

TEST_CASE("PromptInput - Ctrl+W deletes previous word", "[prompt_input][edit]") {
    std::string content = "hello world";
    int cursor = static_cast<int>(content.size());  // at end

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special({23}));  // Ctrl+W

    REQUIRE(content == "hello ");
    REQUIRE(cursor == 6);
}

TEST_CASE("PromptInput - Alt+Backspace deletes previous word", "[prompt_input][edit]") {
    std::string content = "hello world";
    int cursor = static_cast<int>(content.size());

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special("\x1B\x7f"));  // Alt+Backspace

    REQUIRE(content == "hello ");
    REQUIRE(cursor == 6);
}

TEST_CASE("PromptInput - Ctrl+W skips whitespace then deletes word", "[prompt_input][edit]") {
    std::string content = "hello   world";
    int cursor = static_cast<int>(content.size());

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special({23}));  // Ctrl+W — delete "world"

    REQUIRE(content == "hello   ");
    REQUIRE(cursor == 8);
}

TEST_CASE("PromptInput - Alt+D deletes next word", "[prompt_input][edit]") {
    std::string content = "hello world";
    int cursor = 0;

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special("\x1B" "d"));  // Alt+D

    REQUIRE(content == " world");
    REQUIRE(cursor == 0);
}

TEST_CASE("PromptInput - Ctrl+Delete deletes next word", "[prompt_input][edit]") {
    std::string content = "hello world";
    int cursor = 6;  // at start of "world"

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special("\x1B[3;5~"));  // Ctrl+Delete

    REQUIRE(content == "hello ");
    REQUIRE(cursor == 6);
}

TEST_CASE("PromptInput - Alt+Delete deletes next word", "[prompt_input][edit]") {
    std::string content = "hello world";
    int cursor = 6;  // at start of "world"

    auto comp = make_input(content, cursor);
    comp->OnEvent(Event::Special("\x1B[3;3~"));  // Alt+Delete

    REQUIRE(content == "hello ");
    REQUIRE(cursor == 6);
}

TEST_CASE("PromptInput - Alt+D at end of content is no-op", "[prompt_input][edit]") {
    std::string content = "hello";
    int cursor = static_cast<int>(content.size());

    auto comp = make_input(content, cursor);
    bool handled = comp->OnEvent(Event::Special("\x1B" "d"));

    REQUIRE(!handled);
    REQUIRE(content == "hello");
}

TEST_CASE("PromptInput - empty placeholder keeps cursor at the left edge",
          "[prompt_input][render]") {
    std::string content;
    int cursor = 0;

    InputOption opt;
    opt.content = &content;
    opt.placeholder = "Ask anything";
    opt.cursor_position = &cursor;
    opt.multiline = false;

    auto comp = tui::PromptInput(&content, "Ask anything", opt);

    auto screen = Screen::Create(Dimension::Fixed(24), Dimension::Fixed(1));
    Render(screen, comp->Render());

    REQUIRE(screen.cursor().x == 0);
    REQUIRE(screen.cursor().y == 0);
}

TEST_CASE("PromptInput - Ctrl+D deletes character to the right", "[prompt_input][edit]") {
    std::string content = "hello";
    int cursor = 1;  // at 'e'

    auto comp = make_input(content, cursor);
    bool handled = comp->OnEvent(Event::Special({4}));  // Ctrl+D

    REQUIRE(handled);
    REQUIRE(content == "hllo");
    REQUIRE(cursor == 1);
}

TEST_CASE("PromptInput - Ctrl+D at end of content is no-op", "[prompt_input][edit]") {
    std::string content = "hello";
    int cursor = static_cast<int>(content.size());

    auto comp = make_input(content, cursor);
    bool handled = comp->OnEvent(Event::Special({4}));  // Ctrl+D

    REQUIRE(!handled);
    REQUIRE(content == "hello");
}

TEST_CASE("PromptInput - Ctrl+W at start of content is no-op", "[prompt_input][edit]") {
    std::string content = "hello";
    int cursor = 0;

    auto comp = make_input(content, cursor);
    bool handled = comp->OnEvent(Event::Special({23}));

    REQUIRE(!handled);
    REQUIRE(content == "hello");
}

// ── Newline insertion ─────────────────────────────────────────────────────────

TEST_CASE("PromptInput - Shift+Enter inserts newline without submitting",
          "[prompt_input][newline]") {
    bool entered = false;
    std::string content = "hello";
    int cursor = static_cast<int>(content.size());

    InputOption opt;
    opt.content = &content;
    opt.cursor_position = &cursor;
    opt.multiline = true;
    opt.on_enter = [&entered]() { entered = true; };
    auto comp = tui::PromptInput(&content, "", opt);

    comp->OnEvent(Event::Special("\x1B[27;2;13~"));  // Shift+Enter

    REQUIRE(content == "hello\n");
    REQUIRE(!entered);  // must NOT submit
}

TEST_CASE("PromptInput - Ctrl+Enter inserts newline without submitting",
          "[prompt_input][newline]") {
    bool entered = false;
    std::string content = "hello";
    int cursor = static_cast<int>(content.size());

    InputOption opt;
    opt.content = &content;
    opt.cursor_position = &cursor;
    opt.multiline = true;
    opt.on_enter = [&entered]() { entered = true; };
    auto comp = tui::PromptInput(&content, "", opt);

    comp->OnEvent(Event::Special("\x1B[27;5;13~"));  // Ctrl+Enter

    REQUIRE(content == "hello\n");
    REQUIRE(!entered);
}

TEST_CASE("PromptInput - Alt+Enter inserts newline without submitting",
          "[prompt_input][newline]") {
    bool entered = false;
    std::string content = "hello";
    int cursor = static_cast<int>(content.size());

    InputOption opt;
    opt.content = &content;
    opt.cursor_position = &cursor;
    opt.multiline = true;
    opt.on_enter = [&entered]() { entered = true; };
    auto comp = tui::PromptInput(&content, "", opt);

    comp->OnEvent(Event::Special("\x1B\n"));  // Alt+Enter

    REQUIRE(content == "hello\n");
    REQUIRE(!entered);
}

TEST_CASE("PromptInput - bracketed multiline paste inserts newlines without submitting",
          "[prompt_input][paste][newline]") {
    bool entered = false;
    std::string content;
    int cursor = 0;

    InputOption opt;
    opt.content = &content;
    opt.cursor_position = &cursor;
    opt.multiline = false;
    opt.on_enter = [&entered]() { entered = true; };
    auto comp = tui::PromptInput(&content, "", opt);

    comp->OnEvent(Event::Special("\x1B[200~"));  // bracketed paste start
    comp->OnEvent(Event::Character('a'));
    comp->OnEvent(Event::Return);                // newline within paste
    comp->OnEvent(Event::Character('b'));
    comp->OnEvent(Event::Special("\x1B[201~"));  // bracketed paste end

    REQUIRE(content == "a\nb");
    REQUIRE(cursor == 3);
    REQUIRE(!entered);
}

TEST_CASE("PromptInput - fast return after typed character inserts newline without submit",
          "[prompt_input][paste][newline]") {
    bool entered = false;
    std::string content;
    int cursor = 0;

    InputOption opt;
    opt.content = &content;
    opt.cursor_position = &cursor;
    opt.multiline = false;
    opt.on_enter = [&entered]() { entered = true; };
    auto comp = tui::PromptInput(&content, "", opt);

    comp->OnEvent(Event::Character('x'));
    comp->OnEvent(Event::Return);

    REQUIRE(content == "x\n");
    REQUIRE(!entered);
}

TEST_CASE("PromptInput - return submits after paste safety window elapses",
          "[prompt_input][paste][newline]") {
    bool entered = false;
    std::string content = "hello";
    int cursor = static_cast<int>(content.size());

    InputOption opt;
    opt.content = &content;
    opt.cursor_position = &cursor;
    opt.multiline = false;
    opt.on_enter = [&entered]() { entered = true; };
    auto comp = tui::PromptInput(&content, "", opt);

    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    comp->OnEvent(Event::Return);

    REQUIRE(entered);
    REQUIRE(content == "hello");
}

// ── Multiline Ctrl+K / Ctrl+U ────────────────────────────────────────────────

TEST_CASE("PromptInput - Ctrl+K in multiline deletes to end of current line only",
          "[prompt_input][edit][multiline]") {
    std::string content = "line one\nline two";
    int cursor = 4;  // after "line" in first line

    auto comp = make_input(content, cursor, true);
    comp->OnEvent(Event::Special({11}));  // Ctrl+K

    REQUIRE(content == "line\nline two");
    REQUIRE(cursor == 4);
}

TEST_CASE("PromptInput - Ctrl+U in multiline deletes to start of current line only",
          "[prompt_input][edit][multiline]") {
    std::string content = "line one\nline two";
    // cursor at 14 = position of 't' in "two" (after "line ": 9 + 5 = 14)
    int cursor = 14;

    auto comp = make_input(content, cursor, true);
    comp->OnEvent(Event::Special({21}));  // Ctrl+U — deletes "line " leaving "two"

    REQUIRE(content == "line one\ntwo");
    REQUIRE(cursor == 9);
}

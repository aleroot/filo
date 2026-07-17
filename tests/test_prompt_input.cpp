#include <catch2/catch_test_macros.hpp>

#include "core/context/MentionPathUtils.hpp"
#include "core/utils/UriUtils.hpp"
#include "tui/PromptInput.hpp"

#include <ftxui/dom/elements.hpp>
#include <ftxui/component/component_options.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/screen/screen.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>

using namespace ftxui;
namespace fs = std::filesystem;

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

static void paste_text(ftxui::Component& comp, std::string_view text) {
    comp->OnEvent(Event::Special("\x1B[200~"));
    for (const char ch : text) {
        if (ch == '\n') {
            comp->OnEvent(Event::Return);
        } else {
            comp->OnEvent(Event::Character(ch));
        }
    }
    comp->OnEvent(Event::Special("\x1B[201~"));
}

static fs::path make_prompt_input_temp_file(std::string_view filename) {
    const auto dir = fs::temp_directory_path()
        / ("filo_prompt_input_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const auto file = dir / filename;
    std::ofstream out(file);
    out << "x";
    return file;
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

TEST_CASE("PromptInput - bracketed pasted file path becomes mention",
          "[prompt_input][paste][file]") {
    const auto file = make_prompt_input_temp_file("drag image.png");

    std::string content = "See ";
    int cursor = static_cast<int>(content.size());
    auto comp = make_input(content, cursor);

    paste_text(comp, core::context::escape_unquoted_mention_path(file.string()));

    REQUIRE(content == "See " + core::context::format_unquoted_mention(file.string()));
    REQUIRE(cursor == static_cast<int>(content.size()));

    fs::remove_all(file.parent_path());
}

TEST_CASE("PromptInput - bracketed pasted file URI becomes mention",
          "[prompt_input][paste][file]") {
    const auto file = make_prompt_input_temp_file("drag uri.png");

    std::string content;
    int cursor = 0;
    auto comp = make_input(content, cursor);

    paste_text(comp, "file://" + core::utils::uri::percent_encode_uri_path(file.string()));

    REQUIRE(content == core::context::format_unquoted_mention(file.string()));
    REQUIRE(cursor == static_cast<int>(content.size()));

    fs::remove_all(file.parent_path());
}

TEST_CASE("PromptInput - ordinary bracketed paste stays plain text",
          "[prompt_input][paste][file]") {
    std::string content;
    int cursor = 0;
    auto comp = make_input(content, cursor);

    paste_text(comp, "Look at /not/a/real/file.png please");

    REQUIRE(content == "Look at /not/a/real/file.png please");
    REQUIRE(cursor == static_cast<int>(content.size()));
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

// ── Large-document virtualization ────────────────────────────────────────────
// Regression guard for the O(document) per-frame render bug: feeding FTXUI's
// non-virtualizing `frame` every line of a huge paste made each keystroke scale
// with the whole document. These unit tests pin the correctness guarantee that
// the cursor line stays visible no matter how large the document. The
// timing-sensitive performance guard lives in the integration suite.

// Renders `comp` twice. The first call assigns `box_` (via reflect); the
// second call is the one that actually exercises the virtualized window built
// from that geometry — i.e. the steady-state production path on every frame
// after the first.
static std::string render_twice(ftxui::Component& comp, int width, int height) {
    std::string out;
    for (int i = 0; i < 2; ++i) {
        Screen screen(width, height);
        Render(screen, comp->Render());
        out = screen.ToString();
    }
    return out;
}

TEST_CASE("PromptInput - large document keeps cursor line visible",
          "[prompt_input][perf]") {
    // The cursor line must remain on screen regardless of document size and
    // cursor position, including on the steady-state (box_-populated) frame.
    constexpr int kLines = 2000;
    const std::string marker = "ZZZ_MARKER_ZZZ";

    for (int target_line : {0, kLines / 2, kLines - 1}) {
        std::string content;
        int cursor = 0;
        for (int i = 0; i < kLines; ++i) {
            if (i == target_line) {
                cursor = static_cast<int>(content.size());
                content += marker;
            } else {
                content += "padding line number " + std::to_string(i);
            }
            content += '\n';
        }

        auto comp = make_input(content, cursor, true);
        const std::string rendered = render_twice(comp, 90, 24);

        INFO("cursor line " << target_line << " of " << kLines);
        REQUIRE(rendered.find(marker) != std::string::npos);
    }
}

TEST_CASE("PromptInput - large document keeps cursor line visible in a small box",
          "[prompt_input][perf]") {
    // Mirror the production layout: the input shares the screen with a flex
    // history pane, so it only gets a few rows. The virtualized window must
    // still keep the cursor line on screen.
    constexpr int kLines = 2000;
    const std::string marker = "ZZZ_MARKER_ZZZ";
    std::string content;
    for (int i = 0; i < kLines; ++i) {
        if (i == kLines - 1) content += marker;
        else content += "padding line " + std::to_string(i);
        content += '\n';
    }
    int cursor = static_cast<int>(content.size());

    auto input = make_input(content, cursor, true);
    auto layout = Container::Vertical({
        Renderer([] { return text("") | flex; }),
        input,
    });
    // Production always focuses the input (MainApp calls TakeFocus); the frame
    // only centres on the cursor when the input owns focus.
    input->TakeFocus();
    auto renderer = Renderer(layout, [&] {
        return vbox({layout->Render() | flex, separator(), text("status")});
    });

    std::string rendered;
    for (int i = 0; i < 2; ++i) {
        Screen screen(90, 24);
        Render(screen, renderer->Render());
        rendered = screen.ToString();
    }
    REQUIRE(rendered.find(marker) != std::string::npos);
}

TEST_CASE("PromptInput - large paste never squeezes surrounding chrome off screen",
          "[prompt_input][layout]") {
    // Regression: the virtualized line window used to propagate its full
    // height as a minimum layout requirement.  After a multi-hundred-line
    // paste the prompt box would grow until it displaced the banner, the
    // history pane, and even the top window border.  The input must instead
    // cap its height and scroll internally.
    constexpr int kLines = 800;
    std::string content;
    for (int i = 0; i < kLines; ++i) {
        content += "pasted line " + std::to_string(i) + '\n';
    }
    int cursor = static_cast<int>(content.size());

    auto input = make_input(content, cursor, true);
    input->TakeFocus();

    const std::string banner_marker = "BANNER_MARKER";
    const std::string status_marker = "STATUS_MARKER";
    auto renderer = Renderer(input, [&] {
        return vbox({
            text(banner_marker),
            text("") | flex,   // history pane
            input->Render(),
            text(status_marker),
        });
    });

    std::string rendered;
    // Render several frames so box_ feedback (via reflect) reaches steady
    // state; the old bug only manifested once box_ began inflating the
    // virtualization window.
    for (int i = 0; i < 4; ++i) {
        Screen screen(90, 24);
        Render(screen, renderer->Render());
        rendered = screen.ToString();
    }

    REQUIRE(rendered.find(banner_marker) != std::string::npos);
    REQUIRE(rendered.find(status_marker) != std::string::npos);
    // Cursor (end of paste) must still be visible inside the capped viewport.
    REQUIRE(rendered.find("pasted line " + std::to_string(kLines - 1))
            != std::string::npos);
}

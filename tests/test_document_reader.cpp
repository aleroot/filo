#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/DocumentLink.hpp"
#include "tui/DocumentPane.hpp"
#include "tui/DocumentReader.hpp"
#include "tui/DocumentView.hpp"
#include "tui/MarkdownRenderer.hpp"

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <atomic>
#include <map>
#include <random>
#include <unistd.h>
#include <tuple>
#include <unordered_map>
#include <string>
#include <thread>

using Catch::Matchers::ContainsSubstring;
using tui::DocumentTarget;

namespace {

[[nodiscard]] std::string strip_sgr(const std::string& raw) {
    std::string out;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\x1b' && i + 1 < raw.size() && raw[i + 1] == ']') {
            // OSC 8 hyperlinks: skip to the string terminator.
            const auto end = raw.find("\x1b\\", i);
            if (end == std::string::npos) break;
            i = end + 1;
            continue;
        }
        if (raw[i] == '\x1b' && i + 1 < raw.size() && raw[i + 1] == '[') {
            i += 2;
            while (i < raw.size() && !(raw[i] >= '@' && raw[i] <= '~')) ++i;
            continue;
        }
        out.push_back(raw[i]);
    }
    return out;
}

[[nodiscard]] std::string draw(ftxui::Element element, int width, int height) {
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(width),
                                        ftxui::Dimension::Fixed(height));
    ftxui::Render(screen, element);
    return strip_sgr(screen.ToString());
}

/// A directory of its own, removed with everything in it.
class TempDir {
public:
    TempDir()
        : path_(std::filesystem::temp_directory_path()
                / std::format("filo-doc-{}-{}-{}", ::getpid(), std::random_device{}(),
                              counter_.fetch_add(1))) {
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }
    std::filesystem::path write(const std::string& name, const std::string& content) const {
        const auto file = path_ / name;
        std::ofstream(file, std::ios::binary) << content;
        return file;
    }

private:
    static inline std::atomic<int> counter_{0};
    std::filesystem::path path_;
};

[[nodiscard]] std::string numbered(int lines) {
    std::string content;
    for (int i = 1; i <= lines; ++i) content += std::format("Line {}\n", i);
    return content;
}

[[nodiscard]] ftxui::Event click(int x, int y, ftxui::Mouse::Motion motion = ftxui::Mouse::Pressed) {
    ftxui::Mouse mouse;
    mouse.button = ftxui::Mouse::Left;
    mouse.motion = motion;
    mouse.x = x;
    mouse.y = y;
    return ftxui::Event::Mouse("", mouse);
}

} // namespace

TEST_CASE("Local file links are read in every spelling models use", "[document][links]") {
    using tui::parse_document_link;
    CHECK(parse_document_link("/abs/app.py") == DocumentTarget{"/abs/app.py", std::nullopt});
    CHECK(parse_document_link("/abs/app.py:12") == DocumentTarget{"/abs/app.py", 12});
    CHECK(parse_document_link("/abs/app.py:12:3") == DocumentTarget{"/abs/app.py", 12});
    CHECK(parse_document_link("src/app.py#L12") == DocumentTarget{"src/app.py", 12});
    CHECK(parse_document_link("src/app.py#L12-L20") == DocumentTarget{"src/app.py", 12});
    CHECK(parse_document_link("src/app.py#L5C2") == DocumentTarget{"src/app.py", 5});
    CHECK(parse_document_link("file:///abs/a.md") == DocumentTarget{"/abs/a.md", std::nullopt});
    CHECK(parse_document_link("file://localhost/abs/a.md")
          == DocumentTarget{"/abs/a.md", std::nullopt});
    CHECK(parse_document_link("</abs/My Project/Report.md:3>")
          == DocumentTarget{"/abs/My Project/Report.md", 3});
    CHECK(parse_document_link("/abs/My%20Report.md") == DocumentTarget{"/abs/My Report.md", std::nullopt});
    CHECK(parse_document_link("README.md \"the readme\"") == DocumentTarget{"README.md", std::nullopt});
    CHECK(parse_document_link("~/notes.md") == DocumentTarget{"~/notes.md", std::nullopt});
    CHECK(parse_document_link("README.md:7") == DocumentTarget{"README.md", 7});

    // Not files.
    CHECK_FALSE(parse_document_link("https://example.com/a.md"));
    CHECK_FALSE(parse_document_link("mailto:someone@example.com"));
    CHECK_FALSE(parse_document_link("vscode://file/abs/a.md"));
    CHECK_FALSE(parse_document_link("#section"));
    CHECK_FALSE(parse_document_link(""));
}

TEST_CASE("A relative link resolves against its base, and ~ against home", "[document][links]") {
    const auto resolved = tui::resolve_document_target({"docs/../src/a.cpp", 4}, "/work");
    CHECK(resolved.path == std::filesystem::path("/work/src/a.cpp"));
    CHECK(resolved.line == 4);
    CHECK(tui::resolve_document_target({"/abs/b.md", {}}, "/work").path == "/abs/b.md");
}

TEST_CASE("A file link shows its label only; a web link keeps its URL", "[document][markdown]") {
    const auto output = draw(
        tui::render_markdown("See [bench.md](/tmp/x/bench.md:3) and [docs](https://example.com)."),
        100, 3);
    CHECK_THAT(output, ContainsSubstring("bench.md"));
    CHECK_FALSE(output.contains("/tmp/x/bench.md"));
    CHECK_THAT(output, ContainsSubstring("(https://example.com)"));
}

TEST_CASE("Markdown links record their boxes under their scope, wrapped ones in pieces",
          "[document][markdown]") {
    std::unordered_map<std::string, ftxui::Box> boxes;
    auto element = tui::render_markdown(
        "First [a long link label that wraps](/w/a.md) then [b](https://b.example).",
        tui::MarkdownRenderOptions{.links = {.hitboxes = &boxes, .scope = "msg-1"}});
    std::ignore = draw(element, 24, 6);

    std::map<std::size_t, std::size_t> pieces;
    for (const auto& [key, box] : boxes) {
        const auto ref = tui::parse_markdown_link_key(key);
        REQUIRE(ref);
        CHECK(ref->scope == "msg-1");
        CHECK(ref->href == (ref->ordinal == 0 ? "/w/a.md" : "https://b.example"));
        CHECK(box.x_max >= box.x_min);
        ++pieces[ref->ordinal];
    }
    REQUIRE(pieces.size() == 2);
    CHECK(pieces[0] > 1);  // the label wrapped over several rows
    CHECK_FALSE(tui::parse_markdown_link_key("turn-file-change:abc"));
}

TEST_CASE("A scroll window shows the rows it is given, from its offset", "[document][scroll]") {
    // The window keeps views into the content, which outlives the frame.
    const std::string content = numbered(40);
    tui::ScrollMetrics metrics;
    auto window = tui::render_document_content(content, tui::DocumentFormat::Markdown, {}, 10,
                                               &metrics);
    const auto output = draw(window, 30, 5);
    CHECK(metrics.content_rows >= 40);
    CHECK(metrics.viewport_rows == 5);
    CHECK(metrics.offset == 10);
    CHECK(metrics.fresh);
    CHECK_THAT(output, ContainsSubstring("Line 11"));
    CHECK_THAT(output, ContainsSubstring("Line 15"));
    CHECK_FALSE(output.contains("Line 16"));

    // Past the end it stops at the last page.
    auto past = tui::render_document_content(content, tui::DocumentFormat::Text, {}, 500,
                                             &metrics);
    CHECK_THAT(draw(past, 30, 5), ContainsSubstring("40 Line 40"));
    CHECK(metrics.offset == 35);

    // A line asked for is brought into view, a third of the way down.
    auto anchored = tui::render_document_content(content, tui::DocumentFormat::Text, {}, 0,
                                                 &metrics, 20);
    std::ignore = draw(anchored, 30, 9);
    CHECK(metrics.anchored);
    CHECK(metrics.offset == 19 - 3);
}

TEST_CASE("A Markdown table is laid out whole before it is scrolled", "[document][scroll]") {
    const std::string table = "| Names | Scan |\n|---|---|\n| 1,000 | 50.5 |\n| 10,000 | 450.6 |\n";
    tui::ScrollMetrics metrics;
    const std::string document = "# Title\n\n" + table;
    auto window = tui::render_document_content(document, tui::DocumentFormat::Markdown,
                                               {}, 3, &metrics);
    const auto output = draw(window, 40, 6);
    // The rows in view are the table's own, borders and all: no raw pipes.
    CHECK_FALSE(output.contains("|---|"));
    CHECK_THAT(output, ContainsSubstring("450.6"));
}

TEST_CASE("The scroll keys stay within the document", "[document][scroll]") {
    int offset = 0;
    CHECK(tui::handle_scroll_event(offset, ftxui::Event::PageDown, 10, 25));
    CHECK(offset == 10);
    CHECK(tui::handle_scroll_event(offset, ftxui::Event::End, 10, 25));
    CHECK(offset == 15);
    CHECK(tui::handle_scroll_event(offset, ftxui::Event::ArrowDown, 10, 25));
    CHECK(offset == 15);
    CHECK(tui::handle_scroll_event(offset, ftxui::Event::Character('k'), 10, 25));
    CHECK(offset == 14);
    CHECK(tui::handle_scroll_event(offset, ftxui::Event::Home, 10, 25));
    CHECK(offset == 0);
    CHECK_FALSE(tui::handle_scroll_event(offset, ftxui::Event::Character('z'), 10, 25));
}

TEST_CASE("Find matches lines ignoring case", "[document][find]") {
    CHECK(tui::find_document_lines("alpha\nBeta\nalphabet\n", "ALPHA") == std::vector<int>{1, 3});
    CHECK(tui::find_document_lines("alpha", "").empty());
    CHECK(tui::document_line_count("a\nb\n") == 2);
    CHECK(tui::document_line_count("a\nb") == 2);
    CHECK(tui::document_format("NOTES.MD") == tui::DocumentFormat::Markdown);
    CHECK(tui::document_format("main.cpp") == tui::DocumentFormat::Text);
}

TEST_CASE("The reader keeps one tab per file and reads a changed file again",
          "[document][reader]") {
    TempDir dir;
    const auto a = dir.write("a.md", "# A\n");
    const auto b = dir.write("b.txt", "one\ntwo\n");
    tui::DocumentReader reader;
    reader.open({a, {}});
    reader.open({b, 2});
    reader.open({a, {}});
    REQUIRE(reader.tabs().size() == 2);
    CHECK(reader.selected_index() == 0);
    CHECK(reader.tabs()[1].marked_line == 2);
    CHECK(reader.tabs()[0].format == tui::DocumentFormat::Markdown);

    // Same size, later time: still a change.
    reader.selected()->offset = 4;
    const auto generation = reader.selected()->generation;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    dir.write("a.md", "# B\n");
    std::filesystem::last_write_time(
        a, std::filesystem::last_write_time(a) + std::chrono::seconds(2));
    CHECK(reader.refresh(std::chrono::steady_clock::now() + std::chrono::hours(1)));
    CHECK(reader.selected()->content == "# B\n");
    CHECK(reader.selected()->generation > generation);
    CHECK(reader.selected()->offset == 4);  // the place is kept

    reader.close_selected();
    REQUIRE(reader.tabs().size() == 1);
    CHECK(reader.selected()->path == b);
}

TEST_CASE("The reader says why a file cannot be shown", "[document][reader]") {
    TempDir dir;
    tui::DocumentReader reader;
    reader.open({dir.path() / "missing.md", {}});
    CHECK_THAT(reader.selected()->problem, ContainsSubstring("does not exist"));
    reader.open({dir.write("blob.bin", std::string("PK\0\0data", 8)), {}});
    CHECK_THAT(reader.selected()->problem, ContainsSubstring("binary"));
    reader.open({dir.path(), {}});
    CHECK_THAT(reader.selected()->problem, ContainsSubstring("directory"));
}

TEST_CASE("The reader drops the oldest tab past its limit, never the one in front",
          "[document][reader]") {
    TempDir dir;
    tui::DocumentReader reader;
    for (std::size_t i = 0; i <= tui::DocumentReader::kMaxTabs; ++i) {
        reader.open({dir.write(std::format("{}.txt", i), "x"), {}});
    }
    CHECK(reader.tabs().size() == tui::DocumentReader::kMaxTabs);
    CHECK(reader.tabs().front().path.filename() == "1.txt");
    CHECK(reader.selected()->path.filename()
          == std::format("{}.txt", tui::DocumentReader::kMaxTabs));
}

TEST_CASE("Find marks the first match in view and steps through the rest", "[document][find]") {
    tui::DocumentTab tab{.content = "x\nneedle\ny\nneedle\nz\nneedle\n"};
    tui::DocumentFind find{.query = "needle"};
    REQUIRE(tui::find_in_document(find, tab, 3));
    CHECK(tab.marked_line == 4);
    tui::step_document_find(find, tab, 1);
    CHECK(tab.marked_line == 6);
    tui::step_document_find(find, tab, 1);
    CHECK(tab.marked_line == 2);
    tui::step_document_find(find, tab, -1);
    CHECK(tab.marked_line == 6);

    // The file was read again: the next step searches what it says now.
    tab.content = "needle\n";
    ++tab.generation;
    tui::step_document_find(find, tab, 1);
    CHECK(find.matches == std::vector<int>{1});
    CHECK(tab.marked_line == 1);
}

TEST_CASE("The pane follows a link to a file beside the document", "[document][pane]") {
    TempDir dir;
    dir.write("other.md", "# Other\n");
    const auto index = dir.write("index.md", "Read [the other one](other.md:1) next.\n");
    tui::DocumentPaneState pane;
    pane.show({index, {}});
    REQUIRE(pane.focused);

    // Draw a frame so the pane knows where its links are.
    std::ignore = draw(tui::render_document_pane(pane), 60, 12);
    CHECK(tui::handle_document_pane_event(pane, ftxui::Event::Tab).kind
          == tui::DocumentPaneAction::Kind::handled);
    CHECK(pane.reader.selected()->selected_link == 0);
    std::ignore = draw(tui::render_document_pane(pane), 60, 12);
    std::ignore = tui::handle_document_pane_event(pane, ftxui::Event::Return);
    REQUIRE(pane.reader.tabs().size() == 2);
    CHECK(pane.reader.selected()->path == dir.path() / "other.md");
    CHECK(pane.reader.selected()->marked_line == 1);
}

TEST_CASE("The pane hands web links to its host and owns keys only while focused",
          "[document][pane]") {
    TempDir dir;
    const auto doc = dir.write("doc.md", "See [site](https://example.com).\n");
    tui::DocumentPaneState pane;
    pane.show({doc, {}});
    const auto output = draw(tui::render_document_pane(pane), 70, 10);
    CHECK_THAT(output, ContainsSubstring("doc.md"));
    CHECK_THAT(output, ContainsSubstring("Esc back"));

    std::ignore = tui::handle_document_pane_event(pane, ftxui::Event::Tab);
    std::ignore = draw(tui::render_document_pane(pane), 60, 10);
    const auto action = tui::handle_document_pane_event(pane, ftxui::Event::Return);
    CHECK(action.kind == tui::DocumentPaneAction::Kind::open_url);
    CHECK(action.url == "https://example.com");

    // Esc lets go of the link first, then of the keyboard.
    std::ignore = tui::handle_document_pane_event(pane, ftxui::Event::Escape);
    CHECK(pane.focused);
    std::ignore = tui::handle_document_pane_event(pane, ftxui::Event::Escape);
    CHECK_FALSE(pane.focused);
    CHECK(tui::handle_document_pane_event(pane, ftxui::Event::Character('a')).kind
          == tui::DocumentPaneAction::Kind::ignored);
    CHECK(tui::handle_document_pane_event(pane, ftxui::Event::F6).kind
          == tui::DocumentPaneAction::Kind::handled);
    CHECK(pane.focused);

    // q closes the pane; the file stays for /open to bring back.
    std::ignore = tui::handle_document_pane_event(pane, ftxui::Event::Character('q'));
    CHECK_FALSE(pane.open);
    CHECK_FALSE(pane.reader.empty());
}

TEST_CASE("Clicking a tab's × closes it, and the last one closes the pane", "[document][pane]") {
    TempDir dir;
    tui::DocumentPaneState pane;
    pane.show({dir.write("a.txt", "a"), {}});
    pane.show({dir.write("b.txt", "b"), {}});
    std::ignore = draw(tui::render_document_pane(pane), 60, 8);
    REQUIRE(pane.close_boxes.size() == 2);
    const auto close_b = pane.close_boxes[1];
    std::ignore = tui::handle_document_pane_event(pane, click(close_b.x_min, close_b.y_min));
    REQUIRE(pane.reader.tabs().size() == 1);
    std::ignore = draw(tui::render_document_pane(pane), 60, 8);
    const auto close_a = pane.close_boxes[0];
    std::ignore = tui::handle_document_pane_event(pane, click(close_a.x_min, close_a.y_min));
    CHECK(pane.reader.empty());
    CHECK_FALSE(pane.open);
}

TEST_CASE("Ctrl+C reaches the app while the find prompt is open", "[document][pane]") {
    TempDir dir;
    tui::DocumentPaneState pane;
    pane.show({dir.write("f.txt", "text"), {}});
    std::ignore = tui::handle_document_pane_event(pane, ftxui::Event::Character('/'));
    REQUIRE(pane.find.editing);
    CHECK(tui::handle_document_pane_event(pane, ftxui::Event::Special("\x03")).kind
          == tui::DocumentPaneAction::Kind::ignored);
}

TEST_CASE("A file's C1 control characters never reach the terminal", "[document][reader]") {
    TempDir dir;
    tui::DocumentReader reader;
    reader.open({dir.write("c1.txt", "a\xC2\x9B" "31mb \xC3\xA9"), {}});
    CHECK(reader.selected()->content == "a31mb \xC3\xA9");
}

TEST_CASE("Keys that arrive before the next frame each move the pane", "[document][pane]") {
    TempDir dir;
    tui::DocumentPaneState pane;
    pane.show({dir.write("n.txt", numbered(100)), {}});
    std::ignore = draw(tui::render_document_pane(pane), 60, 14);
    std::ignore = tui::handle_document_pane_event(pane, ftxui::Event::ArrowDown);
    std::ignore = tui::handle_document_pane_event(pane, ftxui::Event::ArrowDown);
    CHECK(pane.reader.selected()->offset == 2);
}

TEST_CASE("The pane splits a wide terminal and takes a narrow one whole", "[document][pane]") {
    CHECK_FALSE(tui::document_pane_split_width(100));
    REQUIRE(tui::document_pane_split_width(160));
    CHECK(*tui::document_pane_split_width(160) == 72);
    CHECK(*tui::document_pane_split_width(400) == 110);
}

TEST_CASE("A file the agent shows never takes the keyboard or the user's tab",
          "[document][pane][agent]") {
    using Placement = tui::DocumentPaneState::Placement;
    TempDir dir;
    const auto plan = dir.write("plan.md", "# Plan\n");
    const auto notes = dir.write("notes.md", "# Notes\n");
    tui::DocumentPaneState pane;

    // A wide terminal: the pane opens beside the transcript, unfocused.
    CHECK(pane.show_for_agent({plan, 3}, /*can_split=*/true) == Placement::front);
    CHECK(pane.open);
    CHECK_FALSE(pane.focused);
    CHECK(pane.reader.selected()->path == plan);

    // While the user reads in the focused pane, a new file waits behind.
    pane.focused = true;
    CHECK(pane.show_for_agent({notes, {}}, true) == Placement::behind);
    CHECK(pane.reader.tabs().size() == 2);
    CHECK(pane.reader.selected()->path == plan);
    CHECK(pane.focused);

    // The file already in front is simply shown again.
    CHECK(pane.show_for_agent({plan, {}}, true) == Placement::front);
}

TEST_CASE("On a narrow terminal the agent's file waits for F6", "[document][pane][agent]") {
    using Placement = tui::DocumentPaneState::Placement;
    TempDir dir;
    const auto plan = dir.write("plan.md", "# Plan\n");
    tui::DocumentPaneState pane;

    CHECK(pane.show_for_agent({plan, {}}, /*can_split=*/false) == Placement::hidden);
    CHECK_FALSE(pane.open);
    CHECK_FALSE(pane.focused);
    // Hidden, the pane leaves every other key and click to the app.
    CHECK(tui::handle_document_pane_event(pane, ftxui::Event::Character('a')).kind
          == tui::DocumentPaneAction::Kind::ignored);
    CHECK(tui::handle_document_pane_event(pane, click(0, 0)).kind
          == tui::DocumentPaneAction::Kind::ignored);

    CHECK(tui::handle_document_pane_event(pane, ftxui::Event::F6).kind
          == tui::DocumentPaneAction::Kind::handled);
    CHECK(pane.open);
    CHECK(pane.focused);
    CHECK(pane.reader.selected()->path == plan);
}

TEST_CASE("A tab opened in the background leaves the selected one in front",
          "[document][reader]") {
    TempDir dir;
    tui::DocumentReader reader;
    reader.open({dir.write("a.txt", "a"), {}}, /*select=*/false);
    CHECK(reader.selected()->path == dir.path() / "a.txt");  // The first is always in front.
    reader.open({dir.write("b.txt", "b"), 1}, /*select=*/false);
    CHECK(reader.selected()->path == dir.path() / "a.txt");
    CHECK(reader.tabs().back().marked_line == 1);
}

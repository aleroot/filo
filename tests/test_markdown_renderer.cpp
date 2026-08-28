/**
 * @file test_markdown_renderer.cpp
 * @brief Unit tests for tui/MarkdownRenderer.
 *
 * Covers:
 *  - Inline parser: bold, italic, bold-italic, inline-code, strikethrough,
 *    links, backslash escapes, underscore word-boundary rules
 *  - Block parser: headings (H1-H6), fenced/indented code, unordered/ordered
 *    lists, blockquotes, tables, horizontal rules, paragraphs, blank lines
 *  - render_markdown(): smoke tests (no crash, non-null element)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/MarkdownRenderer.hpp"

#include <ftxui/screen/screen.hpp>

#include <string>

using namespace tui;

// ============================================================================
// Smoke helper: render must not throw and must return a non-null element
// ============================================================================

static void smoke(std::string_view md)
{
    REQUIRE_NOTHROW(render_markdown(md));
    auto el = render_markdown(md);
    REQUIRE(el != nullptr);
}

static std::string strip_ansi(std::string_view input)
{
    std::string out;
    out.reserve(input.size());

    for (std::size_t i = 0; i < input.size();) {
        if (input[i] == '\x1b' && i + 1 < input.size() && input[i + 1] == '[') {
            i += 2;
            while (i < input.size()) {
                const char ch = input[i++];
                if (ch >= '@' && ch <= '~') {
                    break;
                }
            }
            continue;
        }

        out.push_back(input[i]);
        ++i;
    }

    return out;
}

static std::string render_text(std::string_view md, int width)
{
    auto el = render_markdown(md);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(width),
                                        ftxui::Dimension::Fixed(8));
    ftxui::Render(screen, el);
    return strip_ansi(screen.ToString());
}

static std::string render_document(std::string_view md, int width, int height)
{
    auto el = render_markdown(md);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(width),
                                        ftxui::Dimension::Fixed(height));
    ftxui::Render(screen, el);
    return strip_ansi(screen.ToString());
}

// ============================================================================
// Empty / trivial input
// ============================================================================

TEST_CASE("render_markdown — empty string", "[md][smoke]")
{
    smoke("");
}

TEST_CASE("render_markdown — whitespace only", "[md][smoke]")
{
    smoke("   \n  \n");
}

TEST_CASE("render_markdown — plain text", "[md][smoke]")
{
    smoke("Hello, world!");
}

TEST_CASE("render_markdown — control bytes in streamed text", "[md][sanitize]")
{
    std::string noisy = "Chunk A";
    noisy.push_back('\x03');      // ETX
    noisy += " + Chunk B";
    noisy.push_back('\x1B');      // ESC
    noisy += "[31m";
    smoke(noisy);
}

TEST_CASE("render_markdown — malformed UTF-8 fragments", "[md][sanitize]")
{
    std::string malformed = "prefix ";
    malformed.push_back(static_cast<char>(0xC3));  // Truncated 2-byte lead.
    malformed += " middle ";
    malformed.push_back(static_cast<char>(0xF0));  // Truncated 4-byte lead.
    malformed += " suffix";
    smoke(malformed);
}

// ============================================================================
// Headings
// ============================================================================

TEST_CASE("render_markdown — H1", "[md][heading]")
{
    smoke("# Heading One");
}

TEST_CASE("render_markdown — H2", "[md][heading]")
{
    smoke("## Heading Two");
}

TEST_CASE("render_markdown — H3", "[md][heading]")
{
    smoke("### Heading Three");
}

TEST_CASE("render_markdown — H4-H6", "[md][heading]")
{
    smoke("#### H4");
    smoke("##### H5");
    smoke("###### H6");
}

TEST_CASE("render_markdown — heading with inline bold", "[md][heading]")
{
    smoke("## **Bold** heading");
}

TEST_CASE("render_markdown — hashes without space are not headings", "[md][heading]")
{
    smoke("#not-a-heading");
}

// ============================================================================
// Horizontal rule
// ============================================================================

TEST_CASE("render_markdown — horizontal rule dashes", "[md][hr]")
{
    smoke("---");
}

TEST_CASE("render_markdown — horizontal rule stars", "[md][hr]")
{
    smoke("***");
}

TEST_CASE("render_markdown — horizontal rule underscores", "[md][hr]")
{
    smoke("___");
}

// ============================================================================
// Fenced code blocks
// ============================================================================

TEST_CASE("render_markdown — fenced code no language", "[md][code]")
{
    smoke("```\nint x = 42;\n```");
}

TEST_CASE("render_markdown — fenced code with language", "[md][code]")
{
    smoke("```cpp\nvoid foo() {}\n```");
}

TEST_CASE("render_markdown — fenced code empty body", "[md][code]")
{
    smoke("```python\n```");
}

TEST_CASE("render_markdown — fenced code tilde fence", "[md][code]")
{
    smoke("~~~bash\necho hello\n~~~");
}

TEST_CASE("render_markdown — fenced code unclosed (eof)", "[md][code]")
{
    // Unclosed fence: renderer must not crash
    smoke("```cpp\nvoid foo();");
}

TEST_CASE("render_markdown — info string cannot close an enclosing fence",
          "[md][code][regression]")
{
    const auto output = render_document(
        "```markdown\n"
        "## Proposed changes\n\n"
        "```swift\n"
        "let value = 42\n"
        "```\n\n"
        "After the example.\n"
        "```",
        60,
        16);

    // The bug treated ```swift as the outer fence's closer and dropped it.
    // CommonMark requires a closer to have whitespace only after its markers.
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("```swift"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("let value = 42"));
}

TEST_CASE("render_markdown — closing fence follows CommonMark shape",
          "[md][code][commonmark]")
{
    const auto output = render_document(
        "  ````cpp\n"
        "  first\n"
        "``` trailing text\n"
        "```\n"
        "   ````\t\n"
        "after",
        60,
        12);

    // A shorter run and a run with non-whitespace trailing content are code.
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("``` trailing text"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("```"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("after"));
}

TEST_CASE("render_markdown — invalid backtick info string stays literal",
          "[md][code][commonmark]")
{
    auto element = render_markdown(
        "```lang`invalid\n"
        "plain paragraph");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(60),
                                        ftxui::Dimension::Fixed(6));
    ftxui::Render(screen, element);
    const auto output = strip_ansi(screen.ToString());

    // A valid fenced block would render a window border in the first cell.
    // This invalid opener remains ordinary inline text instead.
    REQUIRE(screen.CellAt(0, 0).character == "`");
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("plain paragraph"));
}

// ============================================================================
// Indented code block
// ============================================================================

TEST_CASE("render_markdown — indented code block", "[md][code]")
{
    smoke("    int x = 0;\n    return x;");
}

// ============================================================================
// Inline code
// ============================================================================

TEST_CASE("render_markdown — inline code single backtick", "[md][inline]")
{
    smoke("Call `foo()` now.");
}

TEST_CASE("render_markdown — inline code double backtick", "[md][inline]")
{
    smoke("Use ``a `b` c`` here.");
}

TEST_CASE("render_markdown — inline code unclosed backtick is literal", "[md][inline]")
{
    smoke("Text with ` alone.");
}

// ============================================================================
// Inline bold / italic
// ============================================================================

TEST_CASE("render_markdown — bold double star", "[md][inline]")
{
    auto el = render_markdown("This is **bold** text.");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(40),
                                        ftxui::Dimension::Fixed(2));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("This is bold text."));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("**"));
    REQUIRE(screen.CellAt(8, 0).bold);   // 'b' of bold
    REQUIRE_FALSE(screen.CellAt(7, 0).bold); // space before bold
}

TEST_CASE("render_markdown — italic single star", "[md][inline]")
{
    auto el = render_markdown("This is *italic* text.");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(40),
                                        ftxui::Dimension::Fixed(2));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("This is italic text."));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("*"));
    REQUIRE(screen.CellAt(8, 0).italic);
}

TEST_CASE("render_markdown — bold-italic triple star", "[md][inline]")
{
    auto el = render_markdown("***both***");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(20),
                                        ftxui::Dimension::Fixed(2));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("both"));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("*"));
    REQUIRE(screen.CellAt(0, 0).bold);
    REQUIRE(screen.CellAt(0, 0).italic);
}

TEST_CASE("render_markdown — bold double underscore", "[md][inline]")
{
    auto el = render_markdown("This is __bold__ text.");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(40),
                                        ftxui::Dimension::Fixed(2));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("This is bold text."));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("__"));
    REQUIRE(screen.CellAt(8, 0).bold);
}

TEST_CASE("render_markdown — italic single underscore", "[md][inline]")
{
    auto el = render_markdown("This is _italic_ text.");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(40),
                                        ftxui::Dimension::Fixed(2));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("This is italic text."));
    REQUIRE(screen.CellAt(8, 0).italic);
}

TEST_CASE("render_markdown — underscore inside word is literal", "[md][inline]")
{
    // foo_bar_baz should not parse underscores as italic markers
    const auto output = render_text("foo_bar_baz", 20);
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("foo_bar_baz"));
}

TEST_CASE("render_markdown — unclosed bold is literal", "[md][inline]")
{
    const auto output = render_text("**not closed", 20);
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("**not closed"));
}

TEST_CASE("render_markdown — multiple bold spans", "[md][inline]")
{
    auto el = render_markdown("**a** and **b**");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(20),
                                        ftxui::Dimension::Fixed(2));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("a and b"));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("**"));
    REQUIRE(screen.CellAt(0, 0).bold);
    REQUIRE(screen.CellAt(6, 0).bold); // 'b'
}

TEST_CASE("render_markdown — bold and italic mixed", "[md][inline]")
{
    const auto output = render_text("**bold** and *italic* and `code`", 40);
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("bold"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("italic"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("code"));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("**"));
}

TEST_CASE("render_markdown — nested bold inside italic keeps no markers", "[md][inline][nested]")
{
    auto el = render_markdown("*italic and **bold** mix*");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(40),
                                        ftxui::Dimension::Fixed(2));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("italic and bold mix"));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("*"));
    REQUIRE(screen.CellAt(0, 0).italic);
    REQUIRE(screen.CellAt(11, 0).bold);   // 'b' of bold
    REQUIRE(screen.CellAt(11, 0).italic);
    REQUIRE_FALSE(screen.CellAt(16, 0).bold); // 'm' of outer italic text
    REQUIRE(screen.CellAt(16, 0).italic);
}

TEST_CASE("render_markdown — nested italic inside bold keeps no markers", "[md][inline][nested]")
{
    auto el = render_markdown("**bold and *italic* mix**");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(40),
                                        ftxui::Dimension::Fixed(2));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("bold and italic mix"));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("*"));
    REQUIRE(screen.CellAt(0, 0).bold);
    REQUIRE(screen.CellAt(9, 0).italic); // 'i' of italic after "bold and "
    REQUIRE(screen.CellAt(9, 0).bold);
}

TEST_CASE("render_markdown — bold inside link label", "[md][inline][nested]")
{
    auto el = render_markdown("[**bold link**](https://example.com)");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(50),
                                        ftxui::Dimension::Fixed(2));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("bold link"));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("**"));
    REQUIRE(screen.CellAt(0, 0).bold);
    REQUIRE(screen.CellAt(0, 0).underlined);
}

TEST_CASE("render_markdown — code inside link label stays visibly linked", "[md][inline][nested]")
{
    auto el = render_markdown("[`code`](https://example.com)");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(50),
                                        ftxui::Dimension::Fixed(2));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("code"));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("`"));
    REQUIRE(screen.CellAt(1, 0).underlined); // code tokens have one-cell padding
}

TEST_CASE("render_markdown — bold spans soft-broken paragraph lines", "[md][inline][nested]")
{
    auto el = render_markdown("start **bold\nacross** end");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(40),
                                        ftxui::Dimension::Fixed(4));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("start bold"));
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("across end"));
    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("**"));
    REQUIRE(screen.CellAt(6, 0).bold); // 'b' of bold on first visual row
    REQUIRE(screen.CellAt(0, 1).bold); // 'a' of across on second row
}

TEST_CASE("render_markdown — bold spans blockquote soft lines", "[md][inline][nested]")
{
    auto el = render_markdown("> **bold\n> across**");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(40),
                                        ftxui::Dimension::Fixed(4));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("**"));
    REQUIRE(screen.CellAt(2, 0).bold);
    REQUIRE(screen.CellAt(2, 1).bold);
}

TEST_CASE("render_markdown — bold spans unordered-list continuation", "[md][inline][nested]")
{
    auto el = render_markdown("- **bold\n  across**");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(40),
                                        ftxui::Dimension::Fixed(4));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("**"));
    REQUIRE(screen.CellAt(2, 0).bold);
    REQUIRE(screen.CellAt(2, 1).bold);
}

TEST_CASE("render_markdown — bold spans ordered-list continuation", "[md][inline][nested]")
{
    auto el = render_markdown("1. **bold\n   across**");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(40),
                                        ftxui::Dimension::Fixed(4));
    ftxui::Render(screen, el);
    const auto output = strip_ansi(screen.ToString());

    REQUIRE_THAT(output, !Catch::Matchers::ContainsSubstring("**"));
    REQUIRE(screen.CellAt(3, 0).bold);
    REQUIRE(screen.CellAt(3, 1).bold);
}

// ============================================================================
// Strikethrough
// ============================================================================

TEST_CASE("render_markdown — strikethrough", "[md][inline]")
{
    smoke("~~deleted~~");
}

TEST_CASE("render_markdown — strikethrough unclosed is literal", "[md][inline]")
{
    smoke("~~ not closed");
}

// ============================================================================
// Links
// ============================================================================

TEST_CASE("render_markdown — link basic", "[md][inline]")
{
    smoke("[OpenAI](https://openai.com)");
}

TEST_CASE("render_markdown — link with styled text", "[md][inline]")
{
    smoke("[**bold link**](https://example.com)");
}

TEST_CASE("render_markdown — unclosed bracket is literal", "[md][inline]")
{
    smoke("[not a link");
}

TEST_CASE("render_markdown — link no paren url is literal", "[md][inline]")
{
    smoke("[text] no parens");
}

// ============================================================================
// Backslash escapes
// ============================================================================

TEST_CASE("render_markdown — backslash escapes star", "[md][inline]")
{
    smoke("\\*not italic\\*");
}

TEST_CASE("render_markdown — backslash escapes backtick", "[md][inline]")
{
    smoke("\\`not code\\`");
}

// ============================================================================
// Ordered lists
// ============================================================================

TEST_CASE("render_markdown — ordered list", "[md][list]")
{
    smoke("1. first\n2. second\n3. third");
}

TEST_CASE("render_markdown — ordered list with inline", "[md][list]")
{
    smoke("1. **Step one**\n2. Run `make`\n3. Done");
}

TEST_CASE("render_markdown — ordered list loose (blank lines)", "[md][list]")
{
    // Items separated by blank lines should stay in the same list and increment
    smoke("1. item one\n\n2. item two\n\n3. item three");
}

TEST_CASE("render_markdown — ordered list with continuation lines", "[md][list]")
{
    smoke("1. item one\n   more text for one\n2. item two");
}

TEST_CASE("render_markdown — ordered list restarts after non-list block", "[md][list]")
{
    smoke("1. item one\n\nParagraph\n\n1. new list");
}

// ============================================================================
// Unordered lists
// ============================================================================

TEST_CASE("render_markdown — unordered list dash", "[md][list]")
{
    smoke("- item one\n- item two\n- item three");
}

TEST_CASE("render_markdown — unordered list star", "[md][list]")
{
    smoke("* alpha\n* beta");
}

TEST_CASE("render_markdown — unordered list plus", "[md][list]")
{
    smoke("+ a\n+ b");
}

TEST_CASE("render_markdown — unordered list nested indent", "[md][list]")
{
    smoke("- parent\n  - child\n    - grandchild");
}

TEST_CASE("render_markdown — unordered list loose", "[md][list]")
{
    smoke("- item one\n\n- item two");
}

TEST_CASE("render_markdown — loose continuation does not gain a bullet", "[md][list]")
{
    auto el = render_markdown("- item\n\n  continuation");
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(30),
                                        ftxui::Dimension::Fixed(4));
    ftxui::Render(screen, el);

    REQUIRE(screen.CellAt(2, 2).character == "c");
}

TEST_CASE("render_markdown — unordered list with continuation", "[md][list]")
{
    smoke("- item one\n  continuation line\n- item two");
}

TEST_CASE("render_markdown — list items with inline markup", "[md][list]")
{
    smoke("- **bold item**\n- *italic item*\n- `code item`");
}

// ============================================================================
// Blockquotes
// ============================================================================

TEST_CASE("render_markdown — blockquote single line", "[md][blockquote]")
{
    smoke("> This is quoted.");
}

TEST_CASE("render_markdown — blockquote multi line", "[md][blockquote]")
{
    smoke("> Line one\n> Line two");
}

TEST_CASE("render_markdown — blockquote with inline", "[md][blockquote]")
{
    smoke("> **Note:** this is *important*.");
}

// ============================================================================
// Tables
// ============================================================================

TEST_CASE("render_markdown — simple table", "[md][table]")
{
    smoke("| A | B |\n|---|---|\n| 1 | 2 |");
}

TEST_CASE("render_markdown — table no trailing pipe", "[md][table]")
{
    smoke("| Col1 | Col2\n|------|------\n| val1 | val2");
}

TEST_CASE("render_markdown — table with inline in cells", "[md][table]")
{
    smoke("| **Header** | `code` |\n|---|---|\n| *italic* | normal |");
}

TEST_CASE("render_markdown — table single column", "[md][table]")
{
    smoke("| Solo |\n|------|\n| row1 |\n| row2 |");
}

// ============================================================================
// Multi-block documents
// ============================================================================

TEST_CASE("render_markdown — heading then paragraph", "[md][multi]")
{
    smoke("# Title\n\nSome text here.");
}

TEST_CASE("render_markdown — code block then list", "[md][multi]")
{
    smoke("```bash\nls -la\n```\n\n- file1\n- file2");
}

TEST_CASE("render_markdown — full document", "[md][multi]")
{
    smoke(R"md(
# Project Overview

This is a **C++26** project using [FTXUI](https://github.com/ArthurSonzogni/FTXUI).

## Features

- Fast rendering
- *Markdown* support
- `inline code`

## Usage

```cpp
auto el = render_markdown("# Hello");
```

> **Note:** requires a terminal that supports true colour.

### Table of contents

| Section | Description |
|---------|-------------|
| Install | How to build |
| API     | Public interface |

---

1. Clone the repo
2. Run `cmake -B build`
3. `cmake --build build`
)md");
}

TEST_CASE("render_markdown — streaming partial response (no trailing newline)", "[md][multi]")
{
    smoke("Here is a partial **bold");
}

TEST_CASE("render_markdown — blank lines between blocks", "[md][multi]")
{
    smoke("Para one.\n\nPara two.\n\nPara three.");
}

TEST_CASE("render_markdown — mixed inline spans wrap instead of clipping", "[md][wrap]")
{
    const auto output = render_text(
        "This sentence has **bold text** and must keep tailword visible.",
        32);
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("tailword"));
}

TEST_CASE("render_markdown — link lines wrap instead of clipping", "[md][wrap]")
{
    const auto output = render_text(
        "Prefix [link text](https://example.com/path) suffixword",
        28);
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("suffixword"));
}

TEST_CASE("render_markdown — heading lines wrap instead of clipping", "[md][wrap]")
{
    const auto output = render_text(
        "## Heading with **bold** marker and headingtail",
        28);
    REQUIRE_THAT(output, Catch::Matchers::ContainsSubstring("headingtail"));
}

// ============================================================================
// Custom base colour
// ============================================================================

TEST_CASE("render_markdown — custom base colour", "[md][colour]")
{
    REQUIRE_NOTHROW(render_markdown("Hello **world**", ftxui::Color::GrayLight));
}

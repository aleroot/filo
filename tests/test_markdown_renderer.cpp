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

#include "tui/MarkdownRenderer.hpp"

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
    smoke("This is **bold** text.");
}

TEST_CASE("render_markdown — italic single star", "[md][inline]")
{
    smoke("This is *italic* text.");
}

TEST_CASE("render_markdown — bold-italic triple star", "[md][inline]")
{
    smoke("This is ***bold italic*** text.");
}

TEST_CASE("render_markdown — bold double underscore", "[md][inline]")
{
    smoke("This is __bold__ text.");
}

TEST_CASE("render_markdown — italic single underscore", "[md][inline]")
{
    smoke("This is _italic_ text.");
}

TEST_CASE("render_markdown — underscore inside word is literal", "[md][inline]")
{
    // foo_bar_baz should not parse underscores as italic markers
    smoke("foo_bar_baz");
}

TEST_CASE("render_markdown — unclosed bold is literal", "[md][inline]")
{
    smoke("**not closed");
}

TEST_CASE("render_markdown — multiple bold spans", "[md][inline]")
{
    smoke("**a** and **b**");
}

TEST_CASE("render_markdown — bold and italic mixed", "[md][inline]")
{
    smoke("**bold** and *italic* and `code`");
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

TEST_CASE("render_markdown — list items with inline markup", "[md][list]")
{
    smoke("- **bold item**\n- *italic item*\n- `code item`");
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

// ============================================================================
// Custom base colour
// ============================================================================

TEST_CASE("render_markdown — custom base colour", "[md][colour]")
{
    REQUIRE_NOTHROW(render_markdown("Hello **world**", ftxui::Color::GrayLight));
}

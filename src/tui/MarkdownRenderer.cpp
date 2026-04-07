#include "MarkdownRenderer.hpp"
#include "TuiTheme.hpp"

#include <ftxui/dom/elements.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <vector>

using namespace ftxui;

namespace tui {
namespace {

// ============================================================================
// Colours used only by the markdown renderer
// ============================================================================

inline const Color kMdCodeFg    = Color::RGB(255, 210, 100);
inline const Color kMdCodeBg    = Color::RGB(32,  32,  40);
inline const Color kMdCodeBorder= Color::RGB(72,  72,  88);
inline const Color kMdLinkFg    = Color::RGB(100, 200, 255);
inline const Color kMdQuoteFg   = Color::GrayLight;
inline const Color kMdQuoteBar  = Color::RGB(90, 90, 110);
inline const Color kMdH1Fg      = Color::White;
inline const Color kMdH3Fg      = Color::RGB(255, 246, 61);   // ColorYellowBright
inline const Color kMdH4Fg      = Color::RGB(255, 192, 32);   // ColorYellowDark
inline const Color kMdTableHdrFg= Color::White;
inline const Color kMdTableRowFg= Color::GrayLight;
inline const Color kMdBulletFg  = Color::RGB(255, 192, 32);   // ColorYellowDark
inline const Color kMdNumFg     = Color::RGB(255, 192, 32);

// ============================================================================
// Text sanitization for FTXUI
// ============================================================================

static bool is_utf8_continuation(unsigned char c) {
    return (c & 0b1100'0000u) == 0b1000'0000u;
}

static std::size_t utf8_sequence_length(unsigned char lead) {
    if ((lead & 0b1110'0000u) == 0b1100'0000u) return 2;
    if ((lead & 0b1111'0000u) == 0b1110'0000u) return 3;
    if ((lead & 0b1111'1000u) == 0b1111'0000u) return 4;
    return 0;
}

static bool is_valid_utf8_sequence(std::string_view input,
                                   std::size_t pos,
                                   std::size_t len) {
    if (pos + len > input.size()) return false;

    const unsigned char b0 = static_cast<unsigned char>(input[pos]);
    uint32_t cp = 0;

    if (len == 2) {
        const unsigned char b1 = static_cast<unsigned char>(input[pos + 1]);
        if (!is_utf8_continuation(b1)) return false;
        cp = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
        return cp >= 0x80u;
    }

    if (len == 3) {
        const unsigned char b1 = static_cast<unsigned char>(input[pos + 1]);
        const unsigned char b2 = static_cast<unsigned char>(input[pos + 2]);
        if (!is_utf8_continuation(b1) || !is_utf8_continuation(b2)) return false;
        cp = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
        if (cp < 0x800u) return false;                       // overlong
        if (cp >= 0xD800u && cp <= 0xDFFFu) return false;    // surrogate half
        return true;
    }

    if (len == 4) {
        const unsigned char b1 = static_cast<unsigned char>(input[pos + 1]);
        const unsigned char b2 = static_cast<unsigned char>(input[pos + 2]);
        const unsigned char b3 = static_cast<unsigned char>(input[pos + 3]);
        if (!is_utf8_continuation(b1)
            || !is_utf8_continuation(b2)
            || !is_utf8_continuation(b3)) return false;
        cp = ((b0 & 0x07u) << 18)
           | ((b1 & 0x3Fu) << 12)
           | ((b2 & 0x3Fu) << 6)
           | (b3 & 0x3Fu);
        if (cp < 0x10000u) return false;                     // overlong
        return cp <= 0x10FFFFu;
    }

    return false;
}

static std::string sanitize_for_ftxui(std::string_view input) {
    std::string out;
    out.reserve(input.size());

    for (std::size_t i = 0; i < input.size();) {
        const unsigned char c = static_cast<unsigned char>(input[i]);

        // Keep visible ASCII + '\n' + '\t'. Drop other ASCII control bytes.
        if (c < 0x80u) {
            if (c == '\r') {
                // Normalize CRLF/CR to '\n' to avoid cursor-jump artifacts.
                if (i + 1 < input.size() && input[i + 1] == '\n') ++i;
                out.push_back('\n');
                ++i;
                continue;
            }
            if (c == '\n' || c == '\t' || (c >= 0x20u && c != 0x7Fu)) {
                out.push_back(static_cast<char>(c));
            }
            ++i;
            continue;
        }

        const std::size_t len = utf8_sequence_length(c);
        if (len == 0 || !is_valid_utf8_sequence(input, i, len)) {
            ++i;   // drop malformed byte
            continue;
        }

        out.append(input.substr(i, len));
        i += len;
    }

    return out;
}

// ============================================================================
// Inline span types
// ============================================================================

enum class SpanKind { Normal, Bold, Italic, BoldItalic, Code, Strikethrough, Link };

struct TextSpan {
    SpanKind    kind = SpanKind::Normal;
    std::string text;
    std::string href; // Link only
};

// Find the next exact run of `delim_char` repeated `delim_len` times,
// where the run is not immediately preceded or followed by `delim_char`.
static std::size_t find_closing_delim(std::string_view sv,
                                      char delim_char,
                                      int  delim_len,
                                      std::size_t from)
{
    const std::size_t n = sv.size();
    while (from + static_cast<std::size_t>(delim_len) <= n) {
        bool match = true;
        for (int k = 0; k < delim_len; ++k) {
            if (sv[from + k] != delim_char) { match = false; break; }
        }
        if (match) {
            const bool not_preceded = (from == 0 || sv[from - 1] != delim_char);
            const std::size_t after = from + static_cast<std::size_t>(delim_len);
            const bool not_followed = (after >= n || sv[after] != delim_char);
            if (not_preceded && not_followed) return from;
        }
        ++from;
    }
    return std::string_view::npos;
}

// Find closing backtick run of exactly `tick_len` backticks.
static std::size_t find_closing_backticks(std::string_view sv,
                                          std::size_t tick_len,
                                          std::size_t from)
{
    const std::size_t n = sv.size();
    while (from + tick_len <= n) {
        if (sv[from] == '`') {
            std::size_t run = 0;
            while (from + run < n && sv[from + run] == '`') ++run;
            if (run == tick_len) return from;
            from += run;
        } else {
            ++from;
        }
    }
    return std::string_view::npos;
}

// ============================================================================
// Inline parser
// ============================================================================

static std::vector<TextSpan> parse_inline(std::string_view text)
{
    std::vector<TextSpan> spans;
    std::string current;
    std::size_t i = 0;
    const std::size_t n = text.size();

    auto flush = [&]() {
        if (!current.empty()) {
            spans.push_back({.kind = SpanKind::Normal, .text = std::move(current), .href = {}});
            current.clear();
        }
    };

    while (i < n) {
        const char c = text[i];

        // ── Backslash escape ─────────────────────────────────────────────────
        if (c == '\\' && i + 1 < n) {
            current += text[i + 1];
            i += 2;
            continue;
        }

        // ── Inline code (backtick run) ────────────────────────────────────────
        if (c == '`') {
            std::size_t tick_len = 0;
            while (i + tick_len < n && text[i + tick_len] == '`') ++tick_len;
            std::size_t j = find_closing_backticks(text, tick_len, i + tick_len);
            if (j != std::string_view::npos) {
                flush();
                spans.push_back({.kind = SpanKind::Code,
                    .text = std::string(text.substr(i + tick_len, j - i - tick_len)),
                    .href = {}});
                i = j + tick_len;
                continue;
            }
            current += c; ++i;
            continue;
        }

        // ── Link [text](url) ─────────────────────────────────────────────────
        if (c == '[') {
            std::size_t bracket_end = text.find(']', i + 1);
            if (bracket_end != std::string_view::npos &&
                bracket_end + 1 < n && text[bracket_end + 1] == '(') {
                std::size_t paren_end = text.find(')', bracket_end + 2);
                if (paren_end != std::string_view::npos) {
                    flush();
                    TextSpan s;
                    s.kind = SpanKind::Link;
                    s.text = std::string(text.substr(i + 1, bracket_end - i - 1));
                    s.href = std::string(text.substr(bracket_end + 2,
                                                      paren_end - bracket_end - 2));
                    spans.push_back(std::move(s));
                    i = paren_end + 1;
                    continue;
                }
            }
            current += c; ++i;
            continue;
        }

        // ── Strikethrough ~~text~~ ───────────────────────────────────────────
        if (c == '~' && i + 1 < n && text[i + 1] == '~') {
            std::size_t j = find_closing_delim(text, '~', 2, i + 2);
            if (j != std::string_view::npos) {
                flush();
                spans.push_back({.kind = SpanKind::Strikethrough,
                    .text = std::string(text.substr(i + 2, j - i - 2)),
                    .href = {}});
                i = j + 2;
                continue;
            }
            current += c; ++i;
            continue;
        }

        // ── Bold/italic (* and _) ────────────────────────────────────────────
        if (c == '*' || c == '_') {
            // Count the run
            std::size_t run = 0;
            while (i + run < n && text[i + run] == c) ++run;

            // For '_', only trigger at word boundary (not inside words)
            bool boundary_ok = true;
            if (c == '_') {
                const bool prev_alnum = (i > 0) &&
                    std::isalnum(static_cast<unsigned char>(text[i - 1]));
                boundary_ok = !prev_alnum;
            }

            bool handled = false;
            if (boundary_ok) {
                // Try longest match first: ***  **  *
                for (int try_len : {3, 2, 1}) {
                    if (static_cast<std::size_t>(try_len) > run) continue;
                    std::size_t j = find_closing_delim(text, c, try_len,
                                                        i + static_cast<std::size_t>(try_len));
                    if (j == std::string_view::npos) continue;

                    // For '_', also require end boundary
                    if (c == '_') {
                        const std::size_t after = j + static_cast<std::size_t>(try_len);
                        const bool next_alnum = (after < n) &&
                            std::isalnum(static_cast<unsigned char>(text[after]));
                        if (next_alnum) continue;
                    }

                    flush();
                    const std::string content(text.substr(
                        i + static_cast<std::size_t>(try_len),
                        j - i - static_cast<std::size_t>(try_len)));
                    SpanKind kind = try_len == 3 ? SpanKind::BoldItalic
                                  : try_len == 2 ? SpanKind::Bold
                                                 : SpanKind::Italic;
                    spans.push_back({.kind = kind, .text = content, .href = {}});
                    i = j + static_cast<std::size_t>(try_len);
                    handled = true;
                    break;
                }
            }

            if (!handled) {
                for (std::size_t k = 0; k < run; ++k) current += c;
                i += run;
            }
            continue;
        }

        current += c;
        ++i;
    }

    if (!current.empty()) {
        spans.push_back({.kind = SpanKind::Normal, .text = std::move(current), .href = {}});
    }
    return spans;
}

// ============================================================================
// Span → FTXUI element
// ============================================================================

static Element render_span(const TextSpan& span, Color base)
{
    switch (span.kind) {
        case SpanKind::Normal:
            return ftxui::text(span.text) | ftxui::color(base);
        case SpanKind::Bold:
            return ftxui::text(span.text) | ftxui::color(base) | ftxui::bold;
        case SpanKind::Italic:
            return ftxui::text(span.text) | ftxui::color(base) | italic;
        case SpanKind::BoldItalic:
            return ftxui::text(span.text) | ftxui::color(base) | ftxui::bold | italic;
        case SpanKind::Strikethrough:
            return ftxui::text(span.text) | ftxui::color(base) | strikethrough;
        case SpanKind::Code:
            return hbox({
                ftxui::text(" " + span.text + " ")
                    | ftxui::color(kMdCodeFg)
                    | ftxui::bgcolor(kMdCodeBg),
            });
        case SpanKind::Link:
            return hbox({
                ftxui::text(span.text)
                    | ftxui::color(kMdLinkFg)
                    | underlined,
                ftxui::text(" (" + span.href + ")")
                    | ftxui::color(Color::GrayDark)
                    | dim,
            });
    }
    return ftxui::text(span.text) | ftxui::color(base); // unreachable
}

// Render one line of inline markdown.
// Plain text → paragraph() which wraps.
// Styled / mixed → hbox() (no wrap, preserves decoration).
static Element render_inline_line(std::string_view line, Color base)
{
    if (line.empty()) return ftxui::text("");

    auto spans = parse_inline(line);
    if (spans.empty()) return ftxui::text("") | ftxui::color(base);

    // Single span: use paragraph() for wrapping where possible
    if (spans.size() == 1) {
        const auto& s = spans[0];
        switch (s.kind) {
            case SpanKind::Normal:
                return paragraph(s.text) | ftxui::color(base) | xflex;
            case SpanKind::Bold:
                return paragraph(s.text) | ftxui::color(base) | ftxui::bold | xflex;
            case SpanKind::Italic:
                return paragraph(s.text) | ftxui::color(base) | italic | xflex;
            case SpanKind::BoldItalic:
                return paragraph(s.text) | ftxui::color(base) | ftxui::bold | italic | xflex;
            case SpanKind::Strikethrough:
                return paragraph(s.text) | ftxui::color(base) | strikethrough | xflex;
            default: break;
        }
    }

    // Multi-span or special span types: hbox
    Elements elems;
    elems.reserve(spans.size());
    for (const auto& s : spans) {
        elems.push_back(render_span(s, base));
    }
    return hbox(std::move(elems)) | xflex;
}

// ============================================================================
// Block types & parser
// ============================================================================

enum class BlockKind {
    Paragraph,
    Heading,
    FencedCode,
    IndentedCode,
    UnorderedList,
    OrderedList,
    Blockquote,
    Table,
    HorizontalRule,
    BlankLine,
};

struct Block {
    BlockKind            kind     = BlockKind::Paragraph;
    int                  level    = 1;   // heading level 1-6
    std::string          language;       // fenced code language tag
    std::vector<std::string> lines;
};

// ── Line classifiers ─────────────────────────────────────────────────────────

static bool is_blank(std::string_view s)
{
    for (char c : s) if (!std::isspace(static_cast<unsigned char>(c))) return false;
    return true;
}

static int heading_level(std::string_view s)
{
    int lv = 0;
    while (lv < static_cast<int>(s.size()) && s[lv] == '#') ++lv;
    if (lv >= 1 && lv <= 6 &&
        lv < static_cast<int>(s.size()) && s[lv] == ' ')
        return lv;
    return 0;
}

static bool is_fenced_code_start(std::string_view s)
{
    return s.size() >= 3 &&
           ((s[0] == '`' && s[1] == '`' && s[2] == '`') ||
            (s[0] == '~' && s[1] == '~' && s[2] == '~'));
}

static bool is_indented_code(std::string_view s)
{
    return s.size() >= 4 &&
           s[0] == ' ' && s[1] == ' ' && s[2] == ' ' && s[3] == ' ';
}

static bool is_hr(std::string_view s)
{
    int dashes = 0, stars = 0, underscores = 0;
    for (char c : s) {
        if      (c == '-') ++dashes;
        else if (c == '*') ++stars;
        else if (c == '_') ++underscores;
        else if (c != ' ') return false;
    }
    return (dashes >= 3 || stars >= 3 || underscores >= 3) &&
           (dashes == 0 || stars == 0) && (stars == 0 || underscores == 0);
}

static bool is_ul(std::string_view s)
{
    const auto p = s.find_first_not_of(' ');
    return p != std::string_view::npos &&
           (s[p] == '-' || s[p] == '*' || s[p] == '+') &&
           p + 1 < s.size() && s[p + 1] == ' ';
}

static bool is_ol(std::string_view s)
{
    const auto p = s.find_first_not_of(' ');
    if (p == std::string_view::npos) return false;
    std::size_t j = p;
    while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
    return j > p && j + 1 < s.size() && s[j] == '.' && s[j + 1] == ' ';
}

static bool is_bq(std::string_view s)
{
    const auto p = s.find_first_not_of(' ');
    return p != std::string_view::npos && s[p] == '>';
}

static bool is_table_row(std::string_view s)
{
    const auto p = s.find_first_not_of(' ');
    return p != std::string_view::npos && s[p] == '|';
}

static bool is_table_separator(std::string_view s)
{
    bool pipe = false, dash = false;
    for (char c : s) {
        if      (c == '|') pipe = true;
        else if (c == '-') dash = true;
        else if (c != ' ' && c != ':') return false;
    }
    return pipe && dash;
}

static int list_leading_spaces(std::string_view s)
{
    int sp = 0;
    while (sp < static_cast<int>(s.size()) && s[sp] == ' ') ++sp;
    return sp;
}

static std::string list_item_content(std::string_view s)
{
    const auto p = s.find_first_not_of(' ');
    if (p == std::string_view::npos) return {};

    if (is_ol(s)) {
        std::size_t j = p;
        while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
        // j is at '.', j+1 is at ' '
        return std::string(s.substr(j + 2));
    }
    if (is_ul(s)) {
        return std::string(s.substr(p + 2));
    }

    return std::string(s.substr(p));
}

// Tells us if a line starts a new block (used to terminate paragraphs early)
static bool is_block_start(std::string_view s)
{
    return is_blank(s)       ||
           heading_level(s)  ||
           is_fenced_code_start(s) ||
           is_hr(s)          ||
           is_ul(s)          ||
           is_ol(s)          ||
           is_bq(s)          ||
           is_table_row(s);
}

// ── Block parser ──────────────────────────────────────────────────────────────

static std::vector<Block> parse_blocks(std::string_view text)
{
    // Split into lines
    std::vector<std::string_view> lines;
    {
        std::size_t start = 0;
        while (start <= text.size()) {
            const std::size_t end = text.find('\n', start);
            lines.push_back(end == std::string_view::npos
                            ? text.substr(start)
                            : text.substr(start, end - start));
            if (end == std::string_view::npos) break;
            start = end + 1;
        }
    }

    std::vector<Block> blocks;
    std::size_t i = 0;
    const std::size_t total = lines.size();

    while (i < total) {
        const auto line = lines[i];

        // ── Blank line ────────────────────────────────────────────────────────
        if (is_blank(line)) {
            // Deduplicate consecutive blanks
            if (blocks.empty() || blocks.back().kind != BlockKind::BlankLine)
                blocks.push_back({.kind = BlockKind::BlankLine, .level = 0, .language = {}, .lines = {}});
            ++i; continue;
        }

        // ── ATX heading ───────────────────────────────────────────────────────
        {
            const int lv = heading_level(line);
            if (lv) {
                Block b;
                b.kind  = BlockKind::Heading;
                b.level = lv;
                b.lines.emplace_back(line.substr(static_cast<std::size_t>(lv + 1)));
                blocks.push_back(std::move(b));
                ++i; continue;
            }
        }

        // ── Horizontal rule (before UL to avoid mis-classifying "---") ────────
        if (is_hr(line) && !is_ul(line)) {
            blocks.push_back({.kind = BlockKind::HorizontalRule, .level = 0, .language = {}, .lines = {}});
            ++i; continue;
        }

        // ── Fenced code block ─────────────────────────────────────────────────
        if (is_fenced_code_start(line)) {
            const char fc = line[0];
            std::size_t fence_len = 0;
            while (fence_len < line.size() && line[fence_len] == fc) ++fence_len;

            Block b;
            b.kind     = BlockKind::FencedCode;
            b.language = std::string(line.substr(fence_len));
            // Trim language tag
            {
                auto& lang = b.language;
                while (!lang.empty() && std::isspace(static_cast<unsigned char>(lang.front())))
                    lang.erase(lang.begin());
                while (!lang.empty() && std::isspace(static_cast<unsigned char>(lang.back())))
                    lang.pop_back();
            }
            ++i;
            while (i < total) {
                const auto cl = lines[i];
                if (cl.size() >= fence_len) {
                    bool closing = true;
                    for (std::size_t k = 0; k < fence_len; ++k) {
                        if (cl[k] != fc) { closing = false; break; }
                    }
                    if (closing) { ++i; break; }
                }
                b.lines.emplace_back(cl);
                ++i;
            }
            blocks.push_back(std::move(b));
            continue;
        }

        // ── Indented code block ───────────────────────────────────────────────
        if (is_indented_code(line)) {
            Block b;
            b.kind = BlockKind::IndentedCode;
            while (i < total && (is_indented_code(lines[i]) || is_blank(lines[i]))) {
                b.lines.emplace_back(is_blank(lines[i]) ? "" : lines[i].substr(4));
                ++i;
            }
            // Strip trailing blank lines
            while (!b.lines.empty() && b.lines.back().empty()) b.lines.pop_back();
            blocks.push_back(std::move(b));
            continue;
        }

        // ── Table ─────────────────────────────────────────────────────────────
        if (is_table_row(line)) {
            Block b;
            b.kind = BlockKind::Table;
            while (i < total && is_table_row(lines[i])) {
                if (!is_table_separator(lines[i]))
                    b.lines.emplace_back(lines[i]);
                ++i;
            }
            blocks.push_back(std::move(b));
            continue;
        }

        // ── Blockquote ────────────────────────────────────────────────────────
        if (is_bq(line)) {
            Block b;
            b.kind = BlockKind::Blockquote;
            while (i < total && is_bq(lines[i])) {
                const auto bql = lines[i];
                const auto p   = bql.find_first_not_of(' ');
                // Strip "> " or ">"
                std::size_t content_start = p + 1;
                if (content_start < bql.size() && bql[content_start] == ' ')
                    ++content_start;
                b.lines.emplace_back(bql.substr(content_start));
                ++i;
            }
            blocks.push_back(std::move(b));
            continue;
        }

        // ── Unordered list ────────────────────────────────────────────────────
        if (is_ul(line)) {
            Block b;
            b.kind = BlockKind::UnorderedList;
            while (i < total) {
                if (is_blank(lines[i])) {
                    size_t next = i + 1;
                    while (next < total && is_blank(lines[next])) ++next;
                    if (next < total && (is_ul(lines[next]) || list_leading_spaces(lines[next]) >= 2)) {
                        for (size_t k = i; k < next; ++k) b.lines.emplace_back("");
                        i = next;
                        continue;
                    }
                    break;
                }
                b.lines.emplace_back(lines[i]);
                ++i;
            }
            blocks.push_back(std::move(b));
            continue;
        }

        // ── Ordered list ──────────────────────────────────────────────────────
        if (is_ol(line)) {
            Block b;
            b.kind = BlockKind::OrderedList;
            while (i < total) {
                if (is_blank(lines[i])) {
                    size_t next = i + 1;
                    while (next < total && is_blank(lines[next])) ++next;
                    if (next < total && (is_ol(lines[next]) || list_leading_spaces(lines[next]) >= 2)) {
                        for (size_t k = i; k < next; ++k) b.lines.emplace_back("");
                        i = next;
                        continue;
                    }
                    break;
                }
                b.lines.emplace_back(lines[i]);
                ++i;
            }
            blocks.push_back(std::move(b));
            continue;
        }

        // ── Paragraph ─────────────────────────────────────────────────────────
        {
            Block b;
            b.kind = BlockKind::Paragraph;
            while (i < total && !is_block_start(lines[i])) {
                b.lines.emplace_back(lines[i]);
                ++i;
            }
            if (!b.lines.empty()) blocks.push_back(std::move(b));
            continue;
        }
    }

    return blocks;
}

// ============================================================================
// Block renderers
// ============================================================================

static Element render_heading(const Block& block)
{
    const std::string_view content = block.lines.empty()
        ? std::string_view{}
        : std::string_view{block.lines.front()};
    auto spans = parse_inline(content);

    auto make_inline = [&](Color fg, bool do_bold, bool do_underline,
                           bool do_underline_double) -> Element
    {
        Elements es;
        for (const auto& s : spans) {
            Element e = render_span(s, fg);
            if (do_bold)            e = std::move(e) | ftxui::bold;
            if (do_underline_double)e = std::move(e) | underlinedDouble;
            else if (do_underline)  e = std::move(e) | underlined;
            es.push_back(std::move(e));
        }
        return hbox(std::move(es)) | xflex;
    };

    Elements out;
    switch (block.level) {
        case 1:
            out.push_back(make_inline(kMdH1Fg, true,  false, true));
            out.push_back(separatorHeavy() | ftxui::color(ColorYellowDark));
            break;
        case 2:
            out.push_back(make_inline(kMdH1Fg, true,  true,  false));
            break;
        case 3:
            out.push_back(make_inline(kMdH3Fg, true,  false, false));
            break;
        case 4:
            out.push_back(make_inline(kMdH4Fg, true,  false, false));
            break;
        default: // 5-6
            out.push_back(make_inline(Color::GrayLight, false, false, false) | dim);
            break;
    }
    return vbox(std::move(out));
}

static Element render_code_block(const Block& block)
{
    std::vector<Element> rows;
    rows.reserve(block.lines.size() + 1);

    for (const auto& cl : block.lines) {
        rows.push_back(
            hbox({
                ftxui::text("  "),
                ftxui::text(cl) | ftxui::color(Color::RGB(220, 220, 200)),
                filler(),
            }) | xflex
        );
    }
    if (rows.empty()) {
        rows.push_back(ftxui::text("  ") | ftxui::color(Color::RGB(220, 220, 200)));
    }

    Element content = vbox(std::move(rows));

    if (!block.language.empty()) {
        auto title = hbox({
            ftxui::text("  "),
            ftxui::text(block.language) | ftxui::color(kMdCodeFg) | ftxui::bold,
            ftxui::text("  "),
        });
        return UiWindow(std::move(title), std::move(content));
    }
    return std::move(content) | UiBorder(kMdCodeBorder);
}

static std::vector<std::string> split_table_cells(std::string_view row)
{
    // Strip leading/trailing pipe and whitespace
    auto s = row.find_first_not_of(' ');
    if (s != std::string_view::npos && row[s] == '|') row = row.substr(s + 1);
    auto e = row.find_last_not_of(' ');
    if (e != std::string_view::npos && row[e] == '|') row = row.substr(0, e);

    std::vector<std::string> cells;
    std::size_t pos = 0;
    while (pos <= row.size()) {
        const std::size_t sep = row.find('|', pos);
        const auto cell = sep == std::string_view::npos
                          ? row.substr(pos) : row.substr(pos, sep - pos);
        const auto cs = cell.find_first_not_of(' ');
        const auto ce = cell.find_last_not_of(' ');
        cells.emplace_back(cs == std::string_view::npos
                           ? "" : cell.substr(cs, ce - cs + 1));
        if (sep == std::string_view::npos) break;
        pos = sep + 1;
    }
    return cells;
}

static Element render_table(const Block& block)
{
    if (block.lines.empty()) return emptyElement();

    // Parse all rows into a 2-D grid of cell strings
    std::vector<std::vector<std::string>> rows;
    rows.reserve(block.lines.size());
    std::size_t max_cols = 0;
    for (const auto& l : block.lines) {
        auto cells = split_table_cells(l);
        max_cols = std::max(max_cols, cells.size());
        rows.push_back(std::move(cells));
    }
    if (rows.empty() || max_cols == 0) return emptyElement();

    std::vector<Elements> grid;
    grid.reserve(rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const Color fg = (r == 0) ? kMdTableHdrFg : kMdTableRowFg;
        const bool hdr = (r == 0);
        Elements row_elems;
        row_elems.reserve(max_cols);
        for (std::size_t c = 0; c < max_cols; ++c) {
            const std::string& cell = c < rows[r].size() ? rows[r][c] : "";
            Element cell_el = hbox({
                ftxui::text(" "),
                render_inline_line(cell, fg),
                ftxui::text(" "),
            }) | xflex;
            if (hdr) cell_el = std::move(cell_el) | ftxui::bold;
            row_elems.push_back(std::move(cell_el));
        }
        grid.push_back(std::move(row_elems));
    }

    return gridbox(std::move(grid)) | UiBorder(Color::GrayDark) | xflex;
}

static Element render_blockquote(const Block& block, Color /*base*/)
{
    std::vector<Element> rows;
    rows.reserve(block.lines.size());
    for (const auto& l : block.lines) {
        rows.push_back(
            hbox({
                ftxui::text("▌ ") | ftxui::color(kMdQuoteBar),
                render_inline_line(l, kMdQuoteFg),
            }) | xflex
        );
    }
    if (rows.empty()) return emptyElement();
    return vbox(std::move(rows));
}

static Element render_unordered_list(const Block& block, Color base)
{
    static constexpr std::array<const char*, 3> bullets = {"• ", "◦ ", "▸ "};

    std::vector<Element> rows;
    rows.reserve(block.lines.size());
    for (const auto& l : block.lines) {
        if (is_blank(l)) {
            rows.push_back(ftxui::text(""));
            continue;
        }
        const int spaces = list_leading_spaces(l);
        if (is_ul(l)) {
            const int level  = std::min(spaces / 2, 2);
            const std::string indent(static_cast<std::size_t>(level * 2), ' ');
            rows.push_back(
                hbox({
                    ftxui::text(indent),
                    ftxui::text(bullets[level]) | ftxui::color(kMdBulletFg),
                    render_inline_line(list_item_content(l), base),
                }) | xflex
            );
        } else {
            // Continuation line
            rows.push_back(
                hbox({
                    ftxui::text(std::string(static_cast<size_t>(spaces), ' ')),
                    render_inline_line(l.substr(static_cast<size_t>(spaces)), base),
                }) | xflex
            );
        }
    }
    return vbox(std::move(rows));
}

static Element render_ordered_list(const Block& block, Color base)
{
    std::vector<Element> rows;
    rows.reserve(block.lines.size());
    int num = 1;
    for (const auto& l : block.lines) {
        if (is_blank(l)) {
            rows.push_back(ftxui::text(""));
            continue;
        }
        const int spaces = list_leading_spaces(l);
        if (is_ol(l)) {
            const std::string indent(static_cast<std::size_t>(spaces), ' ');
            const std::string marker = std::format("{}. ", num++);
            rows.push_back(
                hbox({
                    ftxui::text(indent),
                    ftxui::text(marker) | ftxui::color(kMdNumFg),
                    render_inline_line(list_item_content(l), base),
                }) | xflex
            );
        } else {
            // Continuation line
            rows.push_back(
                hbox({
                    ftxui::text(std::string(static_cast<size_t>(spaces), ' ')),
                    render_inline_line(l.substr(static_cast<size_t>(spaces)), base),
                }) | xflex
            );
        }
    }
    return vbox(std::move(rows));
}

static Element render_paragraph(const Block& block, Color base)
{
    if (block.lines.empty()) return emptyElement();
    std::vector<Element> rows;
    rows.reserve(block.lines.size());
    for (const auto& l : block.lines) {
        rows.push_back(l.empty()
                       ? ftxui::text("")
                       : render_inline_line(l, base));
    }
    return vbox(std::move(rows)) | xflex;
}

static Element render_block(const Block& block, Color base)
{
    switch (block.kind) {
        case BlockKind::BlankLine:      return ftxui::text("");
        case BlockKind::HorizontalRule: return separatorHeavy() | ftxui::color(Color::GrayDark);
        case BlockKind::Heading:        return render_heading(block);
        case BlockKind::FencedCode:     [[fallthrough]];
        case BlockKind::IndentedCode:   return render_code_block(block);
        case BlockKind::Table:          return render_table(block);
        case BlockKind::Blockquote:     return render_blockquote(block, base);
        case BlockKind::UnorderedList:  return render_unordered_list(block, base);
        case BlockKind::OrderedList:    return render_ordered_list(block, base);
        case BlockKind::Paragraph:      return render_paragraph(block, base);
    }
    return emptyElement(); // unreachable
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

Element render_markdown(std::string_view text, Color base_color)
{
    if (text.empty()) return ftxui::text("") | ftxui::color(base_color);

    // Shield FTXUI from malformed UTF-8 and raw control bytes that can show up
    // in streamed model output and cause visual artifacts while rendering.
    const std::string safe_text = sanitize_for_ftxui(text);
    if (safe_text.empty()) return ftxui::text("") | ftxui::color(base_color);

    const auto blocks = parse_blocks(safe_text);
    if (blocks.empty()) return ftxui::text("") | ftxui::color(base_color);

    Elements elements;
    elements.reserve(blocks.size());
    for (const auto& b : blocks) {
        elements.push_back(render_block(b, base_color));
    }
    return vbox(std::move(elements)) | xflex;
}

} // namespace tui

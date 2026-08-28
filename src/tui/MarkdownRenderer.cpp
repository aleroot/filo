#include "MarkdownRenderer.hpp"
#include "TuiTheme.hpp"

#include <ftxui/dom/elements.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace ftxui;

namespace tui {
namespace {

// ============================================================================
// Colours used only by the markdown renderer
// ============================================================================

inline constexpr RgbColor kMdCodeFg{255, 210, 100};
inline constexpr RgbColor kMdCodeBg{32, 32, 40};
inline constexpr RgbColor kMdCodeBorder{72, 72, 88};
inline constexpr RgbColor kMdLinkFg{100, 200, 255};
inline const Color kMdQuoteFg   = Color::GrayLight;
inline constexpr RgbColor kMdQuoteBar{90, 90, 110};
inline const Color kMdH1Fg      = Color::White;
inline constexpr RgbColor kMdH3Fg{255, 246, 61};   // ColorYellowBright
inline constexpr RgbColor kMdH4Fg{255, 192, 32};   // ColorYellowDark
inline const Color kMdTableHdrFg= Color::White;
inline const Color kMdTableRowFg= Color::GrayLight;
inline constexpr RgbColor kMdBulletFg{255, 192, 32};   // ColorYellowDark
inline constexpr RgbColor kMdNumFg{255, 192, 32};

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

// A span has one rendering role and a set of composable styles. Keeping these
// separate prevents invalid role combinations (for example code + link URL)
// without reintroducing a BoldItalic-style combinatorial enum.
enum class SpanKind {
    Text,
    Code,
    LinkUrl,
};

struct SpanStyle {
    bool bold   = false;
    bool italic = false;
    bool strike = false;
    bool link   = false;
};

struct TextSpan {
    SpanKind kind = SpanKind::Text;
    SpanStyle style;
    std::string text;
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

static std::vector<TextSpan> parse_inline(std::string_view text);

// Re-parse emphasis/link/strike bodies so nested markers are not left literal.
// Code spans stay leaf nodes (content is shown verbatim with code styling).
static void append_nested(std::vector<TextSpan>& out,
                          std::string_view content,
                          SpanStyle inherited)
{
    if (content.empty()) return;

    auto inner = parse_inline(content);
    if (inner.empty()) {
        TextSpan s;
        s.text = std::string(content);
        s.style = inherited;
        out.push_back(std::move(s));
        return;
    }

    for (auto& s : inner) {
        // Generated URL suffixes are presentation metadata, not label content.
        // Code keeps its own face/colours, but can still carry link underline.
        if (s.kind == SpanKind::Text) {
            s.style.bold = s.style.bold || inherited.bold;
            s.style.italic = s.style.italic || inherited.italic;
            s.style.strike = s.style.strike || inherited.strike;
        }
        if (inherited.link && s.kind != SpanKind::LinkUrl) s.style.link = true;
        out.push_back(std::move(s));
    }
}

static std::vector<TextSpan> parse_inline(std::string_view text)
{
    std::vector<TextSpan> spans;
    std::string current;
    std::size_t i = 0;
    const std::size_t n = text.size();

    auto flush = [&]() {
        if (!current.empty()) {
            spans.push_back({.text = std::move(current)});
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
                TextSpan s;
                s.kind = SpanKind::Code;
                s.text = std::string(text.substr(i + tick_len, j - i - tick_len));
                spans.push_back(std::move(s));
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
                    const auto label = text.substr(i + 1, bracket_end - i - 1);
                    const std::string href(text.substr(bracket_end + 2,
                                                       paren_end - bracket_end - 2));
                    append_nested(spans, label, {.link = true});
                    if (!href.empty()) {
                        TextSpan url;
                        url.kind = SpanKind::LinkUrl;
                        url.text = " (" + href + ")";
                        spans.push_back(std::move(url));
                    }
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
                append_nested(spans, text.substr(i + 2, j - i - 2),
                              {.strike = true});
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
                    const auto content = text.substr(
                        i + static_cast<std::size_t>(try_len),
                        j - i - static_cast<std::size_t>(try_len));
                    const bool add_bold = try_len >= 2;
                    const bool add_italic = (try_len == 1 || try_len == 3);
                    append_nested(spans, content,
                                  {.bold = add_bold, .italic = add_italic});
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
        spans.push_back({.text = std::move(current)});
    }
    return spans;
}

static void append_wrapping_tokens(std::vector<TextSpan>& out,
                                   const TextSpan& prototype,
                                   std::string_view text)
{
    std::size_t start = 0;
    while (start < text.size()) {
        const bool whitespace = std::isspace(static_cast<unsigned char>(text[start])) != 0;
        std::size_t end = start + 1;
        while (end < text.size()) {
            const bool next_is_space =
                std::isspace(static_cast<unsigned char>(text[end])) != 0;
            if (next_is_space != whitespace) break;
            ++end;
        }

        TextSpan token = prototype;
        token.text = std::string(text.substr(start, end - start));
        // Keep whitespace unstyled so hflow gaps do not stretch bold/underline.
        if (whitespace) {
            token.kind = SpanKind::Text;
            token.style = {};
        }
        out.push_back(std::move(token));
        start = end;
    }
}

static std::vector<TextSpan> tokenize_spans_for_wrapping(const std::vector<TextSpan>& spans)
{
    std::vector<TextSpan> tokens;
    tokens.reserve(spans.size() * 2);

    for (const auto& span : spans) {
        if (span.kind == SpanKind::Code) {
            TextSpan token = span;
            token.text = " " + span.text + " ";
            tokens.push_back(std::move(token));
            continue;
        }
        append_wrapping_tokens(tokens, span, span.text);
    }

    return tokens;
}

static Element apply_span_style(Element e, const TextSpan& span, Color base)
{
    switch (span.kind) {
        case SpanKind::Text:
            e = std::move(e) | ftxui::color(
                span.style.link ? Color{kMdLinkFg} : base);
            break;
        case SpanKind::Code:
            e = std::move(e) | ftxui::color(kMdCodeFg) | ftxui::bgcolor(kMdCodeBg);
            break;
        case SpanKind::LinkUrl:
            e = std::move(e) | ftxui::color(Color::GrayDark) | dim;
            break;
    }

    if (span.style.link)   e = std::move(e) | underlined;
    if (span.style.bold)   e = std::move(e) | ftxui::bold;
    if (span.style.italic) e = std::move(e) | italic;
    if (span.style.strike) e = std::move(e) | strikethrough;
    return e;
}

static Element render_wrapping_token(const TextSpan& token, Color base)
{
    return apply_span_style(ftxui::text(token.text), token, base);
}

static Element render_span_row(const std::vector<TextSpan>& row, Color base)
{
    if (row.empty()) return ftxui::text("");

    // Single plain/styled span (no code/link): paragraph() wraps cleanly.
    if (row.size() == 1) {
        const auto& s = row[0];
        if (s.kind == SpanKind::Text && !s.style.link) {
            return apply_span_style(paragraph(s.text), s, base) | xflex;
        }
    }

    const auto tokens = tokenize_spans_for_wrapping(row);
    Elements elems;
    elems.reserve(tokens.size());
    for (const auto& token : tokens) {
        elems.push_back(render_wrapping_token(token, base));
    }
    return hflow(std::move(elems)) | xflex;
}

// Parse once across soft newlines, then split the flattened styled spans back
// into visual rows. Block renderers can add their own prefix to each row.
static std::vector<Element> render_inline_rows(std::string_view text, Color base)
{
    auto spans = parse_inline(text);

    // Split styled spans into rows on embedded '\n'.
    std::vector<std::vector<TextSpan>> rows(1);
    for (const auto& span : spans) {
        std::size_t start = 0;
        while (start <= span.text.size()) {
            const std::size_t nl = span.text.find('\n', start);
            const auto piece = (nl == std::string::npos)
                ? span.text.substr(start)
                : span.text.substr(start, nl - start);

            if (!piece.empty()) {
                TextSpan part = span;
                part.text = piece;
                rows.back().push_back(std::move(part));
            }

            if (nl == std::string::npos) break;
            rows.emplace_back();
            start = nl + 1;
        }
    }

    std::vector<Element> lines;
    lines.reserve(rows.size());
    for (const auto& row : rows) {
        lines.push_back(render_span_row(row, base));
    }
    return lines;
}

// Render one or more soft-broken lines without block-specific prefixes.
static Element render_inline_line(std::string_view line, Color base)
{
    if (line.empty()) return ftxui::text("");

    auto rows = render_inline_rows(line, base);
    if (rows.size() == 1) return std::move(rows.front());
    return vbox(std::move(rows)) | xflex;
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

struct CodeFence {
    char marker = '`';
    std::size_t length = 0;
    std::size_t indentation = 0;
    std::string_view info;
};

static std::size_t leading_spaces(std::string_view line, std::size_t limit)
{
    std::size_t count = 0;
    while (count < line.size() && count < limit && line[count] == ' ') ++count;
    return count;
}

// CommonMark 0.31.2 section 4.5: an opening fence has at least three matching
// markers and at most three leading spaces. Backtick info strings cannot
// themselves contain a backtick.
static std::optional<CodeFence> parse_opening_fence(std::string_view line)
{
    const std::size_t indentation = leading_spaces(line, 4);
    if (indentation == 4 || indentation == line.size()) return std::nullopt;

    const char marker = line[indentation];
    if (marker != '`' && marker != '~') return std::nullopt;

    std::size_t end = indentation;
    while (end < line.size() && line[end] == marker) ++end;
    const std::size_t length = end - indentation;
    if (length < 3) return std::nullopt;

    auto info = line.substr(end);
    if (marker == '`' && info.find('`') != std::string_view::npos)
        return std::nullopt;

    while (!info.empty() && (info.front() == ' ' || info.front() == '\t'))
        info.remove_prefix(1);
    while (!info.empty() && (info.back() == ' ' || info.back() == '\t'))
        info.remove_suffix(1);

    return CodeFence{
        .marker = marker,
        .length = length,
        .indentation = indentation,
        .info = info,
    };
}

// A closing fence must use the same marker, be at least as long as the opener,
// and contain nothing after the marker run except spaces or tabs. In
// particular, "```swift" is another opening fence, never a closing one.
static bool is_closing_fence(std::string_view line, const CodeFence& opening)
{
    const std::size_t indentation = leading_spaces(line, 4);
    if (indentation == 4 || indentation == line.size() ||
        line[indentation] != opening.marker)
        return false;

    std::size_t end = indentation;
    while (end < line.size() && line[end] == opening.marker) ++end;
    if (end - indentation < opening.length) return false;

    for (; end < line.size(); ++end) {
        if (line[end] != ' ' && line[end] != '\t') return false;
    }
    return true;
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
           parse_opening_fence(s).has_value() ||
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
        if (const auto fence = parse_opening_fence(line)) {
            Block b;
            b.kind     = BlockKind::FencedCode;
            b.language = std::string(fence->info);
            ++i;
            while (i < total) {
                const auto cl = lines[i];
                if (is_closing_fence(cl, *fence)) { ++i; break; }

                // Up to the opener's indentation is removed from content.
                const std::size_t content_indent =
                    leading_spaces(cl, fence->indentation);
                b.lines.emplace_back(cl.substr(content_indent));
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

    auto make_inline = [&](Color fg, bool do_bold, bool do_underline,
                           bool do_underline_double) -> Element
    {
        Element e = render_inline_line(content, fg);
        if (do_bold)             e = std::move(e) | ftxui::bold;
        if (do_underline_double) e = std::move(e) | underlinedDouble;
        else if (do_underline)   e = std::move(e) | underlined;
        return e;
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

static std::string join_soft_lines(const std::vector<std::string>& lines)
{
    std::string joined;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (i != 0) joined.push_back('\n');
        joined += lines[i];
    }
    return joined;
}

static Element render_blockquote(const Block& block, Color /*base*/)
{
    if (block.lines.empty()) return emptyElement();

    const auto joined = join_soft_lines(block.lines);
    auto inline_rows = render_inline_rows(joined, kMdQuoteFg);
    std::vector<Element> rows;
    rows.reserve(inline_rows.size());
    for (auto& inline_row : inline_rows) {
        rows.push_back(
            hbox({
                ftxui::text("▌ ") | ftxui::color(kMdQuoteBar),
                std::move(inline_row),
            }) | xflex
        );
    }
    return vbox(std::move(rows));
}

static Element render_list(const Block& block, Color base)
{
    static constexpr std::array<const char*, 3> bullets = {"• ", "◦ ", "▸ "};
    const bool ordered = block.kind == BlockKind::OrderedList;

    std::vector<Element> rows;
    rows.reserve(block.lines.size());
    int num = 1;
    for (std::size_t i = 0; i < block.lines.size();) {
        const auto& l = block.lines[i];
        if (is_blank(l)) {
            rows.push_back(ftxui::text(""));
            ++i;
            continue;
        }
        const bool is_item = ordered ? is_ol(l) : is_ul(l);
        if (!is_item) {
            const int spaces = list_leading_spaces(l);
            rows.push_back(
                hbox({
                    ftxui::text(std::string(static_cast<std::size_t>(spaces), ' ')),
                    render_inline_line(l.substr(static_cast<std::size_t>(spaces)), base),
                }) | xflex
            );
            ++i;
            continue;
        }

        // Parse each logical item together with its continuation lines so
        // emphasis can cross a soft newline without leaking into the next item.
        const std::size_t end = [&] {
            std::size_t next = i + 1;
            while (next < block.lines.size() && !is_blank(block.lines[next])) {
                const bool next_is_item = ordered
                    ? is_ol(block.lines[next])
                    : is_ul(block.lines[next]);
                if (next_is_item) break;
                ++next;
            }
            return next;
        }();

        std::vector<std::string> content_lines;
        content_lines.reserve(end - i);
        content_lines.push_back(list_item_content(l));
        for (std::size_t j = i + 1; j < end; ++j) {
            const int continuation_spaces = list_leading_spaces(block.lines[j]);
            content_lines.emplace_back(block.lines[j].substr(
                static_cast<std::size_t>(continuation_spaces)));
        }
        auto inline_rows = render_inline_rows(join_soft_lines(content_lines), base);

        const int spaces = list_leading_spaces(l);
        const int level = std::min(spaces / 2, 2);
        const std::string indent = ordered
            ? std::string(static_cast<std::size_t>(spaces), ' ')
            : std::string(static_cast<std::size_t>(level * 2), ' ');
        const std::string marker = ordered
            ? std::format("{}. ", num++)
            : bullets[level];
        const Color marker_color = ordered ? Color{kMdNumFg} : Color{kMdBulletFg};
        for (std::size_t j = 0; j < inline_rows.size(); ++j) {
            if (j == 0) {
                rows.push_back(
                    hbox({
                        ftxui::text(indent),
                        ftxui::text(marker) | ftxui::color(marker_color),
                        std::move(inline_rows[j]),
                    }) | xflex
                );
                continue;
            }

            const auto& continuation = block.lines[i + j];
            const int continuation_spaces = list_leading_spaces(continuation);
            rows.push_back(
                hbox({
                    ftxui::text(std::string(
                        static_cast<std::size_t>(continuation_spaces), ' ')),
                    std::move(inline_rows[j]),
                }) | xflex
            );
        }
        i = end;
    }
    return vbox(std::move(rows));
}

static Element render_paragraph(const Block& block, Color base)
{
    if (block.lines.empty()) return emptyElement();

    // Join soft-broken lines so emphasis/links can span model line wraps
    // (e.g. "start **bold\nacross** end"). Embedded '\n' is preserved as a
    // visual row break inside render_inline_line.
    return render_inline_line(join_soft_lines(block.lines), base);
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
        case BlockKind::UnorderedList:  [[fallthrough]];
        case BlockKind::OrderedList:    return render_list(block, base);
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

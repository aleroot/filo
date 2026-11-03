#include "Frontmatter.hpp"

#include "../logging/Logger.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/Hex.hpp"
#include "../utils/InterprocessFile.hpp"
#include "../utils/StringUtils.hpp"
#include "../utils/Utf8.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <format>
#include <set>
#include <span>
#include <utility>

namespace fs = std::filesystem;

namespace core::tools::frontmatter {

namespace {

using core::utils::ascii::is_blank;
namespace str = core::utils::str;

constexpr auto npos = std::string_view::npos;
/// White space within a YAML line. Line breaks are structure, not blanks.
constexpr std::string_view kBlank = " \t";

template <typename T>
using Result = std::expected<T, ParseError>;

using Map = std::map<std::string, Value, std::less<>>;

[[nodiscard]] std::unexpected<ParseError> fail(std::size_t line, std::string message) {
    return std::unexpected(ParseError{.line = line, .message = std::move(message)});
}

/// @p text up to its first blank: an anchor, tag or alias token.
[[nodiscard]] std::string_view first_token(std::string_view text) {
    return text.substr(0, text.find_first_of(kBlank));
}

[[nodiscard]] bool is_blank_line(std::string_view text) {
    return str::trim_left_view(text, kBlank).empty();
}

[[nodiscard]] bool is_comment(std::string_view text) {
    return str::trim_left_view(text, kBlank).starts_with('#');
}

/// Leading spaces. Tabs are never indentation in YAML.
[[nodiscard]] std::size_t leading_spaces(std::string_view text) {
    const auto start = text.find_first_not_of(' ');
    return start == npos ? text.size() : start;
}

/**
 * A line of the frontmatter, or the part of one that holds a node: after
 * `- ` the entry's content starts further right, and @ref column records
 * where, so a collection opened there (`- key: value`) knows its indentation.
 */
struct Line {
    std::string_view text;  ///< Without the line terminator.
    std::size_t number = 0; ///< 1-based.
    std::size_t column = 0; ///< Column of text[0] in the source line.
};
using Lines = std::span<const Line>;

[[nodiscard]] bool is_significant(const Line& line) {
    return !is_blank_line(line.text) && !is_comment(line.text);
}

[[nodiscard]] std::size_t indent(const Line& line) {
    return line.column + leading_spaces(line.text);
}

/// The line without its indentation, keeping its column.
[[nodiscard]] Line content(const Line& line) {
    const auto spaces = leading_spaces(line.text);
    return Line{.text = line.text.substr(spaces), .number = line.number, .column = line.column + spaces};
}

/// The rest of @p line from @p offset on, white space skipped.
[[nodiscard]] Line rest_of(const Line& line, std::size_t offset) {
    const auto rest = line.text.substr(offset);
    const auto skip = rest.find_first_not_of(kBlank);
    const auto start = skip == npos ? rest.size() : skip;
    return Line{.text = rest.substr(start), .number = line.number, .column = line.column + offset + start};
}

[[nodiscard]] Result<void> check_tab(const Line& line) {
    const auto spaces = leading_spaces(line.text);
    if (spaces < line.text.size() && line.text[spaces] == '\t') {
        return fail(line.number, "tab character used for indentation");
    }
    return {};
}

/// Cuts a ` # comment` from plain text: `#` starts a comment only after white
/// space (or at the start).
[[nodiscard]] std::string_view strip_comment(std::string_view text, bool& had_comment) {
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '#' && (i == 0 || is_blank(text[i - 1]))) {
            had_comment = true;
            return str::trim_right_view(text.substr(0, i), kBlank);
        }
    }
    return str::trim_right_view(text, kBlank);
}

/// `-` followed by white space or the end: a block sequence entry.
[[nodiscard]] bool is_sequence_entry(std::string_view text) {
    return !text.empty() && text.front() == '-' && (text.size() == 1 || is_blank(text[1]));
}

/// Why plain text cannot start a plain scalar, or null when it can. YAML
/// reserves the indicator characters; `-`, `?` and `:` start plain text only
/// when followed by a non-space (`-x`, `:x`).
[[nodiscard]] const char* plain_start_error(std::string_view text) {
    switch (text.front()) {
        case ']':
        case '}':
        case ',':
        case '%':
        case '@':
        case '`':
        case '|':
        case '>':
            return "a plain scalar cannot start with a YAML indicator character";
        case '-':
        case '?':
        case ':':
            if (text.size() == 1 || is_blank(text[1])) {
                return "a plain scalar cannot start with a YAML indicator character";
            }
            break;
        default:
            break;
    }
    return nullptr;
}

/// The end of a quoted scalar starting at text[0] on this line, or npos.
[[nodiscard]] std::size_t closing_quote(std::string_view text) {
    const char quote = text.front();
    for (std::size_t i = 1; i < text.size(); ++i) {
        if (quote == '"' && text[i] == '\\') {
            ++i;
        } else if (text[i] == quote) {
            if (quote == '\'' && i + 1 < text.size() && text[i + 1] == '\'') {
                ++i;
            } else {
                return i;
            }
        }
    }
    return npos;
}

/// True when @p text opens a mapping entry: `key: value`, `key:`,
/// `"quoted key": value` or a `? ` complex key.
[[nodiscard]] bool looks_like_mapping_entry(std::string_view text) {
    if (text.empty()) return false;
    std::size_t colon = npos;
    if (text.front() == '"' || text.front() == '\'') {
        const auto close = closing_quote(text);
        if (close == npos) return false;
        colon = close + 1;
        while (colon < text.size() && is_blank(text[colon])) ++colon;
        if (colon >= text.size() || text[colon] != ':') return false;
    } else if (text.front() == '[' || text.front() == '{') {
        return false;
    } else if (text.front() == '?') {
        return text.size() == 1 || is_blank(text[1]);
    } else {
        for (std::size_t i = 0; i < text.size() && colon == npos; ++i) {
            if (text[i] == '#' && i > 0 && is_blank(text[i - 1])) return false;
            if (text[i] == ':') colon = i;
            if (colon != npos && colon + 1 < text.size() && !is_blank(text[colon + 1])) colon = npos;
        }
        if (colon == npos) return false;
    }
    return colon + 1 == text.size() || is_blank(text[colon + 1]);
}

[[nodiscard]] Value nested() { return Value{.kind = Value::Kind::Nested, .scalar = {}, .items = {}}; }

[[nodiscard]] Value scalar(std::string text) {
    return Value{.kind = Value::Kind::Scalar, .scalar = std::move(text), .items = {}};
}

/// Anchored nodes, for aliases. Anchors inside a skipped flow mapping are not
/// recorded, so an alias that may point into one cannot be called undefined.
struct Anchors {
    Map values;
    bool incomplete = false;
};

/**
 * The text of a quoted scalar or flow collection, which may continue on any
 * number of following lines — at any indentation, as libyaml and js-yaml
 * accept. Lines are joined with `\n` and loaded only as the scan reaches
 * them; offsets map back to line numbers and to how many lines were used.
 */
class Source {
public:
    Source(const Line& head, Lines following) : following_(following) {
        text_ = head.text;
        starts_.push_back(0);
        numbers_.push_back(head.number);
    }

    /// True when @p pos is inside the text, loading lines as needed.
    [[nodiscard]] bool has(std::size_t pos) {
        while (pos >= text_.size() && loaded_ < following_.size()) {
            text_ += '\n';
            starts_.push_back(text_.size());
            numbers_.push_back(following_[loaded_].number);
            text_ += following_[loaded_].text;
            ++loaded_;
        }
        return pos < text_.size();
    }

    [[nodiscard]] char operator[](std::size_t pos) const { return text_[pos]; }
    [[nodiscard]] std::size_t loaded_size() const { return text_.size(); }

    [[nodiscard]] std::size_t line_at(std::size_t offset) const { return numbers_[segment_at(offset)]; }

    /// 0 for the head line, k for the k-th following line.
    [[nodiscard]] std::size_t segment_at(std::size_t offset) const {
        const auto it = std::ranges::upper_bound(starts_, offset);
        return static_cast<std::size_t>(it - starts_.begin()) - 1;
    }

    void skip_inline_space(std::size_t& pos) {
        while (has(pos) && is_blank(text_[pos])) ++pos;
    }

    /// White space, line breaks and comments between flow tokens.
    void skip_separation(std::size_t& pos) {
        while (has(pos)) {
            const char ch = text_[pos];
            if (is_blank(ch) || ch == '\n') {
                ++pos;
            } else if (ch == '#' && (pos == 0 || is_blank(text_[pos - 1]) || text_[pos - 1] == '\n')) {
                while (has(pos) && text_[pos] != '\n') ++pos;
            } else {
                break;
            }
        }
    }

    /**
     * A line break inside a quoted or flow scalar, at @p pos: white space
     * around it is dropped (not below @p keep, the end of the last escape)
     * and it becomes a space — or one `\n` per empty line that follows.
     */
    void fold_break(std::size_t& pos, std::string& out, std::size_t keep) {
        while (out.size() > keep && is_blank(out.back())) out.pop_back();
        ++pos;
        skip_inline_space(pos);
        std::size_t empty_lines = 0;
        while (has(pos) && text_[pos] == '\n') {
            ++empty_lines;
            ++pos;
            skip_inline_space(pos);
        }
        if (empty_lines == 0) {
            out += ' ';
        } else {
            out.append(empty_lines, '\n');
        }
    }

private:
    Lines following_;
    std::size_t loaded_ = 0;
    std::string text_;
    std::vector<std::size_t> starts_;
    std::vector<std::size_t> numbers_;
};

[[nodiscard]] Result<std::uint32_t> parse_hex(Source& src, std::size_t pos, std::size_t digits) {
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < digits; ++i) {
        if (!src.has(pos + i)) return fail(src.line_at(pos), "truncated hexadecimal escape");
        const int digit = core::utils::hex_digit_value(static_cast<unsigned char>(src[pos + i]));
        if (digit < 0) return fail(src.line_at(pos + i), "invalid hexadecimal escape");
        value = (value << 4) | static_cast<std::uint32_t>(digit);
    }
    return value;
}

/// A double-quoted scalar starting at the quote at @p pos; @p pos ends past
/// the closing quote.
[[nodiscard]] Result<std::string> parse_double_quoted(Source& src, std::size_t& pos) {
    const auto open_line = src.line_at(pos);
    ++pos;
    std::string out;
    std::size_t keep = 0;  // Escaped white space must survive folding.
    while (src.has(pos)) {
        const char ch = src[pos];
        if (ch == '"') {
            ++pos;
            return out;
        }
        if (ch == '\n') {
            src.fold_break(pos, out, keep);
            continue;
        }
        if (ch != '\\') {
            out += ch;
            ++pos;
            continue;
        }
        if (!src.has(pos + 1)) break;
        const char escape = src[pos + 1];
        pos += 2;
        switch (escape) {
            case '\n':
                // An escaped line break joins the lines without a space;
                // empty lines after it still count.
                src.skip_inline_space(pos);
                while (src.has(pos) && src[pos] == '\n') {
                    out += '\n';
                    ++pos;
                    src.skip_inline_space(pos);
                }
                break;
            case '0': out += '\0'; break;
            case 'a': out += '\a'; break;
            case 'b': out += '\b'; break;
            case 't':
            case '\t': out += '\t'; break;
            case 'n': out += '\n'; break;
            case 'v': out += '\v'; break;
            case 'f': out += '\f'; break;
            case 'r': out += '\r'; break;
            case 'e': out += '\x1B'; break;
            case ' ': out += ' '; break;
            case '"': out += '"'; break;
            case '/': out += '/'; break;
            case '\\': out += '\\'; break;
            case 'N': core::utils::utf8::append(out, 0x85); break;
            case '_': core::utils::utf8::append(out, 0xA0); break;
            case 'L': core::utils::utf8::append(out, 0x2028); break;
            case 'P': core::utils::utf8::append(out, 0x2029); break;
            case 'x':
            case 'u':
            case 'U': {
                const std::size_t digits = escape == 'x' ? 2 : escape == 'u' ? 4 : 8;
                auto cp = parse_hex(src, pos, digits);
                if (!cp) return std::unexpected(cp.error());
                if ((*cp >= 0xD800 && *cp <= 0xDFFF) || *cp > 0x10FFFF) {
                    return fail(src.line_at(pos), "escape is not a Unicode scalar value");
                }
                core::utils::utf8::append(out, *cp);
                pos += digits;
                break;
            }
            default:
                return fail(src.line_at(pos - 1), std::format("invalid escape '\\{}'", escape));
        }
        keep = out.size();
    }
    return fail(open_line, "unterminated double-quoted scalar");
}

/// A single-quoted scalar (`''` is a quote) starting at @p pos.
[[nodiscard]] Result<std::string> parse_single_quoted(Source& src, std::size_t& pos) {
    const auto open_line = src.line_at(pos);
    ++pos;
    std::string out;
    while (src.has(pos)) {
        const char ch = src[pos];
        if (ch == '\'') {
            if (src.has(pos + 1) && src[pos + 1] == '\'') {
                out += '\'';
                pos += 2;
                continue;
            }
            ++pos;
            return out;
        }
        if (ch == '\n') {
            src.fold_break(pos, out, 0);
            continue;
        }
        out += ch;
        ++pos;
    }
    return fail(open_line, "unterminated single-quoted scalar");
}

/// Skips a flow collection (`[...]` or `{...}`) at @p pos, quotes included.
[[nodiscard]] Result<void> skip_flow_collection(Source& src, std::size_t& pos, Anchors& anchors) {
    const auto open_line = src.line_at(pos);
    std::vector<char> closers;
    while (src.has(pos)) {
        const char ch = src[pos];
        if (ch == '[' || ch == '{') {
            closers.push_back(ch == '[' ? ']' : '}');
            ++pos;
        } else if (ch == ']' || ch == '}') {
            if (closers.empty() || closers.back() != ch) {
                return fail(src.line_at(pos), std::format("unbalanced '{}'", ch));
            }
            closers.pop_back();
            ++pos;
            if (closers.empty()) return {};
        } else if (ch == '"' || ch == '\'') {
            auto skipped = ch == '"' ? parse_double_quoted(src, pos) : parse_single_quoted(src, pos);
            if (!skipped) return std::unexpected(skipped.error());
        } else if (ch == '#' && pos > 0 && (is_blank(src[pos - 1]) || src[pos - 1] == '\n')) {
            while (src.has(pos) && src[pos] != '\n') ++pos;
        } else {
            if (ch == '&') anchors.incomplete = true;
            ++pos;
        }
    }
    return fail(open_line, "unterminated flow collection");
}

/// A plain scalar inside a flow collection, ending before `,`, `]` or `}`.
[[nodiscard]] Result<std::string> parse_flow_plain(Source& src, std::size_t& pos) {
    {
        std::string start(1, src[pos]);
        if (src.has(pos + 1)) start += src[pos + 1];
        if (const char* error = plain_start_error(start)) return fail(src.line_at(pos), error);
    }
    std::string out;
    while (src.has(pos)) {
        const char ch = src[pos];
        if (ch == ',' || ch == ']' || ch == '}') break;
        if (ch == '[' || ch == '{') {
            return fail(src.line_at(pos), std::format("unexpected '{}' inside a flow scalar", ch));
        }
        if (ch == '#' && !out.empty() && is_blank(out.back())) {
            while (src.has(pos) && src[pos] != '\n') ++pos;
            continue;
        }
        if (ch == '\n') {
            src.fold_break(pos, out, 0);
            continue;
        }
        out += ch;
        ++pos;
    }
    return std::string(str::trim_view(out, kBlank));
}

/// The anchor, tag or alias token at @p pos (`&name`, `!tag`, `*name`).
[[nodiscard]] std::string read_flow_token(Source& src, std::size_t& pos) {
    std::string token;
    while (src.has(pos) && !is_blank(src[pos]) && src[pos] != '\n' && src[pos] != ','
           && src[pos] != ']' && src[pos] != '}') {
        token += src[pos++];
    }
    return token;
}

struct FlowSequence {
    std::vector<std::string> items;
    bool nested = false;
};

/// A flow sequence starting at the `[` at @p pos.
[[nodiscard]] Result<FlowSequence> parse_flow_sequence(Source& src, std::size_t& pos, Anchors& anchors) {
    const auto open_line = src.line_at(pos);
    ++pos;
    FlowSequence sequence;
    while (true) {
        src.skip_separation(pos);
        if (!src.has(pos)) return fail(open_line, "unterminated flow sequence");
        if (src[pos] == ']') {
            ++pos;
            return sequence;
        }

        std::string anchor;
        while (src.has(pos) && (src[pos] == '&' || src[pos] == '!')) {
            const auto line = src.line_at(pos);
            if (auto token = read_flow_token(src, pos); token.front() == '&') {
                if (token.size() == 1) return fail(line, "empty anchor name");
                anchor = token.substr(1);
            }
            src.skip_separation(pos);
        }
        if (!src.has(pos)) return fail(open_line, "unterminated flow sequence");

        std::optional<Value> item;
        const char ch = src[pos];
        if (!anchor.empty() && (ch == ',' || ch == ']')) {
            item = scalar({});  // `[&a, b]`: the anchor marks an empty entry.
        } else if (ch == '"' || ch == '\'') {
            auto text = ch == '"' ? parse_double_quoted(src, pos) : parse_single_quoted(src, pos);
            if (!text) return std::unexpected(text.error());
            item = scalar(std::move(*text));
        } else if (ch == '[' || ch == '{') {
            if (auto skipped = skip_flow_collection(src, pos, anchors); !skipped) {
                return std::unexpected(skipped.error());
            }
            item = nested();
        } else if (ch == '*') {
            const auto line = src.line_at(pos);
            const auto name = read_flow_token(src, pos).substr(1);
            if (const auto it = anchors.values.find(name); it != anchors.values.end()) {
                item = it->second;
            } else if (anchors.incomplete) {
                item = nested();
            } else {
                return fail(line, std::format("undefined alias '{}'", name));
            }
        } else if (ch == ',') {
            return fail(src.line_at(pos), "empty entry in flow sequence");
        } else {
            auto text = parse_flow_plain(src, pos);
            if (!text) return std::unexpected(text.error());
            item = scalar(std::move(*text));
        }

        if (!anchor.empty()) anchors.values.insert_or_assign(anchor, *item);
        if (item->kind == Value::Kind::Scalar) {
            sequence.items.push_back(std::move(item->scalar));
        } else {
            sequence.nested = true;
        }

        src.skip_separation(pos);
        if (!src.has(pos)) return fail(open_line, "unterminated flow sequence");
        if (src[pos] == ',') {
            ++pos;
        } else if (src[pos] != ']') {
            return fail(src.line_at(pos), "expected ',' or ']' in flow sequence");
        }
    }
}

enum class Chomping { Clip, Strip, Keep };

struct BlockHeader {
    bool folded = false;
    Chomping chomping = Chomping::Clip;
    std::size_t indent = 0;  ///< Explicit indentation indicator; 0 = detect.
};

/// `|` or `>`, then a chomping indicator and an indentation digit in either
/// order, then optionally a comment.
[[nodiscard]] Result<BlockHeader> parse_block_header(std::string_view head, std::size_t line) {
    BlockHeader header{.folded = head.front() == '>'};
    std::size_t i = 1;
    bool seen_chomp = false;
    bool seen_indent = false;
    for (; i < head.size() && !is_blank(head[i]); ++i) {
        const char ch = head[i];
        if ((ch == '-' || ch == '+') && !seen_chomp) {
            header.chomping = ch == '-' ? Chomping::Strip : Chomping::Keep;
            seen_chomp = true;
        } else if (ch >= '1' && ch <= '9' && !seen_indent) {
            header.indent = static_cast<std::size_t>(ch - '0');
            seen_indent = true;
        } else {
            return fail(line, std::format("invalid block scalar header '{}'", head));
        }
    }
    if (const auto rest = str::trim_left_view(head.substr(i), kBlank); !rest.empty() && rest.front() != '#') {
        return fail(line, "unexpected content after block scalar header");
    }
    return header;
}

struct BlockLine {
    std::string_view text;  ///< Without the block indentation.
    bool empty = false;
};

/// Applies folding (for `>`) and chomping to a block scalar's lines.
[[nodiscard]] std::string render_block(const std::vector<BlockLine>& lines, const BlockHeader& header) {
    const auto last_content = std::ranges::find_if(lines.rbegin(), lines.rend(),
                                                   [](const BlockLine& line) { return !line.empty; });
    const auto content_end = last_content.base();
    if (content_end == lines.begin()) {
        // No content: only Keep retains the line breaks.
        return header.chomping == Chomping::Keep ? std::string(lines.size(), '\n') : std::string{};
    }
    const auto trailing_empty = static_cast<std::size_t>(lines.end() - content_end);

    std::string out;
    if (header.folded) {
        // Lines at the block indentation fold into one with a space; empty
        // lines between them become newlines; breaks around more-indented
        // ("spaced") lines are kept.
        std::size_t pending_empty = 0;
        bool have_previous = false;
        bool previous_spaced = false;
        for (auto it = lines.begin(); it != content_end; ++it) {
            if (it->empty) {
                ++pending_empty;
                continue;
            }
            const bool spaced = is_blank(it->text.front());
            if (!have_previous) {
                out.append(pending_empty, '\n');
            } else if (!previous_spaced && !spaced) {
                if (pending_empty == 0) out += ' ';
                else out.append(pending_empty, '\n');
            } else {
                out.append(pending_empty + 1, '\n');
            }
            out += it->text;
            pending_empty = 0;
            have_previous = true;
            previous_spaced = spaced;
        }
    } else {
        for (auto it = lines.begin(); it != content_end; ++it) {
            if (it != lines.begin()) out += '\n';
            out += it->text;
        }
    }

    switch (header.chomping) {
        case Chomping::Strip: break;
        case Chomping::Clip: out += '\n'; break;
        case Chomping::Keep: out.append(trailing_empty + 1, '\n'); break;
    }
    return out;
}

struct Key {
    std::string name;
    std::size_t value_offset = 0;  ///< Where the value starts in the line.
};

/// `key:` at the start of @p line, after any anchor or tag on the key.
[[nodiscard]] Result<Key> parse_key(const Line& whole) {
    Line line = whole;
    while (!line.text.empty() && (line.text.front() == '&' || line.text.front() == '!')) {
        line = rest_of(line, first_token(line.text).size());
    }
    if (line.text.empty()) return fail(line.number, "expected 'key: value'");
    const auto text = line.text;
    const auto skipped = line.column - whole.column;
    const char first = text.front();
    if (first == '?' && (text.size() == 1 || is_blank(text[1]))) {
        return fail(line.number, "complex mapping keys are not supported");
    }
    if (first == '*') return fail(line.number, "aliases as keys are not supported");
    if (first == '[' || first == '{') {
        return fail(line.number, "flow collections as keys are not supported");
    }
    if (first != '"' && first != '\'') {
        if (const char* error = plain_start_error(text)) return fail(line.number, error);
    }

    std::string name;
    std::size_t colon = npos;
    if (first == '"' || first == '\'') {
        Source source(line, {});
        std::size_t pos = 0;
        auto quoted = first == '"' ? parse_double_quoted(source, pos) : parse_single_quoted(source, pos);
        if (!quoted) return std::unexpected(quoted.error());
        while (pos < text.size() && is_blank(text[pos])) ++pos;
        if (pos < text.size() && text[pos] == ':') colon = pos;
        name = std::move(*quoted);
    } else {
        for (std::size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '#' && i > 0 && is_blank(text[i - 1])) break;
            if (text[i] == ':' && (i + 1 == text.size() || is_blank(text[i + 1]))) {
                colon = i;
                break;
            }
        }
        if (colon != npos) name = std::string(str::trim_right_view(text.substr(0, colon), kBlank));
    }

    if (colon == npos || (colon + 1 < text.size() && !is_blank(text[colon + 1]))) {
        return fail(line.number, "expected 'key: value'");
    }
    if (name.empty()) return fail(line.number, "empty key");
    return Key{.name = std::move(name), .value_offset = skipped + colon + 1};
}

/**
 * Recursive descent over the frontmatter lines with a single cursor: every
 * node consumes exactly the lines it owns — by indentation for block nodes,
 * up to the closing delimiter for quoted scalars and flow collections — and
 * the enclosing collection resumes from there.
 */
class Parser {
public:
    explicit Parser(Lines lines) : lines_(lines) {}

    [[nodiscard]] Result<Map> parse() {
        if (const Line* first = peek(); first != nullptr && indent(*first) == 0
                                        && is_sequence_entry(first->text)) {
            return fail(first->number, "frontmatter must be a mapping of keys, not a sequence");
        }
        Map fields;
        if (auto parsed = parse_mapping(0, nullptr, &fields); !parsed) {
            return std::unexpected(parsed.error());
        }
        return fields;
    }

private:
    /// The next line with content, without consuming anything.
    [[nodiscard]] const Line* peek() const {
        for (auto i = pos_; i < lines_.size(); ++i) {
            if (is_significant(lines_[i])) return &lines_[i];
        }
        return nullptr;
    }

    /// Consumes up to and including the line @ref peek returned.
    const Line& take() {
        while (!is_significant(lines_[pos_])) ++pos_;
        return lines_[pos_++];
    }

    /// A block mapping with its keys at @p column. @p first is its first
    /// entry when that shares a line with a sequence indicator.
    [[nodiscard]] Result<void> parse_mapping(std::size_t column, const Line* first, Map* out) {
        std::set<std::string, std::less<>> keys;
        const auto entry = [&](const Line& line) -> Result<void> {
            auto key = parse_key(line);
            if (!key) return std::unexpected(key.error());
            if (!keys.insert(key->name).second) {
                return fail(line.number, std::format("duplicate key '{}'", key->name));
            }
            auto value = parse_value(rest_of(line, key->value_offset), column, false);
            if (!value) return std::unexpected(value.error());
            if (out != nullptr) out->emplace(std::move(key->name), std::move(*value));
            return {};
        };

        if (first != nullptr) {
            if (auto parsed = entry(*first); !parsed) return parsed;
        }
        while (const Line* next = peek()) {
            const auto next_indent = indent(*next);
            if (next_indent < column) break;
            if (auto tab = check_tab(*next); !tab) return tab;
            if (next_indent > column) return fail(next->number, "unexpected indentation");
            if (is_sequence_entry(content(*next).text)) {
                return fail(next->number, "expected 'key: value'");
            }
            if (auto parsed = entry(content(take())); !parsed) return parsed;
        }
        return {};
    }

    /// A block sequence with its `-` at @p column. A compact sequence (one at
    /// its key's own column) ends at the first line that is not an entry.
    [[nodiscard]] Result<Value> parse_sequence(std::size_t column, const Line* first, bool compact) {
        Value sequence{.kind = Value::Kind::Sequence, .scalar = {}, .items = {}};
        bool has_nested = false;
        const auto entry = [&](const Line& line) -> Result<void> {
            auto value = parse_value(rest_of(line, 1), column, true);
            if (!value) return std::unexpected(value.error());
            if (value->kind == Value::Kind::Scalar) {
                sequence.items.push_back(std::move(value->scalar));
            } else {
                has_nested = true;
            }
            return {};
        };

        if (first != nullptr) {
            if (auto parsed = entry(*first); !parsed) return std::unexpected(parsed.error());
        }
        while (const Line* next = peek()) {
            const auto next_indent = indent(*next);
            if (next_indent < column) break;
            if (auto tab = check_tab(*next); !tab) return std::unexpected(tab.error());
            if (next_indent > column) return fail(next->number, "unexpected indentation inside a sequence");
            if (!is_sequence_entry(content(*next).text)) {
                if (compact) break;
                return fail(next->number, "expected a '- ' sequence entry");
            }
            if (auto parsed = entry(content(take())); !parsed) return std::unexpected(parsed.error());
        }
        if (has_nested) return nested();
        return sequence;
    }

    /// The node after `key:` (@p in_sequence false) or `- ` (true); @p head is
    /// the rest of that line and @p parent the column of the key or `-`.
    [[nodiscard]] Result<Value> parse_value(Line head, std::size_t parent, bool in_sequence) {
        // Every nested block node descends through here, so this one bound
        // keeps a hostile `- - - …` ladder from exhausting the stack.
        if (depth_ >= kMaxDepth) {
            return fail(head.number, std::format("nesting deeper than {} levels", kMaxDepth));
        }
        ++depth_;
        struct Ascend {
            std::size_t& depth;
            ~Ascend() { --depth; }
        } ascend{depth_};

        std::string anchor;
        while (!head.text.empty() && (head.text.front() == '&' || head.text.front() == '!')) {
            const auto token = first_token(head.text);
            if (token.front() == '&') {
                if (token.size() == 1) return fail(head.number, "empty anchor name");
                if (!anchor.empty()) return fail(head.number, "a node can have only one anchor");
                anchor = token.substr(1);
            }
            head = rest_of(head, token.size());
        }

        auto value = parse_value_without_properties(head, parent, in_sequence);
        if (value && !anchor.empty()) anchors_.values.insert_or_assign(anchor, *value);
        return value;
    }

    [[nodiscard]] Result<Value>
    parse_value_without_properties(const Line& head, std::size_t parent, bool in_sequence) {
        const auto text = head.text;
        if (!text.empty() && text.front() == '*') return parse_alias(head);

        if (text.empty() || text.front() == '#') {
            const Line* next = peek();
            if (next != nullptr && indent(*next) > parent) {
                if (auto tab = check_tab(*next); !tab) return std::unexpected(tab.error());
                return parse_own_line(content(take()), parent);
            }
            if (next != nullptr && !in_sequence && indent(*next) == parent
                && is_sequence_entry(content(*next).text)) {
                return parse_sequence(parent, nullptr, true);
            }
            return scalar({});
        }

        if (in_sequence) {
            // `- - a` and `- key: value` open a collection inside the entry.
            if (is_sequence_entry(text)) return parse_sequence(head.column, &head, false);
            if (looks_like_mapping_entry(text)) {
                if (auto parsed = parse_mapping(head.column, &head, nullptr); !parsed) {
                    return std::unexpected(parsed.error());
                }
                return nested();
            }
        } else if (is_sequence_entry(text)) {
            return fail(head.number, "a block sequence must start on the line after its key");
        }
        return parse_scalar(head, parent);
    }

    /// A node on its own line below its key or entry.
    [[nodiscard]] Result<Value> parse_own_line(const Line& line, std::size_t parent) {
        if (is_sequence_entry(line.text)) return parse_sequence(line.column, &line, false);
        if (looks_like_mapping_entry(line.text)) {
            if (auto parsed = parse_mapping(line.column, &line, nullptr); !parsed) {
                return std::unexpected(parsed.error());
            }
            return nested();
        }
        return parse_value(line, parent, false);
    }

    [[nodiscard]] Result<Value> parse_alias(const Line& head) {
        const auto token = first_token(head.text);
        if (const auto rest = str::trim_left_view(head.text.substr(token.size()), kBlank); !rest.empty() && rest.front() != '#') {
            return fail(head.number, "unexpected content after an alias");
        }
        const auto name = token.substr(1);
        if (const auto it = anchors_.values.find(name); it != anchors_.values.end()) return it->second;
        if (anchors_.incomplete) return nested();
        return fail(head.number, std::format("undefined alias '{}'", name));
    }

    [[nodiscard]] Result<Value> parse_scalar(const Line& head, std::size_t parent) {
        switch (head.text.front()) {
            case '|':
            case '>': {
                const auto header = parse_block_header(head.text, head.number);
                if (!header) return std::unexpected(header.error());
                auto text = parse_block_scalar(*header, parent);
                if (!text) return std::unexpected(text.error());
                return scalar(std::move(*text));
            }
            case '"':
            case '\'':
            case '[':
            case '{':
                return parse_flow_node(head);
            case '?':
                if (head.text.size() == 1 || is_blank(head.text[1])) {
                    return fail(head.number, "complex mapping keys are not supported");
                }
                break;
            default:
                break;
        }
        if (const char* error = plain_start_error(head.text)) return fail(head.number, error);
        return scalar(parse_plain(head, parent));
    }

    /// A quoted scalar or flow collection; consumes every line it spans.
    [[nodiscard]] Result<Value> parse_flow_node(const Line& head) {
        Source source(head, lines_.subspan(pos_));
        std::size_t pos = 0;
        Value value;
        std::string_view what = "a quoted scalar";
        switch (head.text.front()) {
            case '"':
            case '\'': {
                auto text = head.text.front() == '"' ? parse_double_quoted(source, pos)
                                                     : parse_single_quoted(source, pos);
                if (!text) return std::unexpected(text.error());
                value = scalar(std::move(*text));
                break;
            }
            case '[': {
                auto sequence = parse_flow_sequence(source, pos, anchors_);
                if (!sequence) return std::unexpected(sequence.error());
                value = sequence->nested
                    ? nested()
                    : Value{.kind = Value::Kind::Sequence, .scalar = {}, .items = std::move(sequence->items)};
                what = "a flow sequence";
                break;
            }
            default: {
                if (auto skipped = skip_flow_collection(source, pos, anchors_); !skipped) {
                    return std::unexpected(skipped.error());
                }
                value = nested();
                what = "a flow mapping";
                break;
            }
        }

        // Only white space and a comment may follow on the closing line.
        auto after = pos;
        while (after < source.loaded_size() && is_blank(source[after])) ++after;
        if (after < source.loaded_size() && source[after] != '\n'
            && !(source[after] == '#' && after > pos)) {
            return fail(source.line_at(after), std::format("unexpected content after {}", what));
        }
        pos_ += source.segment_at(pos - 1);
        return value;
    }

    /// A plain scalar: @p head and the more-indented lines that continue it,
    /// folded. A comment ends it.
    [[nodiscard]] std::string parse_plain(const Line& head, std::size_t parent) {
        bool comment_seen = false;
        std::string out(strip_comment(head.text, comment_seen));
        std::size_t pending_breaks = 0;
        while (!comment_seen && pos_ < lines_.size()) {
            const auto& line = lines_[pos_];
            if (is_blank_line(line.text)) {
                ++pending_breaks;
                ++pos_;
                continue;
            }
            if (is_comment(line.text) || indent(line) <= parent) break;
            const auto text = strip_comment(str::trim_left_view(line.text, kBlank), comment_seen);
            if (pending_breaks == 0) out += ' ';
            else out.append(pending_breaks, '\n');
            out += text;
            pending_breaks = 0;
            ++pos_;
        }
        return out;
    }

    /// A block scalar's lines: blank, or indented at least as deep as the
    /// block (given by the header or by its first non-blank line).
    [[nodiscard]] Result<std::string> parse_block_scalar(const BlockHeader& header, std::size_t parent) {
        std::size_t block_indent = header.indent == 0 ? 0 : parent + header.indent;
        if (block_indent == 0) {
            for (auto i = pos_; i < lines_.size(); ++i) {
                if (is_blank_line(lines_[i].text)) continue;
                if (const auto first = indent(lines_[i]); first > parent) block_indent = first;
                break;
            }
        }

        std::vector<BlockLine> lines;
        while (pos_ < lines_.size()) {
            const auto& line = lines_[pos_];
            const auto text = line.text;
            const auto line_indent = indent(line);
            const bool spaces_only = text.find_first_not_of(' ') == npos;
            if (is_blank_line(text) && (block_indent == 0 || line_indent < block_indent
                                   || (spaces_only && text.size() <= block_indent))) {
                lines.push_back({.text = {}, .empty = true});
            } else if (block_indent != 0 && line_indent >= block_indent) {
                lines.push_back({.text = text.substr(block_indent), .empty = false});
            } else if (line_indent > parent && !is_comment(text)) {
                if (auto tab = check_tab(line); !tab) return std::unexpected(tab.error());
                return fail(line.number, "line is indented less than its block scalar");
            } else {
                break;
            }
            ++pos_;
        }
        return render_block(lines, header);
    }

    /// Far beyond any real manifest, far below what the stack can hold.
    static constexpr std::size_t kMaxDepth = 64;

    Lines lines_;
    std::size_t pos_ = 0;
    std::size_t depth_ = 0;
    Anchors anchors_;
};

[[nodiscard]] std::vector<Line> split_lines(std::string_view content) {
    std::vector<Line> lines;
    std::size_t start = 0;
    while (start <= content.size()) {
        const auto end = content.find('\n', start);
        auto text = content.substr(start, end == npos ? npos : end - start);
        if (!text.empty() && text.back() == '\r') text.remove_suffix(1);
        lines.push_back(Line{.text = text, .number = lines.size() + 1, .column = 0});
        if (end == npos) break;
        start = end + 1;
    }
    return lines;
}

[[nodiscard]] bool is_delimiter(std::string_view text, std::string_view marker) {
    return text.starts_with(marker) && is_blank_line(text.substr(marker.size()));
}

} // namespace

const Value* Document::find(std::string_view key) const {
    const auto it = fields.find(key);
    return it == fields.end() ? nullptr : &it->second;
}

std::optional<std::string> Document::get(std::string_view key) const {
    const auto* value = find(key);
    if (value == nullptr || value->kind != Value::Kind::Scalar) return std::nullopt;
    return value->scalar;
}

std::optional<std::vector<std::string>> Document::list(std::string_view key) const {
    const auto* value = find(key);
    if (value == nullptr) return std::nullopt;
    switch (value->kind) {
        case Value::Kind::Scalar:
            return split_list(value->scalar);
        case Value::Kind::Sequence: {
            std::vector<std::string> items;
            for (const auto& item : value->items) {
                if (const auto text = str::trim_ascii_view(item); !text.empty()) items.emplace_back(text);
            }
            return items;
        }
        case Value::Kind::Nested:
            break;
    }
    return std::nullopt;
}

std::expected<Document, ParseError> parse(std::string_view content) {
    if (content.starts_with("\xEF\xBB\xBF")) content.remove_prefix(3);
    const auto lines = split_lines(content);

    std::size_t open = 0;
    while (open < lines.size() && is_blank_line(lines[open].text)) ++open;
    if (open == lines.size() || !is_delimiter(lines[open].text, "---")) {
        return fail(open == lines.size() ? 1 : lines[open].number,
                    "expected a '---' line opening the frontmatter");
    }
    std::size_t close = open + 1;
    while (close < lines.size() && !is_delimiter(lines[close].text, "---")
           && !is_delimiter(lines[close].text, "...")) {
        ++close;
    }
    if (close == lines.size()) {
        return fail(lines[open].number, "frontmatter has no closing '---' line");
    }

    Document document;
    auto fields = Parser(Lines(lines).subspan(open + 1, close - open - 1)).parse();
    if (!fields) return std::unexpected(fields.error());
    document.fields = std::move(*fields);

    std::size_t body_start = close + 1;
    while (body_start < lines.size() && is_blank_line(lines[body_start].text)) ++body_start;
    for (std::size_t j = body_start; j < lines.size(); ++j) {
        if (j > body_start) document.body += '\n';
        document.body += lines[j].text;
    }
    document.body.resize(str::trim_right_view(document.body, " \t\r\n").size());
    return document;
}

std::optional<Document> parse_file(const fs::path& path) {
    const auto content = core::utils::read_file(path);
    if (!content.has_value()) {
        core::logging::warn("frontmatter: cannot open '{}'", path.string());
        return std::nullopt;
    }
    auto parsed = parse(*content);
    if (!parsed) {
        core::logging::warn("frontmatter: {}:{}: {}", path.string(), parsed.error().line,
                            parsed.error().message);
        return std::nullopt;
    }
    return std::move(*parsed);
}

std::optional<bool> parse_bool(std::string_view value) {
    const auto text = str::trim_ascii_view(value);
    const auto any_of = [&](std::initializer_list<std::string_view> spellings) {
        return std::ranges::any_of(spellings, [&](std::string_view spelling) {
            return core::utils::ascii::iequals(text, spelling);
        });
    };
    if (any_of({"true", "yes", "on", "1"})) return true;
    if (any_of({"false", "no", "off", "0"})) return false;
    return std::nullopt;
}

std::vector<std::string> split_list(std::string_view value) {
    const bool comma_separated = value.find(',') != npos;
    std::vector<std::string> items;
    std::size_t start = 0;
    while (start <= value.size()) {
        const auto end = comma_separated ? value.find(',', start)
                                         : value.find_first_of(" \t\r\n", start);
        const auto item = str::trim_ascii_view(value.substr(start, end == npos ? npos : end - start));
        if (!item.empty()) items.emplace_back(item);
        if (end == npos) break;
        start = end + 1;
    }
    return items;
}

FieldReader::FieldReader(const Document& document, fs::path source, std::string_view label)
    : document_(document), source_(std::move(source)), label_(label) {}

std::optional<std::string_view> FieldReader::present(Keys keys) const {
    for (const auto key : keys) {
        if (document_.find(key) != nullptr) return key;
    }
    return std::nullopt;
}

bool FieldReader::has_any(Keys keys) const { return present(keys).has_value(); }

std::optional<std::string> FieldReader::text(Keys keys) const {
    const auto key = present(keys);
    if (!key) return std::nullopt;
    const auto value = document_.get(*key);
    if (!value) {
        warn(std::format("'{}' must be text", *keys.begin()));
        return std::nullopt;
    }
    return std::string(str::trim_ascii_view(*value));
}

std::optional<bool> FieldReader::flag(Keys keys) const {
    const auto value = text(keys);
    if (!value) return std::nullopt;
    const auto parsed = parse_bool(*value);
    if (!parsed) warn(std::format("'{}' must be true or false", *keys.begin()));
    return parsed;
}

std::optional<int> FieldReader::positive_integer(Keys keys) const {
    const auto value = text(keys);
    if (!value) return std::nullopt;
    int parsed = 0;
    const auto* const end = value->data() + value->size();
    if (const auto [stop, ec] = std::from_chars(value->data(), end, parsed);
        ec != std::errc{} || stop != end || parsed <= 0) {
        warn(std::format("'{}' must be a positive integer", *keys.begin()));
        return std::nullopt;
    }
    return parsed;
}

std::optional<std::vector<std::string>> FieldReader::list(Keys keys) const {
    const auto key = present(keys);
    if (!key) return std::nullopt;
    auto value = document_.list(*key);
    if (!value) warn(std::format("'{}' must be a list", *keys.begin()));
    return value;
}

void FieldReader::warn(std::string_view problem) const {
    core::logging::warn("{} '{}': {}", label_, source_.string(), problem);
}

} // namespace core::tools::frontmatter

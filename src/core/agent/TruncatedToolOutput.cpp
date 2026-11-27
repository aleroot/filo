#include "TruncatedToolOutput.hpp"

#include "../utils/JsonUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <format>
#include <vector>

namespace core::agent::tool_output_history {
namespace {

[[nodiscard]] bool is_json_space(char c) noexcept {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

[[nodiscard]] bool is_json_object_or_array(std::string_view text) {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(simdjson::padded_string(text)).get(document) != simdjson::SUCCESS) {
        return false;
    }
    const auto type = document.type();
    return type == simdjson::dom::element_type::OBJECT
        || type == simdjson::dom::element_type::ARRAY;
}

[[nodiscard]] std::string size_label(std::size_t bytes) {
    if (bytes < 1024) return std::format("{} bytes", bytes);
    if (bytes < 1024 * 1024) return std::format("{:.0f} KB", static_cast<double>(bytes) / 1024.0);
    return std::format("{:.1f} MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
}

[[nodiscard]] bool is_high_surrogate_escape(std::string_view escape) {
    // `escape` is a complete "\uXXXX".
    if (escape.size() != 6) return false;
    const char a = static_cast<char>(escape[2] | 0x20);
    const char b = static_cast<char>(escape[3] | 0x20);
    return a == 'd' && b >= '8' && b <= 'b';
}

/// Length of the longest prefix of a JSON string's raw contents that ends on
/// a whole character: never inside an escape, never on half a surrogate pair.
[[nodiscard]] std::size_t complete_string_prefix(std::string_view contents) {
    std::size_t complete = 0;
    std::size_t pos = 0;
    while (pos < contents.size()) {
        if (contents[pos] != '\\') {
            complete = ++pos;
            continue;
        }
        const bool unicode = pos + 1 < contents.size() && contents[pos + 1] == 'u';
        const std::size_t length = unicode ? 6 : 2;
        if (pos + length > contents.size()) break;
        const bool dangling = unicode
            && is_high_surrogate_escape(contents.substr(pos, length))
            && pos + length == contents.size();
        if (dangling) break;
        pos += length;
        complete = pos;
    }
    return complete;
}

/// Where a JSON prefix can be closed into a valid document.
struct PrefixCut {
    /// Prefix length after the last complete value (or opening bracket).
    std::size_t safe_end = 0;
    std::string safe_closers;
    /// Set when the prefix ends inside a string value: where its contents
    /// start, what closes the containers around it, and whether it sits
    /// directly in the root object (the only place a tail can be joined).
    std::optional<std::size_t> open_string_start;
    std::string open_string_closers;
    bool open_string_at_root = false;
};

/// A forgiving single pass over a JSON prefix. It does not validate: the
/// rebuilt document is parsed afterwards, and a prefix that was never JSON
/// simply fails there.
[[nodiscard]] std::optional<PrefixCut> scan_json_prefix(std::string_view text) {
    enum class Expect { Value, Key, Colon, CommaOrEnd };
    struct Frame {
        char closer;
        Expect expect;
    };
    std::vector<Frame> stack;
    PrefixCut cut;
    bool seen_root = false;

    const auto closers = [&] {
        std::string out;
        for (auto it = stack.rbegin(); it != stack.rend(); ++it) out += it->closer;
        return out;
    };
    const auto mark_safe = [&](std::size_t end) {
        cut.safe_end = end;
        cut.safe_closers = closers();
    };
    const auto value_completed = [&](std::size_t end) {
        if (!stack.empty()) stack.back().expect = Expect::CommaOrEnd;
        mark_safe(end);
    };

    std::size_t pos = 0;
    while (pos < text.size()) {
        const char c = text[pos];
        if (is_json_space(c)) {
            ++pos;
            continue;
        }
        const bool expecting_key = !stack.empty() && stack.back().expect == Expect::Key;
        switch (c) {
            case '"': {
                std::size_t end = pos + 1;
                while (end < text.size() && text[end] != '"') {
                    end += text[end] == '\\' ? 2 : 1;
                }
                if (end >= text.size()) {
                    if (!expecting_key) {
                        cut.open_string_start = pos + 1;
                        cut.open_string_closers = closers();
                        cut.open_string_at_root =
                            stack.size() == 1 && stack.front().closer == '}';
                    }
                    return seen_root ? std::optional{cut} : std::nullopt;
                }
                seen_root = true;
                pos = end + 1;
                if (expecting_key) {
                    stack.back().expect = Expect::Colon;
                } else {
                    value_completed(pos);
                }
                continue;
            }
            case '{':
            case '[':
                stack.push_back({c == '{' ? '}' : ']', c == '{' ? Expect::Key : Expect::Value});
                seen_root = true;
                ++pos;
                // An empty array closes harmlessly, but an object opened for
                // an element would come back as a bogus `{}` row; only the
                // root's own brace is a place to stop.
                if (c == '[' || stack.size() == 1) mark_safe(pos);
                continue;
            case '}':
            case ']':
                if (stack.empty()) return std::nullopt;
                stack.pop_back();
                value_completed(++pos);
                continue;
            case ':':
                if (!stack.empty()) stack.back().expect = Expect::Value;
                ++pos;
                continue;
            case ',':
                if (!stack.empty()) {
                    stack.back().expect =
                        stack.back().closer == '}' ? Expect::Key : Expect::Value;
                }
                ++pos;
                continue;
            default: {
                // Number or literal. One that runs to the end of the prefix
                // may itself be cut ("tru", "12" of "1234"), so it is dropped.
                std::size_t end = pos;
                while (end < text.size()
                       && (std::isalnum(static_cast<unsigned char>(text[end]))
                           || text[end] == '-' || text[end] == '+' || text[end] == '.')) {
                    ++end;
                }
                if (end == pos) return std::nullopt;
                if (end >= text.size()) return seen_root ? std::optional{cut} : std::nullopt;
                seen_root = true;
                pos = end;
                value_completed(pos);
                continue;
            }
        }
    }
    return seen_root ? std::optional{cut} : std::nullopt;
}

/// True when `fragment` is the end of a string that closes directly in the
/// root object: after its closing quote come only the root's remaining members.
[[nodiscard]] bool closes_root_string(std::string_view fragment) {
    std::size_t pos = 0;
    while (pos < fragment.size() && fragment[pos] != '"') {
        pos += fragment[pos] == '\\' ? 2 : 1;
    }
    if (pos >= fragment.size()) return false;

    std::string_view rest = fragment.substr(pos + 1);
    while (!rest.empty() && is_json_space(rest.front())) rest.remove_prefix(1);
    while (!rest.empty() && is_json_space(rest.back())) rest.remove_suffix(1);
    std::string members;
    if (rest.starts_with(',')) {
        members = "{" + std::string(rest.substr(1));
    } else if (rest == "}") {
        members = "{}";
    } else {
        return false;
    }
    simdjson::dom::parser parser;
    simdjson::dom::object object;
    return parser.parse(simdjson::padded_string(members)).get(object) == simdjson::SUCCESS;
}

/// Where the tail's share of the cut root string starts, if it can be joined.
/// The tail was cut at an arbitrary byte, so it may open with the remains of
/// an escape ("n" of "\n", "00e9" of "\u00e9"). It is therefore joined
/// from its first line start when it has one: the reader sees whole lines on
/// both sides of the marker. Failing that, a leading quote is read as the
/// escaped character it most likely was rather than as the string's end.
[[nodiscard]] std::optional<std::size_t> tail_join_offset(std::string_view tail) {
    if (tail.empty()) return std::nullopt;
    const bool orphan_first = tail.front() == '"';
    for (const std::size_t offset : {orphan_first ? 1U : 0U, orphan_first ? 0U : 1U}) {
        if (offset > tail.size() || !closes_root_string(tail.substr(offset))) continue;
        for (std::size_t pos = offset; pos + 1 < tail.size() && tail[pos] != '"';) {
            if (tail[pos] != '\\') {
                ++pos;
                continue;
            }
            if (tail[pos + 1] == 'n') return pos + 2;
            pos += 2;
        }
        return offset;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> rebuild_json(std::string_view head,
                                                      std::string_view tail,
                                                      std::string_view marker) {
    const auto cut = scan_json_prefix(head);
    if (!cut) return std::nullopt;

    // Only a string directly in the root object (a command's output, a
    // file's content) is worth keeping half of. A half string deeper down
    // belongs to a list element that is better dropped whole than shown with
    // a cut path or id.
    if (cut->open_string_start && cut->open_string_at_root) {
        const std::size_t start = *cut->open_string_start;
        const std::string_view contents = head.substr(start);
        std::string prefix(head.substr(0, start));
        prefix += contents.substr(0, complete_string_prefix(contents));
        prefix += core::utils::escape_json_string(marker);

        if (const auto offset = tail_join_offset(tail)) {
            std::string joined = prefix + std::string(tail.substr(*offset));
            if (is_json_object_or_array(joined)) return joined;
        }
        std::string closed = std::move(prefix);
        closed += '"';
        closed += cut->open_string_closers;
        if (is_json_object_or_array(closed)) return closed;
    }

    std::string closed(head.substr(0, cut->safe_end));
    closed += cut->safe_closers;
    if (is_json_object_or_array(closed)) return closed;
    return std::nullopt;
}

} // namespace

std::string describe_omission(std::size_t omitted_chars) {
    return size_label(omitted_chars) + " omitted";
}

std::optional<TruncatedToolOutput> reassemble_truncated_output(std::string_view payload) {
    if (!payload.starts_with('{')) return std::nullopt;
    simdjson::dom::parser parser;
    simdjson::dom::object object;
    if (parser.parse(simdjson::padded_string(payload)).get(object) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    std::string_view head;
    std::string_view digest;
    std::int64_t original_chars = 0;
    std::int64_t kept_chars = 0;
    if (object[envelope::kHead].get(head) != simdjson::SUCCESS
        || object[envelope::kDigest].get(digest) != simdjson::SUCCESS
        || object[envelope::kOriginalChars].get(original_chars) != simdjson::SUCCESS
        || object[envelope::kKeptChars].get(kept_chars) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    bool truncated = false;
    const bool marked_truncated =
        object[envelope::kTruncated].get(truncated) == simdjson::SUCCESS && truncated;
    std::string_view error;
    const bool recorded_error = object[envelope::kError].get(error) == simdjson::SUCCESS;
    if (!marked_truncated && !recorded_error) return std::nullopt;

    std::string_view tail;
    core::utils::json::ignore_error(object[envelope::kTail].get(tail));

    TruncatedToolOutput out;
    out.original_chars = static_cast<std::size_t>(std::max<std::int64_t>(original_chars, 0));
    out.omitted_chars = static_cast<std::size_t>(std::max<std::int64_t>(original_chars - kept_chars, 0));
    const std::string marker = std::format("\n[… {} …]\n", describe_omission(out.omitted_chars));

    const std::size_t first = head.find_first_not_of(" \n\r\t");
    const bool json_head = first != std::string_view::npos
        && (head[first] == '{' || head[first] == '[');
    if (!json_head) {
        out.payload = std::string(head) + marker + std::string(tail);
        return out;
    }

    // The writer marks an envelope `error` whenever the text "error" appears
    // anywhere in the output, so that flag is not a verdict. The rebuilt
    // payload is judged like any whole one: by its own root `error`.
    auto rebuilt = rebuild_json(head, tail, marker);
    if (!rebuilt) return std::nullopt;
    out.payload = std::move(*rebuilt);
    return out;
}

} // namespace core::agent::tool_output_history

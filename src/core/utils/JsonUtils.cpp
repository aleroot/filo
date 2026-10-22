#include "JsonUtils.hpp"
#include "JsonWriter.hpp"
#include <simdjson.h>
#include <algorithm>   // std::find_if
#include <array>
#include <charconv>    // std::from_chars
#include <cstring>     // std::memchr
#include <cstdio>      // std::snprintf
#include <initializer_list>
#include <optional>
#include <ranges>

namespace core::utils {

namespace {

[[nodiscard]] bool is_utf8_continuation(unsigned char byte) noexcept {
    return (byte & 0xC0u) == 0x80u;
}

[[nodiscard]] std::size_t valid_utf8_sequence_length(const char* p,
                                                     const char* end) noexcept {
    const auto remaining = static_cast<std::size_t>(end - p);
    const auto b0 = static_cast<unsigned char>(p[0]);

    if (b0 >= 0xC2u && b0 <= 0xDFu) {
        return remaining >= 2
                && is_utf8_continuation(static_cast<unsigned char>(p[1]))
            ? 2
            : 0;
    }

    if (b0 >= 0xE0u && b0 <= 0xEFu) {
        if (remaining < 3) return 0;
        const auto b1 = static_cast<unsigned char>(p[1]);
        const auto b2 = static_cast<unsigned char>(p[2]);
        const bool valid_second =
            (b0 == 0xE0u && b1 >= 0xA0u && b1 <= 0xBFu)
            || (b0 == 0xEDu && b1 >= 0x80u && b1 <= 0x9Fu)
            || ((b0 >= 0xE1u && b0 <= 0xECu)
                && is_utf8_continuation(b1))
            || ((b0 >= 0xEEu && b0 <= 0xEFu)
                && is_utf8_continuation(b1));
        return valid_second && is_utf8_continuation(b2) ? 3 : 0;
    }

    if (b0 >= 0xF0u && b0 <= 0xF4u) {
        if (remaining < 4) return 0;
        const auto b1 = static_cast<unsigned char>(p[1]);
        const auto b2 = static_cast<unsigned char>(p[2]);
        const auto b3 = static_cast<unsigned char>(p[3]);
        const bool valid_second =
            (b0 == 0xF0u && b1 >= 0x90u && b1 <= 0xBFu)
            || ((b0 >= 0xF1u && b0 <= 0xF3u)
                && is_utf8_continuation(b1))
            || (b0 == 0xF4u && b1 >= 0x80u && b1 <= 0x8Fu);
        return valid_second
                && is_utf8_continuation(b2)
                && is_utf8_continuation(b3)
            ? 4
            : 0;
    }

    return 0;
}

} // namespace

void append_escaped_unchecked(std::string& out, std::string_view sv) {
    const char* p         = sv.data();
    const char* const end = p + sv.size();

    while (p != end) {
        const char* q = std::find_if(p, end,[](unsigned char c) noexcept { return kEscapeTable[c] != 0u; });

        // Bulk-append the clean run with a single memcpy-based append.
        out.append(p, q);
        if (q == end) break;

        switch (static_cast<unsigned char>(*q)) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default: {
                // Remaining control characters: emit \u00XX
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x",
                              static_cast<unsigned char>(*q));
                out.append(buf, 6);
            }
        }
        p = q + 1;
    }
}

void append_escaped(std::string& out, std::string_view sv) {
    // simdjson validates in vector-width chunks. Keeping validation separate
    // lets valid external data use the bulk escaper without per-code-point
    // branches; only malformed byte streams enter the repair path below.
    if (simdjson::validate_utf8(sv)) {
        append_escaped_unchecked(out, sv);
        return;
    }

    const char* p = sv.data();
    const char* const end = p + sv.size();
    while (p != end) {
        const char* q = std::find_if(
            p,
            end,
            [](unsigned char c) noexcept {
                return kEscapeTable[c] != 0u || c >= 0x80u;
            });
        out.append(p, q);
        if (q == end) break;

        const auto byte = static_cast<unsigned char>(*q);
        if (byte < 0x80u) {
            append_escaped_unchecked(out, std::string_view(q, 1));
            p = q + 1;
            continue;
        }

        const std::size_t sequence_length =
            valid_utf8_sequence_length(q, end);
        if (sequence_length == 0) {
            // JSON exchanged with providers must be Unicode. Preserve the
            // surrounding output while making an arbitrary byte stream safe
            // to serialize.
            out += "\\ufffd";
            p = q + 1;
        } else {
            out.append(q, sequence_length);
            p = q + sequence_length;
        }
    }
}

void append_escaped_utf8_safe(std::string& out, std::string_view sv) {
    append_escaped(out, sv);
}

void append_string_array(std::string& out, std::span<const std::string> values) {
    out.push_back('[');
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out.push_back(',');
        }
        out.push_back('"');
        append_escaped(out, values[i]);
        out.push_back('"');
    }
    out.push_back(']');
}

std::string repair_utf8(std::string_view sv) {
    if (simdjson::validate_utf8(sv)) return std::string(sv);

    static constexpr std::string_view kReplacement = "\xef\xbf\xbd";
    std::string out;
    out.reserve(sv.size());

    const char* p = sv.data();
    const char* const end = p + sv.size();
    while (p != end) {
        const char* q = std::find_if(
            p,
            end,
            [](unsigned char byte) noexcept { return byte >= 0x80u; });
        out.append(p, q);
        if (q == end) break;

        const std::size_t sequence_length =
            valid_utf8_sequence_length(q, end);
        if (sequence_length == 0) {
            out += kReplacement;
            p = q + 1;
        } else {
            out.append(q, sequence_length);
            p = q + sequence_length;
        }
    }
    return out;
}

namespace json {
namespace {

enum class RootKind {
    Invalid,
    Other,
    Object,
};

[[nodiscard]] std::string_view trim_whitespace(std::string_view input) noexcept {
    while (!input.empty() && is_whitespace(
               static_cast<unsigned char>(input.front()))) {
        input.remove_prefix(1);
    }
    while (!input.empty() && is_whitespace(
               static_cast<unsigned char>(input.back()))) {
        input.remove_suffix(1);
    }
    return input;
}

[[nodiscard]] RootKind root_kind(std::string_view input) {
    if (input.empty()) return RootKind::Invalid;

    thread_local simdjson::dom::parser parser;
    const ParserRetentionGuard parser_guard{parser};
    parser.number_as_string(true);

    simdjson::dom::element document;
    // Exact-length views have no SIMDJSON_PADDING bytes after them.
    // parse_unpadded (simdjson 5) reads only input.size() and does not copy.
    if (parser.parse_unpadded(input).get(document) != simdjson::SUCCESS) {
        return RootKind::Invalid;
    }
    return document.is_object() ? RootKind::Object : RootKind::Other;
}

} // namespace

bool is_valid(std::string_view input) {
    return root_kind(input) != RootKind::Invalid;
}

bool is_object(std::string_view input) {
    input = trim_whitespace(input);
    // Truncated streamed arguments usually fail this check; avoid allocating a
    // parser buffer proportional to their size just to reject them.
    if (input.size() < 2 || input.front() != '{' || input.back() != '}') {
        return false;
    }
    return root_kind(input) == RootKind::Object;
}

std::string_view object_or_empty(std::string_view input) {
    static constexpr std::string_view empty_object = "{}";
    return is_object(input) ? input : empty_object;
}

namespace {

/// Same limit as simdjson's DEFAULT_MAX_DEPTH, so both reject the same nesting.
constexpr std::size_t kMaxScanDepth = 1024;

/// Bytes that end the fast run inside a string: the closing quote, an
/// escape, or a control character JSON forbids unescaped.
inline constexpr auto kStringStopTable = [] noexcept {
    std::array<bool, 256> t{};
    for (unsigned i = 0; i < 0x20u; ++i) t[i] = true;
    t[static_cast<unsigned>('"')] = true;
    t[static_cast<unsigned>('\\')] = true;
    return t;
}();

[[nodiscard]] int hex_value(char c) noexcept {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

[[nodiscard]] bool read_hex4(const char* p, const char* end, std::uint32_t& out) noexcept {
    if (end - p < 4) return false;
    out = 0;
    for (int i = 0; i < 4; ++i) {
        const int digit = hex_value(p[i]);
        if (digit < 0) return false;
        out = (out << 4) | static_cast<std::uint32_t>(digit);
    }
    return true;
}

/// Decodes the escape at p (just past the backslash) into UTF-8 and advances
/// p past it. Returns 0 for an escape simdjson rejects: an unknown letter,
/// bad hex, or an unpaired surrogate.
[[nodiscard]] std::size_t decode_escape(const char*& p, const char* end, char out[4]) noexcept {
    if (p == end) return 0;
    switch (*p++) {
        case '"':  out[0] = '"';  return 1;
        case '\\': out[0] = '\\'; return 1;
        case '/':  out[0] = '/';  return 1;
        case 'b':  out[0] = '\b'; return 1;
        case 'f':  out[0] = '\f'; return 1;
        case 'n':  out[0] = '\n'; return 1;
        case 'r':  out[0] = '\r'; return 1;
        case 't':  out[0] = '\t'; return 1;
        case 'u':  break;
        default:   return 0;
    }
    std::uint32_t code_point = 0;
    if (!read_hex4(p, end, code_point)) return 0;
    p += 4;
    if (code_point >= 0xD800 && code_point < 0xDC00) {
        std::uint32_t low = 0;
        if (end - p < 6 || p[0] != '\\' || p[1] != 'u'
            || !read_hex4(p + 2, end, low) || low < 0xDC00 || low > 0xDFFF) {
            return 0;
        }
        p += 6;
        code_point = (((code_point - 0xD800) << 10) | (low - 0xDC00)) + 0x10000;
    } else if (code_point >= 0xDC00 && code_point <= 0xDFFF) {
        return 0;
    }
    if (code_point < 0x80) {
        out[0] = static_cast<char>(code_point);
        return 1;
    }
    if (code_point < 0x800) {
        out[0] = static_cast<char>(0xC0 | (code_point >> 6));
        out[1] = static_cast<char>(0x80 | (code_point & 0x3F));
        return 2;
    }
    if (code_point < 0x10000) {
        out[0] = static_cast<char>(0xE0 | (code_point >> 12));
        out[1] = static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
        out[2] = static_cast<char>(0x80 | (code_point & 0x3F));
        return 3;
    }
    out[0] = static_cast<char>(0xF0 | (code_point >> 18));
    out[1] = static_cast<char>(0x80 | ((code_point >> 12) & 0x3F));
    out[2] = static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
    out[3] = static_cast<char>(0x80 | (code_point & 0x3F));
    return 4;
}

/// p is on the opening quote; on success it is left just past the closing one.
[[nodiscard]] bool scan_string(const char*& p, const char* end) noexcept {
    ++p;
    while (true) {
        while (p != end && !kStringStopTable[static_cast<unsigned char>(*p)]) ++p;
        if (p == end) return false;
        if (*p == '"') {
            ++p;
            return true;
        }
        if (*p != '\\') return false;  // unescaped control character
        ++p;
        char ignored[4];
        if (decode_escape(p, end, ignored) == 0) return false;
    }
}

[[nodiscard]] bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

[[nodiscard]] bool scan_number(const char*& p, const char* end) noexcept {
    const char* const start = p;
    if (*p == '-') ++p;
    if (p == end || !is_digit(*p)) return false;
    const char* const integer = p;
    if (*p == '0') {
        ++p;
    } else {
        while (p != end && is_digit(*p)) ++p;
    }
    const char* const integer_end = p;
    const char* fraction = p;
    if (p != end && *p == '.') {
        ++p;
        if (p == end || !is_digit(*p)) return false;
        fraction = p;
        while (p != end && is_digit(*p)) ++p;
    }
    const char* const fraction_end = p;
    bool has_exponent = false;
    std::int64_t exponent = 0;
    if (p != end && (*p == 'e' || *p == 'E')) {
        has_exponent = true;
        ++p;
        const bool negative = p != end && *p == '-';
        if (p != end && (*p == '+' || *p == '-')) ++p;
        if (p == end || !is_digit(*p)) return false;
        const char* const exponent_digits = p;
        while (p != end && is_digit(*p)) ++p;
        // Saturate: past 2^62 the sign is all that matters, and adding the
        // leading-digit position below cannot overflow.
        if (!std::from_chars(exponent_digits, p, exponent)) exponent = 1LL << 62;
        if (negative) exponent = -exponent;
    }
    // Integers of any length are valid (simdjson keeps big ones as BIGINT);
    // a fraction or exponent is invalid only when it overflows a double.
    if (fraction == fraction_end && !has_exponent) return true;
    double value = 0;
    if (std::from_chars(start, p, value).ec != std::errc::result_out_of_range) return true;
    // Out of range is overflow or underflow (read as zero, which is fine). The
    // decimal position of the leading significant digit tells which.
    std::int64_t leading = integer_end - integer - 1;
    if (*integer == '0') {
        const char* const significant =
            std::ranges::find_if(fraction, fraction_end, [](char c) { return c != '0'; });
        if (significant == fraction_end) return true;  // zero
        leading = -(significant - fraction) - 1;
    }
    return leading + exponent < 0;
}

[[nodiscard]] bool scan_literal(const char*& p, const char* end, std::string_view word) noexcept {
    if (static_cast<std::size_t>(end - p) < word.size()
        || std::string_view(p, word.size()) != word) {
        return false;
    }
    p += word.size();
    return true;
}

void skip_whitespace(const char*& p, const char* end) noexcept {
    while (p != end && is_whitespace(static_cast<unsigned char>(*p))) ++p;
}

/// Whether the raw contents of a validated string token (quotes excluded)
/// decode to exactly `plain`.
[[nodiscard]] bool string_contents_equal(std::string_view raw, std::string_view plain) noexcept {
    if (raw.find('\\') == std::string_view::npos) return raw == plain;
    const char* p = raw.data();
    const char* const end = p + raw.size();
    std::size_t matched = 0;
    while (p != end) {
        if (*p != '\\') {
            if (matched == plain.size() || plain[matched] != *p) return false;
            ++matched;
            ++p;
            continue;
        }
        ++p;
        char decoded[4];
        const std::size_t length = decode_escape(p, end, decoded);
        if (length == 0 || plain.size() - matched < length
            || plain.substr(matched, length) != std::string_view(decoded, length)) {
            return false;
        }
        matched += length;
    }
    return matched == plain.size();
}

} // namespace

bool find_members(std::string_view input,
                  std::span<const std::string_view> keys,
                  std::span<RawMember> found) noexcept {
    for (auto& member : found) member = {};
    if (!simdjson::validate_utf8(input)) return false;

    const char* p = input.data();
    const char* const end = p + input.size();
    skip_whitespace(p, end);
    if (p == end || *p != '{') return false;

    // One bit per open container: set for arrays. Bounded like simdjson's.
    std::array<bool, kMaxScanDepth> is_array{};
    std::size_t depth = 0;
    std::string_view top_key;
    const char* top_value = nullptr;

    const auto record = [&](const char* value_end, ValueKind kind) {
        for (auto&& [key, slot] : std::views::zip(keys, found)) {
            if (slot.kind == ValueKind::Absent && string_contents_equal(top_key, key)) {
                slot = {.kind = kind,
                        .value = std::string_view(top_value, static_cast<std::size_t>(value_end - top_value))};
            }
        }
    };

    // Each iteration starts at a value: the root, a member value, or an element.
    bool expect_key = false;
    while (true) {
        if (expect_key) {
            if (p == end || *p != '"') return false;
            const char* key_start = p;
            if (!scan_string(p, end)) return false;
            if (depth == 1) {
                top_key = std::string_view(key_start + 1, static_cast<std::size_t>(p - key_start - 2));
            }
            skip_whitespace(p, end);
            if (p == end || *p != ':') return false;
            ++p;
            skip_whitespace(p, end);
            expect_key = false;
        }
        if (p == end) return false;
        if (depth == 1) top_value = p;

        ValueKind kind = ValueKind::Absent;
        bool scalar = true;
        switch (*p) {
            case '{':
            case '[': {
                if (depth == kMaxScanDepth) return false;
                const bool array = *p == '[';
                is_array[depth++] = array;
                ++p;
                skip_whitespace(p, end);
                if (p != end && *p == (array ? ']' : '}')) {
                    // An empty container is complete at once; fall through to
                    // the close handling below with it still open.
                    scalar = false;
                    break;
                }
                expect_key = !array;
                continue;
            }
            case '"':
                if (!scan_string(p, end)) return false;
                kind = ValueKind::String;
                break;
            case 't':
                if (!scan_literal(p, end, "true")) return false;
                kind = ValueKind::True;
                break;
            case 'f':
                if (!scan_literal(p, end, "false")) return false;
                kind = ValueKind::False;
                break;
            case 'n':
                if (!scan_literal(p, end, "null")) return false;
                kind = ValueKind::Null;
                break;
            default:
                if (!scan_number(p, end)) return false;
                kind = ValueKind::Number;
                break;
        }
        if (scalar && depth == 1) record(p, kind);

        // After a value: close as many containers as end here, then expect a
        // separator before the next value.
        while (true) {
            skip_whitespace(p, end);
            if (p == end) return false;
            const bool array = is_array[depth - 1];
            if (*p == (array ? ']' : '}')) {
                ++p;
                --depth;
                if (depth == 0) {
                    skip_whitespace(p, end);
                    return p == end;
                }
                if (depth == 1) record(p, array ? ValueKind::Array : ValueKind::Object);
                continue;
            }
            if (*p != ',') return false;
            ++p;
            skip_whitespace(p, end);
            expect_key = !array;
            break;
        }
    }
}

std::string decode_string(std::string_view raw_token) {
    std::string out;
    if (raw_token.size() < 2) return out;
    const std::string_view contents = raw_token.substr(1, raw_token.size() - 2);
    out.reserve(contents.size());
    const char* p = contents.data();
    const char* const end = p + contents.size();
    while (p != end) {
        const auto* backslash = static_cast<const char*>(
            std::memchr(p, '\\', static_cast<std::size_t>(end - p)));
        if (backslash == nullptr) {
            out.append(p, end);
            break;
        }
        out.append(p, backslash);
        p = backslash + 1;
        char decoded[4];
        out.append(decoded, decode_escape(p, end, decoded));
    }
    return out;
}

std::size_t decode_string_part(std::string_view& contents, std::span<char> out) noexcept {
    std::size_t written = 0;
    const char* p = contents.data();
    const char* const end = p + contents.size();
    while (p != end && written < out.size()) {
        if (*p == '\\') {
            const char* next = p + 1;
            char decoded[4];
            const std::size_t length = decode_escape(next, end, decoded);
            if (length > out.size() - written) break;  // the next part starts here
            std::memcpy(out.data() + written, decoded, length);
            written += length;
            p = next;
            continue;
        }
        const auto* backslash = static_cast<const char*>(
            std::memchr(p, '\\', static_cast<std::size_t>(end - p)));
        const std::size_t run = std::min(
            static_cast<std::size_t>((backslash ? backslash : end) - p), out.size() - written);
        std::memcpy(out.data() + written, p, run);
        written += run;
        p += run;
    }
    contents.remove_prefix(static_cast<std::size_t>(p - contents.data()));
    return written;
}

std::optional<std::int64_t> decode_int64(std::string_view raw_token) noexcept {
    // simdjson's DOM stores fractions and exponents as doubles, and integers
    // beyond int64 as uint64 or big integers: none of them are int64.
    if (raw_token.empty() || raw_token.find_first_of(".eE") != std::string_view::npos) {
        return std::nullopt;
    }
    const char* const end = raw_token.data() + raw_token.size();
    std::int64_t value = 0;
    if (const auto parsed = std::from_chars(raw_token.data(), end, value); !parsed || parsed.ptr != end) {
        return std::nullopt;
    }
    return value;
}

std::optional<RawMember> first_member(
    std::string_view raw_json,
    std::initializer_list<std::string_view> keys,
    bool (*accept)(const RawMember&) noexcept) noexcept {
    // Callers name a handful of aliases; look them up in one pass per chunk.
    constexpr std::size_t kChunk = 8;
    const std::span<const std::string_view> all(keys.begin(), keys.size());
    for (std::size_t offset = 0; offset < all.size(); offset += kChunk) {
        const auto chunk = all.subspan(offset, std::min(kChunk, all.size() - offset));
        std::array<RawMember, kChunk> found{};
        if (!find_members(raw_json, chunk, found)) return std::nullopt;
        for (std::size_t i = 0; i < chunk.size(); ++i) {
            if (accept(found[i])) return found[i];
        }
    }
    return std::nullopt;
}

} // namespace json

namespace {

using core::utils::json::schema::RootRole;

enum class SchemaType {
    String,
    Boolean,
    Integer,
    Number,
    Object,
    Array,
    Null,
};

[[nodiscard]] std::string_view type_name(SchemaType type) noexcept {
    switch (type) {
        case SchemaType::String: return "string";
        case SchemaType::Boolean: return "boolean";
        case SchemaType::Integer: return "integer";
        case SchemaType::Number: return "number";
        case SchemaType::Object: return "object";
        case SchemaType::Array: return "array";
        case SchemaType::Null: return "null";
    }
    return "string";
}

[[nodiscard]] bool has_keyword(simdjson::dom::object object,
                               std::string_view key) {
    simdjson::dom::element value;
    return object[key].get(value) == simdjson::SUCCESS;
}

[[nodiscard]] bool has_any_keyword(
    simdjson::dom::object object,
    std::initializer_list<std::string_view> keys) {
    for (std::string_view key : keys) {
        if (has_keyword(object, key)) return true;
    }
    return false;
}

[[nodiscard]] std::optional<SchemaType> infer_type_from_json_value(
    simdjson::dom::element value) {
    switch (value.type()) {
        case simdjson::dom::element_type::STRING: return SchemaType::String;
        case simdjson::dom::element_type::BOOL: return SchemaType::Boolean;
        case simdjson::dom::element_type::INT64: return SchemaType::Integer;
        case simdjson::dom::element_type::UINT64: return SchemaType::Integer;
        case simdjson::dom::element_type::BIGINT: return SchemaType::Integer;
        case simdjson::dom::element_type::DOUBLE: return SchemaType::Number;
        case simdjson::dom::element_type::OBJECT: return SchemaType::Object;
        case simdjson::dom::element_type::ARRAY: return SchemaType::Array;
        case simdjson::dom::element_type::NULL_VALUE: return SchemaType::Null;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<SchemaType> infer_common_type_from_values(
    simdjson::dom::array values) {
    std::optional<SchemaType> inferred;
    for (simdjson::dom::element value : values) {
        const auto current = infer_type_from_json_value(value);
        if (!current.has_value()) return std::nullopt;
        if (!inferred.has_value()) {
            inferred = *current;
            continue;
        }
        if (*inferred == *current) continue;
        if ((*inferred == SchemaType::Integer && *current == SchemaType::Number)
            || (*inferred == SchemaType::Number && *current == SchemaType::Integer)) {
            inferred = SchemaType::Number;
            continue;
        }
        return std::nullopt;
    }
    return inferred;
}

[[nodiscard]] std::optional<SchemaType> infer_type_from_structural_keywords(
    simdjson::dom::object object) {
    if (has_any_keyword(object, {
            "properties", "additionalProperties", "patternProperties",
            "propertyNames", "required", "minProperties", "maxProperties"})) {
        return SchemaType::Object;
    }
    if (has_any_keyword(object, {
            "items", "prefixItems", "minItems", "maxItems",
            "uniqueItems", "contains"})) {
        return SchemaType::Array;
    }
    if (has_any_keyword(object, {"minLength", "maxLength", "pattern", "format"})) {
        return SchemaType::String;
    }
    if (has_any_keyword(object, {
            "minimum", "maximum", "multipleOf", "exclusiveMinimum",
            "exclusiveMaximum"})) {
        return SchemaType::Number;
    }
    return std::nullopt;
}

[[nodiscard]] bool has_schema_combinator(simdjson::dom::object object) {
    return has_any_keyword(object, {
        "anyOf", "oneOf", "allOf", "not", "if", "then", "else", "$ref"});
}

[[nodiscard]] SchemaType infer_property_type(simdjson::dom::object object) {
    simdjson::dom::array enum_values;
    if (object["enum"].get(enum_values) == simdjson::SUCCESS && enum_values.size() > 0) {
        return infer_common_type_from_values(enum_values).value_or(SchemaType::String);
    }

    simdjson::dom::element const_value;
    if (object["const"].get(const_value) == simdjson::SUCCESS) {
        return infer_type_from_json_value(const_value).value_or(SchemaType::String);
    }

    return infer_type_from_structural_keywords(object).value_or(SchemaType::String);
}

void write_schema_with_property_types(JsonWriter& writer,
                                      simdjson::dom::element element,
                                      bool normalize_property);

void write_schema_array_with_property_types(JsonWriter& writer,
                                            simdjson::dom::array array,
                                            bool normalize_items) {
    auto scope = writer.array();
    bool first = true;
    for (simdjson::dom::element item : array) {
        if (!first) writer.comma();
        write_schema_with_property_types(writer, item, normalize_items);
        first = false;
    }
}

void write_properties_object_with_property_types(JsonWriter& writer,
                                                 simdjson::dom::object object) {
    auto scope = writer.object();
    bool first = true;
    for (auto field : object) {
        if (!first) writer.comma();
        writer.key(field.key);
        write_schema_with_property_types(writer, field.value, true);
        first = false;
    }
}

void write_schema_object_with_property_types(JsonWriter& writer,
                                             simdjson::dom::object object,
                                             bool normalize_property) {
    auto scope = writer.object();
    bool first = true;
    bool has_type = false;
    const bool has_combinator = has_schema_combinator(object);
    for (auto field : object) {
        has_type = has_type || field.key == "type";
        if (!first) writer.comma();
        writer.key(field.key);

        simdjson::dom::object child_obj;
        simdjson::dom::array child_arr;
        if (field.key == "properties"
            && field.value.get(child_obj) == simdjson::SUCCESS) {
            write_properties_object_with_property_types(writer, child_obj);
        } else if (field.key == "items") {
            if (field.value.get(child_obj) == simdjson::SUCCESS) {
                write_schema_object_with_property_types(writer, child_obj, true);
            } else if (field.value.get(child_arr) == simdjson::SUCCESS) {
                write_schema_array_with_property_types(writer, child_arr, true);
            } else {
                writer.raw(simdjson::to_string(field.value));
            }
        } else if (field.key == "additionalProperties"
            && field.value.get(child_obj) == simdjson::SUCCESS) {
            write_schema_object_with_property_types(writer, child_obj, true);
        } else if ((field.key == "anyOf" || field.key == "oneOf" || field.key == "allOf")
            && field.value.get(child_arr) == simdjson::SUCCESS) {
            write_schema_array_with_property_types(writer, child_arr, true);
        } else {
            writer.raw(simdjson::to_string(field.value));
        }
        first = false;
    }

    if (normalize_property && !has_type && !has_combinator) {
        if (!first) writer.comma();
        writer.kv_str("type", type_name(infer_property_type(object)));
    }
}

void write_schema_with_property_types(JsonWriter& writer,
                                      simdjson::dom::element element,
                                      bool normalize_property) {
    simdjson::dom::object object;
    if (element.get(object) == simdjson::SUCCESS) {
        write_schema_object_with_property_types(writer, object, normalize_property);
        return;
    }

    writer.raw(simdjson::to_string(element));
}

} // namespace

namespace json::schema {

std::string ensure_property_types(std::string_view schema,
                                  RootRole root_role) {
    simdjson::dom::parser parser;
    parser.number_as_string(true);
    simdjson::padded_string padded(schema);
    simdjson::dom::element root;
    if (parser.parse(padded).get(root) != simdjson::SUCCESS) {
        return std::string(schema);
    }

    JsonWriter writer(schema.size() + 128);
    write_schema_with_property_types(
        writer, root, root_role == RootRole::Property);
    return std::move(writer).take();
}

} // namespace json::schema

} // namespace core::utils

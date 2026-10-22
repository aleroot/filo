#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <array>

#include <simdjson.h>

namespace core::utils {

// ---------------------------------------------------------------------------
// Compile-time escape classification table (C++20 constexpr lambda).
//
// kEscapeTable[c] != 0  iff  the byte c must be escaped in a JSON string.
// Stored as inline constexpr so every TU shares a single definition with no
// ODR violation and the compiler can inline all accesses to it.
// ---------------------------------------------------------------------------
inline constexpr auto kEscapeTable = [] noexcept {
    std::array<unsigned char, 256> t{};
    for (unsigned i = 0; i < 0x20u; ++i) t[i] = 1u;   // control characters
    t[static_cast<unsigned>('"')]  = 1u;
    t[static_cast<unsigned>('\\')] = 1u;
    return t;
}();

// ---------------------------------------------------------------------------
// append_escaped — safe core primitive.
//
// Appends the JSON-escaped form of sv directly into out and guarantees valid
// UTF-8. Validation uses simdjson's vectorized fast path; malformed bytes take
// the rare repair path and are represented as U+FFFD.
// ---------------------------------------------------------------------------
void append_escaped(std::string& out, std::string_view sv);

// Explicit opt-out for strings whose UTF-8 validity is already guaranteed.
// This performs JSON syntax escaping only. Prefer append_escaped() unless the
// caller owns and proves the invariant.
void append_escaped_unchecked(std::string& out, std::string_view sv);

// Backward-compatible spelling retained for callers that document an external
// byte boundary. It has the same behavior as append_escaped().
void append_escaped_utf8_safe(std::string& out, std::string_view sv);

// ---------------------------------------------------------------------------
// append_string_array — ["a","b"], each value escaped.
//
// Callers that need a field write the name and the colon first: whether the
// pair is preceded by a comma belongs to the document being built, not here.
// ---------------------------------------------------------------------------
void append_string_array(std::string& out, std::span<const std::string> values);

// Returns valid UTF-8, replacing every malformed byte with U+FFFD. Valid input
// is copied without per-code-point decoding after a vectorized validation pass.
[[nodiscard]] std::string repair_utf8(std::string_view sv);

// ---------------------------------------------------------------------------
// escape_json_string — convenience wrapper.
// Accepts std::string_view: implicit conversions from std::string and
// const char* work without any extra overloads.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string escape_json_string(std::string_view sv) {
    std::string out;
    out.reserve(sv.size() + sv.size() / 8);
    append_escaped(out, sv);
    return out;
}

[[nodiscard]] inline std::string escape_json_string_utf8_safe(
    std::string_view sv) {
    std::string out;
    out.reserve(sv.size() + sv.size() / 8);
    append_escaped_utf8_safe(out, sv);
    return out;
}

namespace json {

/// JSON permits exactly these four whitespace characters between tokens.
[[nodiscard]] constexpr bool is_whitespace(unsigned char ch) noexcept {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
}

/// Return whether simdjson accepts input as one complete JSON value.
[[nodiscard]] bool is_valid(std::string_view input);

/// Return whether input parses as a JSON object.
[[nodiscard]] bool is_object(std::string_view input);

/// Return the input object when valid, or a view of "{}" otherwise. The
/// returned view borrows input on success; the fallback has static storage.
[[nodiscard]] std::string_view object_or_empty(std::string_view input);

// ---------------------------------------------------------------------------
// find_members — read top-level fields of a JSON object without a parser.
//
// A DOM parse sizes its tape and string buffers to the whole document, so
// asking a multi-megabyte tool result or argument payload for one small field
// touched memory proportional to the payload, several times per tool call.
// find_members validates the whole document in one pass, as strictly as
// simdjson does for grammar, UTF-8, escapes and nesting depth, and reports
// where the first occurrence of each requested top-level key sits in the
// input. It allocates nothing; values stay raw JSON text.
// ---------------------------------------------------------------------------
enum class ValueKind : std::uint8_t {
    Absent,
    Object,
    Array,
    String,
    Number,
    True,
    False,
    Null,
};

struct RawMember {
    ValueKind kind = ValueKind::Absent;
    /// Raw JSON text of the value; a string keeps its quotes and escapes.
    std::string_view value;
};

/// Returns false unless input is one complete JSON document whose root is an
/// object. found[i] receives the first top-level member named keys[i]; keys
/// compare after unescaping, as simdjson's object lookup does. found is only
/// meaningful when the call returns true.
[[nodiscard]] bool find_members(std::string_view input,
                                std::span<const std::string_view> keys,
                                std::span<RawMember> found) noexcept;

/// Decodes a string token that find_members reported (quotes included).
[[nodiscard]] std::string decode_string(std::string_view raw_token);

/// Decodes the next part of a validated string's @p contents (the token
/// without its quotes) into @p out, as much as fits, and drops that part from
/// @p contents. Returns the bytes written; @p out must hold at least 4. Lets a
/// caller that only forwards the text, to a file say, never hold all of it.
[[nodiscard]] std::size_t decode_string_part(std::string_view& contents,
                                             std::span<char> out) noexcept;

/// Reads a number token that find_members reported as int64, exactly as
/// simdjson's DOM get<int64_t>: integers that fit, nothing else.
[[nodiscard]] std::optional<std::int64_t> decode_int64(std::string_view raw_token) noexcept;

/// What an ondemand parser can iterate for @p text without copying it: the
/// string itself when its spare capacity already holds SIMDJSON_PADDING bytes
/// (simdjson's own std::string view does the same), else a padded copy kept
/// in @p storage. Request lines can be megabytes.
[[nodiscard]] inline simdjson::padded_string_view padded_view(const std::string& text,
                                                              simdjson::padded_string& storage) {
    if (text.capacity() - text.size() >= simdjson::SIMDJSON_PADDING) {
        return simdjson::padded_string_view(text.data(), text.size(), text.capacity());
    }
    storage = simdjson::padded_string(text);
    return storage;
}

// ---------------------------------------------------------------------------
// ignore_error — deliberately discard an optional simdjson lookup.
//
// simdjson marks simdjson_result::get() with warn_unused_result. GCC, unlike
// clang, does not accept `(void)` / static_cast<void> as an acknowledgement of
// that attribute, so every "absent key keeps the fallback" site has to feed the
// error code to a real call instead. Use this whenever the lookup failing is a
// valid outcome; check the code explicitly when it is not.
// ---------------------------------------------------------------------------
inline void ignore_error(simdjson::error_code) noexcept {}

// ---------------------------------------------------------------------------
// ParserRetentionGuard — bound what a reused simdjson parser keeps resident.
//
// A simdjson parser sizes its buffers to the largest document it has parsed
// (about 13x the input for dom, 5x for ondemand) and never shrinks them. A
// thread_local parser therefore pins its high-water mark for the life of the
// thread: one 5 MB tool result left 75 MB resident in a long-running server.
//
// Declare the guard immediately after the parser. When the scope ends, a
// parser that grew past kMaxRetainedCapacity is replaced by a fresh one, which
// releases its buffers. Documents under the limit keep reusing the warm
// buffers, so the common path allocates nothing. Only valid when nothing the
// scope returns still refers into the parser's memory.
// ---------------------------------------------------------------------------
template <typename Parser>
class ParserRetentionGuard {
public:
    /// Input bytes whose buffers may stay resident between parses.
    static constexpr std::size_t kMaxRetainedCapacity = 64 * 1024;

    explicit ParserRetentionGuard(Parser& parser) noexcept : parser_(parser) {}
    ParserRetentionGuard(const ParserRetentionGuard&) = delete;
    ParserRetentionGuard& operator=(const ParserRetentionGuard&) = delete;

    ~ParserRetentionGuard() {
        if (parser_.capacity() > kMaxRetainedCapacity) parser_ = Parser{};
    }

private:
    Parser& parser_;
};

[[nodiscard]] inline bool bool_field(simdjson::dom::object object,
                                     std::string_view key,
                                     bool fallback = false) {
    bool value = fallback;
    ignore_error(object[key].get(value));
    return value;
}

[[nodiscard]] inline std::string string_field(simdjson::dom::object object,
                                              std::string_view key,
                                              std::string_view fallback = {}) {
    std::string_view value;
    if (object[key].get(value) == simdjson::SUCCESS) {
        return std::string(value);
    }
    return std::string(fallback);
}

[[nodiscard]] inline std::optional<std::string>
optional_string_field(const simdjson::dom::object& object,
                      std::string_view key) {
    std::string_view value;
    if (object[key].get(value) == simdjson::SUCCESS) {
        return std::string(value);
    }
    return std::nullopt;
}

[[nodiscard]] inline std::optional<std::string>
first_string_field(const simdjson::dom::object& object,
                   std::initializer_list<std::string_view> keys) {
    for (const auto key : keys) {
        if (auto value = optional_string_field(object, key); value.has_value()) {
            return value;
        }
    }
    return std::nullopt;
}

[[nodiscard]] inline std::optional<std::string>
first_string_field(const simdjson::dom::object& object) {
    for (const auto field : object) {
        std::string_view value;
        if (field.value.get(value) == simdjson::SUCCESS) {
            return std::string(value);
        }
    }
    return std::nullopt;
}

[[nodiscard]] inline std::string first_string_field_or_empty(
    const simdjson::dom::object& object,
    std::initializer_list<std::string_view> keys) {
    return first_string_field(object, keys).value_or(std::string{});
}

[[nodiscard]] inline int64_t int64_field(simdjson::dom::object object,
                                         std::string_view key,
                                         int64_t fallback = 0) {
    int64_t value = fallback;
    ignore_error(object[key].get(value));
    return value;
}

[[nodiscard]] inline int int_field(simdjson::dom::object object,
                                   std::string_view key,
                                   int fallback = 0) {
    return static_cast<int>(int64_field(object, key, fallback));
}

[[nodiscard]] inline std::optional<int>
optional_int_field_clamped(const simdjson::dom::object& object, std::string_view key) {
    int64_t signed_value = 0;
    if (object[key].get(signed_value) == simdjson::SUCCESS) {
        if (signed_value > std::numeric_limits<int>::max()) {
            return std::numeric_limits<int>::max();
        }
        if (signed_value < std::numeric_limits<int>::min()) {
            return std::numeric_limits<int>::min();
        }
        return static_cast<int>(signed_value);
    }

    uint64_t unsigned_value = 0;
    if (object[key].get(unsigned_value) == simdjson::SUCCESS) {
        if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            return std::numeric_limits<int>::max();
        }
        return static_cast<int>(unsigned_value);
    }

    return std::nullopt;
}

[[nodiscard]] inline std::optional<int64_t>
first_int64_field(const simdjson::dom::object& object,
                  std::initializer_list<std::string_view> keys) {
    for (const auto key : keys) {
        int64_t value = 0;
        if (object[key].get(value) == simdjson::SUCCESS) {
            return value;
        }
    }
    return std::nullopt;
}

// The raw_json overloads below read a field without parsing the document
// into a DOM: see find_members. They return the fallback for anything that is
// not a valid JSON object, like the dom::object overloads' callers did.

[[nodiscard]] inline std::string string_field(std::string_view raw_json,
                                              std::string_view key,
                                              std::string_view fallback = {}) {
    const std::string_view keys[] = {key};
    RawMember found[1];
    if (!find_members(raw_json, keys, found) || found[0].kind != ValueKind::String) {
        return std::string(fallback);
    }
    return decode_string(found[0].value);
}

[[nodiscard]] inline bool bool_field(std::string_view raw_json,
                                     std::string_view key,
                                     bool fallback = false) {
    const std::string_view keys[] = {key};
    RawMember found[1];
    if (!find_members(raw_json, keys, found)) return fallback;
    if (found[0].kind == ValueKind::True) return true;
    if (found[0].kind == ValueKind::False) return false;
    return fallback;
}

/// The first of keys, in order, whose member satisfies accept; nullopt when
/// none does or raw_json is not a valid object.
[[nodiscard]] std::optional<RawMember> first_member(
    std::string_view raw_json,
    std::initializer_list<std::string_view> keys,
    bool (*accept)(const RawMember&) noexcept) noexcept;

[[nodiscard]] inline std::optional<std::string>
first_string_field(std::string_view raw_json,
                   std::initializer_list<std::string_view> keys) {
    const auto member = first_member(raw_json, keys, [](const RawMember& m) noexcept {
        return m.kind == ValueKind::String;
    });
    if (!member) return std::nullopt;
    return decode_string(member->value);
}

[[nodiscard]] inline std::string first_string_field_or_empty(
    std::string_view raw_json,
    std::initializer_list<std::string_view> keys) {
    return first_string_field(raw_json, keys).value_or(std::string{});
}

[[nodiscard]] inline std::optional<int64_t>
first_int64_field(std::string_view raw_json,
                  std::initializer_list<std::string_view> keys) {
    const auto member = first_member(raw_json, keys, [](const RawMember& m) noexcept {
        return m.kind == ValueKind::Number && decode_int64(m.value).has_value();
    });
    if (!member) return std::nullopt;
    return decode_int64(member->value);
}

[[nodiscard]] inline int64_t first_int64_field_or_zero(
    std::string_view raw_json,
    std::initializer_list<std::string_view> keys) {
    return first_int64_field(raw_json, keys).value_or(0);
}

namespace schema {

enum class RootRole {
    Container,
    Property,
};

[[nodiscard]] std::string ensure_property_types(
    std::string_view schema,
    RootRole root_role = RootRole::Container);

} // namespace json::schema
} // namespace json

} // namespace core::utils

#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
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

[[nodiscard]] inline bool bool_field(simdjson::dom::object object,
                                     std::string_view key,
                                     bool fallback = false) {
    bool value = fallback;
    static_cast<void>(object[key].get(value));
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
    static_cast<void>(object[key].get(value));
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

[[nodiscard]] inline std::string string_field(std::string_view raw_json,
                                              std::string_view key,
                                              std::string_view fallback = {}) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(raw_json.data(), raw_json.size()).get(doc) != simdjson::SUCCESS) {
        return std::string(fallback);
    }
    simdjson::dom::object object;
    if (doc.get(object) != simdjson::SUCCESS) {
        return std::string(fallback);
    }
    return string_field(object, key, fallback);
}

[[nodiscard]] inline bool bool_field(std::string_view raw_json,
                                     std::string_view key,
                                     bool fallback = false) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(raw_json.data(), raw_json.size()).get(doc) != simdjson::SUCCESS) {
        return fallback;
    }
    simdjson::dom::object object;
    if (doc.get(object) != simdjson::SUCCESS) {
        return fallback;
    }
    return bool_field(object, key, fallback);
}

[[nodiscard]] inline std::optional<std::string>
first_string_field(std::string_view raw_json,
                   std::initializer_list<std::string_view> keys) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(raw_json.data(), raw_json.size()).get(doc) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    simdjson::dom::object object;
    if (doc.get(object) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    return first_string_field(object, keys);
}

[[nodiscard]] inline std::string first_string_field_or_empty(
    std::string_view raw_json,
    std::initializer_list<std::string_view> keys) {
    return first_string_field(raw_json, keys).value_or(std::string{});
}

[[nodiscard]] inline std::optional<int64_t>
first_int64_field(std::string_view raw_json,
                  std::initializer_list<std::string_view> keys) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(raw_json.data(), raw_json.size()).get(doc) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    simdjson::dom::object object;
    if (doc.get(object) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    return first_int64_field(object, keys);
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

#pragma once

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
// append_escaped — core primitive.
//
// Appends the JSON-escaped form of sv directly into out.
// The inner scan loop uses std::find_if with a table-lookup predicate; with
// -O3 -march=native the compiler vectorises this to SSE2/AVX2, processing
// 16–32 bytes per iteration before falling back to the per-character escape
// path.  Clean chunks are bulk-copied with std::string::append.
// ---------------------------------------------------------------------------
void append_escaped(std::string& out, std::string_view sv);

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

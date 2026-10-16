#pragma once

// ---------------------------------------------------------------------------
// JsonValue — a small, mutable, order-preserving JSON tree.
//
// The rest of Filo reads JSON with simdjson (immutable DOM) and writes it with
// JsonWriter (append-only). Neither can express "load this file, replace one
// subtree, write it back", which is exactly what configuration import needs.
// JsonValue fills that gap and nothing more:
//
//  • Objects keep their members in document order, so a round-tripped
//    config.json diffs only where it was actually changed.
//  • Numbers are kept as text: integers exactly as written, floating-point
//    values in simdjson's shortest round-trip form (1e3 becomes 1000.0), so a
//    value never changes even when its spelling does.
//  • Duplicate keys collapse to the first one, which is the one simdjson's
//    lookups (and so the rest of Filo) read.
// ---------------------------------------------------------------------------

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace core::utils {

class JsonValue {
public:
    struct Number {
        std::string text;
        bool operator==(const Number&) const = default;
    };
    using Array = std::vector<JsonValue>;
    using Member = std::pair<std::string, JsonValue>;
    using Object = std::vector<Member>;

    JsonValue() = default;  // null
    JsonValue(bool value) : value_(value) {}
    JsonValue(Number value) : value_(std::move(value)) {}
    JsonValue(std::string value) : value_(std::move(value)) {}
    JsonValue(std::string_view value) : value_(std::string(value)) {}
    JsonValue(const char* value) : value_(std::string(value)) {}
    JsonValue(Array value) : value_(std::move(value)) {}
    JsonValue(Object value) : value_(std::move(value)) {}

    [[nodiscard]] static JsonValue object() { return JsonValue(Object{}); }
    [[nodiscard]] static JsonValue array() { return JsonValue(Array{}); }
    [[nodiscard]] static JsonValue integer(long long value);

    [[nodiscard]] bool is_null() const noexcept { return value_.index() == 0; }
    [[nodiscard]] bool is_bool() const noexcept { return std::holds_alternative<bool>(value_); }
    [[nodiscard]] bool is_number() const noexcept { return std::holds_alternative<Number>(value_); }
    [[nodiscard]] bool is_string() const noexcept { return std::holds_alternative<std::string>(value_); }
    [[nodiscard]] bool is_array() const noexcept { return std::holds_alternative<Array>(value_); }
    [[nodiscard]] bool is_object() const noexcept { return std::holds_alternative<Object>(value_); }

    /// Typed access. Each returns the fallback when the value has another type.
    [[nodiscard]] bool as_bool(bool fallback = false) const noexcept;
    [[nodiscard]] std::string_view as_string(std::string_view fallback = {}) const noexcept;
    [[nodiscard]] std::optional<long long> as_integer() const noexcept;

    /// Container access; calling these on another type is a logic error.
    [[nodiscard]] Array& items();
    [[nodiscard]] const Array& items() const;
    [[nodiscard]] Object& members();
    [[nodiscard]] const Object& members() const;

    /// Object member lookup; nullptr when absent or when this is not an object.
    [[nodiscard]] JsonValue* find(std::string_view key) noexcept;
    [[nodiscard]] const JsonValue* find(std::string_view key) const noexcept;

    /// Replaces the member in place (keeping its position) or appends it.
    /// A non-object value is turned into an empty object first.
    JsonValue& set(std::string_view key, JsonValue value);
    /// Removes the member; returns whether it existed.
    bool erase(std::string_view key);

    /// RFC 6901 JSON Pointers ("/a/b~1c" addresses key "b/c" inside "a").
    /// `pointer_token` escapes one key so it can be appended after a '/'.
    [[nodiscard]] static std::string pointer_token(std::string_view key);
    /// nullptr when the pointer is malformed or nothing is there. Array
    /// elements are addressed by index.
    [[nodiscard]] const JsonValue* at_pointer(std::string_view pointer) const;
    /// Stores `value` at the pointer, creating (or replacing non-object)
    /// intermediate members with objects; only objects are descended. The
    /// empty pointer replaces this value. nullptr when the pointer is malformed.
    JsonValue* set_at_pointer(std::string_view pointer, JsonValue value);

    /// Serialization. indent == 0 produces compact output; otherwise members
    /// and elements go on their own lines indented by `indent` spaces.
    [[nodiscard]] std::string dump(int indent = 2) const;

    /// Parses a complete JSON document. Returns nullopt and sets `error` on
    /// malformed input.
    [[nodiscard]] static std::optional<JsonValue> parse(std::string_view text,
                                                        std::string* error = nullptr);
    /// Reads and parses a file. Returns nullopt and sets `error` when the file
    /// cannot be read (including when it does not exist) or is malformed.
    [[nodiscard]] static std::optional<JsonValue> load(const std::filesystem::path& path,
                                                       std::string* error = nullptr);

    /// Deep equality; object members compare in order.
    friend bool operator==(const JsonValue& lhs, const JsonValue& rhs);

private:
    void dump_to(std::string& out, int indent, int depth) const;

    std::variant<std::monostate, bool, Number, std::string, Array, Object> value_;
};

} // namespace core::utils

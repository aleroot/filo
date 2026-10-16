#include "JsonValue.hpp"

#include "JsonUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <charconv>
#include <format>
#include <ranges>
#include <stdexcept>

namespace core::utils {
namespace {

[[nodiscard]] JsonValue from_element(simdjson::dom::element element) {
    switch (element.type()) {
    case simdjson::dom::element_type::OBJECT: {
        JsonValue::Object members;
        for (auto field : simdjson::dom::object(element)) {
            // Keep the first of duplicate keys: simdjson lookups, and so
            // everything in Filo that reads these files, see that one.
            if (std::ranges::contains(members, field.key, &JsonValue::Member::first)) continue;
            members.emplace_back(std::string(field.key), from_element(field.value));
        }
        return JsonValue(std::move(members));
    }
    case simdjson::dom::element_type::ARRAY: {
        JsonValue::Array items;
        for (auto item : simdjson::dom::array(element)) items.push_back(from_element(item));
        return JsonValue(std::move(items));
    }
    case simdjson::dom::element_type::STRING:
        return JsonValue(std::string(std::string_view(element)));
    case simdjson::dom::element_type::BOOL:
        return JsonValue(bool(element));
    case simdjson::dom::element_type::NULL_VALUE:
        return JsonValue();
    case simdjson::dom::element_type::INT64:
    case simdjson::dom::element_type::UINT64:
    case simdjson::dom::element_type::DOUBLE:
    case simdjson::dom::element_type::BIGINT:
        return JsonValue(JsonValue::Number{simdjson::minify(element)});
    }
    return JsonValue();
}

void newline(std::string& out, int indent, int depth) {
    if (indent <= 0) return;
    out += '\n';
    out.append(static_cast<std::size_t>(indent * depth), ' ');
}

void append_string(std::string& out, std::string_view value) {
    out += '"';
    append_escaped(out, value);
    out += '"';
}

/// The unescaped reference tokens of an RFC 6901 pointer; nullopt when malformed.
[[nodiscard]] std::optional<std::vector<std::string>> pointer_tokens(std::string_view pointer) {
    std::vector<std::string> tokens;
    if (pointer.empty()) return tokens;
    if (pointer.front() != '/') return std::nullopt;
    // Split the whole pointer and drop the empty piece before the leading '/':
    // splitting only the remainder would lose the empty key that "/" names.
    for (const auto part : std::views::split(pointer, '/') | std::views::drop(1)) {
        const std::string_view raw(part.begin(), part.end());
        std::string token;
        for (std::size_t i = 0; i < raw.size(); ++i) {
            if (raw[i] != '~') {
                token += raw[i];
            } else if (i + 1 < raw.size() && (raw[i + 1] == '0' || raw[i + 1] == '1')) {
                token += raw[++i] == '0' ? '~' : '/';
            } else {
                return std::nullopt;
            }
        }
        tokens.push_back(std::move(token));
    }
    return tokens;
}

[[nodiscard]] std::optional<JsonValue> parse_padded(const simdjson::padded_string& text,
                                                    std::string* error) {
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (const auto code = parser.parse(text).get(root); code != simdjson::SUCCESS) {
        if (error) *error = simdjson::error_message(code);
        return std::nullopt;
    }
    return from_element(root);
}

} // namespace

JsonValue JsonValue::integer(long long value) {
    char buffer[32];
    const auto [end, _] = std::to_chars(buffer, buffer + sizeof(buffer), value);
    return JsonValue(Number{std::string(buffer, end)});
}

bool JsonValue::as_bool(bool fallback) const noexcept {
    const auto* value = std::get_if<bool>(&value_);
    return value ? *value : fallback;
}

std::string_view JsonValue::as_string(std::string_view fallback) const noexcept {
    const auto* value = std::get_if<std::string>(&value_);
    return value ? std::string_view(*value) : fallback;
}

std::optional<long long> JsonValue::as_integer() const noexcept {
    const auto* number = std::get_if<Number>(&value_);
    if (!number) return std::nullopt;
    long long value = 0;
    const auto* begin = number->text.data();
    const auto* end = begin + number->text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end) return std::nullopt;
    return value;
}

JsonValue::Array& JsonValue::items() { return std::get<Array>(value_); }
const JsonValue::Array& JsonValue::items() const { return std::get<Array>(value_); }
JsonValue::Object& JsonValue::members() { return std::get<Object>(value_); }
const JsonValue::Object& JsonValue::members() const { return std::get<Object>(value_); }

JsonValue* JsonValue::find(std::string_view key) noexcept {
    auto* object = std::get_if<Object>(&value_);
    if (!object) return nullptr;
    for (auto& [name, value] : *object) {
        if (name == key) return &value;
    }
    return nullptr;
}

const JsonValue* JsonValue::find(std::string_view key) const noexcept {
    return const_cast<JsonValue*>(this)->find(key);
}

JsonValue& JsonValue::set(std::string_view key, JsonValue value) {
    if (!is_object()) value_ = Object{};
    if (auto* existing = find(key)) {
        *existing = std::move(value);
        return *existing;
    }
    auto& object = members();
    object.emplace_back(std::string(key), std::move(value));
    return object.back().second;
}

bool JsonValue::erase(std::string_view key) {
    auto* object = std::get_if<Object>(&value_);
    if (!object) return false;
    const auto old_size = object->size();
    std::erase_if(*object, [key](const Member& member) { return member.first == key; });
    return object->size() != old_size;
}

std::string JsonValue::pointer_token(std::string_view key) {
    std::string token;
    token.reserve(key.size());
    for (const char c : key) {
        if (c == '~') token += "~0";
        else if (c == '/') token += "~1";
        else token += c;
    }
    return token;
}

const JsonValue* JsonValue::at_pointer(std::string_view pointer) const {
    const auto tokens = pointer_tokens(pointer);
    if (!tokens) return nullptr;
    const JsonValue* node = this;
    for (const auto& token : *tokens) {
        if (const auto* array = std::get_if<Array>(&node->value_)) {
            std::size_t index = 0;
            const auto* end = token.data() + token.size();
            const auto [ptr, ec] = std::from_chars(token.data(), end, index);
            if (ec != std::errc{} || ptr != end || index >= array->size()) return nullptr;
            node = &(*array)[index];
        } else if (node = node->find(token); !node) {
            return nullptr;
        }
    }
    return node;
}

JsonValue* JsonValue::set_at_pointer(std::string_view pointer, JsonValue value) {
    const auto tokens = pointer_tokens(pointer);
    if (!tokens) return nullptr;
    if (tokens->empty()) {
        *this = std::move(value);
        return this;
    }
    JsonValue* node = this;
    for (const auto& token : *tokens | std::views::take(tokens->size() - 1)) {
        auto* next = node->find(token);
        node = next && next->is_object() ? next : &node->set(token, object());
    }
    return &node->set(tokens->back(), std::move(value));
}

// Defined here, where JsonValue is complete: a defaulted comparison would ask
// the compiler to resolve the recursive variant comparison mid-definition.
bool operator==(const JsonValue& lhs, const JsonValue& rhs) {
    return lhs.value_ == rhs.value_;
}

std::string JsonValue::dump(int indent) const {
    std::string out;
    dump_to(out, indent, 0);
    return out;
}

void JsonValue::dump_to(std::string& out, int indent, int depth) const {
    std::visit([&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            out += "null";
        } else if constexpr (std::is_same_v<T, bool>) {
            out += value ? "true" : "false";
        } else if constexpr (std::is_same_v<T, Number>) {
            out += value.text;
        } else if constexpr (std::is_same_v<T, std::string>) {
            append_string(out, value);
        } else if constexpr (std::is_same_v<T, Array>) {
            if (value.empty()) {
                out += "[]";
                return;
            }
            out += '[';
            for (std::size_t i = 0; i < value.size(); ++i) {
                if (i > 0) out += ',';
                newline(out, indent, depth + 1);
                value[i].dump_to(out, indent, depth + 1);
            }
            newline(out, indent, depth);
            out += ']';
        } else {
            if (value.empty()) {
                out += "{}";
                return;
            }
            out += '{';
            for (std::size_t i = 0; i < value.size(); ++i) {
                if (i > 0) out += ',';
                newline(out, indent, depth + 1);
                append_string(out, value[i].first);
                out += indent > 0 ? ": " : ":";
                value[i].second.dump_to(out, indent, depth + 1);
            }
            newline(out, indent, depth);
            out += '}';
        }
    }, value_);
}

std::optional<JsonValue> JsonValue::parse(std::string_view text, std::string* error) {
    return parse_padded(simdjson::padded_string(text), error);
}

std::optional<JsonValue> JsonValue::load(const std::filesystem::path& path, std::string* error) {
    simdjson::padded_string text;
    if (simdjson::padded_string::load(path.string()).get(text) != simdjson::SUCCESS) {
        if (error) *error = std::format("cannot read {}", path.string());
        return std::nullopt;
    }
    std::string parse_error;
    auto value = parse_padded(text, &parse_error);
    if (!value && error) *error = std::format("{} is not valid JSON: {}", path.string(), parse_error);
    return value;
}

} // namespace core::utils

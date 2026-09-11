#include "ArgumentCoercion.hpp"

#include "ToolSchemaInternal.hpp"
#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace core::tools::schema {
namespace {

using Element = simdjson::dom::element;
using Object = simdjson::dom::object;

/// Structural recursion guard. Tool schemas are shallow; anything deeper is
/// either hostile or hopeless, and bailing out simply means "no coercion".
constexpr std::size_t kMaxDepth = 16;

/// How many times a string may be decoded into a JSON value. Single encoding
/// is the common model mistake and double encoding happens when a payload
/// crosses a gateway that re-stringifies; beyond that the value is noise.
constexpr int kMaxStringUnwraps = 2;

/// Longest value echoed back inside a diagnostic message.
constexpr std::size_t kMaxQuotedValueChars = 48;

[[nodiscard]] std::string_view trim(std::string_view text) noexcept {
    constexpr std::string_view kSpace = " \t\r\n";
    const auto begin = text.find_first_not_of(kSpace);
    if (begin == std::string_view::npos) return {};
    const auto end = text.find_last_not_of(kSpace);
    return text.substr(begin, end - begin + 1);
}

/// True when a string was meant to carry a JSON array/object payload. Such a
/// string is never wrapped or otherwise reinterpreted when it fails to parse:
/// turning a malformed payload into a plausible-looking value is how coercion
/// stops being a repair and starts being a corruption.
[[nodiscard]] bool looks_like_json_container(std::string_view text) noexcept {
    const auto trimmed = trim(text);
    return !trimmed.empty() && (trimmed.front() == '[' || trimmed.front() == '{');
}

/**
 * Owns every parser and buffer backing a value decoded out of a JSON string.
 *
 * simdjson elements borrow from their parser's DOM, so a decoded payload must
 * outlive the coercion pass that reads it. Deques keep addresses stable as
 * more nested documents are decoded.
 */
class NestedDocuments {
public:
    [[nodiscard]] std::optional<Element> parse(std::string_view json) {
        auto& buffer = buffers_.emplace_back(simdjson::padded_string(json));
        auto& parser = parsers_.emplace_back(
            std::make_unique<simdjson::dom::parser>());
        parser->number_as_string(true);
        Element root;
        if (parser->parse(buffer).get(root) != simdjson::SUCCESS) {
            return std::nullopt;
        }
        return root;
    }

private:
    std::deque<simdjson::padded_string> buffers_;
    std::deque<std::unique_ptr<simdjson::dom::parser>> parsers_;
};

/// Property names declared by an object schema.
[[nodiscard]] std::vector<std::string_view> declared_properties(Object schema) {
    Object properties;
    if (schema["properties"].get(properties) != simdjson::SUCCESS) return {};
    std::vector<std::string_view> names;
    names.reserve(properties.size());
    for (const auto field : properties) names.emplace_back(field.key);
    return names;
}

[[nodiscard]] bool contains(const std::vector<std::string_view>& names,
                            std::string_view name) noexcept {
    return std::ranges::find(names, name) != names.end();
}

/// Required property names declared by an object schema.
[[nodiscard]] std::vector<std::string_view> required_properties(Object schema) {
    simdjson::dom::array required;
    if (schema["required"].get(required) != simdjson::SUCCESS) return {};
    std::vector<std::string_view> names;
    for (Element item : required) {
        std::string_view name;
        if (item.get(name) == simdjson::SUCCESS) names.emplace_back(name);
    }
    return names;
}

[[nodiscard]] bool rejects_additional_properties(Object schema) {
    bool allowed = true;
    return schema["additionalProperties"].get(allowed) == simdjson::SUCCESS
        && !allowed;
}

/**
 * The coercion pass.
 *
 * Every method answers the same question — "can this value be rewritten into
 * something the schema accepts?" — and returns the rewritten JSON text, or
 * nullopt to mean "left unchanged". Verification against the schema happens at
 * each decision point so a strategy is only adopted when it actually works.
 */
class Coercer {
public:
    [[nodiscard]] std::optional<std::string> coerce(Element value,
                                                    Element schema) {
        return coerce_value(value, schema, 0, kMaxStringUnwraps);
    }

    [[nodiscard]] bool validates(std::string_view candidate_json,
                                 Element schema) {
        const auto parsed = documents_.parse(candidate_json);
        return parsed.has_value()
            && !detail::validate_value(*parsed, schema, "$").has_value();
    }

private:
    [[nodiscard]] static std::string serialize(Element value) {
        return simdjson::to_string(value);
    }

    /// Rewritten text when coercion applies, the value's own text otherwise.
    [[nodiscard]] std::string coerce_or_serialize(Element value,
                                                  Element schema,
                                                  std::size_t depth,
                                                  int unwraps) {
        if (auto coerced = coerce_value(value, schema, depth, unwraps)) {
            return std::move(*coerced);
        }
        return serialize(value);
    }

    [[nodiscard]] std::optional<std::string> coerce_value(Element value,
                                                          Element schema_element,
                                                          std::size_t depth,
                                                          int unwraps) {
        if (depth >= kMaxDepth) return std::nullopt;
        Object schema;
        if (schema_element.get(schema) != simdjson::SUCCESS) return std::nullopt;

        // A value of the declared type can still contain mis-shaped children,
        // so recurse first and only then consider changing this level.
        const std::string_view actual = detail::element_type_name(value);
        if (detail::schema_accepts_type(schema_element, actual)) {
            if (Object object; value.get(object) == simdjson::SUCCESS) {
                return coerce_object(object, schema, depth, unwraps);
            }
            if (simdjson::dom::array array; value.get(array) == simdjson::SUCCESS) {
                return coerce_array(array, schema, depth, unwraps);
            }
            return std::nullopt;
        }

        if (std::string_view text; value.get(text) == simdjson::SUCCESS) {
            if (auto decoded = decode_string(text, schema_element, depth, unwraps)) {
                return decoded;
            }
            // A payload that was meant to be a container but does not parse is
            // a malformed argument, not a scalar to be packed into a list.
            if (looks_like_json_container(text)) return std::nullopt;
        }
        return wrap_in_array(value, schema_element, schema, depth, unwraps);
    }

    /// object → object: coerce declared members, then try to adopt flattened
    /// list-item fields the model spread across this object.
    [[nodiscard]] std::optional<std::string> coerce_object(Object object,
                                                           Object schema,
                                                           std::size_t depth,
                                                           int unwraps) {
        Object properties;
        const bool has_properties =
            schema["properties"].get(properties) == simdjson::SUCCESS;
        Element additional_schema;
        const bool has_additional_schema =
            schema["additionalProperties"].get(additional_schema) == simdjson::SUCCESS
            && additional_schema.type() == simdjson::dom::element_type::OBJECT;

        std::vector<std::pair<std::string, std::string>> members;
        members.reserve(object.size());
        bool changed = false;
        for (const auto field : object) {
            Element member_schema;
            const bool known = has_properties
                && properties[field.key].get(member_schema) == simdjson::SUCCESS;
            if (!known && has_additional_schema) member_schema = additional_schema;
            std::string text;
            if (known || has_additional_schema) {
                if (auto coerced =
                        coerce_value(field.value, member_schema, depth + 1, unwraps)) {
                    text = std::move(*coerced);
                    changed = true;
                } else {
                    text = serialize(field.value);
                }
            } else {
                text = serialize(field.value);
            }
            members.emplace_back(std::string(field.key), std::move(text));
        }

        if (gather_flattened_item(object, schema, members)) changed = true;
        if (!changed) return std::nullopt;
        return write_object(std::move(members));
    }

    [[nodiscard]] std::optional<std::string> coerce_array(simdjson::dom::array array,
                                                          Object schema,
                                                          std::size_t depth,
                                                          int unwraps) {
        Element items;
        if (schema["items"].get(items) != simdjson::SUCCESS) return std::nullopt;

        std::vector<std::string> elements;
        elements.reserve(array.size());
        bool changed = false;
        for (Element item : array) {
            if (auto coerced = coerce_value(item, items, depth + 1, unwraps)) {
                elements.push_back(std::move(*coerced));
                changed = true;
            } else {
                elements.push_back(serialize(item));
            }
        }
        if (!changed) return std::nullopt;
        return write_array(elements);
    }

    /// string → declared type. Container payloads are parsed strictly; scalars
    /// are read as complete literals. Every candidate must validate.
    [[nodiscard]] std::optional<std::string> decode_string(std::string_view text,
                                                           Element schema,
                                                           std::size_t depth,
                                                           int unwraps) {
        const std::string_view trimmed = trim(text);
        if (trimmed.empty()) return std::nullopt;

        // Double encoding: a gateway re-stringified an argument that the model
        // had already stringified. Peel exactly one quoted layer and retry.
        if (trimmed.front() == '"' && unwraps > 0) {
            const auto parsed = documents_.parse(trimmed);
            std::string_view inner;
            if (!parsed || parsed->get(inner) != simdjson::SUCCESS) {
                return std::nullopt;
            }
            return decode_string(inner, schema, depth, unwraps - 1);
        }

        if (looks_like_json_container(trimmed) && unwraps > 0) {
            if (const auto parsed = documents_.parse(trimmed)) {
                std::string candidate =
                    coerce_or_serialize(*parsed, schema, depth + 1, unwraps - 1);
                if (validates(candidate, schema)) return candidate;
            }
            return std::nullopt;
        }

        if (detail::schema_accepts_type(schema, "integer")
            || detail::schema_accepts_type(schema, "number")) {
            if (auto number = parse_number_literal(trimmed, schema)) {
                if (validates(*number, schema)) return number;
            }
        }
        if (detail::schema_accepts_type(schema, "boolean")
            && (trimmed == "true" || trimmed == "false")) {
            const std::string candidate(trimmed);
            if (validates(candidate, schema)) return candidate;
        }
        return std::nullopt;
    }

    /// A lone item where a list is expected: `edits: {...}` → `edits: [{...}]`.
    /// Mirrors Ajv's `coerceTypes: "array"`, restricted to values the declared
    /// item schema actually accepts.
    [[nodiscard]] std::optional<std::string> wrap_in_array(Element value,
                                                           Element schema_element,
                                                           Object schema,
                                                           std::size_t depth,
                                                           int unwraps) {
        if (!detail::schema_accepts_type(schema_element, "array")) return std::nullopt;
        Element items;
        if (schema["items"].get(items) != simdjson::SUCCESS) return std::nullopt;

        const std::string item_text =
            coerce_or_serialize(value, items, depth + 1, unwraps);
        std::string candidate = std::format("[{}]", item_text);
        if (!validates(candidate, schema_element)) return std::nullopt;
        return candidate;
    }

    /**
     * Adopts list-item fields flattened onto this object.
     *
     * `{"file_path":"a","old_string":"x","new_string":"y"}` is a single edit
     * written without its wrapper. The rewrite is attempted only when the
     * schema leaves no other reading: the list property is missing, the object
     * forbids additional properties, and every stray field is declared by the
     * item schema, which must find all of its required fields present.
     */
    [[nodiscard]] bool gather_flattened_item(
        Object object,
        Object schema,
        std::vector<std::pair<std::string, std::string>>& members) {
        if (!rejects_additional_properties(schema)) return false;

        const auto declared = declared_properties(schema);
        std::vector<std::string_view> strays;
        for (const auto field : object) {
            if (!contains(declared, field.key)) strays.emplace_back(field.key);
        }
        if (strays.empty()) return false;

        for (const auto& name : required_properties(schema)) {
            Element present;
            if (object[name].get(present) == simdjson::SUCCESS) continue;

            Element list_schema;
            Object list_object;
            Element items;
            Object item_object;
            if (schema["properties"][name].get(list_schema) != simdjson::SUCCESS
                || list_schema.get(list_object) != simdjson::SUCCESS
                || !detail::schema_accepts_type(list_schema, "array")
                || list_object["items"].get(items) != simdjson::SUCCESS
                || items.get(item_object) != simdjson::SUCCESS) {
                continue;
            }

            const auto item_properties = declared_properties(item_object);
            const bool strays_all_declared =
                std::ranges::all_of(strays, [&](std::string_view stray) {
                    return contains(item_properties, stray);
                });
            if (!strays_all_declared) continue;
            const bool item_is_complete =
                std::ranges::all_of(required_properties(item_object),
                                    [&](std::string_view field) {
                                        return contains(strays, field);
                                    });
            if (!item_is_complete) continue;

            std::vector<std::pair<std::string, std::string>> item_members;
            item_members.reserve(strays.size());
            std::erase_if(members, [&](const auto& member) {
                if (!contains(strays, member.first)) return false;
                item_members.emplace_back(member);
                return true;
            });
            members.emplace_back(std::string(name),
                                 std::format("[{}]", write_object(std::move(item_members))));
            return true;
        }
        return false;
    }

    /// Deterministic member ordering keeps coerced arguments comparable with
    /// the canonical form produced by validation.
    [[nodiscard]] static std::string write_object(
        std::vector<std::pair<std::string, std::string>> members) {
        std::ranges::sort(members, {}, &std::pair<std::string, std::string>::first);
        core::utils::JsonWriter writer(128);
        {
            auto scope = writer.object();
            for (std::size_t i = 0; i < members.size(); ++i) {
                if (i > 0) writer.comma();
                writer.key(members[i].first).raw(members[i].second);
            }
        }
        return std::move(writer).take();
    }

    [[nodiscard]] static std::string write_array(
        const std::vector<std::string>& elements) {
        core::utils::JsonWriter writer(128);
        {
            auto scope = writer.array();
            for (std::size_t i = 0; i < elements.size(); ++i) {
                if (i > 0) writer.comma();
                writer.raw(elements[i]);
            }
        }
        return std::move(writer).take();
    }

    /// Complete numeric literal, or nullopt. Trailing text ("821, 1025") is a
    /// model conflating two values and must not be silently truncated to one.
    [[nodiscard]] static std::optional<std::string> parse_number_literal(
        std::string_view text,
        Element schema) {
        const char* begin = text.data();
        const char* end = begin + text.size();
        if (std::int64_t integral = 0;
            std::from_chars(begin, end, integral).ptr == end) {
            return std::to_string(integral);
        }
        if (!detail::schema_accepts_type(schema, "number")) return std::nullopt;
        if (double real = 0.0; std::from_chars(begin, end, real).ptr == end) {
            return std::string(text);
        }
        return std::nullopt;
    }

    NestedDocuments documents_;
};

[[nodiscard]] std::string quote_for_message(std::string_view value) {
    std::string quoted;
    quoted.reserve(std::min(value.size(), kMaxQuotedValueChars) + 8);
    for (const char ch : value.substr(0, kMaxQuotedValueChars)) {
        switch (ch) {
            case '\n': quoted += "\\n"; break;
            case '\r': quoted += "\\r"; break;
            case '\t': quoted += "\\t"; break;
            default: quoted += ch; break;
        }
    }
    if (value.size() > kMaxQuotedValueChars) quoted += "...";
    return quoted;
}

/// Cheap structural diagnosis of a JSON payload that failed to parse: an
/// unclosed string or unbalanced brackets means the model's output was cut
/// short, which calls for a different correction than a quoting mistake.
[[nodiscard]] bool looks_truncated(std::string_view payload) noexcept {
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (const char ch : payload) {
        if (in_string) {
            if (escaped) { escaped = false; continue; }
            if (ch == '\\') escaped = true;
            else if (ch == '"') in_string = false;
            continue;
        }
        switch (ch) {
            case '"': in_string = true; break;
            case '[':
            case '{': ++depth; break;
            case ']':
            case '}': --depth; break;
            default: break;
        }
    }
    return in_string || depth > 0;
}

} // namespace

std::optional<std::string> coerce_arguments(std::string_view arguments_json,
                                            std::string_view schema_json) {
    simdjson::dom::parser arguments_parser;
    arguments_parser.number_as_string(true);
    simdjson::padded_string padded_arguments(arguments_json);
    Element arguments_root;
    if (arguments_parser.parse(padded_arguments).get(arguments_root)
        != simdjson::SUCCESS) {
        return std::nullopt;
    }

    simdjson::dom::parser schema_parser;
    schema_parser.number_as_string(true);
    simdjson::padded_string padded_schema(schema_json);
    Element schema_root;
    if (schema_parser.parse(padded_schema).get(schema_root) != simdjson::SUCCESS) {
        return std::nullopt;
    }

    Coercer coercer;
    auto coerced = coercer.coerce(arguments_root, schema_root);
    if (!coerced.has_value()) return std::nullopt;
    // Final gate: the rewrite is only an improvement if it fully satisfies the
    // contract. Anything less is handed back to the model as a clear error.
    if (!coercer.validates(*coerced, schema_root)) return std::nullopt;
    return coerced;
}

std::string explain_unparsable_arguments(std::string_view raw) {
    const std::string_view trimmed = trim(raw);
    if (trimmed.empty()) return {};
    if (looks_truncated(trimmed)) {
        return " — the payload ends mid-value (unterminated string or bracket),"
               " so the tool call was cut off rather than mis-typed. Send the"
               " call again with less content, splitting large work across"
               " several calls";
    }
    return {};
}

std::string explain_string_value(std::string_view value,
                                 std::string_view expected_type) {
    const std::string_view trimmed = trim(value);
    if (trimmed.empty()) return {};

    if (expected_type == "array" || expected_type == "object") {
        if (!looks_like_json_container(trimmed)) {
            return std::format(
                " — send a JSON {0} value, not a quoted string",
                expected_type);
        }
        if (looks_truncated(trimmed)) {
            return std::format(
                " — the value is a JSON {0} serialized into a string and the"
                " payload is truncated (unterminated string or bracket). Send"
                " the {0} as a real JSON value, and split large content across"
                " smaller calls",
                expected_type);
        }
        return std::format(
            " — the value is a JSON {0} serialized into a string and its JSON"
            " is malformed (most often a quote or backslash escaped at the"
            " wrong level). Send the {0} as a real JSON value instead of"
            " escaping it inside a string",
            expected_type);
    }

    if (expected_type == "integer" || expected_type == "number"
        || expected_type == "boolean") {
        return std::format(
            " — \"{}\" is quoted; send a bare JSON {} value",
            quote_for_message(trimmed),
            expected_type);
    }
    return {};
}

} // namespace core::tools::schema

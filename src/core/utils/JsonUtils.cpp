#include "JsonUtils.hpp"
#include "JsonWriter.hpp"
#include <simdjson.h>
#include <algorithm>   // std::find_if
#include <cstdio>      // std::snprintf
#include <initializer_list>
#include <optional>

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

void append_escaped(std::string& out, std::string_view sv) {
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

void append_escaped_utf8_safe(std::string& out, std::string_view sv) {
    // simdjson validates in vector-width chunks. Keeping validation separate
    // lets valid external data use the bulk escaper without per-code-point
    // branches; only malformed byte streams enter the repair path below.
    if (simdjson::validate_utf8(sv)) {
        append_escaped(out, sv);
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
            append_escaped(out, std::string_view(q, 1));
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

#include "ToolSchema.hpp"

#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace core::tools::schema {
namespace {

using Element = simdjson::dom::element;
using Object = simdjson::dom::object;

void write_canonical(core::utils::JsonWriter& writer, Element value);

void write_canonical_object(core::utils::JsonWriter& writer, Object object) {
    std::vector<std::pair<std::string, Element>> fields;
    for (const auto field : object) {
        fields.emplace_back(std::string(field.key), field.value);
    }
    std::ranges::sort(fields, {}, &std::pair<std::string, Element>::first);

    auto scope = writer.object();
    for (std::size_t i = 0; i < fields.size(); ++i) {
        if (i > 0) writer.comma();
        writer.key(fields[i].first);
        write_canonical(writer, fields[i].second);
    }
}

void write_canonical_array(core::utils::JsonWriter& writer,
                           simdjson::dom::array array) {
    auto scope = writer.array();
    bool first = true;
    for (Element item : array) {
        if (!first) writer.comma();
        write_canonical(writer, item);
        first = false;
    }
}

void write_canonical(core::utils::JsonWriter& writer, Element value) {
    Object object;
    if (value.get(object) == simdjson::SUCCESS) {
        write_canonical_object(writer, object);
        return;
    }
    simdjson::dom::array array;
    if (value.get(array) == simdjson::SUCCESS) {
        write_canonical_array(writer, array);
        return;
    }
    writer.raw(simdjson::to_string(value));
}

[[nodiscard]] std::string canonicalize_json(std::string_view json) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(json);
    Element root;
    if (parser.parse(padded).get(root) != simdjson::SUCCESS) {
        return std::string(json);
    }
    core::utils::JsonWriter writer(json.size() + 128);
    write_canonical(writer, root);
    return std::move(writer).take();
}

[[nodiscard]] bool is_required(Object schema, std::string_view name) {
    simdjson::dom::array required;
    if (schema["required"].get(required) != simdjson::SUCCESS) return false;
    for (Element item : required) {
        std::string_view value;
        if (item.get(value) == simdjson::SUCCESS && value == name) return true;
    }
    return false;
}

[[nodiscard]] std::optional<Element> property_schema(Object schema,
                                                     std::string_view name) {
    Object properties;
    if (schema["properties"].get(properties) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    Element property;
    if (properties[name].get(property) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    return property;
}

[[nodiscard]] bool schema_accepts_type(Element schema, std::string_view wanted) {
    Object object;
    if (schema.get(object) != simdjson::SUCCESS) return true;

    std::string_view type;
    if (object["type"].get(type) == simdjson::SUCCESS) return type == wanted;

    simdjson::dom::array types;
    if (object["type"].get(types) == simdjson::SUCCESS) {
        for (Element item : types) {
            std::string_view candidate;
            if (item.get(candidate) == simdjson::SUCCESS && candidate == wanted) {
                return true;
            }
        }
        return false;
    }

    for (const std::string_view combinator : {"anyOf", "oneOf"}) {
        simdjson::dom::array alternatives;
        if (object[combinator].get(alternatives) == simdjson::SUCCESS) {
            for (Element alternative : alternatives) {
                if (schema_accepts_type(alternative, wanted)) return true;
            }
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::string element_type_name(Element value) {
    switch (value.type()) {
        case simdjson::dom::element_type::ARRAY: return "array";
        case simdjson::dom::element_type::OBJECT: return "object";
        case simdjson::dom::element_type::INT64:
        case simdjson::dom::element_type::UINT64: return "integer";
        case simdjson::dom::element_type::DOUBLE: return "number";
        case simdjson::dom::element_type::STRING: return "string";
        case simdjson::dom::element_type::BOOL: return "boolean";
        case simdjson::dom::element_type::NULL_VALUE: return "null";
    }
    return "unknown";
}

[[nodiscard]] std::optional<std::string> validate_value(
    Element value,
    Element schema,
    std::string_view path);

[[nodiscard]] std::optional<std::string> validate_combinators(
    Element value,
    Object schema,
    std::string_view path) {
    simdjson::dom::array all_of;
    if (schema["allOf"].get(all_of) == simdjson::SUCCESS) {
        for (Element candidate : all_of) {
            if (auto error = validate_value(value, candidate, path)) return error;
        }
    }

    for (const auto [keyword, exact] : {
             std::pair<std::string_view, bool>{"anyOf", false},
             std::pair<std::string_view, bool>{"oneOf", true}}) {
        simdjson::dom::array alternatives;
        if (schema[keyword].get(alternatives) != simdjson::SUCCESS) continue;
        std::size_t matches = 0;
        for (Element candidate : alternatives) {
            if (!validate_value(value, candidate, path).has_value()) ++matches;
        }
        if ((exact && matches != 1) || (!exact && matches == 0)) {
            return std::format("{} does not satisfy {}", path, keyword);
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> validate_object(
    Object value,
    Object schema,
    std::string_view path) {
    simdjson::dom::array required;
    if (schema["required"].get(required) == simdjson::SUCCESS) {
        for (Element item : required) {
            std::string_view name;
            if (item.get(name) != simdjson::SUCCESS) continue;
            Element ignored;
            if (value[name].get(ignored) != simdjson::SUCCESS) {
                return std::format("{} is missing required property '{}'", path, name);
            }
        }
    }

    Object properties;
    const bool has_properties = schema["properties"].get(properties) == simdjson::SUCCESS;
    bool allow_additional = true;
    static_cast<void>(schema["additionalProperties"].get(allow_additional));
    Element additional_schema;
    const bool has_additional_schema =
        schema["additionalProperties"].get(additional_schema) == simdjson::SUCCESS
        && additional_schema.type() == simdjson::dom::element_type::OBJECT;

    for (const auto field : value) {
        Element field_schema;
        const bool known = has_properties
            && properties[field.key].get(field_schema) == simdjson::SUCCESS;
        const std::string child_path = std::format("{}.{}", path, field.key);
        if (known) {
            if (auto error = validate_value(field.value, field_schema, child_path)) return error;
        } else if (has_additional_schema) {
            if (auto error = validate_value(field.value, additional_schema, child_path)) return error;
        } else if (!allow_additional) {
            return std::format("Unknown argument '{}' at {}", field.key, path);
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> validate_value(
    Element value,
    Element schema_element,
    std::string_view path) {
    Object schema;
    if (schema_element.get(schema) != simdjson::SUCCESS) return std::nullopt;
    if (auto error = validate_combinators(value, schema, path)) return error;

    const std::string actual = element_type_name(value);
    if (!schema_accepts_type(schema_element, actual)
        && !(actual == "integer" && schema_accepts_type(schema_element, "number"))) {
        return std::format("{} must be {}, got {}",
                           path,
                           core::utils::json::string_field(schema, "type", "a valid schema type"),
                           actual);
    }

    simdjson::dom::array enum_values;
    if (schema["enum"].get(enum_values) == simdjson::SUCCESS) {
        const std::string canonical_value = canonicalize_json(simdjson::to_string(value));
        bool matched = false;
        for (Element candidate : enum_values) {
            if (canonicalize_json(simdjson::to_string(candidate)) == canonical_value) {
                matched = true;
                break;
            }
        }
        if (!matched) return std::format("{} is not one of the allowed values", path);
    }

    Element constant;
    if (schema["const"].get(constant) == simdjson::SUCCESS
        && canonicalize_json(simdjson::to_string(constant))
            != canonicalize_json(simdjson::to_string(value))) {
        return std::format("{} does not match the required constant", path);
    }

    Object object;
    if (value.get(object) == simdjson::SUCCESS) {
        return validate_object(object, schema, path);
    }

    simdjson::dom::array array;
    if (value.get(array) == simdjson::SUCCESS) {
        Element items;
        if (schema["items"].get(items) == simdjson::SUCCESS) {
            std::size_t index = 0;
            for (Element item : array) {
                if (auto error = validate_value(
                        item, items, std::format("{}[{}]", path, index))) {
                    return error;
                }
                ++index;
            }
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::string strip_optional_nulls(Object arguments,
                                               Object schema) {
    std::vector<std::pair<std::string, Element>> fields;
    for (const auto field : arguments) {
        const auto property = property_schema(schema, field.key);
        const bool strip = field.value.is_null()
            && property.has_value()
            && !is_required(schema, field.key)
            && !schema_accepts_type(*property, "null");
        if (!strip) fields.emplace_back(std::string(field.key), field.value);
    }
    std::ranges::sort(fields, {}, &std::pair<std::string, Element>::first);

    core::utils::JsonWriter writer;
    {
        auto scope = writer.object();
        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (i > 0) writer.comma();
            writer.key(fields[i].first);
            write_canonical(writer, fields[i].second);
        }
    }
    return std::move(writer).take();
}

} // namespace

std::string canonical_input_schema(const ToolDefinition& definition) {
    if (!definition.input_schema.empty()) {
        return canonicalize_json(core::utils::json::schema::ensure_property_types(
            definition.input_schema,
            core::utils::json::schema::RootRole::Container));
    }

    core::utils::JsonWriter writer(512);
    {
        auto root = writer.object();
        writer.kv_str("type", "object").comma().key("properties");
        {
            auto properties = writer.object();
            for (std::size_t i = 0; i < definition.parameters.size(); ++i) {
                if (i > 0) writer.comma();
                const auto& parameter = definition.parameters[i];
                writer.key(parameter.name);
                if (!parameter.schema.empty()) {
                    writer.raw(core::utils::json::schema::ensure_property_types(
                        parameter.schema,
                        core::utils::json::schema::RootRole::Property));
                    continue;
                }
                auto property = writer.object();
                writer.kv_str("type", parameter.type);
                if (!parameter.description.empty()) {
                    writer.comma().kv_str("description", parameter.description);
                }
                if (!parameter.items_schema.empty()) {
                    writer.comma().kv_raw("items", parameter.items_schema);
                }
            }
        }
        writer.comma().key("required");
        {
            auto required = writer.array();
            bool first = true;
            for (const auto& parameter : definition.parameters) {
                if (!parameter.required) continue;
                if (!first) writer.comma();
                writer.str(parameter.name);
                first = false;
            }
        }
        writer.comma().kv_bool("additionalProperties", false);
    }
    return canonicalize_json(std::move(writer).take());
}

std::expected<std::string, std::string> normalize_arguments(
    const ToolDefinition& definition,
    std::string_view raw_arguments) {
    const std::string arguments_json = raw_arguments.empty() ? "{}" : std::string(raw_arguments);
    const std::string schema_json = canonical_input_schema(definition);

    simdjson::dom::parser arguments_parser;
    simdjson::padded_string padded_arguments(arguments_json);
    Element arguments_root;
    if (arguments_parser.parse(padded_arguments).get(arguments_root) != simdjson::SUCCESS) {
        return std::unexpected("arguments are not valid JSON");
    }
    Object arguments;
    if (arguments_root.get(arguments) != simdjson::SUCCESS) {
        return std::unexpected("arguments must be a JSON object");
    }

    simdjson::dom::parser schema_parser;
    simdjson::padded_string padded_schema(schema_json);
    Element schema_root;
    Object schema_object;
    if (schema_parser.parse(padded_schema).get(schema_root) != simdjson::SUCCESS
        || schema_root.get(schema_object) != simdjson::SUCCESS) {
        return std::unexpected("tool has an invalid input schema");
    }

    const std::string normalized = strip_optional_nulls(arguments, schema_object);
    simdjson::dom::parser normalized_parser;
    simdjson::padded_string padded_normalized(normalized);
    Element normalized_root;
    if (normalized_parser.parse(padded_normalized).get(normalized_root) != simdjson::SUCCESS) {
        return std::unexpected("failed to normalize arguments");
    }
    if (auto error = validate_value(normalized_root, schema_root, "$")) {
        return std::unexpected(std::move(*error));
    }
    return normalized;
}

} // namespace core::tools::schema

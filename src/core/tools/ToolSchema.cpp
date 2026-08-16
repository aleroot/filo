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

enum class StructuralRole {
    Schema,  ///< JSON Schema object: "description" is an annotation keyword.
    NameMap, ///< properties / $defs / etc: keys are names, not keywords.
};

void write_structural(core::utils::JsonWriter& writer,
                      Element value,
                      StructuralRole role);

[[nodiscard]] bool is_schema_name_map(std::string_view key) noexcept {
    return key == "properties"
        || key == "patternProperties"
        || key == "$defs"
        || key == "definitions"
        || key == "dependentSchemas";
}

void write_structural_object(core::utils::JsonWriter& writer,
                             Object object,
                             StructuralRole role) {
    std::vector<std::pair<std::string, Element>> fields;
    for (const auto field : object) {
        // Strip the annotation keyword only. A property actually named
        // "description" lives in a NameMap and must stay in the fingerprint.
        if (role == StructuralRole::Schema && field.key == "description") {
            continue;
        }
        fields.emplace_back(std::string(field.key), field.value);
    }
    std::ranges::sort(fields, {}, &std::pair<std::string, Element>::first);

    auto scope = writer.object();
    for (std::size_t i = 0; i < fields.size(); ++i) {
        if (i > 0) writer.comma();
        writer.key(fields[i].first);
        const auto child_role =
            (role == StructuralRole::Schema && is_schema_name_map(fields[i].first))
                ? StructuralRole::NameMap
                : StructuralRole::Schema;
        write_structural(writer, fields[i].second, child_role);
    }
}

void write_structural(core::utils::JsonWriter& writer,
                      Element value,
                      StructuralRole role) {
    Object object;
    if (value.get(object) == simdjson::SUCCESS) {
        write_structural_object(writer, object, role);
        return;
    }
    simdjson::dom::array array;
    if (value.get(array) == simdjson::SUCCESS) {
        auto scope = writer.array();
        bool first = true;
        for (Element item : array) {
            if (!first) writer.comma();
            write_structural(writer, item, StructuralRole::Schema);
            first = false;
        }
        return;
    }
    writer.raw(simdjson::to_string(value));
}

[[nodiscard]] std::string canonicalize_json(std::string_view json) {
    simdjson::dom::parser parser;
    parser.number_as_string(true);
    simdjson::padded_string padded(json);
    Element root;
    if (parser.parse(padded).get(root) != simdjson::SUCCESS) {
        return std::string(json);
    }
    core::utils::JsonWriter writer(json.size() + 128);
    write_canonical(writer, root);
    return std::move(writer).take();
}

/// Parameter path relative to the argument root: "$.items[2].path" becomes
/// "items[2].path". Paths that do not start at the root are kept verbatim.
[[nodiscard]] std::string relative_path(std::string_view path) {
    constexpr std::string_view kRootPrefix = "$.";
    if (path.starts_with(kRootPrefix)) {
        return std::string(path.substr(kRootPrefix.size()));
    }
    if (path == "$") {
        return {};
    }
    return std::string(path);
}

[[nodiscard]] ArgumentIssue make_issue(ArgumentIssueCode code,
                                       std::string parameter,
                                       std::vector<std::string> allowed,
                                       std::string message) {
    return ArgumentIssue{
        .code = code,
        .parameter = std::move(parameter),
        .allowed = std::move(allowed),
        .message = std::move(message),
    };
}

/// Property names declared by an object schema, deterministically ordered.
[[nodiscard]] std::vector<std::string> property_names(Object schema) {
    Object properties;
    if (schema["properties"].get(properties) != simdjson::SUCCESS) {
        return {};
    }
    std::vector<std::string> names;
    names.reserve(properties.size());
    for (const auto field : properties) {
        names.emplace_back(field.key);
    }
    std::ranges::sort(names);
    return names;
}

/// String values declared by an enum schema, in declaration order. Non-string
/// enum members are omitted: they cannot participate in text guidance.
[[nodiscard]] std::vector<std::string> enum_string_values(Object schema) {
    simdjson::dom::array enum_values;
    if (schema["enum"].get(enum_values) != simdjson::SUCCESS) {
        return {};
    }
    std::vector<std::string> values;
    for (Element candidate : enum_values) {
        std::string_view value;
        if (candidate.get(value) == simdjson::SUCCESS) {
            values.emplace_back(value);
        }
    }
    return values;
}

/// Type tokens declared by a schema's "type" member (string or array form).
[[nodiscard]] std::vector<std::string> declared_type_tokens(Element schema_element) {
    Object schema;
    if (schema_element.get(schema) != simdjson::SUCCESS) {
        return {};
    }
    std::vector<std::string> tokens;
    if (std::string_view type;
        schema["type"].get(type) == simdjson::SUCCESS) {
        tokens.emplace_back(type);
        return tokens;
    }
    simdjson::dom::array types;
    if (schema["type"].get(types) == simdjson::SUCCESS) {
        for (Element item : types) {
            std::string_view candidate;
            if (item.get(candidate) == simdjson::SUCCESS) {
                tokens.emplace_back(candidate);
            }
        }
    }
    return tokens;
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
        case simdjson::dom::element_type::UINT64:
        case simdjson::dom::element_type::BIGINT: return "integer";
        case simdjson::dom::element_type::DOUBLE: return "number";
        case simdjson::dom::element_type::STRING: return "string";
        case simdjson::dom::element_type::BOOL: return "boolean";
        case simdjson::dom::element_type::NULL_VALUE: return "null";
    }
    return "unknown";
}

[[nodiscard]] std::optional<ArgumentIssue> validate_value(
    Element value,
    Element schema,
    std::string_view path);

[[nodiscard]] std::optional<ArgumentIssue> validate_combinators(
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
            return make_issue(
                ArgumentIssueCode::CombinatorMismatch,
                relative_path(path),
                {},
                std::format("{} does not satisfy {}", path, keyword));
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<ArgumentIssue> validate_object(
    Object value,
    Object schema,
    std::string_view path) {
    // Unknown arguments are reported before missing required properties: when
    // a model renames a parameter both conditions hold at once, and the
    // rename (unknown argument) is the actionable diagnosis — recovery
    // machinery relies on this precedence to infer replacement lessons.
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
            return make_issue(
                ArgumentIssueCode::UnknownArgument,
                relative_path(child_path),
                property_names(schema),
                std::format("Unknown argument '{}' at {}", field.key, path));
        }
    }

    simdjson::dom::array required;
    if (schema["required"].get(required) == simdjson::SUCCESS) {
        for (Element item : required) {
            std::string_view name;
            if (item.get(name) != simdjson::SUCCESS) continue;
            Element ignored;
            if (value[name].get(ignored) != simdjson::SUCCESS) {
                return make_issue(
                    ArgumentIssueCode::MissingRequired,
                    relative_path(std::format("{}.{}", path, name)),
                    property_names(schema),
                    std::format("{} is missing required property '{}'", path, name));
            }
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<ArgumentIssue> validate_value(
    Element value,
    Element schema_element,
    std::string_view path) {
    Object schema;
    if (schema_element.get(schema) != simdjson::SUCCESS) return std::nullopt;
    if (auto error = validate_combinators(value, schema, path)) return error;

    const std::string actual = element_type_name(value);
    if (!schema_accepts_type(schema_element, actual)
        && !(actual == "integer" && schema_accepts_type(schema_element, "number"))) {
        return make_issue(
            ArgumentIssueCode::TypeMismatch,
            relative_path(path),
            declared_type_tokens(schema_element),
            std::format("{} must be {}, got {}",
                        path,
                        core::utils::json::string_field(schema, "type", "a valid schema type"),
                        actual));
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
        if (!matched) {
            return make_issue(
                ArgumentIssueCode::EnumMismatch,
                relative_path(path),
                enum_string_values(schema),
                std::format("{} is not one of the allowed values", path));
        }
    }

    Element constant;
    if (schema["const"].get(constant) == simdjson::SUCCESS
        && canonicalize_json(simdjson::to_string(constant))
            != canonicalize_json(simdjson::to_string(value))) {
        return make_issue(
            ArgumentIssueCode::ConstMismatch,
            relative_path(path),
            {},
            std::format("{} does not match the required constant", path));
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

[[nodiscard]] bool is_required(Object schema, std::string_view name) {
    simdjson::dom::array required;
    if (schema["required"].get(required) != simdjson::SUCCESS) return false;
    for (Element item : required) {
        std::string_view value;
        if (item.get(value) == simdjson::SUCCESS && value == name) return true;
    }
    return false;
}

[[nodiscard]] std::string strip_optional_nulls(Object arguments,
                                               Object schema) {
    std::vector<std::pair<std::string, Element>> fields;
    for (const auto field : arguments) {
        const bool is_null = field.value.type() == simdjson::dom::element_type::NULL_VALUE;
        Element property;
        const bool declared = schema["properties"][field.key].get(property)
            == simdjson::SUCCESS;
        // Providers often send explicit null for unused optionals. Drop those
        // only when the schema does not accept null — a nullable property uses
        // null as a real value (clear/reset), not an omission.
        if (is_null && declared && !is_required(schema, field.key)
            && !schema_accepts_type(property, "null")) {
            continue;
        }
        fields.emplace_back(std::string(field.key), field.value);
    }
    std::ranges::sort(fields, {}, &std::pair<std::string, Element>::first);
    core::utils::JsonWriter writer(128);
    {
        auto root = writer.object();
        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (i > 0) writer.comma();
            writer.key(fields[i].first);
            write_canonical(writer, fields[i].second);
        }
    }
    return std::move(writer).take();
}

} // namespace

std::string_view issue_code_name(ArgumentIssueCode code) noexcept {
    switch (code) {
        case ArgumentIssueCode::InvalidJson: return "invalid_json";
        case ArgumentIssueCode::NotAnObject: return "not_an_object";
        case ArgumentIssueCode::SchemaInvalid: return "schema_invalid";
        case ArgumentIssueCode::NormalizeFailed: return "normalize_failed";
        case ArgumentIssueCode::UnknownArgument: return "unknown_argument";
        case ArgumentIssueCode::MissingRequired: return "missing_required";
        case ArgumentIssueCode::TypeMismatch: return "type_mismatch";
        case ArgumentIssueCode::EnumMismatch: return "enum_mismatch";
        case ArgumentIssueCode::ConstMismatch: return "const_mismatch";
        case ArgumentIssueCode::CombinatorMismatch: return "combinator_mismatch";
    }
    return "unknown";
}

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

std::string structural_input_schema(const ToolDefinition& definition) {
    simdjson::dom::parser parser;
    parser.number_as_string(true);
    const std::string canonical = canonical_input_schema(definition);
    simdjson::padded_string padded(canonical);
    Element root;
    if (parser.parse(padded).get(root) != simdjson::SUCCESS) {
        return canonical;
    }
    core::utils::JsonWriter writer(canonical.size() + 64);
    write_structural(writer, root, StructuralRole::Schema);
    return std::move(writer).take();
}

std::expected<std::string, ArgumentIssue> validate_arguments(
    const ToolDefinition& definition,
    std::string_view raw_arguments) {
    const std::string arguments_json = raw_arguments.empty() ? "{}" : std::string(raw_arguments);
    const std::string schema_json = canonical_input_schema(definition);

    simdjson::dom::parser arguments_parser;
    arguments_parser.number_as_string(true);
    simdjson::padded_string padded_arguments(arguments_json);
    Element arguments_root;
    if (arguments_parser.parse(padded_arguments).get(arguments_root) != simdjson::SUCCESS) {
        return std::unexpected(make_issue(
            ArgumentIssueCode::InvalidJson, {}, {}, "arguments are not valid JSON"));
    }
    Object arguments;
    if (arguments_root.get(arguments) != simdjson::SUCCESS) {
        return std::unexpected(make_issue(
            ArgumentIssueCode::NotAnObject, {}, {}, "arguments must be a JSON object"));
    }

    simdjson::dom::parser schema_parser;
    schema_parser.number_as_string(true);
    simdjson::padded_string padded_schema(schema_json);
    Element schema_root;
    Object schema_object;
    if (schema_parser.parse(padded_schema).get(schema_root) != simdjson::SUCCESS
        || schema_root.get(schema_object) != simdjson::SUCCESS) {
        return std::unexpected(make_issue(
            ArgumentIssueCode::SchemaInvalid, {}, {}, "tool has an invalid input schema"));
    }

    const std::string normalized = strip_optional_nulls(arguments, schema_object);
    simdjson::dom::parser normalized_parser;
    normalized_parser.number_as_string(true);
    simdjson::padded_string padded_normalized(normalized);
    Element normalized_root;
    if (normalized_parser.parse(padded_normalized).get(normalized_root) != simdjson::SUCCESS) {
        return std::unexpected(make_issue(
            ArgumentIssueCode::NormalizeFailed, {}, {}, "failed to normalize arguments"));
    }
    if (auto error = validate_value(normalized_root, schema_root, "$")) {
        return std::unexpected(std::move(*error));
    }
    return normalized;
}

std::expected<std::string, std::string> normalize_arguments(
    const ToolDefinition& definition,
    std::string_view raw_arguments) {
    auto result = validate_arguments(definition, raw_arguments);
    if (!result.has_value()) {
        return std::unexpected(std::move(result.error().message));
    }
    return std::expected<std::string, std::string>{
        std::in_place, std::move(*result)};
}

} // namespace core::tools::schema

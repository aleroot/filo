#include "StrictToolSchema.hpp"

#include "ToolSchema.hpp"
#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <string_view>

namespace core::tools::schema {
namespace {

using Element = simdjson::dom::element;
using Object = simdjson::dom::object;

/// What a provider's grammar compiler accepts. Expressed as data so the
/// rewriter stays a single algorithm: adding a third dialect is a table entry,
/// not another traversal.
struct DialectRules {
    /// OpenAI's strict mode requires every property in `required`; optionality
    /// is expressed by admitting null instead of by omission.
    bool require_all_properties = false;
    /// `oneOf` is outside both documented subsets; `anyOf` carries the same
    /// meaning for the disjoint alternatives tool schemas actually use.
    bool rewrite_oneof_as_anyof = false;
    /// Maximum object nesting depth, 0 when the provider documents none.
    std::size_t max_object_depth = 0;
};

[[nodiscard]] constexpr DialectRules rules_for(StrictDialect dialect) noexcept {
    switch (dialect) {
        case StrictDialect::OpenAI:
            return {.require_all_properties = true,
                    .rewrite_oneof_as_anyof = true,
                    .max_object_depth = 5};
        case StrictDialect::Anthropic:
            return {.require_all_properties = false,
                    .rewrite_oneof_as_anyof = true,
                    .max_object_depth = 0};
    }
    return {};
}

/// Keywords a grammar cannot enforce. Dropping them widens what the *provider*
/// will emit, never what Filo accepts: validate_arguments() still applies the
/// canonical schema to whatever comes back.
[[nodiscard]] bool is_unenforceable_keyword(std::string_view key) noexcept {
    constexpr std::array kKeywords = std::to_array<std::string_view>({
        "exclusiveMaximum", "exclusiveMinimum", "maxItems", "maxLength",
        "maxProperties", "maximum", "minItems", "minLength", "minProperties",
        "minimum", "multipleOf", "uniqueItems",
    });
    return std::ranges::find(kKeywords, key) != kKeywords.end();
}

/// Keywords whose meaning would be lost by the projection. Their presence ends
/// the attempt: a strictness claim over a schema we could not faithfully
/// translate would be a lie told to the sampler.
[[nodiscard]] bool is_untranslatable_keyword(std::string_view key) noexcept {
    constexpr std::array kKeywords = std::to_array<std::string_view>({
        "$defs", "$ref", "allOf", "contains", "definitions", "dependentRequired",
        "dependentSchemas", "else", "if", "not", "patternProperties",
        "prefixItems", "propertyNames", "then", "unevaluatedItems",
        "unevaluatedProperties",
    });
    return std::ranges::find(kKeywords, key) != kKeywords.end();
}

[[nodiscard]] bool declares(Object schema, std::string_view key) noexcept {
    Element ignored;
    return schema[key].get(ignored) == simdjson::SUCCESS;
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

/**
 * Rewrites a canonical schema into one dialect's subset.
 *
 * Every method returns false rather than emitting an approximation, and the
 * caller discards a partially written buffer, so a refused projection can never
 * reach the wire.
 */
class StrictProjector {
public:
    explicit StrictProjector(DialectRules rules) noexcept : rules_(rules) {}

    /// @param nullable Emit a schema that also admits null (an optional
    ///        property under a dialect that requires every property).
    [[nodiscard]] bool write_schema(core::utils::JsonWriter& writer,
                                    Element node,
                                    bool nullable,
                                    std::size_t depth) {
        Object schema;
        if (node.get(schema) != simdjson::SUCCESS) return false;

        for (const auto field : schema) {
            if (is_untranslatable_keyword(field.key)) return false;
        }
        // A fixed value cannot also admit null without changing its meaning.
        if (nullable && declares(schema, "const")) return false;

        const bool has_properties = declares(schema, "properties");
        if (has_properties && rules_.max_object_depth != 0
            && depth + 1 > rules_.max_object_depth) {
            return false;
        }

        const bool nullable_expressible = declares(schema, "type")
            || declares(schema, "enum")
            || declares(schema, "anyOf")
            || declares(schema, "oneOf");
        if (nullable && !nullable_expressible) return false;

        auto scope = writer.object();
        bool first = true;
        const auto separate = [&writer, &first] {
            if (!first) writer.comma();
            first = false;
        };

        for (const auto field : schema) {
            const std::string_view key = field.key;
            if (is_unenforceable_keyword(key)) continue;
            // Re-emitted below from the projected property set.
            if (key == "required" && rules_.require_all_properties) continue;

            if (key == "type") {
                separate();
                writer.key(key);
                write_type(writer, field.value, nullable);
                continue;
            }
            if (key == "enum") {
                separate();
                writer.key(key);
                write_enum(writer, field.value, nullable);
                continue;
            }
            if (key == "properties") {
                separate();
                writer.key(key);
                if (!write_properties(writer, field.value, schema, depth)) return false;
                continue;
            }
            if (key == "items" || key == "additionalProperties") {
                separate();
                writer.key(key);
                Object nested;
                if (field.value.get(nested) == simdjson::SUCCESS) {
                    if (!write_schema(writer, field.value, false, depth)) return false;
                } else {
                    writer.raw(simdjson::to_string(field.value));
                }
                continue;
            }
            if (key == "anyOf" || key == "oneOf") {
                separate();
                writer.key(rules_.rewrite_oneof_as_anyof ? "anyOf" : key);
                if (!write_alternatives(writer, field.value, nullable, depth)) return false;
                continue;
            }
            separate();
            writer.key(key);
            writer.raw(simdjson::to_string(field.value));
        }

        if (has_properties) {
            if (rules_.require_all_properties) {
                separate();
                writer.key("required");
                write_all_property_names(writer, schema);
            }
            if (!declares(schema, "additionalProperties")) {
                separate();
                writer.kv_bool("additionalProperties", false);
            }
        }
        return true;
    }

private:
    void write_type(core::utils::JsonWriter& writer, Element type, bool nullable) {
        std::string_view single;
        if (type.get(single) == simdjson::SUCCESS) {
            if (!nullable || single == "null") {
                writer.str(single);
                return;
            }
            auto scope = writer.array();
            writer.str(single).comma().str("null");
            return;
        }

        simdjson::dom::array tokens;
        if (type.get(tokens) != simdjson::SUCCESS) {
            writer.raw(simdjson::to_string(type));
            return;
        }
        bool has_null = false;
        for (Element token : tokens) {
            std::string_view value;
            if (token.get(value) == simdjson::SUCCESS && value == "null") has_null = true;
        }
        auto scope = writer.array();
        bool first = true;
        for (Element token : tokens) {
            if (!first) writer.comma();
            first = false;
            writer.raw(simdjson::to_string(token));
        }
        if (nullable && !has_null) {
            if (!first) writer.comma();
            writer.str("null");
        }
    }

    void write_enum(core::utils::JsonWriter& writer, Element values, bool nullable) {
        simdjson::dom::array members;
        if (values.get(members) != simdjson::SUCCESS) {
            writer.raw(simdjson::to_string(values));
            return;
        }
        bool has_null = false;
        for (Element member : members) {
            if (member.type() == simdjson::dom::element_type::NULL_VALUE) has_null = true;
        }
        auto scope = writer.array();
        bool first = true;
        for (Element member : members) {
            if (!first) writer.comma();
            first = false;
            writer.raw(simdjson::to_string(member));
        }
        if (nullable && !has_null) {
            if (!first) writer.comma();
            writer.null_val();
        }
    }

    [[nodiscard]] bool write_properties(core::utils::JsonWriter& writer,
                                        Element properties,
                                        Object parent,
                                        std::size_t depth) {
        Object members;
        if (properties.get(members) != simdjson::SUCCESS) return false;

        auto scope = writer.object();
        bool first = true;
        for (const auto member : members) {
            if (!first) writer.comma();
            first = false;
            writer.key(member.key);
            const bool nullable = rules_.require_all_properties
                && !is_required(parent, member.key);
            if (!write_schema(writer, member.value, nullable, depth + 1)) return false;
        }
        return true;
    }

    [[nodiscard]] bool write_alternatives(core::utils::JsonWriter& writer,
                                          Element alternatives,
                                          bool nullable,
                                          std::size_t depth) {
        simdjson::dom::array members;
        if (alternatives.get(members) != simdjson::SUCCESS) return false;

        auto scope = writer.array();
        bool first = true;
        bool admits_null = false;
        for (Element member : members) {
            if (!first) writer.comma();
            first = false;
            Object schema;
            if (member.get(schema) == simdjson::SUCCESS) {
                std::string_view type;
                if (schema["type"].get(type) == simdjson::SUCCESS && type == "null") {
                    admits_null = true;
                }
            }
            if (!write_schema(writer, member, false, depth)) return false;
        }
        if (nullable && !admits_null) {
            if (!first) writer.comma();
            auto null_scope = writer.object();
            writer.kv_str("type", "null");
        }
        return true;
    }

    void write_all_property_names(core::utils::JsonWriter& writer, Object schema) {
        Object properties;
        auto scope = writer.array();
        if (schema["properties"].get(properties) != simdjson::SUCCESS) return;
        bool first = true;
        for (const auto property : properties) {
            if (!first) writer.comma();
            first = false;
            writer.str(property.key);
        }
    }

    DialectRules rules_;
};

} // namespace

std::string_view strict_dialect_name(StrictDialect dialect) noexcept {
    switch (dialect) {
        case StrictDialect::OpenAI: return "openai";
        case StrictDialect::Anthropic: return "anthropic";
    }
    return "unknown";
}

std::optional<std::string> strict_input_schema(std::string_view canonical_schema,
                                               StrictDialect dialect) {
    simdjson::dom::parser parser;
    parser.number_as_string(true);
    simdjson::padded_string padded(canonical_schema);
    Element root;
    if (parser.parse(padded).get(root) != simdjson::SUCCESS) return std::nullopt;

    core::utils::JsonWriter writer(canonical_schema.size() + 256);
    StrictProjector projector(rules_for(dialect));
    if (!projector.write_schema(writer, root, /*nullable=*/false, /*depth=*/0)) {
        return std::nullopt;
    }
    return std::move(writer).take();
}

std::optional<std::string> strict_input_schema(const ToolDefinition& definition,
                                               StrictDialect dialect) {
    return strict_input_schema(canonical_input_schema(definition), dialect);
}

} // namespace core::tools::schema

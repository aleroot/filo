#pragma once

/**
 * @file
 * @brief Internal validation primitives shared inside the tool-schema layer.
 *
 * These entry points exist so argument *coercion* can ask the one authoritative
 * validator whether a rewritten value is acceptable, instead of re-implementing
 * type rules that would inevitably drift. They are not part of the tool-facing
 * API: only translation units of the tool-schema layer may include this header.
 */

#include "ToolSchema.hpp"

#include <simdjson.h>

#include <optional>
#include <string_view>

namespace core::tools::schema::detail {

/// JSON Schema type token for a parsed value ("array", "integer", ...).
[[nodiscard]] std::string_view element_type_name(
    simdjson::dom::element value) noexcept;

/// True when @p schema admits a value of JSON Schema type @p wanted. A schema
/// without a "type" keyword admits everything its combinators admit.
[[nodiscard]] bool schema_accepts_type(simdjson::dom::element schema,
                                       std::string_view wanted);

/// Full structural validation of @p value against @p schema. @p path is the
/// JSONPath-style prefix used in diagnostics ("$" for the argument root).
/// Returns the first issue, or nullopt when the value satisfies the schema.
[[nodiscard]] std::optional<ArgumentIssue> validate_value(
    simdjson::dom::element value,
    simdjson::dom::element schema,
    std::string_view path);

} // namespace core::tools::schema::detail

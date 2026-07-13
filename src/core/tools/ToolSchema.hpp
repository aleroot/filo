#pragma once

#include "Tool.hpp"

#include <expected>
#include <string>
#include <string_view>

namespace core::tools::schema {

/**
 * Build the authoritative JSON Schema for a tool invocation.
 *
 * A supplied ToolDefinition::input_schema is preserved (apart from deterministic
 * JSON member ordering and safe type inference for underspecified properties).
 * Otherwise the schema is generated from ToolDefinition::parameters.
 */
[[nodiscard]] std::string canonical_input_schema(const ToolDefinition& definition);

/**
 * Normalize and validate model-generated arguments against the authoritative
 * tool schema. Object keys are emitted deterministically. A null supplied for
 * an optional, non-nullable top-level property is treated as an omitted value;
 * this safely bridges providers that represent unused optional arguments with
 * explicit nulls.
 *
 * The error value is a human-readable validation message, not a JSON payload.
 */
[[nodiscard]] std::expected<std::string, std::string> normalize_arguments(
    const ToolDefinition& definition,
    std::string_view raw_arguments);

} // namespace core::tools::schema

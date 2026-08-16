#pragma once

#include "Tool.hpp"

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools::schema {

/**
 * Machine-readable classification of a rejected tool invocation.
 *
 * The historical normalize_arguments() contract collapses validation failures
 * into a single human-readable string, which loses the structure recovery
 * machinery needs (which parameter offended, what the schema declared).
 * validate_arguments() preserves that structure; normalize_arguments() is now
 * a compatibility adapter over it.
 */
enum class ArgumentIssueCode {
    InvalidJson,        ///< The raw arguments are not parsable JSON.
    NotAnObject,        ///< The arguments are JSON but not an object.
    SchemaInvalid,      ///< The tool definition carries an unusable schema.
    NormalizeFailed,    ///< Internal normalization unexpectedly failed.
    UnknownArgument,    ///< An argument is not declared by the schema.
    MissingRequired,    ///< A required argument is absent.
    TypeMismatch,       ///< An argument's JSON type is not accepted.
    EnumMismatch,       ///< An argument value is outside the allowed enum.
    ConstMismatch,      ///< An argument value differs from the required const.
    CombinatorMismatch, ///< A oneOf/anyOf constraint is not satisfied.
};

/// Stable, persisted identifier for an issue code (e.g. "unknown_argument").
[[nodiscard]] std::string_view issue_code_name(ArgumentIssueCode code) noexcept;

/// Structured description of one rejected tool invocation.
struct ArgumentIssue {
    ArgumentIssueCode code = ArgumentIssueCode::InvalidJson;
    /// Offending parameter path relative to the argument root (e.g. "query",
    /// "items[2].path"). Empty when no specific parameter is at fault.
    std::string parameter;
    /// Candidate names/values declared at the offending schema position in a
    /// deterministic order: property names for UnknownArgument/MissingRequired,
    /// enum string values for EnumMismatch, expected type tokens for
    /// TypeMismatch. Empty when the schema offers no candidates.
    std::vector<std::string> allowed;
    /// Human-readable description (the historical validation message).
    std::string message;
};

/**
 * Normalize and validate model-generated arguments against the authoritative
 * tool schema, preserving the structured failure classification.
 *
 * Behavior is identical to normalize_arguments(); only the error payload type
 * differs. See normalize_arguments() for the full contract.
 *
 * The error value's message is human-readable, not a JSON payload.
 */
[[nodiscard]] std::expected<std::string, ArgumentIssue> validate_arguments(
    const ToolDefinition& definition,
    std::string_view raw_arguments);

/**
 * Normalize and validate model-generated arguments against the authoritative
 * tool schema. Object keys are emitted deterministically. A null supplied for
 * an optional, non-nullable top-level property is treated as an omitted value;
 * this safely bridges providers that represent unused optional arguments with
 * explicit nulls. A null for a property that accepts null is preserved.
 *
 * The error value is a human-readable validation message, not a JSON payload.
 */
[[nodiscard]] std::expected<std::string, std::string> normalize_arguments(
    const ToolDefinition& definition,
    std::string_view raw_arguments);

/**
 * Build the authoritative JSON Schema for a tool invocation.
 *
 * A supplied ToolDefinition::input_schema is preserved (apart from deterministic
 * JSON member ordering and safe type inference for underspecified properties).
 * Otherwise the schema is generated from ToolDefinition::parameters.
 */
[[nodiscard]] std::string canonical_input_schema(const ToolDefinition& definition);

/**
 * Canonical input schema with every schema-annotation "description" removed.
 *
 * Descriptions guide the model but never change argument validity, so this is
 * the stable *structural* identity of a tool contract: it survives description
 * rewording and changes exactly when the contract changes. A property actually
 * named "description" is part of the contract and is kept. Used to scope
 * learned tool-recovery lessons to a schema version.
 */
[[nodiscard]] std::string structural_input_schema(const ToolDefinition& definition);

} // namespace core::tools::schema

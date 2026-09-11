#pragma once

/**
 * @file
 * @brief Decides when tool schemas may be sent under a provider's strict
 *        (constrained-decoding) contract.
 *
 * Strict tool schemas are not a capability every model advertises over the
 * wire: neither vendor's model listing reports it, so the decision is made from
 * documented model families plus an explicit configuration switch. Guessing
 * wrong is not a soft failure — a provider that does not recognise the strict
 * contract rejects the whole request — which is why the default is off and the
 * model table lists only generations documented as supporting it.
 */

#include "../tools/StrictToolSchema.hpp"

#include <optional>
#include <string_view>

namespace core::llm {

/// Wire formats that carry a per-tool strictness flag.
enum class ToolSchemaWire {
    OpenAI,     ///< Chat Completions and Responses: `"strict": true` on the function.
    Anthropic,  ///< Messages: `"strict": true` alongside `input_schema`.
};

/**
 * Whether @p model's generation documents provider-side strict tool use.
 *
 * Pure and case-insensitive. Unknown models answer false: an unrecognised
 * identifier is treated as unsupported, never as probably-fine.
 */
[[nodiscard]] bool model_supports_strict_tools(ToolSchemaWire wire,
                                               std::string_view model) noexcept;

/**
 * The dialect to project tool schemas into, or nullopt to stay on the
 * permissive wire format.
 *
 * @param enabled The resolved configuration switch, passed in so the decision
 *                stays a pure function of its inputs.
 */
[[nodiscard]] std::optional<core::tools::schema::StrictDialect> strict_tool_dialect(
    ToolSchemaWire wire,
    std::string_view model,
    bool enabled) noexcept;

/// strict_tool_dialect() against the active configuration's `strict_tool_schemas`.
[[nodiscard]] std::optional<core::tools::schema::StrictDialect>
configured_strict_tool_dialect(ToolSchemaWire wire, std::string_view model);

} // namespace core::llm

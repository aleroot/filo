#pragma once

/**
 * @file
 * @brief Projection of a tool contract into a provider's strict-decoding subset.
 *
 * Providers that expose constrained decoding for tool calls (OpenAI
 * `"strict": true`, Anthropic strict tool use) compile the supplied schema into
 * a grammar and mask the sampler with it, which makes an out-of-shape argument
 * unrepresentable rather than merely invalid. That is the only place where the
 * "array serialized into a string" class of failure can be prevented instead of
 * repaired, but it is bought with a narrower JSON Schema dialect: bounds
 * keywords are rejected, some structures are unsupported, and OpenAI requires
 * every property to be listed as required.
 *
 * This unit rewrites the canonical schema into that subset. Two properties make
 * the trade sound:
 *
 *  - It never *tightens* the contract. A projection that cannot be expressed
 *    without changing what the model is allowed to send is refused (nullopt),
 *    and the caller falls back to the permissive wire format.
 *  - Where it *loosens* the contract — a stripped `maxLength`, an optional made
 *    nullable — the authoritative validator still runs on the way in, so the
 *    end-to-end contract is unchanged. The grammar owns shape; validation owns
 *    everything else.
 */

#include "Tool.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace core::tools::schema {

/// Constrained-decoding dialects. They differ in how much of JSON Schema the
/// provider's grammar compiler accepts, not in what the tool means.
enum class StrictDialect {
    OpenAI,     ///< Every property required; optionals expressed as null unions.
    Anthropic,  ///< Optional properties stay optional; no bounds keywords.
};

/// Stable identifier for a dialect, for diagnostics and tests.
[[nodiscard]] std::string_view strict_dialect_name(StrictDialect dialect) noexcept;

/**
 * Projects @p definition's canonical input schema into @p dialect's subset.
 *
 * @return The projected schema, or nullopt when the contract cannot be
 *         expressed in that subset — in which case the tool must be sent
 *         without a strictness claim rather than with a weakened one.
 */
[[nodiscard]] std::optional<std::string> strict_input_schema(
    const ToolDefinition& definition,
    StrictDialect dialect);

/**
 * Overload for callers that already hold the canonical schema (every wire
 * serializer does), so a projection costs one parse instead of two.
 */
[[nodiscard]] std::optional<std::string> strict_input_schema(
    std::string_view canonical_schema,
    StrictDialect dialect);

} // namespace core::tools::schema

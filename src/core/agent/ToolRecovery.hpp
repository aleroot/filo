#pragma once

#include "../memory/ToolRecoveryTypes.hpp"
#include "../tools/Tool.hpp"
#include "../tools/ToolSchema.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::agent::recovery {

// Persistence types live on the memory port. Re-exported so algorithm code
// and tests keep writing recovery::RecoveryKey / RecoveryLesson.
using RecoveryKey = core::memory::RecoveryKey;
using RecoveryLesson = core::memory::RecoveryLesson;
using core::memory::clamp_hint;
using core::memory::kMaxHintChars;
using core::memory::kRuntimeFailureCode;

/// Maximum number of candidate names listed in a single hint.
inline constexpr std::size_t kMaxListedCandidates = 8;

/// How many model steps a pending fail→success observation stays eligible.
/// Beyond this window the later success is unlikely to be a correction of the
/// earlier failure, and pairing them would manufacture a bogus lesson.
inline constexpr int kMaxObservationStepDistance = 4;

/// Guidance attached to a failing tool call. `learned` distinguishes hints
/// recalled from the persistent recovery memory (proven across sessions) from
/// hints deduced deterministically from the schema on the spot.
struct RecoveryHint {
    std::string instruction;
    bool learned = false;
};

/// Stable 64-bit FNV-1a over the structural input schema, hex encoded.
/// Not a security boundary — a collision's worst case is one irrelevant hint
/// that schema validation still rejects — so no crypto dependency is warranted.
[[nodiscard]] std::string schema_fingerprint(
    const core::tools::ToolDefinition& definition);

/// Builds the exact-match identity for a schema-detected argument issue.
[[nodiscard]] RecoveryKey make_key(
    std::string_view tool,
    const core::tools::ToolDefinition& definition,
    const core::tools::schema::ArgumentIssue& issue);

/**
 * Deterministic, stateless guidance for a rejected tool call.
 *
 * Derives a hint purely from the structured issue plus the tool contract:
 * nearest-parameter suggestions for unknown arguments (fixing the very first
 * failure without any learned state), required-parameter reminders, accepted
 * enum values, and expected types. Returns nullopt when the issue carries
 * nothing actionable.
 */
[[nodiscard]] std::optional<RecoveryHint> advise(
    const core::tools::schema::ArgumentIssue& issue,
    const core::tools::ToolDefinition& definition);

/**
 * Infers a lesson from a schema-detected failure followed by a successful
 * call to the same tool.
 *
 * A replacement is inferred only when the offending parameter disappears from
 * the successful call and exactly one new parameter appears; otherwise the
 * lesson just advises removing the parameter. All other issue codes carry
 * sufficient deterministic guidance in their own error text and are
 * deliberately not memorized.
 */
[[nodiscard]] std::optional<RecoveryLesson> derive_validation_lesson(
    std::string_view tool,
    const core::tools::ToolDefinition& definition,
    const core::tools::schema::ArgumentIssue& issue,
    std::string_view failed_arguments,
    std::string_view successful_arguments);

/**
 * Infers a lesson from a runtime failure (arguments validated but the tool
 * reported an error) followed by a successful call to the same tool.
 *
 * Stricter than the validation policy because there is no schema verdict to
 * anchor on: the two calls must be structurally identical except for exactly
 * one difference — a single removed parameter, or a single rename whose value
 * is unchanged. Anything else is ambiguous noise and yields no lesson.
 */
[[nodiscard]] std::optional<RecoveryLesson> derive_runtime_lesson(
    std::string_view tool,
    const core::tools::ToolDefinition& definition,
    std::string_view failed_arguments,
    std::string_view successful_arguments);

/// True when a tool result payload reports failure per the Tool result
/// contract (top-level "error" member or "isError": true).
[[nodiscard]] bool result_indicates_error(std::string_view tool_result);

/**
 * Appends a "recovery_hint" member to a JSON-object tool error payload.
 *
 * The payload is re-emitted with deterministic member ordering; when the
 * payload is not a JSON object it is returned unchanged rather than risking
 * corruption of a tool's own output format.
 */
[[nodiscard]] std::string augment_error_payload(std::string_view tool_result,
                                                std::string_view hint);

} // namespace core::agent::recovery

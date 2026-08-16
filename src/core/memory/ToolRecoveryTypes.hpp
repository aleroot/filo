#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace core::memory {

/// Issue-code namespace reserved for lessons inferred from runtime failures
/// (calls that validated but failed inside the tool). Persisted alongside
/// schema-derived codes in lesson keys.
inline constexpr std::string_view kRuntimeFailureCode = "runtime_failure";

/// Upper bound on a single hint's length. Hints are built from tool-supplied
/// parameter names, which are untrusted input for MCP tools, so every producer
/// clamps: an unbounded hint would inflate both the persisted store and the
/// failure payload sent back to the model.
inline constexpr std::size_t kMaxHintChars = 160;

/**
 * Identifies one recoverable argument mistake for a specific structural
 * version of a tool contract.
 *
 * The schema fingerprint scopes lessons to a contract version: reworded
 * descriptions never invalidate a lesson, while any structural change (renamed
 * or retyped parameter) does, because the fingerprint changes and stale
 * lessons simply never match again.
 */
struct RecoveryKey {
    std::string tool;
    std::string schema_fingerprint;
    std::string issue_code;
    std::string parameter;

    [[nodiscard]] friend bool operator==(const RecoveryKey&,
                                         const RecoveryKey&) = default;
};

/// A sanitized correction derived from a failed call followed by a successful
/// call. Raw argument values are never retained — only parameter names.
struct RecoveryLesson {
    RecoveryKey key;
    std::string hint;
};

/// Truncates `text` to kMaxHintChars without splitting a UTF-8 sequence,
/// appending an ellipsis when characters were dropped.
[[nodiscard]] std::string clamp_hint(std::string text);

} // namespace core::memory

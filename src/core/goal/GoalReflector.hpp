#pragma once

#include "GoalTypes.hpp"
#include "GoalVerifier.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace core::goal {

// ---------------------------------------------------------------------------
// GoalReflector — turns failures into reusable verbal lessons (Reflexion:
// store natural-language critiques and condition the next attempt on them,
// rather than retrying blindly).
//
// Output contract of the critique model:
//   CRITIQUE:
//   <what went wrong and why>
//   LESSONS:
//   - <actionable lesson for the next attempt>
//   - ...
// Parsing is deliberately forgiving; a deterministic heuristic reflection is
// synthesized when no model is configured, so the retry loop always improves.
// ---------------------------------------------------------------------------

struct Reflection {
    std::string critique;
    std::vector<std::string> lessons;
};

class GoalReflector {
public:
    /// @param complete judge/critique model; may be empty (heuristic fallback).
    explicit GoalReflector(CompletionFn complete);

    /// Critique a failed node against the verdict that failed it.
    [[nodiscard]] Reflection reflect(const Node& node,
                                     const Verdict& verdict,
                                     const VerifyContext& context) const;

    /// Render accumulated lessons as retry context prepended to the next
    /// work attempt. Empty when the node has no lessons yet.
    [[nodiscard]] static std::string build_retry_context(const Node& node);

    /// Model-free fallback: critique = verdict reason, one actionable lesson.
    [[nodiscard]] static Reflection heuristic(const Node& node, const Verdict& verdict);

    /// Parse the CRITIQUE/LESSONS contract. Tolerates missing markers.
    [[nodiscard]] static Reflection parse_response(std::string_view response);

private:
    CompletionFn complete_;
};

} // namespace core::goal

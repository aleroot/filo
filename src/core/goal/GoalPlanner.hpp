#pragma once

#include "GoalGraph.hpp"
#include "GoalVerifier.hpp"

#include <expected>
#include <span>
#include <string>
#include <string_view>

namespace core::goal {

enum class PlanProfile {
  /// Full manually-operated goal graph, including verification nodes.
  FullGoal,
  /// AUTO preflight graph. Exploration is executed by the scheduler; the
  /// parent agent remains the exclusive writer and owns final verification.
  AutoExecution,
};

// ---------------------------------------------------------------------------
// GoalPlanner — decomposes a goal objective into a validated DAG before any
// execution happens (plan-first: agents that plan first solve what agents
// that rush can't).
//
// The model emits a strict JSON contract; the planner parses, sanitizes, and
// structurally validates it. Any failure (no model, malformed JSON, invalid
// graph) degrades gracefully to fallback_linear_plan(): a minimal
// work -> verify chain that still carries the full verify/reflect semantics.
// Planning therefore never blocks a goal from running.
// ---------------------------------------------------------------------------
class GoalPlanner {
public:
  /// @param complete planning model; may be empty (fallback plans only).
  explicit GoalPlanner(CompletionFn complete);

  /// Build a validated plan graph for @p objective. @p context is optional
  /// grounding text (repo summary, active goal note) injected into the prompt.
  [[nodiscard]] std::expected<GoalGraph, std::string>
  plan(std::string_view objective, std::string_view context = {},
       PlanProfile profile = PlanProfile::FullGoal,
       std::span<const std::string> fallback_verification_recipe_ids = {}) const;

  /// Deterministic minimal plan: one work node (budget 3 attempts) feeding
  /// one verify node whose failure requeues the work node.
  [[nodiscard]] static GoalGraph
  fallback_linear_plan(std::string_view objective,
                       PlanProfile profile = PlanProfile::FullGoal,
                       std::span<const std::string> verification_recipe_ids = {});

  /// Parse the planner JSON contract into a validated graph.
  /// Exposed for testing and for the replan path.
  [[nodiscard]] static std::expected<GoalGraph, std::string>
  parse_plan_json(std::string_view json, std::string_view objective);

private:
  CompletionFn complete_;
};

} // namespace core::goal

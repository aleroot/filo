#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::goal {

// ---------------------------------------------------------------------------
// GoalTypes — shared value types for the goal-graph engine.
//
// A goal is a versioned directed acyclic graph (DAG) of typed nodes:
//   plan -> decompose the objective before executing        (rule: planning)
//   work -> act on the repo through tools/subagents         (rule: tool use)
//   verify -> check a contract, deterministic command first (rule: tool use)
//   reflect -> critique a failure into reusable lessons     (rule: reflection)
//   fan_in / gate / replan -> synthesis, human approval, repair
//
// The graph itself stays acyclic: retry loops are expressed as per-node
// attempt budgets managed by the scheduler, never as back-edges.
// ---------------------------------------------------------------------------

using NodeId = std::uint32_t;
inline constexpr NodeId kInvalidNodeId = 0;

enum class NodeKind : std::uint8_t {
    Plan,
    Work,
    Verify,
    Reflect,
    FanIn,
    Gate,
    Replan,
};

[[nodiscard]] inline std::string_view to_string(NodeKind kind) noexcept {
    switch (kind) {
    case NodeKind::Plan:    return "plan";
    case NodeKind::Work:    return "work";
    case NodeKind::Verify:  return "verify";
    case NodeKind::Reflect: return "reflect";
    case NodeKind::FanIn:   return "fan_in";
    case NodeKind::Gate:    return "gate";
    case NodeKind::Replan:  return "replan";
    }
    return "work";
}

[[nodiscard]] inline std::optional<NodeKind> node_kind_from_string(
    std::string_view kind) noexcept {
    if (kind == "plan") return NodeKind::Plan;
    if (kind == "work") return NodeKind::Work;
    if (kind == "verify") return NodeKind::Verify;
    if (kind == "reflect") return NodeKind::Reflect;
    if (kind == "fan_in" || kind == "fanin") return NodeKind::FanIn;
    if (kind == "gate") return NodeKind::Gate;
    if (kind == "replan") return NodeKind::Replan;
    return std::nullopt;
}

enum class NodeState : std::uint8_t {
    Pending,
    Ready,
    Running,
    Succeeded,
    Failed,
    Skipped,
    Blocked,
};

[[nodiscard]] inline std::string_view to_string(NodeState state) noexcept {
    switch (state) {
    case NodeState::Pending:   return "pending";
    case NodeState::Ready:     return "ready";
    case NodeState::Running:   return "running";
    case NodeState::Succeeded: return "succeeded";
    case NodeState::Failed:    return "failed";
    case NodeState::Skipped:   return "skipped";
    case NodeState::Blocked:   return "blocked";
    }
    return "pending";
}

[[nodiscard]] inline NodeState node_state_from_string(std::string_view state) noexcept {
    if (state == "ready") return NodeState::Ready;
    if (state == "running") return NodeState::Running;
    if (state == "succeeded") return NodeState::Succeeded;
    if (state == "failed") return NodeState::Failed;
    if (state == "skipped") return NodeState::Skipped;
    if (state == "blocked") return NodeState::Blocked;
    return NodeState::Pending;
}

[[nodiscard]] inline bool is_terminal(NodeState state) noexcept {
    return state == NodeState::Succeeded || state == NodeState::Failed
        || state == NodeState::Skipped || state == NodeState::Blocked;
}

/// Edge semantics. Always/OnPass form hard dependencies that gate readiness;
/// OnFail/OnExhausted are control-flow routes activated by the scheduler.
enum class EdgeCondition : std::uint8_t {
    Always,
    OnPass,
    OnFail,
    OnExhausted,
};

[[nodiscard]] inline std::string_view to_string(EdgeCondition condition) noexcept {
    switch (condition) {
    case EdgeCondition::Always:      return "always";
    case EdgeCondition::OnPass:      return "on_pass";
    case EdgeCondition::OnFail:      return "on_fail";
    case EdgeCondition::OnExhausted: return "on_exhausted";
    }
    return "always";
}

[[nodiscard]] inline std::optional<EdgeCondition> edge_condition_from_string(
    std::string_view condition) noexcept {
    if (condition == "always" || condition == "next") return EdgeCondition::Always;
    if (condition == "on_pass" || condition == "pass") return EdgeCondition::OnPass;
    if (condition == "on_fail" || condition == "fail") return EdgeCondition::OnFail;
    if (condition == "on_exhausted" || condition == "exhausted") {
        return EdgeCondition::OnExhausted;
    }
    return std::nullopt;
}

[[nodiscard]] inline bool is_dependency(EdgeCondition condition) noexcept {
    return condition == EdgeCondition::Always || condition == EdgeCondition::OnPass;
}

enum class RunState : std::uint8_t {
    Idle,
    Planning,
    Running,
    Paused,
    Completed,
    Failed,
    Blocked,
};

[[nodiscard]] inline std::string_view to_string(RunState state) noexcept {
    switch (state) {
    case RunState::Idle:      return "idle";
    case RunState::Planning:  return "planning";
    case RunState::Running:   return "running";
    case RunState::Paused:    return "paused";
    case RunState::Completed: return "completed";
    case RunState::Failed:    return "failed";
    case RunState::Blocked:   return "blocked";
    }
    return "idle";
}

// Workspace access is an execution capability, not a suggestion to the model.
// SharedRead nodes may be scheduled concurrently. ExclusiveWrite nodes require
// the repository's single-writer lease and are always serialized.
enum class WorkspaceAccess : std::uint8_t {
  SharedRead,
  ExclusiveWrite,
};

[[nodiscard]] inline std::string_view
to_string(WorkspaceAccess access) noexcept {
  switch (access) {
  case WorkspaceAccess::SharedRead:
    return "shared_read";
  case WorkspaceAccess::ExclusiveWrite:
    return "exclusive_write";
  }
  return "exclusive_write";
}

[[nodiscard]] inline std::optional<WorkspaceAccess>
workspace_access_from_string(std::string_view access) noexcept {
  if (access == "shared_read" || access == "read_only" || access == "read") {
    return WorkspaceAccess::SharedRead;
  }
  if (access == "exclusive_write" || access == "workspace_write" ||
      access == "write") {
    return WorkspaceAccess::ExclusiveWrite;
  }
  return std::nullopt;
}

[[nodiscard]] inline RunState run_state_from_string(std::string_view state) noexcept {
    if (state == "planning") return RunState::Planning;
    if (state == "running") return RunState::Running;
    if (state == "paused") return RunState::Paused;
    if (state == "completed") return RunState::Completed;
    if (state == "failed") return RunState::Failed;
    if (state == "blocked") return RunState::Blocked;
    return RunState::Idle;
}

[[nodiscard]] inline bool is_terminal(RunState state) noexcept {
    return state == RunState::Completed || state == RunState::Failed
        || state == RunState::Blocked;
}

// ---------------------------------------------------------------------------
// Node — one typed unit of work with a verifiable contract.
// ---------------------------------------------------------------------------
struct Node {
  static constexpr std::size_t kMaxNameChars = 128;
  static constexpr std::size_t kMaxDirectiveChars = 4096;
  static constexpr std::size_t kMaxCheckChars = 1024;
  static constexpr std::size_t kMaxAcceptanceChars = 2048;
  static constexpr std::size_t kMaxResultChars = 2048;
  static constexpr std::size_t kMaxLessonChars = 512;
  static constexpr std::size_t kMaxLessons = 8;
  static constexpr std::size_t kMaxVerificationRecipes = 16;
  static constexpr int kMaxAttempts = 8;

  NodeId id = kInvalidNodeId;
  NodeKind kind = NodeKind::Work;
  std::string name;          ///< short unique label, e.g. "migrate-net"
  std::string directive;     ///< what to do (work) / what was decomposed (plan)
  std::string check_command; ///< verify: deterministic shell check, empty =
                             ///< model-judged
  std::string acceptance; ///< contract text for the model verifier / reflector
  int max_attempts =
      2; ///< retry budget for the scheduler (clamped to kMaxAttempts)
  int attempts = 0;
  NodeState state = NodeState::Pending;
  WorkspaceAccess workspace_access = WorkspaceAccess::ExclusiveWrite;
  bool parallelizable = false; ///< only valid with SharedRead access
  NodeId retry_target =
      kInvalidNodeId; ///< verify/reflect: work node requeued on failure
  /// Trusted recipe ids resolved by the verification catalog. The graph
  /// never stores caller-provided shell strings for new plans.
  std::vector<std::string> verification_recipe_ids;
  std::string result_summary; ///< filled on completion (clamped)
  std::vector<std::string>
      lessons; ///< accumulated reflections (Reflexion memory)
};

struct Edge {
    NodeId from = kInvalidNodeId;
    NodeId to = kInvalidNodeId;
    EdgeCondition condition = EdgeCondition::Always;
};

} // namespace core::goal

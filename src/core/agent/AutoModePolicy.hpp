#pragma once

#include "../llm/routing/AutoClassifier.hpp"

#include <cstddef>
#include <string>
#include <string_view>

namespace core::agent {

enum class AutoExecutionPath {
  Direct,
  Orchestrated,
};

[[nodiscard]] std::string_view to_string(AutoExecutionPath path) noexcept;

struct AutoModeContext {
  std::size_t history_tokens = 0;
  int turn_count = 0;
  bool has_tool_history = false;
};

struct AutoModeDecision {
  AutoExecutionPath path = AutoExecutionPath::Direct;
  core::llm::routing::TaskType task_type = core::llm::routing::TaskType::Simple;
  double complexity = 0.0;
  bool parallel_exploration = false;
  bool verification_required_after_mutation = false;
  std::string reason;
};

// Pure policy: it selects an execution topology but owns no agent, UI, or
// provider state. That makes AUTO deterministic, testable, and reusable by
// every frontend.
class AutoModePolicy {
public:
  AutoModePolicy() noexcept = default;
  explicit AutoModePolicy(
      core::llm::routing::AutoClassifier classifier) noexcept;

  [[nodiscard]] AutoModeDecision
  decide(std::string_view prompt, AutoModeContext context = {}) const noexcept;

  [[nodiscard]] std::string
  execution_contract(const AutoModeDecision &decision) const;

private:
  core::llm::routing::AutoClassifier classifier_;
};

} // namespace core::agent

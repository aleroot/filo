#pragma once

#include "../goal/GoalPlanner.hpp"
#include "../goal/GoalScheduler.hpp"

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace core::agent {

struct AutoGraphFinding {
  std::string node_name;
  bool succeeded = false;
  std::string evidence;
};

struct AutoGraphPreparation {
  std::string rendered_graph;
  std::vector<AutoGraphFinding> findings;
  std::vector<std::string> warnings;

  [[nodiscard]] bool explored() const noexcept { return !findings.empty(); }
};

// Harness-owned preflight for orchestrated AUTO turns. The planner may propose
// a graph, but this class validates it and executes only shared-read frontier
// nodes. The parent agent remains the sole writer in the user's checkout.
class AutoGraphOrchestrator {
public:
  using ExploreFn = std::function<core::goal::WorkOutcome(
      const core::goal::Node &, std::string_view)>;

  struct Hooks {
    core::goal::CompletionFn complete;
    ExploreFn explore;
    std::function<void(const core::goal::GoalEvent &)> on_event;
    std::function<bool()> cancellation_requested;
  };

  [[nodiscard]] AutoGraphPreparation
  prepare(std::string_view objective, std::string_view repository_context,
          Hooks hooks) const noexcept;

  [[nodiscard]] static std::string
  render_for_prompt(const AutoGraphPreparation &preparation);
};

} // namespace core::agent

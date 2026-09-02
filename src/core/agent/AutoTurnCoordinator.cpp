#include "AutoTurnCoordinator.hpp"

#include "PermissionGate.hpp"
#include "../logging/Logger.hpp"
#include "../tools/ToolNames.hpp"

#include <algorithm>
#include <ranges>
#include <utility>

namespace core::agent {

namespace {

[[nodiscard]] bool writer_expected(
    core::llm::routing::TaskType task_type) noexcept {
  using core::llm::routing::TaskType;
  return task_type == TaskType::CodeGen || task_type == TaskType::Debugging ||
         task_type == TaskType::Architecture;
}

[[nodiscard]] bool mutates_workspace(const AutoToolIntent &tool) {
  if (!tool.approved)
    return false;
  if (core::tools::names::is_file_modification_tool(tool.name))
    return true;
  if (core::tools::names::is_subagent_tool(tool.name))
    return !tool.read_only_subagent;
  if (core::tools::names::is_terminal_tool(tool.name)) {
    const auto command = CommandSafetyPolicy::extract_shell_command(
        tool.arguments);
    return CommandSafetyPolicy::classify(command) != CommandSafetyClass::Safe;
  }
  return tool.destructive_hint;
}

} // namespace

AutoTurnCoordinator::AutoTurnCoordinator(
    std::shared_ptr<core::scm::WorkspaceLeaseRegistry> workspace_leases)
    : workspace_(std::move(workspace_leases)) {}

std::unique_ptr<AutoTurnState> AutoTurnCoordinator::start(
    std::string_view prompt,
    AutoModeContext context,
    const std::filesystem::path &workspace_root) const {
  auto turn = std::unique_ptr<AutoTurnState>(new AutoTurnState);
  turn->decision = policy_.decide(prompt, context);
  auto discovery = verification_.discover(workspace_root);
  turn->verification_recipes = std::move(discovery.recipes);
  turn->verification_warnings = std::move(discovery.warnings);
  turn->verification_config_invalid =
      discovery.project_config_present && !discovery.project_config_valid;
  return turn;
}

std::size_t AutoTurnCoordinator::prepare(
    AutoTurnState &turn,
    std::string_view objective,
    std::string_view repository_context,
    const std::filesystem::path &workspace_root,
    AutoGraphOrchestrator::Hooks hooks) const {
  const core::scm::GitWorkspaceCoordinator::AcquireOptions options{
      .cancellation_requested = hooks.cancellation_requested,
  };
  // SharedRead first: the read-only frontier must be able to run (and nested
  // readers must not block on a parent exclusive lock). ExclusiveWrite is
  // taken only after exploration, or later when a mutating tool is approved.
  turn.repository_lease.emplace(workspace_.acquire(
      workspace_root, core::goal::WorkspaceAccess::SharedRead, options));
  if (!turn.repository_lease->owns_lock()) {
    core::logging::warn(
        "[AUTO] could not acquire a shared repository lease for {}",
        workspace_root.string());
  }

  std::size_t findings = 0;
  if (turn.decision.path == AutoExecutionPath::Orchestrated && hooks.complete) {
    std::string planning_context(repository_context);
    planning_context += core::verification::Catalog::render_for_prompt(
        turn.verification_recipes);
    const auto preparation = graph_.prepare(
        objective, planning_context, std::move(hooks));
    findings = preparation.findings.size();
    turn.graph_prompt_context =
        AutoGraphOrchestrator::render_for_prompt(preparation);
  }

  if (writer_expected(turn.decision.task_type) &&
      turn.repository_lease.has_value()) {
    auto upgrade = workspace_.upgrade_to_exclusive(
        std::move(*turn.repository_lease), workspace_root, options);
    turn.repository_lease = std::move(upgrade.lease);
    if (!upgrade.upgraded) {
      core::logging::warn(
          "[AUTO] could not take the exclusive repository writer lease for {}; "
          "continuing without writer exclusion",
          workspace_root.string());
    }
  }

  turn.repository_baseline = workspace_.capture(workspace_root);
  if (turn.repository_baseline.has_value()) {
    turn.repository_prompt_context =
        core::scm::GitWorkspaceCoordinator::render_preflight(
            *turn.repository_baseline);
  }
  return findings;
}

std::string AutoTurnCoordinator::prompt_suffix(
    const AutoTurnState &turn) const {
  std::string result = policy_.execution_contract(turn.decision);
  result += turn.graph_prompt_context;
  result += turn.repository_prompt_context;
  result += core::verification::Catalog::render_for_prompt(
      turn.verification_recipes);
  result += core::verification::Catalog::render_warnings(
      turn.verification_warnings);
  return result;
}

const AutoModeDecision &AutoTurnCoordinator::decision(
    const AutoTurnState &turn) const noexcept {
  return turn.decision;
}

bool AutoTurnCoordinator::mutation_observed(
    const AutoTurnState &turn) const noexcept {
  return turn.quality.mutation_observed();
}

WorkspaceWriterState AutoTurnCoordinator::prepare_tool_batch(
    AutoTurnState &turn,
    const std::filesystem::path &workspace_root,
    std::span<const AutoToolIntent> tools,
    std::function<bool()> cancellation_requested) const {
  // Bind any receipt this batch mints to the pre-batch mutation counter, so a
  // write executing concurrently with a verification command cannot be treated
  // as covered by it.
  turn.quality.begin_tool_batch();

  if (!std::ranges::any_of(tools, mutates_workspace)) {
    return WorkspaceWriterState::NotRequired;
  }
  if (!turn.repository_lease.has_value()) {
    return WorkspaceWriterState::Unavailable;
  }
  if (turn.repository_lease->owns_lock() &&
      turn.repository_lease->access() ==
          core::goal::WorkspaceAccess::ExclusiveWrite) {
    // Already the exclusive writer for this turn; nothing to arrange.
    return WorkspaceWriterState::NotRequired;
  }

  auto upgrade = workspace_.upgrade_to_exclusive(
      std::move(*turn.repository_lease), workspace_root,
      core::scm::GitWorkspaceCoordinator::AcquireOptions{
          .cancellation_requested = std::move(cancellation_requested),
      });
  turn.repository_lease = std::move(upgrade.lease);
  return upgrade.upgraded ? WorkspaceWriterState::Acquired
                          : WorkspaceWriterState::Unavailable;
}

bool AutoTurnCoordinator::holds_workspace_lock(
    const AutoTurnState &turn) const noexcept {
  return turn.repository_lease.has_value() &&
         turn.repository_lease->owns_lock();
}

std::optional<core::goal::WorkspaceAccess>
AutoTurnCoordinator::workspace_access(const AutoTurnState &turn) const noexcept {
  if (!turn.repository_lease.has_value() ||
      !turn.repository_lease->owns_lock()) {
    return std::nullopt;
  }
  return turn.repository_lease->access();
}

void AutoTurnCoordinator::observe_tool(
    AutoTurnState &turn,
    const AutoToolObservation &observation) const noexcept {
  turn.quality.observe_tool(
      observation.name, observation.arguments, observation.result,
      observation.succeeded, observation.mutation_hint,
      observation.trusted_verification_receipts);
}

core::session::TurnCompletionResult AutoTurnCoordinator::evaluate_completion(
    AutoTurnState &turn,
    std::string_view response,
    const std::filesystem::path &workspace_root,
    const CompletionGate &run_completion_gate,
    const VerificationRunner &run_verification) const {
  // The tool batches are done; completion-gate receipts are produced
  // sequentially and bind to the live mutation counter.
  turn.quality.end_tool_batch();

  auto latest = verification_.discover(workspace_root);
  turn.verification_recipes = std::move(latest.recipes);
  turn.verification_warnings = std::move(latest.warnings);
  turn.verification_config_invalid =
      latest.project_config_present && !latest.project_config_valid;

  if (turn.repository_baseline.has_value() &&
      turn.quality.mutation_observed()) {
    const auto audit = workspace_.audit(*turn.repository_baseline);
    if (audit.requires_reconciliation() &&
        !core::scm::GitWorkspaceCoordinator::has_reconciliation_statement(
            response)) {
      if (turn.repository_reconciliation_followups++ == 0) {
        return {
            .action = core::session::TurnCompletionAction::Continue,
            .status = "AUTO · repository reconciliation required",
            .message = core::scm::GitWorkspaceCoordinator::
                reconciliation_follow_up(audit),
        };
      }
      return {
          .action = core::session::TurnCompletionAction::Fail,
          .message = "AUTO repository gate failed: repository transition "
                     "was not reconciled.",
      };
    }
  }

  const core::session::TurnCompletionResult external = run_completion_gate
      ? run_completion_gate()
      : core::session::TurnCompletionResult{};
  if (external.action != core::session::TurnCompletionAction::Complete) {
    return external;
  }

  const bool verification_exception_allowed =
      turn.verification_config_invalid || turn.verification_recipes.empty();
  if (!turn.decision.verification_required_after_mutation ||
      !turn.quality.mutation_observed() ||
      external.quality_gate_satisfied ||
      (verification_exception_allowed &&
       AutoQualityLedger::has_explicit_exception(response))) {
    return {};
  }

  if (run_verification && !turn.verification_config_invalid &&
      turn.quality.needs_verification(turn.verification_recipes)) {
    for (const auto &recipe_id :
         core::verification::Catalog::default_quality_gate(
             turn.verification_recipes)) {
      const auto receipt = run_verification(recipe_id);
      if (receipt.has_value()) {
        turn.quality.observe_verification_receipt(*receipt);
      } else {
        core::logging::warn(
            "[AUTO] completion quality recipe '{}' could not run: {}",
            recipe_id, receipt.error());
      }
    }
  }

  if (!turn.verification_config_invalid &&
      !turn.quality.needs_verification(turn.verification_recipes)) {
    return {
        .status = "AUTO · completion quality gate passed",
    };
  }

  if (turn.quality_followups++ == 0) {
    return {
        .action = core::session::TurnCompletionAction::Continue,
        .status = "AUTO · verification required",
        .message = AutoQualityLedger::verification_follow_up(),
    };
  }
  return {
      .action = core::session::TurnCompletionAction::Fail,
      .message = "AUTO quality gate failed: workspace changes have no "
                 "successful fresh verification evidence.",
  };
}

} // namespace core::agent

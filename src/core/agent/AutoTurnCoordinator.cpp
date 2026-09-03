#include "AutoTurnCoordinator.hpp"

#include "PermissionGate.hpp"
#include "../logging/Logger.hpp"
#include "../tools/ToolNames.hpp"
#include "../scm/ScmFactory.hpp"

#include <algorithm>
#include <exception>
#include <format>
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
  turn->objective = prompt;
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
  turn.hooks = hooks;
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

  // Boost keeps a pristine copy of the turn's starting state so a failing
  // check can later be attributed to this turn rather than to a build the
  // user had already broken. Creating it is a cheap `git worktree add`; it is
  // only ever *run* when something actually fails.
  if (turn.decision.boost && !turn.verification_recipes.empty()) {
    turn.baseline_worktree =
        core::scm::EphemeralWorktree::create(workspace_root, "baseline");
  }

  std::size_t findings = 0;
  if (turn.decision.path == AutoExecutionPath::Orchestrated && hooks.complete) {
    std::string planning_context(repository_context);
    planning_context += core::verification::Catalog::render_for_prompt(
        turn.verification_recipes);
    const auto preparation = graph_.prepare(
        objective, planning_context, hooks, turn.decision.boost);
    findings = preparation.findings.size();
    turn.graph_prompt_context =
        AutoGraphOrchestrator::render_for_prompt(preparation);

    // Phase 2 implementation workstreams: bounded parallel candidates, each
    // built and checked in its own throwaway worktree. Their diffs come back
    // as evidence only — the parent stays the single writer in this checkout.
    // Gated on the same predicate as the writer lease: a pure analysis turn
    // should not pay for two candidate builds it will never adopt.
    if (turn.decision.boost && hooks.implement &&
        writer_expected(turn.decision.task_type)) {
      turn.candidates = BoostPipeline::implement(
          objective, turn.graph_prompt_context, turn.verification_recipes,
          hooks);
      turn.candidate_prompt_context =
          BoostPipeline::render_candidates(turn.candidates);
    }
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
  result += turn.candidate_prompt_context;
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

std::size_t AutoTurnCoordinator::candidate_count(
    const AutoTurnState &turn) const noexcept {
  return turn.candidates.size();
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

// Re-runs the read-only investigation wave against concrete failure evidence.
// Bounded by the Boost round counter, so it cannot loop.
void AutoTurnCoordinator::reinvestigate(AutoTurnState &turn,
                                        std::string_view diagnostics) const {
  if (!turn.hooks.explore || !turn.hooks.complete)
    return;
  if (turn.hooks.cancellation_requested && turn.hooks.cancellation_requested())
    return;
  std::string context =
      "The previous attempt did not satisfy verification or independent review. "
      "Investigate this specific evidence; do not restate the original plan.\n";
  context += std::string(diagnostics.substr(0, 8192));
  const auto preparation =
      graph_.prepare(turn.objective, context, turn.hooks, true);
  auto refreshed = AutoGraphOrchestrator::render_for_prompt(preparation);
  if (!refreshed.empty())
    turn.graph_prompt_context = std::move(refreshed);
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
    if (turn.decision.boost && external.action == core::session::TurnCompletionAction::Continue)
      return turn.boost_pipeline.retry(external.message);
    return external;
  }

  const bool verification_exception_allowed =
      turn.verification_config_invalid || turn.verification_recipes.empty();
  bool quality_satisfied = !turn.decision.verification_required_after_mutation ||
      !turn.quality.mutation_observed() || external.quality_gate_satisfied;
  const bool exception = verification_exception_allowed &&
      AutoQualityLedger::has_explicit_exception(response);
  turn.verification_evidence = external.quality_gate_satisfied
      ? "A trusted completion quality hook passed after the latest mutations.\n"
      : "";
  if (!turn.quality.mutation_observed())
    turn.verification_evidence += "No workspace mutations observed; independently verify the investigation.\n";
  if (exception)
    turn.verification_evidence += "Verification exception claimed; independently assess the limitation.\n";

  // Attribution. A check that was already red on the pristine turn-start copy
  // is not this turn's defect, and must not consume its correction rounds.
  const auto already_failing = [&](std::string_view recipe_id) {
    if (!turn.baseline_worktree.has_value() ||
        !turn.baseline_worktree->valid() || !run_verification)
      return false;
    const std::string id(recipe_id);
    if (std::ranges::find(turn.preexisting_failures, id) !=
        turn.preexisting_failures.end())
      return true;
    const auto baseline =
        run_verification(recipe_id, turn.baseline_worktree->root());
    if (!baseline.has_value() || baseline->passed())
      return false;
    turn.preexisting_failures.push_back(id);
    return true;
  };

  // Scope the reviewer's diff. Without this it credits the turn for edits the
  // user had already made, or blames it for them.
  const auto preexisting_change_note = [&]() -> std::string {
    if (!turn.repository_baseline.has_value())
      return {};
    if (turn.repository_baseline->changes.empty())
      return "\nThe workspace was clean when this turn started, so the whole "
             "patch above is this turn's work.\n";
    std::string note =
        "\nThese paths were ALREADY modified before this turn started. Do not "
        "credit or blame this turn for them, and do not require them to be "
        "justified by the objective:\n";
    for (const auto &change : turn.repository_baseline->changes)
      note += std::format("  {} {}\n", change.status_code, change.path);
    return note;
  };

  bool failed_verification = false;
  if (!quality_satisfied && !exception && run_verification &&
      !turn.verification_config_invalid &&
      turn.quality.needs_verification(turn.verification_recipes)) {
    for (const auto &recipe_id :
         core::verification::Catalog::default_quality_gate(turn.verification_recipes)) {
      if (turn.hooks.cancellation_requested && turn.hooks.cancellation_requested())
        return {.action = core::session::TurnCompletionAction::Fail,
                .message = "Verification cancelled."};
      const auto receipt = run_verification(recipe_id, {});
      if (receipt.has_value()) {
        turn.quality.observe_verification_receipt(*receipt);
        const bool preexisting =
            !receipt->passed() && already_failing(recipe_id);
        failed_verification =
            failed_verification || (!receipt->passed() && !preexisting);
        turn.verification_evidence += receipt->recipe_id + " (exit " +
            std::to_string(receipt->exit_code) + "): " + receipt->command +
            "\n" + receipt->evidence.substr(0, 4096) + "\n";
        if (preexisting)
          turn.verification_evidence +=
              "^ This check already failed on the unmodified turn-start state: "
              "it is pre-existing, not caused by this turn. State it; do not "
              "silently absorb unrelated repair work into this request.\n";
      } else {
        failed_verification = true;
        turn.verification_evidence += recipe_id + ": " + receipt.error().substr(0, 4096) + "\n";
      }
    }
  }
  // A gate that was already red before the turn must not consume correction
  // rounds: the turn is answerable, the repository merely is not green.
  const bool only_preexisting_failures =
      !failed_verification && !turn.preexisting_failures.empty();
  quality_satisfied = quality_satisfied || exception ||
      only_preexisting_failures ||
      (!turn.verification_config_invalid &&
       !turn.quality.needs_verification(turn.verification_recipes));
  if (quality_satisfied) {
    if (turn.decision.boost) {
      if (turn.quality.mutation_observed() && !exception && !external.quality_gate_satisfied)
        turn.verification_evidence += "Trusted recipe requirements are satisfied for the latest mutation.\n";
      try {
        if (auto scm = core::scm::ScmFactory::create(workspace_root)) {
          turn.verification_evidence += "\nCurrent repository status:\n" + scm->get_status_summary();
          if (const auto patch = scm->review_patch()) {
            turn.verification_evidence += "\nCurrent tracked patch:\n" + *patch;
            // Without this, a reviewer credits the turn for edits the user had
            // already made, or blames it for them.
            turn.verification_evidence += preexisting_change_note();
          } else {
            turn.verification_evidence += "\nSCM patch unavailable; inspect source files directly.\n";
          }
        }
      } catch (const std::exception &error) {
        turn.verification_evidence += "\nRepository evidence unavailable: " + std::string(error.what());
      }
      auto verdict = turn.boost_pipeline.review(
          {.objective = turn.objective,
           .candidate = response,
           .evidence = turn.repository_prompt_context +
                       turn.verification_evidence +
                       turn.quality.verification_summary(),
           .mutated = turn.quality.mutation_observed()},
          turn.hooks);
      if (verdict.action == core::session::TurnCompletionAction::Continue) {
        // Phase 3 feeds diagnostics back into Phase 2 rather than only into the
        // parent's prompt: a round that repeats the same reasoning over the same
        // evidence tends to repeat the same mistake.
        reinvestigate(turn, verdict.message);
        verdict.message += prompt_suffix(turn);
      }
      return verdict;
    }
    return {.status = turn.quality.mutation_observed() && !exception &&
                            !external.quality_gate_satisfied
                        ? "AUTO · completion quality gate passed" : ""};
  }

  // Escalate a failed AUTO quality gate into the same Boost pipeline. This
  // decision is harness-owned and is limited to this turn.
  if (!turn.decision.boost && failed_verification && turn.hooks.explore) {
    turn.decision.boost = true;
    turn.decision.path = AutoExecutionPath::Orchestrated;
    turn.decision.parallel_exploration = true;
    turn.decision.reason = "BOOST automatically activated after failed verification";
    reinvestigate(turn, turn.verification_evidence);
  }
  if (turn.decision.boost) {
    auto retry = turn.boost_pipeline.retry(
        AutoQualityLedger::verification_follow_up() + "\n" + turn.verification_evidence);
    if (retry.action == core::session::TurnCompletionAction::Continue)
      retry.message += prompt_suffix(turn);
    return retry;
  }
  if (turn.quality_followups++ == 0) {
    return {.action = core::session::TurnCompletionAction::Continue,
            .status = "AUTO · verification required",
            .message = AutoQualityLedger::verification_follow_up() + "\n" +
                       turn.verification_evidence};
  }
  return {.action = core::session::TurnCompletionAction::Fail,
          .message = "AUTO quality gate failed: workspace changes have no "
                     "successful fresh verification evidence."};
}

} // namespace core::agent

#pragma once

#include "AutoGraphOrchestrator.hpp"
#include "AutoModePolicy.hpp"
#include "AutoQualityLedger.hpp"
#include "../scm/GitWorkspaceCoordinator.hpp"
#include "../session/TurnCompletion.hpp"
#include "../verification/Verification.hpp"

#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::agent {

// The generic Agent loop owns transport and conversation mechanics. Everything
// specific to an AUTO transaction lives in this aggregate and is manipulated
// only by AutoTurnCoordinator.
class AutoTurnState final {
public:
  AutoTurnState(const AutoTurnState &) = delete;
  AutoTurnState &operator=(const AutoTurnState &) = delete;

private:
  friend class AutoTurnCoordinator;
  AutoTurnState() = default;

  AutoModeDecision decision;
  std::string graph_prompt_context;
  std::string repository_prompt_context;
  std::optional<core::scm::RepositorySnapshot> repository_baseline;
  std::optional<core::scm::GitWorkspaceCoordinator::Lease> repository_lease;
  int repository_reconciliation_followups = 0;
  std::vector<core::verification::Recipe> verification_recipes;
  std::vector<std::string> verification_warnings;
  bool verification_config_invalid = false;
  AutoQualityLedger quality;
  int quality_followups = 0;
};

struct AutoToolIntent {
  std::string_view name;
  std::string_view arguments;
  bool approved = false;
  bool read_only_subagent = false;
  bool destructive_hint = false;
};

// Outcome of arranging writer exclusion for one tool batch. `Unavailable` is
// deliberately distinct from `NotRequired`: a mutating batch that could not
// take the exclusive lease still runs, but the caller must be able to say so
// instead of silently implying exclusion it never had.
enum class WorkspaceWriterState {
  NotRequired,
  Acquired,
  Unavailable,
};

struct AutoToolObservation {
  std::string_view name;
  std::string_view arguments;
  std::string_view result;
  bool succeeded = false;
  bool mutation_hint = false;
  bool trusted_verification_receipts = false;
};

// Application service for one AUTO transaction. It composes pure routing,
// graph preparation, repository coordination and quality policy behind a
// narrow façade; it does not own an Agent, history, UI callbacks or tools.
class AutoTurnCoordinator final {
public:
  using VerificationRunner = std::function<
      std::expected<core::verification::Receipt, std::string>(std::string_view)>;
  using CompletionGate =
      std::function<core::session::TurnCompletionResult()>;

  explicit AutoTurnCoordinator(
      std::shared_ptr<core::scm::WorkspaceLeaseRegistry> workspace_leases = {});

  [[nodiscard]] std::unique_ptr<AutoTurnState>
  start(std::string_view prompt,
        AutoModeContext context,
        const std::filesystem::path &workspace_root) const;

  [[nodiscard]] std::size_t prepare(
      AutoTurnState &turn,
      std::string_view objective,
      std::string_view repository_context,
      const std::filesystem::path &workspace_root,
      AutoGraphOrchestrator::Hooks hooks) const;

  [[nodiscard]] std::string prompt_suffix(const AutoTurnState &turn) const;
  [[nodiscard]] const AutoModeDecision &decision(
      const AutoTurnState &turn) const noexcept;
  [[nodiscard]] bool mutation_observed(
      const AutoTurnState &turn) const noexcept;

  [[nodiscard]] WorkspaceWriterState prepare_tool_batch(
      AutoTurnState &turn,
      const std::filesystem::path &workspace_root,
      std::span<const AutoToolIntent> tools,
      std::function<bool()> cancellation_requested = {}) const;

  [[nodiscard]] bool holds_workspace_lock(
      const AutoTurnState &turn) const noexcept;
  [[nodiscard]] std::optional<core::goal::WorkspaceAccess>
  workspace_access(const AutoTurnState &turn) const noexcept;

  void observe_tool(AutoTurnState &turn,
                    const AutoToolObservation &observation) const noexcept;

  [[nodiscard]] core::session::TurnCompletionResult evaluate_completion(
      AutoTurnState &turn,
      std::string_view response,
      const std::filesystem::path &workspace_root,
      const CompletionGate &run_completion_gate,
      const VerificationRunner &run_verification) const;

private:
  AutoModePolicy policy_;
  AutoGraphOrchestrator graph_;
  core::scm::GitWorkspaceCoordinator workspace_;
  core::verification::Catalog verification_;
};

} // namespace core::agent

#pragma once

#include "AutoGraphOrchestrator.hpp"
#include "../llm/routing/AutoClassifier.hpp"
#include "../session/TurnCompletion.hpp"
#include "../verification/Verification.hpp"

#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::agent {

// A turn-scoped execution policy, independent of the selected model and mode.
// The coordinator owns mutations/receipts; this pipeline owns independent
// review and the bounded correction loop. A model cannot self-certify a pass.
class BoostPipeline {
public:
  static constexpr int kMaxRounds = 3;
  static constexpr int kMaxSteps = 96;
  // Two isolated candidates keep the parallel cost bounded: a third rarely
  // adds a distinct approach but always multiplies build time.
  static constexpr std::size_t kMaxCandidates = 2;

  // Why a turn was routed to Boost. Carrying the score keeps the automatic
  // decision explainable in the AUTO contract instead of being a silent
  // keyword match the user cannot audit.
  struct Activation {
    bool active = false;
    int score = 0;
    std::string reason;
  };

  using Candidate = BoostCandidate;

  struct ReviewRequest {
    std::string_view objective;
    std::string_view candidate;
    std::string_view evidence;
    // A turn that changed nothing cannot leave the workspace broken, so an
    // unavailable reviewer degrades instead of discarding usable work.
    bool mutated = false;
  };

  [[nodiscard]] static std::optional<std::string_view>
  command_task(std::string_view text) noexcept;
  [[nodiscard]] static Activation
  should_activate(std::string_view prompt,
                  const core::llm::routing::ClassificationResult &classified);
  [[nodiscard]] static std::string execution_contract();
  [[nodiscard]] static std::string_view reasoning_effort(std::string_view selected) noexcept {
    return selected == "max" || selected == "xhigh" || selected == "ultra"
        ? selected : std::string_view("high");
  }

  /// Phase 2 implementation workstreams cost two full quality-gate runs in cold
  /// isolated worktrees, which is cheap for a scripted test suite and very
  /// expensive for a large compiled project. That trade-off is a property of the
  /// project, so it is opt-in per workspace via `boost_candidates: true` in
  /// `.filo/settings.json`. Investigations and independent review are
  /// unaffected either way.
  [[nodiscard]] static bool
  candidates_enabled(const std::filesystem::path &workspace_root) noexcept;

  // Phase 2: run bounded parallel implementation candidates in isolation and
  // render them as harness evidence for the parent writer.
  [[nodiscard]] static std::vector<Candidate>
  implement(std::string_view objective, std::string_view strategy,
            std::span<const core::verification::Recipe> recipes,
            const AutoGraphOrchestrator::Hooks &hooks);
  [[nodiscard]] static std::string
  render_candidates(std::span<const Candidate> candidates);

  [[nodiscard]] core::session::TurnCompletionResult retry(
      std::string_view diagnostics);
  [[nodiscard]] core::session::TurnCompletionResult review(
      const ReviewRequest &request, const AutoGraphOrchestrator::Hooks &hooks);

  [[nodiscard]] int round() const noexcept { return corrections_; }

private:
  int corrections_ = 0;
};

} // namespace core::agent

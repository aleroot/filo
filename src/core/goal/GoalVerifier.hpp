#pragma once

#include "GoalTypes.hpp"

#include <expected>
#include <functional>
#include <string>
#include <string_view>

namespace core::goal {

// ---------------------------------------------------------------------------
// GoalVerifier — tool-grounded acceptance checks (CRITIC principle:
// verification must be anchored in tools, not internal confirmation).
//
// Strategy composition:
//   1. ShellCheckVerifier — deterministic: run the node's check command,
//      exit code 0 means pass. Preferred whenever a command exists.
//   2. ModelCheckVerifier — a strict judge model reading tool-produced
//      evidence. Fallback only, never the primary signal.
//   GoalVerifier routes between them per node (deterministic first).
//
// Both strategies depend on injected callables, keeping this module free of
// shell/LLM dependencies and trivially testable (Dependency Inversion).
// ---------------------------------------------------------------------------

struct CommandResult {
    int exit_code = -1;
    std::string output; ///< combined stdout+stderr, clamped
};

struct RecipeResult {
  bool passed = false;
  std::string command;
  std::string evidence;
  std::string error;
};

using CommandRunner = std::function<CommandResult(std::string_view command)>;
using RecipeRunner = std::function<RecipeResult(std::string_view recipe_id)>;
using CompletionFn =
    std::function<std::expected<std::string, std::string>(std::string_view prompt)>;

struct VerifyContext {
    std::string work_output;   ///< summary produced by the work node under check
    std::string workspace_dir; ///< session working directory, for prompt grounding
};

struct Verdict {
    bool passed = false;
    bool deterministic = false; ///< true when produced by an executed command
    std::string reason;         ///< short human-readable explanation
    std::string evidence;       ///< clamped tool output the verdict is based on
};

class IVerifier {
public:
    virtual ~IVerifier() = default;
    [[nodiscard]] virtual Verdict verify(const Node& node,
                                         const VerifyContext& context) const = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
};

/// Deterministic verification through an injected command runner.
class ShellCheckVerifier final : public IVerifier {
public:
    static constexpr std::size_t kMaxEvidenceChars = 2048;

    explicit ShellCheckVerifier(CommandRunner runner);

    [[nodiscard]] Verdict verify(const Node& node,
                                 const VerifyContext& context) const override;
    [[nodiscard]] std::string_view name() const noexcept override { return "shell"; }
    [[nodiscard]] bool available() const noexcept { return static_cast<bool>(runner_); }

private:
    CommandRunner runner_;
};

/// Model-judged verification with a strict output contract (PASS/FAIL + reason).
class ModelCheckVerifier final : public IVerifier {
public:
    explicit ModelCheckVerifier(CompletionFn complete);

    [[nodiscard]] Verdict verify(const Node& node,
                                 const VerifyContext& context) const override;
    [[nodiscard]] std::string_view name() const noexcept override { return "model"; }
    [[nodiscard]] bool available() const noexcept { return static_cast<bool>(complete_); }

    /// Parse the judge contract: first non-empty line starts with PASS or FAIL.
    [[nodiscard]] static Verdict parse_response(std::string_view response);

private:
    CompletionFn complete_;
};

/// Deterministic-first composite: shell when the node carries a check command
/// and a runner is available, model otherwise.
class GoalVerifier final : public IVerifier {
public:
    GoalVerifier(CommandRunner runner, CompletionFn complete);
    GoalVerifier(RecipeRunner recipe_runner, CommandRunner legacy_runner,
                 CompletionFn complete);

    [[nodiscard]] Verdict verify(const Node& node,
                                 const VerifyContext& context) const override;
    [[nodiscard]] std::string_view name() const noexcept override { return "chained"; }

private:
  RecipeRunner recipe_runner_;
  ShellCheckVerifier shell_;
  ModelCheckVerifier model_;
};

/// Default POSIX command runner (popen/pclose, combined output, exit status).
/// Tests inject fakes; production wiring may replace this with a ShellTool
/// adapter without touching the verifier.
[[nodiscard]] CommandRunner make_popen_command_runner();

} // namespace core::goal

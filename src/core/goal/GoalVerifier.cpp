#include "GoalVerifier.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <format>

#if defined(_WIN32)
#define FILO_GOAL_POPEN _popen
#define FILO_GOAL_PCLOSE _pclose
#else
#include <sys/wait.h>
#include <unistd.h>
#define FILO_GOAL_POPEN ::popen
#define FILO_GOAL_PCLOSE ::pclose
#endif

namespace core::goal {

namespace {

[[nodiscard]] std::string clamp_evidence(std::string_view output) {
    if (output.size() <= ShellCheckVerifier::kMaxEvidenceChars) {
        return std::string(output);
    }
    // Keep the tail: failure diagnostics live at the end of command output.
    constexpr std::string_view kMarker = "... [truncated]\n";
    std::string out(kMarker);
    out.append(output.substr(output.size()
                             - (ShellCheckVerifier::kMaxEvidenceChars - kMarker.size())));
    return out;
}

[[nodiscard]] std::string_view trim(std::string_view value) noexcept {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

/// POSIX single-quote escaping: wrap in '...', closing and reopening around
/// any embedded quote. The workspace path is session-controlled rather than
/// model-controlled, but it still must not be able to terminate the `cd`
/// argument and append a second command.
[[nodiscard]] std::string shell_quote(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 2);
    out += '\'';
    for (const char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += '\'';
    return out;
}

#if !defined(_WIN32)
/// Wall-clock ceiling for one deterministic check. Without it a hung check
/// (a server that never exits, a command waiting on stdin) stalls the wave
/// thread — and therefore the whole goal run — forever. Generous enough for a
/// cold full build plus test suite.
constexpr int kCheckTimeoutSeconds = 900;

/// Locate a coreutils-style `timeout` binary once. Absent on a stock macOS,
/// where checks then run unbounded exactly as they did before.
[[nodiscard]] const std::string& timeout_binary() {
    static const std::string path = [] -> std::string {
        for (const char* candidate : {"/usr/bin/timeout", "/bin/timeout",
                                      "/usr/local/bin/timeout",
                                      "/opt/homebrew/bin/gtimeout",
                                      "/usr/local/bin/gtimeout"}) {
            if (::access(candidate, X_OK) == 0) {
                return candidate;
            }
        }
        return {};
    }();
    return path;
}
#endif

} // namespace

// ---------------------------------------------------------------------------
// ShellCheckVerifier
// ---------------------------------------------------------------------------

ShellCheckVerifier::ShellCheckVerifier(CommandRunner runner)
    : runner_(std::move(runner)) {}

Verdict ShellCheckVerifier::verify(const Node& node, const VerifyContext& context) const {
    Verdict verdict;
    verdict.deterministic = true;

    if (!runner_) {
        verdict.reason = "no command runner configured";
        return verdict;
    }
    if (node.check_command.empty()) {
        verdict.reason = "node has no deterministic check command";
        return verdict;
    }

    std::string command(node.check_command);
    if (!context.workspace_dir.empty()) {
        command = std::format("cd {} && {}", shell_quote(context.workspace_dir), command);
    }

    const CommandResult result = runner_(command);
    verdict.passed = result.exit_code == 0;
    verdict.evidence = clamp_evidence(result.output);
    verdict.reason = verdict.passed
        ? std::format("`{}` exited 0", node.check_command)
        : std::format("`{}` exited {}", node.check_command, result.exit_code);
    return verdict;
}

// ---------------------------------------------------------------------------
// ModelCheckVerifier
// ---------------------------------------------------------------------------

ModelCheckVerifier::ModelCheckVerifier(CompletionFn complete)
    : complete_(std::move(complete)) {}

Verdict ModelCheckVerifier::parse_response(std::string_view response) {
    Verdict verdict;
    verdict.deterministic = false;

    const std::string_view trimmed = trim(response);
    const auto line_end = trimmed.find('\n');
    const std::string_view first_line = trim(trimmed.substr(0, line_end));

    const auto starts_with = [](std::string_view text, std::string_view prefix) {
        return text.size() >= prefix.size()
            && text.substr(0, prefix.size()) == prefix;
    };
    if (starts_with(first_line, "PASS")) {
        verdict.passed = true;
    } else if (starts_with(first_line, "FAIL")) {
        verdict.passed = false;
    } else {
        verdict.passed = false;
        verdict.reason = "judge response did not start with PASS or FAIL";
        verdict.evidence = clamp_evidence(response);
        return verdict;
    }

    std::string_view rest = line_end == std::string_view::npos
        ? std::string_view{}
        : trim(trimmed.substr(line_end + 1));
    verdict.reason = rest.empty() ? std::string(first_line) : std::string(rest);
    verdict.evidence = clamp_evidence(response);
    return verdict;
}

Verdict ModelCheckVerifier::verify(const Node& node, const VerifyContext& context) const {
    if (!complete_) {
        Verdict verdict;
        verdict.reason = "no judge model configured";
        return verdict;
    }

    std::string prompt;
    prompt += "You are a strict verification judge for an autonomous coding agent.\n"
              "Decide whether the stated acceptance criteria hold, based ONLY on the\n"
              "evidence below. Do not assume anything the evidence does not show.\n\n";
    prompt += "Acceptance criteria:\n";
    prompt += node.acceptance.empty() ? node.directive : node.acceptance;
    prompt += "\n\nEvidence (work summary and tool output):\n";
    prompt += context.work_output.empty() ? "(no work output captured)" : context.work_output;
    prompt += "\n\nRespond with EXACTLY this contract:\n"
              "Line 1: PASS or FAIL\n"
              "Line 2+: one short paragraph explaining the verdict.\n";

    const auto response = complete_(prompt);
    if (!response.has_value()) {
        Verdict verdict;
        verdict.reason = std::format("judge model error: {}", response.error());
        return verdict;
    }
    return parse_response(*response);
}

// ---------------------------------------------------------------------------
// GoalVerifier — deterministic-first routing
// ---------------------------------------------------------------------------

GoalVerifier::GoalVerifier(CommandRunner runner, CompletionFn complete)
    : GoalVerifier({}, std::move(runner), std::move(complete)) {}

GoalVerifier::GoalVerifier(RecipeRunner recipe_runner,
                           CommandRunner legacy_runner, CompletionFn complete)
    : recipe_runner_(std::move(recipe_runner)),
      shell_(std::move(legacy_runner)), model_(std::move(complete)) {}

Verdict GoalVerifier::verify(const Node &node,
                             const VerifyContext &context) const {
  if (!node.verification_recipe_ids.empty()) {
    Verdict verdict;
    if (!recipe_runner_) {
      // Fail closed, but never as `deterministic`: no command ran, so this is
      // a configuration error, not tool-grounded evidence that the work is
      // wrong. Reflection and the UI treat deterministic verdicts as proof.
      verdict.reason = "no trusted verification recipe runner configured";
      return verdict;
    }
    verdict.deterministic = true;

    std::string evidence;
    for (const auto &recipe_id : node.verification_recipe_ids) {
      const RecipeResult result = recipe_runner_(recipe_id);
      if (!evidence.empty()) {
        evidence += '\n';
      }
      evidence +=
          std::format("[{}] {}", recipe_id,
                      result.evidence.empty() ? result.error : result.evidence);
      if (!result.passed) {
        verdict.reason =
            result.error.empty()
                ? std::format("verification recipe '{}' failed", recipe_id)
                : std::format("verification recipe '{}' failed: {}", recipe_id,
                              result.error);
        verdict.evidence = clamp_evidence(evidence);
        return verdict;
      }
    }
    verdict.passed = true;
    verdict.reason = std::format("{} trusted verification recipe(s) passed",
                                 node.verification_recipe_ids.size());
    verdict.evidence = clamp_evidence(evidence);
    return verdict;
  }
  if (!node.check_command.empty() && shell_.available()) {
    return shell_.verify(node, context);
  }
  if (model_.available()) {
    return model_.verify(node, context);
  }
  if (shell_.available()) {
    return shell_.verify(node, context);
  }
  Verdict verdict;
  verdict.reason =
      "no verifier available (neither command runner nor judge model)";
  return verdict;
}

// ---------------------------------------------------------------------------
// Default POSIX command runner
// ---------------------------------------------------------------------------

CommandRunner make_popen_command_runner() {
    return [](std::string_view command) -> CommandResult {
        CommandResult result;
        std::string shell_command;
#if !defined(_WIN32)
        if (!timeout_binary().empty()) {
            // -k: escalate to SIGKILL if the check ignores the initial SIGTERM.
            // A timed-out check exits 124, so it simply reads as a failed
            // verdict and feeds the reflector like any other failure.
            shell_command = std::format("{} -k 10 {} /bin/sh -c {} 2>&1",
                                        timeout_binary(), kCheckTimeoutSeconds,
                                        shell_quote(command));
        } else
#endif
        {
            shell_command = std::string(command) + " 2>&1";
        }

        FILE* pipe = FILO_GOAL_POPEN(shell_command.c_str(), "r");
        if (pipe == nullptr) {
            result.exit_code = 127;
            result.output = "failed to spawn shell";
            return result;
        }

        std::array<char, 4096> buffer{};
        std::string output;
        output.reserve(8192);
        while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
            output.append(buffer.data());
            if (output.size() > ShellCheckVerifier::kMaxEvidenceChars * 4) {
                // Bound memory on noisy commands; the tail is what matters.
                output.erase(0, output.size() - ShellCheckVerifier::kMaxEvidenceChars * 2);
            }
        }

        const int status = FILO_GOAL_PCLOSE(pipe);
#if defined(_WIN32)
        result.exit_code = status;
#else
        result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128;
#endif
        result.output = std::move(output);
        return result;
    };
}

} // namespace core::goal

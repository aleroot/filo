#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace core::permissions {

// ---------------------------------------------------------------------------
// ShellWord — one word of a simple command, after quote removal.
//
// The shell finishes some words only at run time: `$HOME/x`, `*.cpp`,
// `"$(git rev-parse HEAD)"`. A static reader cannot know their value, so it
// keeps what it can prove — the text before the first expansion — and says
// how much is unknown. That prefix is enough to answer the questions a policy
// asks: "is this the program `git`?" needs a literal word; "could this be the
// option --pre?" is settled whenever the prefix shows it cannot be.
// ---------------------------------------------------------------------------
struct ShellWord {
    /// The word up to its first expansion; the whole word when it has none.
    std::string text = {};
    /// The word contains an expansion ($NAME, ${NAME}, $(…), a glob or a
    /// brace), so the rest of its value is decided at run time.
    bool expands = false;
    /// One of those expansions is unquoted, so the shell may turn the word
    /// into zero or several words.
    bool splits = false;

    [[nodiscard]] bool is_literal() const noexcept { return !expands; }

    friend bool operator==(const ShellWord&, const ShellWord&) = default;
};

/// The words of one simple command: the program, then its arguments.
using ShellCommand = std::vector<ShellWord>;

// ---------------------------------------------------------------------------
// parse_shell_commands — the simple commands a bash command line runs.
//
// This is deliberately a small, closed grammar, not a bash implementation:
// it accepts what can be analysed statically and rejects everything else with
// a reason, so callers fail closed (ask) rather than guess.
//
// Accepted:
//   • words built from bare text, '…', "…" and backslash escapes;
//   • $NAME, ${NAME}, special parameters, globs and braces (as expansions);
//   • $(…) — its commands are returned too, because they run as well;
//   • the operators && || ; | |& and newlines, plus # comments;
//   • redirections that cannot write a file: `< file`, `> /dev/null`,
//     `2>/dev/null`, `&>/dev/null`, `2>&1`, `>&-`.
//
// Rejected: any other redirection, heredocs, process substitution,
// subshells and ( ), backticks, $'…', $"…", ${NAME…} with operators, $((…)),
// background `&`, unterminated quotes, and empty commands around operators.
//
// Reserved words (if, for, while, …) are returned as ordinary command names;
// a policy that has no rule for them asks, which is the safe reading.
// ---------------------------------------------------------------------------
[[nodiscard]] std::expected<std::vector<ShellCommand>, std::string>
parse_shell_commands(std::string_view line);

} // namespace core::permissions

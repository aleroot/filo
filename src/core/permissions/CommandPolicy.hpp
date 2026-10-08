#pragma once

#include "ShellCommandParser.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::permissions {

enum class CommandDecision : std::uint8_t {
    Allow,  // runs without asking
    Ask,    // the user approves it first
};

// ---------------------------------------------------------------------------
// CommandPolicy — decides which shell commands may run without asking.
//
// The knowledge lives in rules documents (JSON), not in code: the built-in
// one is src/core/permissions/default_command_rules.json, and a user may add
// <config dir>/command_rules.json (read at startup). A document is
//
//   { "wrappers":       [ { "program", "flags", "options_with_value",
//                           "positionals", "justification" } ],
//     "global_options": [ { "program", "flags", "options_with_value",
//                           "justification" } ],
//     "rules":          [ { "pattern", "decision", "unless_options",
//                           "max_positionals", "justification",
//                           "match", "not_match" } ] }
//
//   pattern          the command's leading words. Each element is a word,
//                    a list of alternative words, or {"regex": "…"} (full
//                    match; not for the program itself).
//   decision         "allow" or "ask".
//   unless_options   the rule does not match if any argument may set one
//                    of these options (see may_set_option for spellings).
//   max_positionals  the rule does not match with more non-option arguments.
//   match/not_match  example commands that must / must not match the rule;
//                    at least one "match" is required.
//
// Unknown keys are errors, so a typo cannot silently drop a safeguard. A
// later layer's wrapper or global_options entry replaces an earlier one for
// the same program; rules accumulate.
//
// Semantics, in the order they apply:
//   1. The line is parsed by parse_shell_commands; anything it cannot
//      analyse asks.
//   2. Declared wrappers (`timeout 5 …`) are removed, so their command is
//      what gets judged. A wrapper whose options are not understood asks.
//   3. Declared global options (`git -C dir …`) are skipped before matching.
//   4. A command is allowed only when some rule allows it and no matching
//      rule asks: the strictest decision wins, across every layer.
//   5. A line is allowed only when every command in it is, including those
//      inside $(…). Nothing matched means ask — there is no denylist.
//
// Every rule carries examples that must and must not match it; they are
// checked whenever a layer is loaded, so a rule cannot silently drift from
// its intent.
//
// A CommandPolicy is an immutable value; copies share their rules.
// ---------------------------------------------------------------------------
class CommandPolicy {
public:
    /// A policy without rules: it asks before every command.
    CommandPolicy();

    /// This policy plus one rules document. The document's examples are
    /// checked against the combined policy, and any error rejects the whole
    /// document, naming the offending rule.
    [[nodiscard]] std::expected<CommandPolicy, std::string>
    with_layer(std::string_view json) const;

    /// The simple commands `line` runs, wrappers removed, or why it cannot
    /// be analysed.
    [[nodiscard]] std::expected<std::vector<ShellCommand>, std::string>
    commands(std::string_view line) const;

    /// Never throws: anything that prevents a decision is an Ask.
    [[nodiscard]] CommandDecision decide(const ShellCommand& command) const noexcept;
    [[nodiscard]] CommandDecision decide(std::string_view line) const noexcept;

    struct Definition;

private:
    explicit CommandPolicy(std::shared_ptr<const Definition> definition) noexcept;

    std::shared_ptr<const Definition> definition_;
};

/// The built-in rules document, embedded from default_command_rules.json.
[[nodiscard]] std::string_view default_command_rules() noexcept;

/// Where a user's own rules layer is read from.
[[nodiscard]] std::string user_command_rules_path();

/// The policy Filo enforces: the built-in rules plus the user's layer.
/// Loaded once; a layer that fails to load is logged and left out, which can
/// only make the policy stricter.
[[nodiscard]] const CommandPolicy& command_policy();

/// The `command` argument of a run_terminal_command call, read exactly as
/// ShellTool reads it (a real JSON parse, so escapes such as \u0026 decode).
[[nodiscard]] std::optional<std::string> shell_command_argument(std::string_view tool_args_json);

/// True when command_policy() lets this run_terminal_command call run
/// without asking. Missing or malformed arguments, and any failure, ask.
[[nodiscard]] bool shell_call_is_allowed(std::string_view tool_args_json) noexcept;

} // namespace core::permissions

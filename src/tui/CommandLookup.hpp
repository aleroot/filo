#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tui {

/// Splits a command line into words, honouring single and double quotes and
/// backslash escapes, without evaluating any expansion or substitution. The
/// original command stays untouched for shell execution; this is only for
/// inspecting it.
///
/// Returns nullopt when the quoting is unbalanced, which means the command is
/// not the one the user meant to write.
[[nodiscard]] std::optional<std::vector<std::string>> shell_words(std::string_view command);

/// True when the first word of `command` names an executable reachable from the
/// current PATH, or is itself an executable path.
[[nodiscard]] bool command_is_executable(std::string_view command);

/// The exit code a @c std::system status describes: the child's own code, 128
/// plus the signal when one killed it, or -1 when it could not be run at all.
[[nodiscard]] int child_exit_code(int status);

} // namespace tui

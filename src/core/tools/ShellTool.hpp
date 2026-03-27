#pragma once

#include "Tool.hpp"
#include "shell/IShellExecutor.hpp"
#include "shell/ShellExecutorFactory.hpp"
#include <memory>
#include <mutex>
#include <string>

namespace core::tools {

/**
 * @brief Tool that executes shell commands in a persistent bash session.
 *
 * Exposed to MCP clients as @c run_terminal_command.
 *
 * ### Execution model
 * Unlike a stateless fork-per-command approach, ShellTool maintains a single
 * long-lived bash process.  State persists across calls:
 *
 *  - **Working directory**: @c cd changes are visible in subsequent commands.
 *  - **Environment variables**: @c export persists for the lifetime of the
 *    session.
 *  - **Shell functions / aliases**: defined once, usable in later calls.
 *  - **Activated environments**: @c source @c venv/bin/activate, @c nvm @c use,
 *    etc. take effect for all subsequent commands.
 *
 * If bash exits (e.g. the user command called @c exit @c N) the next call to
 * execute() transparently restarts the session.  The @c working_dir parameter
 * runs the command inside a subshell (@c (cd … && …)) so the session's current
 * directory is not affected.
 *
 * ### Shell backend
 * The concrete executor is selected at compile time via ShellExecutorFactory:
 *   - POSIX (Linux, macOS, WSL): PosixShellExecutor (persistent bash session)
 *   - Future native Windows: WindowsShellExecutor (not yet implemented)
 *
 * ### Result JSON
 * @code{.json}
 * // Success
 * {"output": "<stdout+stderr>", "exit_code": 0}
 *
 * // Failure (non-zero exit is NOT an error response — use exit_code to detect)
 * {"output": "<stderr output>", "exit_code": 1}
 *
 * // Tool-level error (bad args, bad working_dir, session failure)
 * {"error": "<message>"}
 * @endcode
 *
 * @note @c destructiveHint and @c openWorldHint are set to @c true — this
 *       tool can perform any action the user's shell can, including network
 *       access, file writes, and process management.
 */
class ShellTool : public Tool {
public:
    // Constructs with the platform-appropriate executor (selected at compile time).
    ShellTool() : executor_(shell::make_shell_executor()) {}

    // Dependency-injection constructor — used in tests to supply a mock executor.
    explicit ShellTool(std::unique_ptr<shell::IShellExecutor> executor)
        : executor_(std::move(executor)) {}

    ToolDefinition get_definition() const override;
    std::string    execute(const std::string& json_args) override;

private:
    std::unique_ptr<shell::IShellExecutor> executor_;
    std::mutex                             executor_mutex_;
};

} // namespace core::tools

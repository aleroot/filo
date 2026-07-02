#pragma once

#include "Tool.hpp"
#include "TempFileAccessRegistry.hpp"
#include "shell/IShellExecutor.hpp"
#include "shell/ShellExecutorFactory.hpp"
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

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
    struct ActiveCommand {
        std::string session_id;
        std::string command;
        std::string working_dir;
        std::string tool_call_id;
        std::chrono::steady_clock::time_point started_at;
    };

    ShellTool();

    explicit ShellTool(std::shared_ptr<TempFileAccessRegistry> temp_file_access_registry);

    // Dependency-injection constructor — used in tests to supply a mock executor.
    explicit ShellTool(std::unique_ptr<shell::IShellExecutor> executor);

    ShellTool(std::unique_ptr<shell::IShellExecutor> executor,
              std::shared_ptr<TempFileAccessRegistry> temp_file_access_registry);

    static void clear_mcp_session(std::string_view session_id);
    static bool interrupt_mcp_session(std::string_view session_id);
    static std::vector<ActiveCommand> active_commands();

    ToolDefinition get_definition() const override;
    std::string execute(const std::string& json_args, const core::context::SessionContext& context) override;
    std::string execute(const std::string& json_args, const ToolInvocationContext& invocation) override;
    void clear_session_state(std::string_view session_id) override;

private:
    std::string execute_impl(const std::string& json_args,
                             const core::context::SessionContext& context,
                             std::string_view tool_call_id);

    std::unique_ptr<shell::IShellExecutor> executor_;
    std::mutex                             executor_mutex_;
    std::shared_ptr<TempFileAccessRegistry> temp_file_access_registry_;
};

} // namespace core::tools

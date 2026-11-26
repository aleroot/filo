#pragma once

#include "IShellExecutor.hpp"

#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools::shell {

// Runtime-owned persistent processes. ShellTool handles arguments and policy;
// this owner handles session isolation, activity, cancellation and retirement.
class ShellSessions final {
public:
    struct ActiveCommand {
        std::string session_id;
        std::string command;
        std::string working_dir;
        std::string tool_call_id;
        std::chrono::steady_clock::time_point started_at;
    };

    ShellSessions();
    explicit ShellSessions(std::unique_ptr<IShellExecutor> anonymous_executor);
    ~ShellSessions();
    ShellSessions(const ShellSessions&) = delete;
    ShellSessions& operator=(const ShellSessions&) = delete;

    IShellExecutor::Result run(ActiveCommand command,
        const std::filesystem::path& workspace, const core::landrun::LandrunPolicy& policy,
        std::chrono::milliseconds timeout, const std::function<void(int)>& before_dispatch = {});

    // Command sessions (codex-rs unified exec): a command runs in its own
    // process, and a call returns after its yield time with what the command
    // wrote so far. One still running keeps an id a later call continues it
    // by, writing to its stdin or only reading. Sessions belong to the
    // conversation's shell session: clear, move, interrupt and quiesce reach
    // them too.

    /// What one call on a command session returned.
    struct CommandStep {
        std::string output;
        /// Set once the command has ended; the session is then gone.
        std::optional<int> exit_code;
        /// Set while the command keeps running.
        std::optional<int> process_id;
        std::chrono::duration<double> wall_time{};
        /// Set, and nothing else, when the call could not run.
        std::string error;
    };
    /// At most this many sessions are kept; past it an ended or the least
    /// recently used idle one is dropped (codex-rs MAX_UNIFIED_EXEC_PROCESSES).
    static constexpr std::size_t kMaxCommandSessions = 64;

    /// Starts @p command (its working_dir, or @p workspace) and waits up to
    /// @p yield for it to end.
    CommandStep start_command(ActiveCommand command, const std::filesystem::path& workspace,
        const core::landrun::LandrunPolicy& policy, std::chrono::milliseconds yield,
        const std::function<void(int)>& before_dispatch = {});
    /// Writes @p input (nothing only reads) to the session @p process_id of
    /// @p session_id and waits up to @p yield for more output.
    CommandStep continue_command(std::string_view session_id, int process_id,
        std::string_view input, std::chrono::milliseconds yield, std::string tool_call_id = {});
    void clear(std::string_view session_id);
    void move(std::string_view from, std::string_view to);
    bool interrupt(std::string_view session_id);
    [[nodiscard]] std::vector<ActiveCommand> active_commands() const;
    [[nodiscard]] bool quiesce_session(std::string_view session_id);
    // Only listed, idle sessions are retired. Destruction occurs outside the
    // registry lock; a running command keeps its session alive independently.
    std::optional<std::chrono::steady_clock::time_point> reap_idle(
        std::span<const std::string> session_ids, std::chrono::steady_clock::duration idle_timeout);

private:
    struct State;
    struct CommandEntry;
    CommandStep step_command(int id, const std::shared_ptr<CommandEntry>& entry,
        std::string_view input, std::chrono::milliseconds yield, std::string tool_call_id);
    std::unique_ptr<State> state_;
};

} // namespace core::tools::shell

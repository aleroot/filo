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
    void clear(std::string_view session_id);
    bool interrupt(std::string_view session_id);
    [[nodiscard]] std::vector<ActiveCommand> active_commands() const;
    [[nodiscard]] bool quiesce_workspace(const std::filesystem::path& root);
    // Only listed, idle sessions are retired. Destruction occurs outside the
    // registry lock; a running command keeps its session alive independently.
    std::optional<std::chrono::steady_clock::time_point> reap_idle(
        std::span<const std::string> session_ids, std::chrono::steady_clock::duration idle_timeout);

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace core::tools::shell

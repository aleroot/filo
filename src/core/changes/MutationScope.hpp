#pragma once

#include "../context/SessionContext.hpp"
#include "../tools/Tool.hpp"

#include <filesystem>
#include <string_view>
#include <utility>
#include <vector>

namespace core::changes {

/// The filesystem paths one tool call may change. Paths are absolute. A
/// directory stands for every file beneath it.
struct MutationScope {
    std::vector<std::filesystem::path> paths;
    /// (source, destination) pairs the call may move.
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> moves;
    /// The call may write, but where cannot be enumerated up front: a shell
    /// command, a script, a verification build, an MCP tool. The tracker
    /// reconciles the paths the turn already knows and reports the summary as
    /// incomplete, because paths such a call creates cannot be discovered.
    bool unbounded = false;

    [[nodiscard]] bool empty() const noexcept { return paths.empty() && !unbounded; }
};

/// Scope of one tool call.
///
/// Built-in file tools resolve to the exact paths they touch. Every other tool
/// is judged by `annotations`, its own declared contract and the same one the
/// MCP dispatcher and subagent profiles read: a tool that promises it never
/// writes to disk is inert, and anything else that Filo has no rule for may
/// write anywhere, so it is unbounded rather than silently ignored.
[[nodiscard]] MutationScope mutation_scope(
    std::string_view tool_name,
    std::string_view json_args,
    const core::context::SessionContext& context,
    const core::tools::ToolAnnotations& annotations);

} // namespace core::changes

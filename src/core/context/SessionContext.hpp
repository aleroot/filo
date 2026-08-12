#pragma once

#include "../memory/MemoryPolicy.hpp"
#include "../workspace/SessionWorkspace.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace core::workspace {
class PathVisibility;
}

namespace core::context {

enum class SessionTransport {
    unspecified,
    cli,
    mcp_stdio,
    mcp_http,
    mcp_client,
};

struct SessionContext {
    SessionContext(std::string session_id,
                   core::workspace::SessionWorkspace workspace,
                   SessionTransport transport = SessionTransport::unspecified,
                   std::shared_ptr<const core::workspace::PathVisibility> path_visibility = {})
        : session_id(std::move(session_id))
        , workspace(std::move(workspace))
        , transport(transport)
        , path_visibility(std::move(path_visibility)) {}

    std::string session_id;
    core::workspace::SessionWorkspace workspace;
    SessionTransport transport;
    std::shared_ptr<const core::workspace::PathVisibility> path_visibility;
    core::memory::MemoryThreadPolicy memory_policy;

    [[nodiscard]] const core::workspace::SessionWorkspace& workspace_view() const noexcept;
    [[nodiscard]] const core::workspace::WorkspaceSnapshot& effective_workspace() const noexcept;
    [[nodiscard]] std::filesystem::path resolve_path(
        const std::filesystem::path& target_path) const;
    [[nodiscard]] bool allows_read(const std::filesystem::path& target_path) const;
    [[nodiscard]] bool allows_write(const std::filesystem::path& target_path) const;

    // Extends the workspace and invalidates visibility derived from its roots.
    std::size_t extend_workspace(
        const std::vector<std::filesystem::path>& paths);

    // Replaces the primary workspace root and invalidates visibility derived
    // from it. Returns false when the candidate directory is invalid or
    // already the current primary; see SessionWorkspace::set_primary.
    bool set_workspace_primary(const std::filesystem::path& new_primary);
};

[[nodiscard]] SessionContext make_session_context(
    core::workspace::WorkspaceSnapshot snapshot,
    SessionTransport transport = SessionTransport::unspecified,
    std::string session_id = {});

} // namespace core::context

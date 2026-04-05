#pragma once

#include "../workspace/SessionWorkspace.hpp"

#include <filesystem>
#include <string>
#include <utility>

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
                   SessionTransport transport = SessionTransport::unspecified)
        : session_id(std::move(session_id))
        , workspace(std::move(workspace))
        , transport(transport) {}

    std::string session_id;
    core::workspace::SessionWorkspace workspace;
    SessionTransport transport;

    [[nodiscard]] const core::workspace::SessionWorkspace& workspace_view() const noexcept;
    [[nodiscard]] const core::workspace::WorkspaceSnapshot& effective_workspace() const noexcept;
    [[nodiscard]] std::filesystem::path resolve_path(
        const std::filesystem::path& target_path) const;
    [[nodiscard]] bool is_path_allowed(const std::filesystem::path& target_path) const;
};

[[nodiscard]] SessionContext make_session_context(
    core::workspace::WorkspaceSnapshot snapshot,
    SessionTransport transport = SessionTransport::unspecified,
    std::string session_id = {});

} // namespace core::context

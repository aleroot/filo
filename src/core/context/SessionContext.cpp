#include "SessionContext.hpp"

namespace core::context {

const core::workspace::SessionWorkspace& SessionContext::workspace_view() const noexcept {
    return workspace;
}

const core::workspace::WorkspaceSnapshot& SessionContext::effective_workspace() const noexcept {
    return workspace.snapshot();
}

std::filesystem::path SessionContext::resolve_path(const std::filesystem::path& target_path) const {
    return workspace_view().resolve_path(target_path);
}

bool SessionContext::allows_read(const std::filesystem::path& target_path) const {
    return workspace_view().allows_read(target_path);
}

bool SessionContext::allows_write(const std::filesystem::path& target_path) const {
    return workspace_view().allows_write(target_path);
}

std::size_t SessionContext::extend_workspace(
    const std::vector<std::filesystem::path>& paths) {
    const auto added = workspace.add_additional_paths(paths);
    if (added > 0) {
        path_visibility.reset();
    }
    return added;
}

bool SessionContext::set_workspace_primary(const std::filesystem::path& new_primary) {
    const bool changed = workspace.set_primary(new_primary);
    if (changed) {
        path_visibility.reset();
    }
    return changed;
}

SessionContext make_session_context(core::workspace::WorkspaceSnapshot snapshot,
                                    SessionTransport transport,
                                    std::string session_id) {
    return SessionContext(
        std::move(session_id),
        core::workspace::SessionWorkspace(std::move(snapshot)),
        transport);
}

} // namespace core::context

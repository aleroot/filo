#include "Workspace.hpp"
#include "SessionWorkspace.hpp"

namespace core::workspace {

void Workspace::initialize(std::filesystem::path primary, std::vector<std::filesystem::path> additional, bool enforce) {
    const auto normalized = SessionWorkspace::normalize_snapshot(WorkspaceSnapshot{
        .primary = std::move(primary),
        .additional = std::move(additional),
        .enforce = enforce,
        .version = 0,
    });

    primary_ = normalized.primary;
    additional_ = normalized.additional;
    enforce_ = normalized.enforce;
}

WorkspaceSnapshot Workspace::snapshot() const {
    return WorkspaceSnapshot{
        .primary = primary_,
        .additional = additional_,
        .enforce = enforce_,
        .version = 0,
    };
}

std::filesystem::path Workspace::resolve_path(const std::filesystem::path& target_path) const {
    return SessionWorkspace(snapshot()).resolve_path(target_path);
}

bool Workspace::is_path_allowed(const std::filesystem::path& target_path) const {
    return SessionWorkspace(snapshot()).is_path_allowed(target_path);
}

} // namespace core::workspace

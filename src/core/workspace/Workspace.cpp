#include "Workspace.hpp"
#include "SessionWorkspace.hpp"

namespace core::workspace {

void Workspace::initialize(std::filesystem::path primary,
                           std::vector<std::filesystem::path> additional,
                           bool enforce,
                           FileAccessScope scratch) {
    const auto normalized = SessionWorkspace::normalize_snapshot(WorkspaceSnapshot{
        .primary = std::move(primary),
        .additional = std::move(additional),
        .enforce = enforce,
        .version = 0,
        .scratch = std::move(scratch),
    });

    primary_ = normalized.primary;
    additional_ = normalized.additional;
    enforce_ = normalized.enforce;
    scratch_ = normalized.scratch;
}

WorkspaceSnapshot Workspace::snapshot() const {
    return WorkspaceSnapshot{
        .primary = primary_,
        .additional = additional_,
        .enforce = enforce_,
        .version = 0,
        .scratch = scratch_,
    };
}

std::vector<std::filesystem::path> Workspace::ordered_roots() const {
    return core::workspace::ordered_roots(primary_, additional_);
}

std::filesystem::path Workspace::resolve_path(const std::filesystem::path& target_path) const {
    return SessionWorkspace(snapshot()).resolve_path(target_path);
}

bool Workspace::allows_read(const std::filesystem::path& target_path) const {
    return SessionWorkspace(snapshot()).allows_read(target_path);
}

bool Workspace::allows_write(const std::filesystem::path& target_path) const {
    return SessionWorkspace(snapshot()).allows_write(target_path);
}

} // namespace core::workspace
